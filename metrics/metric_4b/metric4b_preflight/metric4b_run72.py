"""Reproducible, resumable Metric 4b 72-sector NSGA-II runner.

Place beside metric4b_preflight.py in bird_layout_opt/metric4b_preflight.
Keep the original `model9_code`, `original_results`, and `preflight_reports` there.

   python metric4b_run72.py verify
   python metric4b_run72.py smoke
   python metric4b_run72.py run       # only after approval
   python metric4b_run72.py resume    # interrupted production job only

The ONLY independent scientific-setting change is wind.n_sectors = 12 -> 72.
Diagnostics, serialised checkpoints and archiving do not change objectives.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'model9_code'
ORIGINAL = ROOT / 'original_results'
REPORTS = ROOT / 'preflight_reports'
OUTPUT = ROOT.parent / 'metric_4b_72sectors'
for directory in (str(CODE), str(ROOT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from config import Scenario, WindResource, OptSettings
from floris_aep import FLORISAEPModel
from model8_meso_response import BehaviourScenario
from model9_rotor_transit import (
    DiscreteFlightHeightDistribution, RotorGeometry, RotorTransitModel,
)
from model9_batching import summarise_model9_batches
from model9_aep_problem import Model9AEPRotorTransitProblem
from model9_optimisation_utils import (
    coordinate_columns, layout_frame, load_layout_csv,
    load_initial_population, validate_layout, serialisable,
)
from run_model9_optimisation import make_populations
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.termination import get_termination

try:
    import cloudpickle as serializer
    SERIALIZER = 'cloudpickle'
except ImportError:
    serializer = pickle
    SERIALIZER = 'pickle'

SECTORS = 72
N_CONDITIONS = 504
POPULATION = 80
GENERATIONS = 100
SEED = 142
BATCHES = 5
BIRDS_PER_BATCH = 2000
FRESH_BATCHES_IN_ORIGINAL_ARCHIVE = 5
FRESH_BIRDS_IN_ORIGINAL_ARCHIVE = 10000
CHECKPOINT_EVERY = 5
AEP_CEILING = 854.4956585617186
AEP_LOSS_REFERENCE = 137.96914764866568
BIRD_REFERENCE = 8.704996418223796e-4
HV_REFERENCE = [2.0, 2.0]  # diagnostic only; not used by selection
FHD = CODE / 'data' / 'northern_gannet_Johnston2014_FHD.csv'
FHD_METADATA = CODE / 'data' / 'northern_gannet_Johnston2014_FHD_metadata.json'
FLORIS_CONFIG = CODE / 'floris_config' / 'gch_iea15mw.yaml'
TURBINE_REFERENCE = CODE / 'floris_config' / 'iea_15MW_REFERENCE.yaml'
ENERGY_LAYOUT = CODE / 'data' / 'energy_endpoint_layout.csv'
INITIAL = CODE / 'data' / 'model8_initial_population.csv'
ORIGINAL_INITIAL = ORIGINAL / 'selected_initial_population.csv'
ORIGINAL_SETTINGS = ORIGINAL / 'run_settings.json'


class PreflightStop(RuntimeError):
    pass


def require(condition: bool, message: str):
    if not condition:
        raise PreflightStop(message)


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def tracked_files():
    paths = [
        Path(__file__).resolve(), ORIGINAL_SETTINGS, ORIGINAL_INITIAL,
        ORIGINAL / 'pareto_decision_vectors.npy',
        INITIAL, ENERGY_LAYOUT, FHD, FHD_METADATA, FLORIS_CONFIG,
        TURBINE_REFERENCE, CODE / 'data' / 'model8_reference_run_settings.json',
    ]
    paths.extend(sorted(CODE.glob('*.py')))
    return sorted(set(p.resolve() for p in paths))


def hashes() -> dict[str, str]:
    out = {}
    for path in tracked_files():
        require(path.is_file(), f'Missing source/input: {path}')
        out[path.relative_to(ROOT).as_posix()] = sha256(path)
    return out


def archive_settings() -> dict:
    require(ORIGINAL_SETTINGS.is_file(), 'Original settings not found')
    return json.loads(ORIGINAL_SETTINGS.read_text(encoding='utf-8'))


def approx_equal(a, b, tol=5e-13):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(approx_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(approx_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(np.isclose(a, b, atol=tol, rtol=0))
    return a == b


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.temporary')
    with temporary.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def write_json(path: Path, data: Any):
    atomic_bytes(path, json.dumps(serialisable(data), indent=2, default=str).encode('utf-8'))


class DiagnosticCallback(Callback):
    """Read-only per-generation diagnostics. NSGA-II uses its default operators."""
    def __init__(self):
        super().__init__()
        self._hv = HV(ref_point=np.array(HV_REFERENCE))

    def notify(self, algorithm):
        optimum = algorithm.opt
        f = np.asarray(optimum.get('F'), dtype=float).reshape(-1, 2)
        finite = f[np.all(np.isfinite(f), axis=1)]
        generation = int(algorithm.n_gen)
        values = {
            'generation': generation,
            'function_evaluations': int(algorithm.evaluator.n_eval),
            'pareto_size': int(len(finite)),
            'minimum_normalised_aep_loss': float(np.min(finite[:, 0])) if len(finite) else float('nan'),
            'minimum_normalised_Q_4b': float(np.min(finite[:, 1])) if len(finite) else float('nan'),
            'hypervolume': float(self._hv.do(finite)) if len(finite) else float('nan'),
        }
        for key, val in values.items():
            self.data.setdefault(key, []).append(val)


def setup(*, smoke=False, record_evaluations=True):
    old = archive_settings()
    require(old['seed'] == SEED, 'Original seed differs')
    profile = old['profile']
    require(profile['population'] == POPULATION and profile['generations'] == GENERATIONS,
            'Original population/generations differ')
    require(profile['objective_batches'] == BATCHES and
            profile['objective_trajectories_per_batch'] == BIRDS_PER_BATCH,
            'Original optimisation sample differs')
    require(profile['fresh_batches'] == FRESH_BATCHES_IN_ORIGINAL_ARCHIVE and
            profile['fresh_trajectories_per_batch'] == FRESH_BIRDS_IN_ORIGINAL_ARCHIVE,
            'Original follow-up validation profile differs')
    require(old['objective_trajectory_seeds'] == list(range(SEED, SEED + BATCHES)),
            'Original trajectory seeds differ')
    require(old['objective_flight_height_seeds'] == list(range(10_000 + SEED, 10_000 + SEED + BATCHES)),
            'Original flight-height seeds differ')
    require(package_version('pymoo') == old['runtime_versions']['pymoo'] == '0.6.1.5',
            'pymoo version mismatch')
    require(package_version('floris') == old['runtime_versions']['floris'] == '4.6.5',
            'FLORIS version mismatch')

    scenario = Scenario(
        wind=WindResource(n_sectors=SECTORS),
        opt=OptSettings(pop_size=POPULATION, n_gen=GENERATIONS, seed=SEED,
                        objective_batch_count=BATCHES,
                        objective_trajectories_per_batch=BIRDS_PER_BATCH,
                        fresh_batch_count=FRESH_BATCHES_IN_ORIGINAL_ARCHIVE,
                        fresh_trajectories_per_batch=FRESH_BIRDS_IN_ORIGINAL_ARCHIVE),
    )
    aep = FLORISAEPModel(scenario, FLORIS_CONFIG, turbine_reference_path=TURBINE_REFERENCE)
    require(aep.number_of_conditions == N_CONDITIONS, 'Expected 504 conditions')
    ceiling = float(aep.ideal_aep(scenario.site.n_turbines))
    require(abs(ceiling - AEP_CEILING) <= 1e-7, 'Wake-free AEP ceiling differs')
    require(abs(float(old['problem']['aep_loss_reference_gwh']) - AEP_LOSS_REFERENCE) < 1e-12,
            'AEP scaling reference differs')
    require(abs(float(old['problem']['rotor_transit_reference']) - BIRD_REFERENCE) <= 1e-16,
            'Bird scaling reference differs')
    require(sha256(FLORIS_CONFIG) == old['aep_model']['config_sha256'],
            'FLORIS YAML differs from original')
    require(sha256(FHD) == old['flight_height_distribution']['input_sha256'],
            'FHD file differs from original')

    metadata = json.loads(FHD_METADATA.read_text(encoding='utf-8'))
    fhd = DiscreteFlightHeightDistribution.from_csv(
        FHD, name='northern_gannet_Johnston2014_Gannet.est',
        source=metadata['primary_citation'])
    require(fhd.input_sha256 == metadata['sha256'], 'FHD metadata SHA-256 mismatch')
    rotor = RotorGeometry(rotor_diameter_m=aep.floris_rotor_diameter_m,
                          hub_height_m=aep.floris_hub_height_m)
    birds = scenario.birds
    behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=rotor.rotor_diameter_m,
        response_probability=birds.response_probability,
        response_radius_D=birds.response_radius_D,
        encounter_radius_D=birds.encounter_radius_D,
        deflection_angle_deg=birds.deflection_angle_deg,
        return_length_D=birds.return_length_D,
        step_size_D=birds.step_size_D)
    model = RotorTransitModel(
        behaviour, rotor,
        rotor_plane_heading_deg_toward=birds.heading_deg_toward,
        flight_height_distribution=fhd)
    populations = make_populations(
        distribution=fhd, scenario=scenario, behaviour=behaviour,
        batch_count=(1 if smoke else BATCHES),
        trajectories_per_batch=(120 if smoke else BIRDS_PER_BATCH),
        first_trajectory_seed=SEED, first_height_seed=10_000 + SEED)
    minimum_spacing = 4.0 * rotor.rotor_diameter_m
    baseline = load_layout_csv(
        ENERGY_LAYOUT, number_of_turbines=12, site_width_m=6000.,
        site_height_m=6000., minimum_spacing_m=minimum_spacing)
    if not smoke:
        result = summarise_model9_batches(model, baseline, populations)
        computed_reference = float(result['E_RT_CE'])
        require(result['N_failed'] == 0, 'Baseline bird simulations failed')
        require(abs(computed_reference - BIRD_REFERENCE) <= 5e-16,
                f'Fixed Q_4b reference mismatches: {computed_reference!r}')
    problem = Model9AEPRotorTransitProblem(
        scenario=scenario, aep_model=aep, rotor_transit_model=model,
        trajectory_populations=populations, aep_ceiling_gwh=ceiling,
        aep_loss_reference_gwh=AEP_LOSS_REFERENCE,
        rotor_transit_reference=BIRD_REFERENCE,
        bird_objective_metric='E_RT_CE', record_evaluations=record_evaluations)
    vectors, labels, rows = load_initial_population(
        INITIAL, population_size=POPULATION, seed=SEED,
        number_of_turbines=12, site_width_m=6000.,
        site_height_m=6000., minimum_spacing_m=minimum_spacing)
    saved = pd.read_csv(ORIGINAL_INITIAL)
    archived_x = saved[coordinate_columns(12)].to_numpy(dtype=float)
    require(archived_x.shape == vectors.shape and
            np.max(np.abs(archived_x - vectors)) <= 1e-9,
            'Original initial population not reproduced')
    require(saved['seed_source'].astype(str).tolist() == labels,
            'Original initial-population family labels differ')
    require(rows == old['initial_population_source_rows'],
            'Original initial-population source order differs')
    return old, problem, vectors, labels, rows, baseline


def scientific_diff(old: dict, problem: Model9AEPRotorTransitProblem) -> list[dict]:
    """All recorded scientific fields must agree except derived wind arrays."""
    new_p = problem.settings_summary()
    new_a = problem.aep_model.settings_summary()
    changed = []
    for name, prior in old['problem'].items():
        if name not in new_p or not approx_equal(prior, new_p[name]):
            changed.append({'path': 'problem.' + name, 'original': prior,
                            'new': new_p.get(name, 'MISSING')})
    derived_from_n_sectors = {
        'number_of_direction_sectors', 'number_of_combined_conditions',
        'wind_directions_deg', 'wind_speeds_ms', 'condition_frequencies',
    }
    expected_environment_paths = {'config_path','installed_turbine_path',
                                  'bundled_turbine_reference_path'}
    for name, prior in old['aep_model'].items():
        if name in derived_from_n_sectors or name in expected_environment_paths:
            continue
        if name not in new_a or not approx_equal(prior, new_a[name]):
            changed.append({'path': 'aep_model.' + name, 'original': prior,
                            'new': new_a.get(name, 'MISSING')})
    if not (old['aep_model']['number_of_direction_sectors'] == 12 and
            new_a['number_of_direction_sectors'] == 72 and
            old['aep_model']['number_of_combined_conditions'] == 84 and
            new_a['number_of_combined_conditions'] == 504):
        changed.append({'path': 'wind_direction_grid',
                        'original': [old['aep_model']['number_of_direction_sectors'],
                                     old['aep_model']['number_of_combined_conditions']],
                        'new': [new_a['number_of_direction_sectors'],
                                new_a['number_of_combined_conditions']]})
    # Wind sector count is the only independent scientific change.
    return changed


def proposed_settings(old: dict, problem: Model9AEPRotorTransitProblem):
    new = copy.deepcopy(old)
    for k in ('anchors','anchor_definitions','anchor_selection_note',
              'elapsed_seconds','function_evaluations','training_pareto_size',
              'fresh_nondominated_count','robust_nondominated_95pct_count'):
        new.pop(k, None)
    new['run_status'] = 'prepared; not yet independently evaluated'
    new['model'] = 'Metric 4b — FLORIS AEP versus rotor-transit intensity'
    new['objective_labels'] = ['AEP', 'Q_4b (rotor transits per entering bird)']
    new['n_sectors'] = SECTORS
    new['pymoo_version'] = package_version('pymoo')
    new['floris_version'] = package_version('floris')
    new['python_version'] = platform.python_version()
    new['aep_model'] = problem.aep_model.settings_summary()
    new['problem'] = problem.settings_summary()
    new['wind_input'] = 'programmatic 72-sector x 7-speed-bin weights; no wind CSV'
    new['checkpoint_interval_generations'] = CHECKPOINT_EVERY
    new['checkpoint_serializer'] = SERIALIZER
    new['hypervolume_reference_normalised'] = HV_REFERENCE
    new['archived_pymoo_version'] = old['runtime_versions']['pymoo']
    new['source_provenance_note'] = (
        'Original optimisation records code v4; supplied modules are v7. '
        'Two archived 12-sector endpoint bird scores and AEP were numerically '
        'reproduced in local pre-flight, but byte-for-byte v4 source equivalence '
        'cannot be established.')
    new['independent_confirmation'] = 'Not performed by optimisation runner'
    new['source_file_hashes'] = 'See file_hashes.json'
    return new


def algorithm_with_saved_population(vectors, *, pop=POPULATION):
    # Exactly the original NSGA-II operator invocation: inherited pymoo defaults.
    return NSGA2(pop_size=pop, sampling=vectors[:pop].copy(), eliminate_duplicates=True)


def save_checkpoint(algorithm, path: Path, manifest: dict):
    state = {
        'version': 1,
        'algorithm': algorithm,
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
        'hashes': manifest,
        'pymoo': package_version('pymoo'),
        'generation': (int(algorithm.callback.data['generation'][-1])
                       if algorithm.callback and algorithm.callback.data.get('generation') else 0),
    }
    buffer = serializer.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    previous = path.with_name('checkpoint_previous.pkl')
    if path.is_file():
        atomic_bytes(previous, path.read_bytes())
    atomic_bytes(path, buffer)
    with path.open('rb') as f:
        saved = pickle.load(f)
    require(saved['generation'] == state['generation'], 'Saved checkpoint failed read-back')


def load_checkpoint(path: Path, manifest: dict):
    require(path.is_file(), f'Missing checkpoint {path}')
    with path.open('rb') as f:
        state = pickle.load(f)
    require(state['version'] == 1, 'Checkpoint schema mismatch')
    require(state['hashes'] == manifest, 'Code/input SHA-256 changed since run; STOP')
    require(state['pymoo'] == package_version('pymoo'), 'Installed pymoo version changed; STOP')
    np.random.set_state(state['numpy_random_state'])
    random.setstate(state['python_random_state'])
    return state['algorithm']


def callback_equal(a, b):
    for key in a:
        if key not in b or not approx_equal(a[key], b[key], tol=0):
            return False
    return a.keys() == b.keys()


def verify():
    print('VERIFY | Metric 4b 72-sector production settings (no NSGA-II)', flush=True)
    REPORTS.mkdir(exist_ok=True)
    for report in ('audit_report.json', 'bird_report.json', 'aep_report.json', 'benchmark_report.json'):
        path = REPORTS / report
        require(path.exists() and json.loads(path.read_text(encoding='utf-8'))['status'] == 'PASS'
                if report != 'benchmark_report.json' else path.exists(),
                f'Pre-flight report missing or failed: {report}')
    old, problem, vectors, labels, rows, baseline = setup()
    differences = scientific_diff(old, problem)
    require(not differences, f'Scientific-setting mismatches: {differences}')
    manifest = hashes()
    audit = {
        'status': 'PASS',
        'original_sectors': 12, 'requested_sectors': 72,
        'original_conditions': 84, 'new_conditions': 504,
        'other_scientific_differences': differences,
        'original_fhd_hash': old['flight_height_distribution']['input_sha256'],
        'original_yaml_hash': old['aep_model']['config_sha256'],
        'original_pymoo': old['runtime_versions']['pymoo'],
        'installed_pymoo': package_version('pymoo'),
        'initial_population_max_abs_difference_m': float(np.max(np.abs(vectors - pd.read_csv(ORIGINAL_INITIAL)[coordinate_columns(12)].to_numpy(dtype=float)))),
        'source_hashes': manifest,
        'checkpoint_resume_test': 'PENDING',
        'proposed_settings': proposed_settings(old, problem),
    }
    write_json(REPORTS / 'run72_verify.json', audit)
    print('PASS | All source-recorded scientific settings identical except 12 -> 72 sectors')
    print('PASS | 80 initial layouts, optimiser seed 142, 5x2,000 optimisation sample')
    print('PASS | Locked Q_4b reference and FLORIS wake-free AEP')
    print('PASS | Archived and current pymoo 0.6.1.5, FLORIS 4.6.5')
    print('INFO | v4->v7 code provenance cannot be proven byte-for-byte')
    print('NO OPTIMISATION STARTED')


def smoke():
    print('SMOKE | 8-individual, 3-generation checkpoint equivalence test', flush=True)
    path = REPORTS / 'run72_verify.json'
    require(path.exists(), 'Run verify first')
    audit = json.loads(path.read_text(encoding='utf-8'))
    require(audit['status'] == 'PASS' and audit['source_hashes'] == hashes(),
            'Sources changed since verify')
    # Only the smoke test uses fewer sampled birds and fewer generations.
    _old, p1, initial1, _labels, _rows, _baseline = setup(smoke=True, record_evaluations=False)
    ref = minimize(p1, algorithm_with_saved_population(initial1, pop=8),
                   termination=('n_gen',3), seed=SEED, callback=DiagnosticCallback(),
                   save_history=False, verbose=False)
    _old2, p2, initial2, _labels2, _rows2, _baseline2 = setup(smoke=True, record_evaluations=False)
    algorithm = algorithm_with_saved_population(initial2, pop=8)
    algorithm.setup(p2, termination=get_termination('n_gen',3), seed=SEED,
                    callback=DiagnosticCallback(), verbose=False)
    algorithm.next()
    checkpoint = REPORTS / 'run72_smoke_checkpoint.pkl'
    save_checkpoint(algorithm, checkpoint, hashes())
    del algorithm
    algorithm = load_checkpoint(checkpoint, hashes())
    while algorithm.has_next():
        algorithm.next()
    equal_x = np.array_equal(np.asarray(algorithm.pop.get('X')), np.asarray(ref.algorithm.pop.get('X')))
    equal_f = np.array_equal(np.asarray(algorithm.pop.get('F')), np.asarray(ref.algorithm.pop.get('F')))
    equal_hist = callback_equal(algorithm.callback.data, ref.algorithm.callback.data)
    audit['checkpoint_resume_test'] = 'PASS' if (equal_x and equal_f and equal_hist) else 'FAIL'
    audit['smoke_equal_final_X_exact'] = bool(equal_x)
    audit['smoke_equal_final_F_exact'] = bool(equal_f)
    audit['smoke_equal_convergence_history_exact'] = bool(equal_hist)
    write_json(path, audit)
    require(equal_x and equal_f and equal_hist, 'Checkpoint/resume differs from uninterrupted NSGA-II; STOP')
    print('PASS | Exact final population X, objective F, and convergence history')
    print('PASS | Checkpoint round-trip from disk')
    print('NO PRODUCTION OPTIMISATION STARTED')


def write_results(algorithm, output: Path):
    result = algorithm.result()
    require(result.X is not None and result.F is not None, 'No feasible Pareto solutions')
    x = np.atleast_2d(np.asarray(result.X, dtype=float))
    f = np.atleast_2d(np.asarray(result.F, dtype=float))
    np.save(output / 'pareto_decision_vectors.npy', x)
    np.save(output / 'pareto_normalised_objectives.npy', f)
    pop = algorithm.pop
    np.savez_compressed(output / 'final_population.npz',
                        X=np.asarray(pop.get('X'), dtype=float),
                        F=np.asarray(pop.get('F'), dtype=float),
                        G=np.asarray(pop.get('G'), dtype=float),
                        CV=np.asarray(pop.get('CV'), dtype=float))
    columns = coordinate_columns(12)
    full = pd.DataFrame(x, columns=columns)
    full['solution_index'] = np.arange(len(x))
    full['aep_gwh'] = AEP_CEILING - AEP_LOSS_REFERENCE * f[:,0]
    full['Q_4b'] = BIRD_REFERENCE * f[:,1]
    full['f0_normalised_aep_loss'] = f[:,0]
    full['f1_normalised_Q_4b'] = f[:,1]
    full.to_csv(output / 'pareto_points.csv', index=False)
    for label, idx in [('highest-AEP', full['aep_gwh'].idxmax()),
                       ('lowest-risk', full['Q_4b'].idxmin())]:
        layout_frame(x[int(idx)].reshape(12,2)).to_csv(output / f'{label}_layout.csv', index=False)
    pd.DataFrame(algorithm.callback.data).to_csv(output / 'hypervolume.csv', index=False)
    algorithm.problem.evaluation_dataframe().to_csv(output / 'all_evaluated_candidates.csv', index=False)
    last_gen = int(algorithm.callback.data['generation'][-1])
    evaluations = int(algorithm.evaluator.n_eval)
    require(last_gen == GENERATIONS and evaluations == GENERATIONS * POPULATION,
            f'Run did not complete 100 x 80: gen={last_gen}, eval={evaluations}')
    write_json(output / 'training_summary.json', {
        'terminology': 'Optimisation sample; independent evaluation sample',
        'non_dominated_layouts': int(len(full)),
        'highest_aep_gwh': float(full.aep_gwh.max()),
        'lowest_risk_Q_4b': float(full.Q_4b.min()),
        'generations': last_gen, 'function_evaluations': evaluations,
        'final_population_size': int(len(pop)),
        'original_followup_validation': '5 x 10000; not run by this production driver',
        'independent_confirmation': 'Not run; protocol is 15 x 100000 birds/layout with paired p=0.6, p=0 and fixed height z=150 m',
    })
    print(f'Saved {len(full)} non-dominated layouts and final 80-person population.', flush=True)


def run(resume=False):
    path = REPORTS / 'run72_verify.json'
    require(path.exists(), 'Run verify then smoke before production')
    checked = json.loads(path.read_text(encoding='utf-8'))
    require(checked['status'] == 'PASS' and checked['checkpoint_resume_test'] == 'PASS',
            'Verify and checkpoint smoke must PASS')
    manifest = hashes()
    require(checked['source_hashes'] == manifest, 'Code/inputs changed since verification')
    current_checkpoint = OUTPUT / 'checkpoint_latest.pkl'
    if not resume:
        require(not OUTPUT.exists(), f'Output already exists; refusing to overwrite: {OUTPUT}')
        old, problem, vectors, labels, rows, baseline = setup()
        mismatches = scientific_diff(old, problem)
        require(not mismatches, f'Unexpected scientific mismatch: {mismatches}')
        OUTPUT.mkdir(parents=False, exist_ok=False)
        write_json(OUTPUT / 'file_hashes.json', manifest)
        settings = proposed_settings(old, problem)
        settings['run_status'] = 'optimisation_started'
        write_json(OUTPUT / 'run_settings.json', settings)
        write_json(OUTPUT / 'settings_diff.json', {
            'independent_scientific_change': {'n_sectors': [12,72]},
            'derived_changes': {'wind_conditions': [84,504],
                                'wind_directions_deg': 'regenerated 5-degree spacing',
                                'condition_frequencies': 'recomputed from same weighting formula'},
            'other_scientific_differences': mismatches,
            'operational_additions': ['full resumable checkpoint', 'file hashes',
                                      'full final population', 'diagnostic hypervolume'],
            'source_provenance_limitation': 'Uploaded v7 source; original was v4',
        })
        init = pd.DataFrame(vectors, columns=coordinate_columns(12))
        init['seed_source'] = labels
        init['source_row_model8'] = rows
        init.to_csv(OUTPUT / 'selected_initial_population.csv', index=False)
        layout_frame(baseline).to_csv(OUTPUT / 'baseline_energy_layout.csv', index=False)
        algorithm = algorithm_with_saved_population(vectors)
        algorithm.setup(problem, termination=get_termination('n_gen',GENERATIONS),
                        seed=SEED, callback=DiagnosticCallback(), verbose=False)
        save_checkpoint(algorithm, current_checkpoint, manifest)
        print(f'START | New output folder: {OUTPUT}', flush=True)
    else:
        require(OUTPUT.is_dir(), 'No production run folder to resume')
        require(json.loads((OUTPUT / 'file_hashes.json').read_text(encoding='utf-8')) == manifest,
                'Production source hashes mismatch')
        try:
            algorithm = load_checkpoint(current_checkpoint, manifest)
        except Exception as exc:
            print('WARNING | Latest checkpoint not readable:', str(exc), flush=True)
            algorithm = load_checkpoint(OUTPUT / 'checkpoint_previous.pkl', manifest)
        done = algorithm.callback.data.get('generation', [])
        print(f'RESUME | Saved generation: {done[-1] if done else 0}', flush=True)
    started = time.perf_counter()
    while algorithm.has_next():
        algorithm.next()
        done = int(algorithm.callback.data['generation'][-1])
        print(f'Generation {done}/{GENERATIONS}; elapsed this session {(time.perf_counter()-started)/60:.1f} min', flush=True)
        if done % CHECKPOINT_EVERY == 0 or not algorithm.has_next():
            save_checkpoint(algorithm, current_checkpoint, manifest)
            print(f'  Checkpoint saved after generation {done}', flush=True)
    print('OPTIMISATION COMPLETE | Writing Pareto layouts and final population', flush=True)
    write_results(algorithm, OUTPUT)
    settings = json.loads((OUTPUT / 'run_settings.json').read_text(encoding='utf-8'))
    settings['run_status'] = 'completed'
    settings['function_evaluations'] = algorithm.evaluator.n_eval
    settings['training_pareto_size'] = len(np.atleast_2d(algorithm.result().X))
    write_json(OUTPUT / 'run_settings.json', settings)
    write_json(OUTPUT / 'completion_status.json', {
        'status': 'optimisation_finished',
        'n_sectors': SECTORS, 'generations': GENERATIONS,
        'function_evaluations': int(algorithm.evaluator.n_eval),
        'independent_evaluation': 'not yet run',
    })
    print('DONE | Optimisation complete. Separate independent evaluation is next.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('verify','smoke','run','resume'))
    choice = parser.parse_args().command
    try:
        if choice == 'verify':
            verify()
        elif choice == 'smoke':
            smoke()
        elif choice == 'run':
            run(resume=False)
        else:
            run(resume=True)
    except Exception as exc:
        REPORTS.mkdir(parents=True, exist_ok=True)
        error_path = REPORTS / f'run72_{choice}_ERROR.txt'
        error_path.write_text(traceback.format_exc(), encoding='utf-8')
        print('STOP |', type(exc).__name__, str(exc), flush=True)
        print('ERROR DETAILS:', error_path, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
