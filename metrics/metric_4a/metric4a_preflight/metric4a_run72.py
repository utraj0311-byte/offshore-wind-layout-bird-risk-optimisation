"""Metric 4a, 72-sector reproducible NSGA-II runner.

Add this script beside metric4a_preflight.py inside the already verified
bird_layout_opt/metric4a_preflight directory. Unmodified scientific modules
and original result archives remain in that directory.

Commands (from that directory, in floris-env):
    python metric4a_run72.py verify
    python metric4a_run72.py smoke
    python metric4a_run72.py run       # ONLY AFTER approving pre-flight
    python metric4a_run72.py resume    # after an interrupted run

Scientific difference from archived run: wind.n_sectors 12 -> 72 ONLY.
Checkpointing and reporting are operational changes, not model changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import pickle
import platform
import random
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'model8_code'
for entry in (ROOT, SOURCE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from config import Scenario
from floris_aep import FLORISAEPModel
from model8_aep_problem import Model8AEPEncounterProblem
from model8_batching import summarise_meso_batches
from model8_meso_response import BehaviourScenario, MesoResponseModel
from run_model8_multiobjective import (
    FeasibleSeededSampling, HypervolumeCallback, build_pareto_dataframe,
    fresh_validate_pareto, generate_population_batches, load_layout_csv,
    layout_is_feasible,
)
from run_model8_validation import make_fixed_layouts
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

try:
    import cloudpickle as serialiser
    SERIALISER = 'cloudpickle'
except ImportError:
    serialiser = pickle
    SERIALISER = 'pickle'

SECTORS = 72
POP = 80
GENERATIONS = 100
SEED = 42
LOCKED_AEP_CEILING = 854.4956585617186
LOCKED_AEP_REF = 137.96914764866568
LOCKED_BIRD_REF = 0.6854
CHECKPOINT_INTERVAL = 5
ARCHIVE = ROOT / 'archives' / 'original_optimisation.zip'
ORIGINAL_CONFIRMATION = ROOT / 'archives' / 'original_confirmation.zip'
CONFIG_YAML = ROOT / 'floris_config' / 'gch_iea15mw.yaml'
ENERGY_LAYOUT = ROOT / 'model10_energy_seed_layout.csv'
REPORTS = ROOT / 'preflight_reports'
OUTPUT = ROOT.parent / 'metric_4a_72sectors'
FILES_TO_HASH = [
    ROOT / 'config.py', ROOT / 'floris_aep.py', CONFIG_YAML, ENERGY_LAYOUT,
    ARCHIVE, ORIGINAL_CONFIRMATION,
    *sorted(SOURCE.glob('*.py')), Path(__file__).resolve(),
]


class PreflightStop(RuntimeError):
    pass


def assert_true(condition, message):
    if not condition:
        raise PreflightStop(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hashes():
    return {str(f.relative_to(ROOT)).replace('\\', '/'): sha256(f)
            for f in FILES_TO_HASH}


def load_archive(name):
    with zipfile.ZipFile(ARCHIVE) as file:
        entries = [f for f in file.namelist() if f.endswith('/' + name)]
        assert_true(len(entries) == 1, 'Cannot uniquely locate original ' + name)
        return file.read(entries[0])


def original_data():
    settings = json.loads(load_archive('run_settings.json'))
    initial = pd.read_csv(io.BytesIO(load_archive('initial_population.csv')))
    return settings, initial


def version(package):
    return importlib.metadata.version(package)


def equal(a, b, tol=1e-10):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equal(a[k], b[k], tol) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return bool(np.isclose(a, b, atol=tol, rtol=0.0))
    return a == b


def make_setup(*, smoke=False, record_evaluations=True):
    old, original_initial = original_data()
    scenario = Scenario()
    scenario.site.n_turbines = 12
    scenario.site.width_m = 6000.0
    scenario.site.height_m = 6000.0
    scenario.site.min_spacing_D = 4.0
    scenario.site.migration_axis_deg = 225.0
    scenario.wind.n_sectors = SECTORS
    # No change to direction centre, speed bins, Weibull weights, FLORIS YAML.
    aep = FLORISAEPModel(scenario, config_path=CONFIG_YAML)
    D = float(aep.floris_rotor_diameter_m)
    scenario.turbine.rotor_diameter_m = D
    assert_true(aep.number_of_conditions == 504, 'Not 504 FLORIS wind conditions')
    ceiling = float(aep.ideal_aep(12))
    assert_true(abs(ceiling - LOCKED_AEP_CEILING) <= 1e-7,
                f'Wake-free AEP changed: {ceiling:.12f}')
    assert_true(version('floris') == '4.6.5', 'FLORIS version is not 4.6.5')
    assert_true(version('pymoo') == '0.6.1.5', 'pymoo version differs from verified 0.6.1.5')

    energy = load_layout_csv(ENERGY_LAYOUT, 12)
    assert_true(layout_is_feasible(energy, width_m=6000.0, height_m=6000.0,
                                  minimum_spacing_m=4.0 * D), 'Energy seed infeasible')
    b = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=D, response_probability=0.6,
        response_radius_D=2.0, encounter_radius_D=1.0,
        deflection_angle_deg=45.0, return_length_D=8.0, step_size_D=0.2,
    )
    # Original runner used exactly 5 x 2000 birds; the smoke test alone
    # uses a smaller sample to keep verification quick.
    birds_per_batch = 120 if smoke else 2000
    batches = 1 if smoke else 5
    pops = generate_population_batches(
        scenario=scenario, heading_deg_toward=225.0,
        trajectories_per_batch=birds_per_batch, batch_count=batches,
        longitudinal_margin_m=b.response_radius_m + b.return_length_m,
        first_seed=SEED,
    )
    no_response = BehaviourScenario(
        response_probability=0.0, response_radius_m=b.response_radius_m,
        encounter_radius_m=b.encounter_radius_m,
        deflection_angle_deg=b.deflection_angle_deg,
        return_length_m=b.return_length_m, step_size_m=b.step_size_m,
        max_steps_factor=b.max_steps_factor,
    )
    bird_ref = float(summarise_meso_batches(
        MesoResponseModel(no_response), energy, pops)['E_enc'])
    if smoke:
        # Smoke-test scaling is intentionally local to the smoke test.
        assert_true(bird_ref > 0.0, 'Smoke test has zero bird reference')
    else:
        assert_true(bird_ref == LOCKED_BIRD_REF,
                    f'Original 0.6854 reference changed: {bird_ref!r}')

    problem = Model8AEPEncounterProblem(
        scenario=scenario, aep_model=aep,
        meso_model=MesoResponseModel(b), trajectory_populations=pops,
        rotor_diameter_m=D, aep_ceiling_gwh=ceiling,
        aep_loss_reference_gwh=LOCKED_AEP_REF,
        encounter_reference=bird_ref, record_evaluations=record_evaluations,
    )
    fixed = make_fixed_layouts(site_width_m=6000., site_height_m=6000.,
                               bird_heading_deg_toward=225.,
                               minimum_spacing_m=4.0 * D)
    sampler = FeasibleSeededSampling(
        scenario=scenario, rotor_diameter_m=D,
        exact_seed_layouts={'energy': energy, **fixed}, seed=SEED)
    if not smoke:
        generated = sampler._do(problem, POP)
        saved = original_initial.drop(columns='seed_source').to_numpy(float)
        max_error = float(np.max(np.abs(generated - saved)))
        assert_true(max_error <= 1e-9, f'Initial population changed by {max_error} m')
        assert_true(sampler.last_labels == original_initial.seed_source.tolist(),
                    'Initial population seed-family labels differ')
    return old, problem, sampler, bird_ref, ceiling


def scientific_diff(original, problem, sectors):
    """Compare all source-recorded scientific parameters; only sectors may differ.

    Total wind conditions is a derived change: 84 -> 504, not independent.
    """
    now_aep = problem.aep_model.settings_summary()
    old_aep = original['aep_model']
    now_problem = problem.settings_summary()
    differences = []
    for key, before in original['problem'].items():
        if key not in now_problem or not equal(before, now_problem[key]):
            differences.append({'path': 'problem.' + key, 'original': before,
                                'proposed': now_problem.get(key, 'MISSING')})
    for key, before in old_aep.items():
        if key in ('number_of_direction_sectors', 'number_of_combined_conditions', 'config_path'):
            continue
        if key not in now_aep or not equal(before, now_aep[key]):
            differences.append({'path': 'aep_model.' + key, 'original': before,
                                'proposed': now_aep.get(key, 'MISSING')})
    for key, value in {
        'seed': SEED, 'population': POP, 'generations': GENERATIONS,
        'objective_batches': 5, 'objective_trajectories_per_batch': 2000,
        'fresh_batches': 3, 'fresh_trajectories_per_batch': 10000,
        'validation_z': 1.96, 'bird_heading_deg_toward': 225.0,
        'objective_random_number_seeds': [42, 43, 44, 45, 46],
        'fresh_random_number_seeds': [10042, 10043, 10044],
    }.items():
        if not equal(original[key], value):
            differences.append({'path': key, 'original': original[key], 'proposed': value})
    if sectors != 72 or old_aep['number_of_direction_sectors'] != 12:
        differences.append({'path': 'aep_model.number_of_direction_sectors',
                            'original': old_aep['number_of_direction_sectors'], 'proposed': sectors})
    if old_aep['number_of_combined_conditions'] != 84 or now_aep['number_of_combined_conditions'] != 504:
        differences.append({'path': 'aep_model.number_of_combined_conditions',
                            'original': old_aep['number_of_combined_conditions'],
                            'proposed': now_aep['number_of_combined_conditions']})
    return differences


def proposed_settings(original, problem):
    settings = copy.deepcopy(original)
    settings['model'] = 'Metric 4a — horizontal trajectories and encounters'
    settings['objectives'] = ['maximise AEP', 'minimise Q_4a']
    settings['aep_model'] = problem.aep_model.settings_summary()
    settings['aep_model']['config_path'] = str(CONFIG_YAML)
    settings['problem'] = problem.settings_summary()
    settings['site_width_m'] = 6000.0
    settings['site_height_m'] = 6000.0
    settings['n_sectors'] = 72
    settings['pymoo_version'] = version('pymoo')
    settings['python_version'] = platform.python_version()
    settings['floris_version'] = version('floris')
    settings['numpy_version'] = version('numpy')
    settings['wind_condition_count'] = 504
    settings['run_code_hashes_sha256'] = 'See file_hashes.json (fixed at run launch)'
    settings['checkpoint_interval_generations'] = CHECKPOINT_INTERVAL
    settings['checkpoint_serialiser'] = SERIALISER
    settings['history_pymoo_version_known'] = False
    settings['provenance_note'] = ('Original pymoo version and historical original YAML hash were not '
                                   'recorded; strict numerical reproduction was passed on this laptop.')
    return settings


def atomic_write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.temporary')
    with open(temp, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_json(path, data):
    atomic_write_bytes(path, json.dumps(data, indent=2, default=str).encode('utf-8'))


def make_algorithm(sampling, population=POP):
    # Exactly original NSGA-II invocation: all other operator settings default.
    return NSGA2(pop_size=population, sampling=sampling, eliminate_duplicates=True)


def save_checkpoint(algorithm, path, *, manifest):
    state = {
        'format_version': 1, 'algorithm': algorithm,
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
        'file_hashes': manifest,
        'last_callback_generation': (
            int(algorithm.callback.data['generation'][-1])
            if algorithm.callback and algorithm.callback.data['generation'] else 0),
        'pymoo_version': version('pymoo'),
    }
    packed = serialiser.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    existing = path
    backup = path.with_name(path.stem + '_previous' + path.suffix)
    if existing.exists():
        # Copy old checkpoint only after new bytes were produced successfully.
        atomic_write_bytes(backup, existing.read_bytes())
    atomic_write_bytes(existing, packed)
    # Test the newly saved bytes can deserialize now rather than discovering
    # corrupt/unpicklable checkpoints after a long laptop run.
    with open(existing, 'rb') as handle:
        reloaded = pickle.load(handle)
    assert_true(reloaded['last_callback_generation'] == state['last_callback_generation'],
                'Checkpoint generation did not survive deserialisation')


def load_checkpoint(path, manifest):
    assert_true(path.exists(), f'Missing checkpoint: {path}')
    with open(path, 'rb') as handle:
        state = pickle.load(handle)
    assert_true(state['format_version'] == 1, 'Checkpoint version unsupported')
    assert_true(state['pymoo_version'] == version('pymoo'), 'pymoo changed since checkpoint')
    assert_true(state['file_hashes'] == manifest, 'Source file changed since checkpoint; STOP')
    np.random.set_state(state['numpy_random_state'])
    random.setstate(state['python_random_state'])
    return state['algorithm']


def progress_loop(algorithm, checkpoint_path, manifest, *, interval=CHECKPOINT_INTERVAL):
    started = time.perf_counter()
    while algorithm.has_next():
        algorithm.next()
        completed = int(algorithm.callback.data['generation'][-1])
        print(f'Generation {completed}/{GENERATIONS}: elapsed {((time.perf_counter()-started)/60):.1f} min',
              flush=True)
        if completed % interval == 0 or not algorithm.has_next():
            save_checkpoint(algorithm, checkpoint_path, manifest=manifest)
            print(f'  Checkpoint saved after generation {completed}', flush=True)
    return algorithm


def write_training_outputs(algorithm, output):
    result = algorithm.result()
    assert_true(result.X is not None, 'No feasible Pareto layouts found')
    x = np.atleast_2d(np.asarray(result.X, dtype=float))
    f = np.atleast_2d(np.asarray(result.F, dtype=float))
    np.save(output / 'pareto_decision_vectors.npy', x)
    np.save(output / 'pareto_normalised_objectives.npy', f)
    # Save full last-generation population, including dominated individuals.
    pop = algorithm.pop
    np.savez_compressed(output / 'final_population.npz',
                        X=np.asarray(pop.get('X'), dtype=float),
                        F=np.asarray(pop.get('F'), dtype=float),
                        G=np.asarray(pop.get('G'), dtype=float),
                        CV=np.asarray(pop.get('CV'), dtype=float))
    pareto = build_pareto_dataframe(x, f, algorithm.problem)
    pareto.to_csv(output / 'pareto_points.csv', index=False)
    for label, row in [('highest-AEP', pareto.loc[pareto.aep_gwh.idxmax()]),
                       ('lowest-risk', pareto.loc[pareto.E_enc.idxmin()])]:
        index = int(row.solution_index)
        xy = x[index].reshape(12, 2)
        pd.DataFrame({'turbine_id': np.arange(12), 'x_m': xy[:, 0],
                      'y_m': xy[:, 1]}).to_csv(output / f'{label}_layout.csv', index=False)
    pd.DataFrame(algorithm.callback.data).to_csv(output / 'hypervolume.csv', index=False)
    algorithm.problem.evaluation_dataframe().to_csv(
        output / 'all_evaluated_candidates.csv', index=False)
    write_json(output / 'training_summary.json', {
        'non_dominated_layouts': int(len(pareto)),
        'highest_aep_gwh': float(pareto.aep_gwh.max()),
        'lowest_risk_Q_4a': float(pareto.E_enc.min()),
        'generations': int(algorithm.callback.data['generation'][-1]),
        'final_population_size': len(pop),
        'fresh_validation': 'To be evaluated separately on original seeds 10042-10044',
        'independent_confirmation': 'Not yet run; original protocol seeds 20042-20061',
    })
    print(f'Saved {len(pareto)} Pareto layouts and all {len(pop)} final population members.')
    return x, pareto


def verify():
    REPORTS.mkdir(exist_ok=True)
    old, problem, sampler, ref, ceiling = make_setup()
    diffs = scientific_diff(old, problem, SECTORS)
    assert_true(not diffs, 'Scientific settings differ beyond sectors: ' + str(diffs))
    assert_true(old['aep_model']['number_of_direction_sectors'] == 12, 'Not a 12-sector original')
    assert_true(problem.aep_model.number_of_conditions == 504, 'Not 504 conditions')
    assert_true(ref == LOCKED_BIRD_REF, 'Bird reference changed')
    assert_true(ceiling == LOCKED_AEP_CEILING, 'Wake-free reference changed')
    assert_true((ROOT / 'preflight_reports' / 'aep_report.json').exists(),
                'Run the earlier AEP preflight check first')
    assert_true((ROOT / 'preflight_reports' / 'bird_report.json').exists(),
                'Run the earlier bird preflight check first')
    assert_true((ROOT / 'preflight_reports' / 'audit_report.json').exists(),
                'Run the earlier archive audit first')
    h = hashes()
    report = {
        'status': 'PASS', 'original_sector_count': 12,
        'new_sector_count': 72, 'original_wind_conditions': 84,
        'new_wind_conditions': 504,
        'scientific_differences_other_than_sector_resolution': diffs,
        'derived_wind_condition_difference': '84 -> 504 (72 x 7)',
        'same_initial_population': True,
        'pymoo_version_laptop': version('pymoo'),
        'pymoo_version_original': 'NOT RECORDED',
        'floris_version': version('floris'),
        'objective_reference_Q_4a': ref,
        'wake_free_aep_gwh': ceiling,
        'source_hashes': h,
        'proposed_run_settings': proposed_settings(old, problem),
        'checkpoint_resume_smoke_test': 'PENDING',
    }
    write_json(REPORTS / 'run72_verify.json', report)
    print('PASS | 72 sectors, 504 conditions, same sampling/behaviour/objectives/NSGA-II settings')
    print('PASS | Initial 80-layout population matches original to 1e-9 m')
    print('PASS | fixed bird reference 0.6854 and fixed wake-free AEP')
    print('PASS | Scientific parameter differences: sector count only (12 -> 72)')
    print(f'INFO | Original pymoo version unknown; laptop version {version("pymoo")}')
    print('INFO | report:', REPORTS / 'run72_verify.json')
    print('NO OPTIMISATION STARTED')


def smoke():
    print('SMOKE | Tiny artificial-length test, NOT the production run', flush=True)
    prior = REPORTS / 'run72_verify.json'
    assert_true(prior.is_file(), 'Run verify first')
    report = json.loads(prior.read_text(encoding='utf-8'))
    assert_true(report.get('status') == 'PASS', 'Verify did not pass')
    assert_true(report['source_hashes'] == hashes(), 'Source hashes changed since verify')
    # Runtime reduced ONLY for smoke; do not write these settings to production folder.
    original, p1, s1, _r1, _c1 = make_setup(smoke=True, record_evaluations=False)
    run_reference = minimize(p1, make_algorithm(s1, population=8),
                             termination=('n_gen', 3), seed=SEED,
                             callback=HypervolumeCallback(), verbose=False)
    assert_true(run_reference.X is not None, 'Reference tiny run has no feasible layouts')
    _original, p2, s2, _r2, _c2 = make_setup(smoke=True, record_evaluations=False)
    alg = make_algorithm(s2, population=8)
    alg.setup(p2, termination=get_termination('n_gen', 3), seed=SEED,
              callback=HypervolumeCallback(), verbose=False)
    alg.next()
    smoke_path = REPORTS / 'run72_smoke_checkpoint.pkl'
    manifest = hashes()
    save_checkpoint(alg, smoke_path, manifest=manifest)
    del alg
    alg = load_checkpoint(smoke_path, manifest)
    while alg.has_next():
        alg.next()
    observed_pop_x = np.asarray(alg.pop.get('X'), dtype=float)
    ref_pop_x = np.asarray(run_reference.algorithm.pop.get('X'), dtype=float)
    observed_pop_f = np.asarray(alg.pop.get('F'), dtype=float)
    ref_pop_f = np.asarray(run_reference.algorithm.pop.get('F'), dtype=float)
    # Exact count/float sequence is expected within this same environment.
    same_x = np.array_equal(observed_pop_x, ref_pop_x)
    same_f = np.array_equal(observed_pop_f, ref_pop_f)
    same_hv = equal(alg.callback.data, run_reference.algorithm.callback.data, tol=0)
    report['checkpoint_resume_smoke_test'] = ('PASS' if same_x and same_f and same_hv else 'FAIL')
    report['smoke_same_final_population_X_exact'] = bool(same_x)
    report['smoke_same_final_population_F_exact'] = bool(same_f)
    report['smoke_same_hypervolume_history_exact'] = bool(same_hv)
    report['smoke_config'] = {'pop': 8, 'generations': 3,
                              'bird_batches': 1, 'birds_per_batch': 120,
                              'n_sectors': 72,
                              'NOTE': 'Smoke settings are NEVER used for the main run'}
    write_json(REPORTS / 'run72_verify.json', report)
    assert_true(report['checkpoint_resume_smoke_test'] == 'PASS',
                'Checkpoint/resume diverged from uninterrupted original NSGA-II. STOP.')
    print('PASS | original pymoo minimize() vs interrupted/resumed loop: exact final X, F and HV history')
    print('PASS | checkpoint successfully read back from disk')
    print('NO 8,000-EVALUATION OPTIMISATION STARTED')


def run(resuming):
    verify_path = REPORTS / 'run72_verify.json'
    assert_true(verify_path.exists(), 'First run verify and smoke')
    checked = json.loads(verify_path.read_text(encoding='utf-8'))
    assert_true(checked.get('status') == 'PASS', 'Settings verify failed')
    assert_true(checked.get('checkpoint_resume_smoke_test') == 'PASS',
                'Checkpoint resume smoke test must PASS before production')
    manifest = hashes()
    assert_true(manifest == checked['source_hashes'], 'Source files changed since smoke. STOP')
    checkpoint = OUTPUT / 'checkpoint_latest.pkl'
    if not resuming:
        assert_true(not OUTPUT.exists(), f'Output exists, refusing to overwrite: {OUTPUT}')
        old, problem, sampler, ref, ceiling = make_setup()
        diffs = scientific_diff(old, problem, SECTORS)
        assert_true(not diffs, f'Scientific diff failed: {diffs}')
        OUTPUT.mkdir(parents=False, exist_ok=False)
        write_json(OUTPUT / 'file_hashes.json', manifest)
        write_json(OUTPUT / 'run_settings.json', proposed_settings(old, problem))
        write_json(OUTPUT / 'settings_diff.json', {
            'changed_independent_scientific_input': {'n_sectors': [12, 72]},
            'derived_input_grid': {'number_of_combined_conditions': [84, 504]},
            'other_scientific_differences': diffs,
            'reporting_and_checkpoint_metadata_added': True,
            'previous_pymoo_version_recorded': False,
        })
        init = sampler._do(problem, POP)
        initial_labels = sampler.last_labels
        columns = [f'{xy}_{i}_m' for i in range(12) for xy in ('x','y')]
        pd.DataFrame(init, columns=columns).assign(seed_source=initial_labels).to_csv(
            OUTPUT / 'initial_population.csv', index=False)
        alg = make_algorithm(sampler)
        alg.setup(problem, termination=get_termination('n_gen', GENERATIONS), seed=SEED,
                  callback=HypervolumeCallback(), verbose=False)
        # checkpoint after setup, before any expensive generation evaluation.
        save_checkpoint(alg, checkpoint, manifest=manifest)
        print('START | New output folder:', OUTPUT, flush=True)
    else:
        assert_true(OUTPUT.exists(), 'No existing run output to resume')
        assert_true(json.loads((OUTPUT / 'file_hashes.json').read_text()) == manifest,
                    'File hash manifest mismatch. STOP')
        try:
            alg = load_checkpoint(checkpoint, manifest)
        except Exception:
            backup = OUTPUT / 'checkpoint_latest_previous.pkl'
            print('WARNING | latest checkpoint failed. Trying previous checkpoint.', flush=True)
            alg = load_checkpoint(backup, manifest)
        print('RESUME | Last saved generation',
              (alg.callback.data['generation'][-1] if alg.callback.data['generation'] else 0),
              flush=True)
    progress_loop(alg, checkpoint, manifest)
    print('OPTIMISATION COMPLETE | Saving all final results', flush=True)
    write_training_outputs(alg, OUTPUT)
    write_json(OUTPUT / 'completion_status.json', {
        'status': 'optimisation_finished',
        'n_sectors': 72,
        'generations': GENERATIONS,
        'fresh_validation': 'Not run in this stage',
        'independent_confirmation': 'Not run in this stage',
    })
    print('DONE | Full Pareto and final population archived. Confirmation is a separate step.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['verify', 'smoke', 'run', 'resume'])
    args = parser.parse_args()
    try:
        if args.command == 'verify':
            verify()
        elif args.command == 'smoke':
            smoke()
        elif args.command == 'run':
            run(resuming=False)
        else:
            run(resuming=True)
    except Exception as exc:
        REPORTS.mkdir(exist_ok=True)
        failure = REPORTS / f'run72_{args.command}_ERROR.txt'
        failure.write_text(traceback.format_exc(), encoding='utf-8')
        print('STOP |', type(exc).__name__, str(exc))
        print('ERROR DETAILS:', failure)
        sys.exit(1)


if __name__ == '__main__':
    main()
