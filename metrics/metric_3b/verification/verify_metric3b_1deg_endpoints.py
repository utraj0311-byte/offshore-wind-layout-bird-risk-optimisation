"""Post-run Metric 3b endpoint check: 72 (5°) versus 360 (1°) sectors.

Run from the bird_layout_opt project root inside floris-env:
    python verify_metric3b_1deg_endpoints.py

READS the original completed Metric 3b archive. No optimisation.
Saves a new, separate CSV; never overwrites existing files.
"""
from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from config import Scenario
from floris_aep import FLORISAEPModel


ROOT = Path.cwd()
RUN = ROOT / 'results' / 'metric_3b_72sectors' / 'heading225_W6000_seed48_pop160_gen250'
OUT = ROOT / 'results' / 'metric_3b_72sectors' / 'validation_1deg_endpoints.csv'
TOLERANCE_SAVED_GWH = 0.001
TOLERANCE_1_DEG_PERCENT = 0.2
WAKE_FREE_EXPECTED = 854.495658562


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_aep_model(n_sectors: int, settings: dict, yaml_path: Path) -> FLORISAEPModel:
    scenario = Scenario()
    scenario.site.n_turbines = int(settings['problem']['number_of_turbines'])
    scenario.site.width_m = float(settings['site_width_m'])
    scenario.site.height_m = float(settings['site_height_m'])
    scenario.site.min_spacing_D = float(settings['problem']['minimum_spacing_D'])
    scenario.site.wind_from_deg = float(settings['aep_model']['dominant_wind_from_deg'])
    scenario.turbine.rotor_diameter_m = float(settings['problem']['rotor_diameter_m'])
    scenario.wind.n_sectors = n_sectors
    model = FLORISAEPModel(scenario, config_path=yaml_path)
    assert model.number_of_conditions == n_sectors * int(settings['aep_model']['number_of_speed_bins'])
    return model


def main() -> None:
    if not RUN.is_dir():
        raise SystemExit(f'STOP: completed run folder missing: {RUN}')
    if OUT.exists():
        raise SystemExit(f'STOP: output already exists; refusing to overwrite: {OUT}')
    settings = json.loads((RUN / 'run_settings.json').read_text(encoding='utf-8'))
    expected_hashes = json.loads((RUN / 'source_file_hashes.json').read_text(encoding='utf-8'))
    for filename, archived_name in [
        ('config.py', 'config.py'),
        ('floris_aep.py', 'floris_aep.py'),
    ]:
        local = ROOT / filename
        if not local.exists() or sha256(local) != expected_hashes[archived_name]['sha256']:
            raise SystemExit(f'STOP: {filename} differs from completed-run snapshot.')

    floris_version = version('floris')
    if floris_version != str(settings.get('floris_version')):
        raise SystemExit(f'STOP: FLORIS version changed ({floris_version}).')
    if int(settings['aep_model']['number_of_direction_sectors']) != 72:
        raise SystemExit('STOP: expected completed 72-sector archive.')
    cfg = RUN / 'source_snapshots' / 'floris_configuration.yaml'
    if sha256(cfg) != expected_hashes['floris_configuration.yaml']['sha256']:
        raise SystemExit('STOP: archived FLORIS YAML checksum mismatch.')

    front = pd.read_csv(RUN / 'pareto_points.csv')
    vectors = np.load(RUN / 'pareto_decision_vectors.npy')
    if len(front) != len(vectors):
        raise SystemExit('STOP: archived front and decision-vector lengths differ.')
    best_energy = int(front['aep_gwh'].idxmax())
    min_q = float(front['N_eff'].min())
    candidates = front.index[np.isclose(front['N_eff'], min_q, rtol=0, atol=1e-12)]
    best_bird = int(front.loc[candidates, 'aep_gwh'].idxmax())
    selected = [('highest AEP', best_energy), ('lowest Q_3b', best_bird)]

    model5 = make_aep_model(72, settings, cfg)
    five_results = []
    print('Checking the saved endpoints at 72 sectors (5 degrees)...', flush=True)
    for label, idx in selected:
        xy = vectors[idx].reshape(-1, 2)
        saved = float(front.loc[idx, 'aep_gwh'])
        got = float(model5.evaluate(xy))
        err = abs(got - saved)
        print(f'  {label}: saved={saved:.9f}  recomputed={got:.9f}  |error|={err:.9f} GWh/yr', flush=True)
        if not np.isfinite(got) or err > TOLERANCE_SAVED_GWH:
            raise SystemExit('STOP: 5-degree archived AEP reproduction failed. No 1-degree evaluations performed.')
        five_results.append(got)

    ideal5 = float(model5.ideal_aep(12))
    print('Checking the endpoints at 360 sectors (1 degree)...', flush=True)
    model1 = make_aep_model(360, settings, cfg)
    ideal1 = float(model1.ideal_aep(12))
    if (abs(ideal5 - WAKE_FREE_EXPECTED) > 1e-6 or
            abs(ideal1 - WAKE_FREE_EXPECTED) > 1e-6):
        raise SystemExit('STOP: wake-free AEP mismatch. No validation file written.')

    records = []
    for (label, idx), at5 in zip(selected, five_results):
        xy = vectors[idx].reshape(-1, 2)
        at1 = float(model1.evaluate(xy))
        diff_percent = 100.0 * abs(at5 - at1) / abs(at1)
        ok = bool(np.isfinite(at1) and diff_percent < TOLERANCE_1_DEG_PERCENT)
        r = front.loc[idx]
        records.append({
            'layout': label,
            'solution_index': int(r['solution_index']),
            'aep_5deg_saved_gwh': float(r['aep_gwh']),
            'aep_5deg_recomputed_gwh': at5,
            'aep_1deg_gwh': at1,
            'relative_difference_percent': diff_percent,
            'pass_below_0p2_percent': ok,
            'Q_3b_rotor_diameters': float(r['N_eff']),
            'projected_rotor_bands': int(r['number_of_merged_intervals']),
            'wake_loss_5deg_percent': 100 * (ideal5 - at5) / ideal5,
            'wake_loss_1deg_percent': 100 * (ideal1 - at1) / ideal1,
        })
        print(f'  {label}: AEP(5°)={at5:.9f}, AEP(1°)={at1:.9f} GWh/yr; '
              f'|difference|/AEP(1°)={diff_percent:.6f}% — {"PASS" if ok else "FAIL"}', flush=True)

    results = pd.DataFrame.from_records(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    print(f'\nSaved (new file; originals unchanged): {OUT}', flush=True)
    if all(row['pass_below_0p2_percent'] for row in records):
        print('PASS: both endpoints satisfy the 1-degree validation criterion.', flush=True)
    else:
        print('VALIDATION FAILED: full 126-layout front requires 1-degree re-evaluation and re-determination of nondominance.', flush=True)


if __name__ == '__main__':
    main()
