"""Controlled Jensen 360-sector re-runs: Metric 2a, 2b, then AEP-only reference.

Run from Anaconda Prompt (project root):
  python metric2_360_tools\run_metric2_360.py preflight
  python metric2_360_tools\run_metric2_360.py run-2a
  python metric2_360_tools\run_metric2_360.py run-2b
  python metric2_360_tools\run_metric2_360.py run-aep

Each run refuses to overwrite an existing output folder. Gate checks are compulsory.
Original project modules are imported, not modified. No optimisation occurs in preflight.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
RESULTS = ROOT / 'jensen_1deg_approved' / 'results'
INPUTS = TOOLS_DIR / 'inputs'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AEP_REF_GWH = 760.663033300808
OLD_JENSEN_BEST_1D_GWH = 734.7418306942395
N = 12
D = 240.0
SITE_M = 6000.0
MIN_SPACING_M = 960.0
EXPECTED_PYMOO = '0.6.1.5'
SPECS = {'2a': {'pop': 60, 'gen': 60, 'seed': 42, 'scale_D': 4.0},
         '2b': {'pop': 80, 'gen': 100, 'seed': 42, 'scale_D': 8.0},
         'aep': {'pop': 160, 'gen': 250, 'seed': 42}}
NAMES = {
    '2a': 'metric_2a_360sectors_seed42_pop60_gen60',
    '2b': 'metric_2b_360sectors_seed42_pop80_gen100',
    'aep': 'metric_aep_only_360sectors_seed42_pop160_gen250',
}
# Original implementation hashes supplied by the user. For config.py, read and
# verify physical parameters because both 12- and 360-sector versions exist.
SOURCE_HASHES = {
    'wake.py': '122f6f86863cb85bfe4da21d4e5433ea7d0b28dc892356cd12da47ed73c070d8',
    'validation_problem.py': '4f80ef448b32456116c24abea9bbb8b692e69e76634660c27ab6ef51866e6817',
    'run_validation.py': 'e86b0cea38e8900aa82db092ffd5011ab5f95c76934cb2c830786e76c51cd981',
    'coupled_risk.py': '57df6141578eff7bebfad8dfcb2fd2335be4a296643e0b6f8a26f0ea15364dc1',
    'pairwise_risk.py': '8370fade49fb039b1dd22bb5253138711f9fe54bdc4f044d1cca676493c6c88d',
    'run_aep_only.py': '907be02c1af8e48f3b259d485ec6cf92f4ca29348e37b4baa812eb39a27fe415',
}
ARCHIVES = {
    '2a': ('model5_original.zip', 'surroundedness_interior_risky_scale4p0D_seed42_pop60_gen60'),
    '2b': ('model6_original.zip', 'scale8p0D_seed42_pop80_gen100'),
}


def check(condition, why):
    if not condition:
        raise RuntimeError('PRE-FLIGHT / SAFETY STOP: ' + str(why))


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    """Only used for run-created files, never for original results."""
    dest = Path(path)
    temp = dest.with_name(dest.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, default=json_default) + '\n', encoding='utf-8')
    temp.replace(dest)


def json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Not JSON serialisable: {type(o)}')


def source_inventory():
    files = ['config.py'] + list(SOURCE_HASHES) + ['aep_only_problem.py', 'risk_only_problem.py']
    found = {}
    for name in files:
        path = ROOT / name
        found[name] = {'exists': path.is_file(),
                       'sha256': sha256(path) if path.is_file() else None}
    # Only require root modules that are used by our active runs. risk_only_problem.py
    # is not imported, as the risk-only optimisation is deliberately NOT repeated.
    required = ['config.py', 'wake.py', 'validation_problem.py',
                'run_validation.py', 'coupled_risk.py', 'pairwise_risk.py',
                'run_aep_only.py', 'aep_only_problem.py']
    for name in required:
        check(found[name]['exists'], f'Missing {ROOT / name}. Supply original file, do not invent it.')
    for name, expected_hash in SOURCE_HASHES.items():
        observed = found[name]['sha256']
        check(observed == expected_hash,
              f'{name} is not the uploaded original. Expected SHA-256 {expected_hash}; found {observed}.')
    # run_validation.py imports hypothetical_risk.py; required at project root.
    check((ROOT / 'hypothetical_risk.py').is_file(),
          'Original run_validation.py requires hypothetical_risk.py at project root.')
    found['hypothetical_risk.py'] = {'exists': True, 'sha256': sha256(ROOT / 'hypothetical_risk.py')}
    return found


def imports():
    from config import Scenario
    from wake import AEPModel
    from coupled_risk import SurroundednessRiskModel
    from pairwise_risk import PairwiseGaussianRiskModel
    from validation_problem import ValidationLayoutProblem
    from run_validation import GridSampling, HVCallback, min_spacing, count_spacing_violations, count_boundary_violations
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.optimize import minimize
    return locals()


def get_scenario(mods, sectors):
    scenario = mods['Scenario']()
    s = scenario.site
    t = scenario.turbine
    w = scenario.wind
    check((s.n_turbines, s.width_m, s.height_m, s.min_spacing_D) == (12, 6000., 6000., 4.),
          'Site or minimum-spacing configuration changed from original.')
    check((t.rotor_diameter_m, t.hub_height_m, t.rated_power_mw,
           t.cut_in_ms, t.rated_ms, t.cut_out_ms) == (240., 150., 15., 3., 10.6, 25.),
          'Original Jensen turbine configuration changed.')
    check((s.wind_from_deg, w.weibull_k, w.weibull_A) == (225., 2., 10.),
          'Wind definition changed from original.')
    check(np.array_equal(w.speed_bins, np.array([4., 6., 8., 10., 12., 14., 16.])),
          'Wind-speed bins changed from original.')
    check(w.n_sectors in (12, 360),
          f'Expected approved 12 or 360 sectors in config.py, found {w.n_sectors}.')
    w.n_sectors = sectors       # ONLY scientific parameter changed for reruns.
    return scenario


def risk_model(mods, metric, scenario):
    if metric == '2a':
        return mods['SurroundednessRiskModel'](scenario=scenario, target_index=0,
              interaction_scale_D=4.0, mode='interior_risky', risk_multiplier=1.0)
    if metric == '2b':
        return mods['PairwiseGaussianRiskModel'](scenario=scenario,
              interaction_scale_D=8.0, risk_multiplier=1.0)
    raise ValueError(metric)


def archive_csv(metric, filename):
    zip_name, folder = ARCHIVES[metric]
    with zipfile.ZipFile(INPUTS / zip_name) as archive:
        return pd.read_csv(io.BytesIO(archive.read(f'{folder}/{filename}')))


def csv_xy(dataframe):
    d = dataframe.copy()
    if 'turbine_id' in d.columns:
        d = d.sort_values('turbine_id', kind='stable')
    xy = d[['x_m', 'y_m']].to_numpy(dtype=float)
    check(xy.shape == (N, 2) and np.isfinite(xy).all(), 'Saved layout must contain 12 valid (x,y) coordinates.')
    return xy


def check_feasible(mods, xy):
    sep = mods['min_spacing'](xy)
    check(mods['count_spacing_violations'](xy, MIN_SPACING_M) == 0,
          f'Layout violates 960 m spacing; minimum={sep:.9f} m.')
    check(mods['count_boundary_violations'](xy, SITE_M, SITE_M) == 0,
          'Layout violates 0–6000 m site bounds.')
    return float(sep)


def old_reference_layouts():
    return {
        '2a_highest_aep': ('2a', 'layout_energy_optimal.csv'),
        '2a_lowest_q': ('2a', 'layout_risk_optimal.csv'),
        '2b_highest_aep': ('2b', 'layout_energy_optimal.csv'),
        '2b_lowest_q': ('2b', 'layout_risk_optimal.csv'),
    }


def prior_archive_aep(metric, anchor):
    df = archive_csv(metric, 'anchor_summary.csv')
    return float(df.loc[df['layout'] == anchor, 'aep_gwh'].iloc[0])


def preflight(mods, source_files, print_report=True):
    check(package_version('pymoo') == EXPECTED_PYMOO,
          f'Require pymoo {EXPECTED_PYMOO} for NEW runs; found {package_version("pymoo")}. '
          'Historical version is not assumed known.')
    scenario12 = get_scenario(mods, 12)
    scenario360 = get_scenario(mods, 360)
    aep12 = mods['AEPModel'](scenario12)
    aep360 = mods['AEPModel'](scenario360)
    check(math.isclose(float(aep12.wake_decay_k), .04, abs_tol=1e-15), 'Jensen kw changed.')
    for wind_speed, expected_ct in [(2., 0.), (4., .8), (10.59, .8), (10.6, .25), (25., .25), (26., 0.)]:
        actual = float(aep12.ct_curve(np.array([wind_speed]))[0])
        check(actual == expected_ct, f'C_T curve changed at U={wind_speed}.')
    check(math.isclose(aep12.ideal_aep(12), AEP_REF_GWH, abs_tol=1e-6),
          f'Unexpected 12-sector wake-free AEP: {aep12.ideal_aep(12):.12f}.')
    check(math.isclose(aep360.ideal_aep(12), AEP_REF_GWH, abs_tol=1e-6),
          f'Unexpected 360-sector wake-free AEP: {aep360.ideal_aep(12):.12f}.')
    results = []
    for name, (metric, fn) in old_reference_layouts().items():
        xy = csv_xy(archive_csv(metric, fn))
        check_feasible(mods, xy)
        original = prior_archive_aep(metric, 'energy_optimal' if 'highest_aep' in name else 'risk_optimal')
        observed = float(aep12.evaluate(xy))
        diff = abs(original-observed)
        check(diff <= .001, f'{name} 12-sector AEP reproduction error {diff:.12f} exceeds .001 GWh.')
        results.append({'layout': name, 'archived_12sector_aep_gwh': original,
                        'recomputed_12sector_aep_gwh': observed, 'abs_error_gwh': diff})
    # Independent AEP-only and risk-only reference layouts: archived values are
    # from Model 6 original summary.json. These are included in the mandatory gate.
    with zipfile.ZipFile(INPUTS / ARCHIVES['2b'][0]) as archive:
        summary = json.loads(archive.read(f'{ARCHIVES["2b"][1]}/summary.json'))
    for key, fn, archkey in [
        ('2b_aep_only_reference', 'aep_only_layout_original.csv', 'independent_aep_reference'),
        ('2b_risk_only_reference', 'risk_only_layout_original.csv', 'independent_risk_reference')]:
        xy = csv_xy(pd.read_csv(INPUTS / fn))
        check_feasible(mods, xy)
        original = float(summary[archkey]['aep_gwh'])
        observed = float(aep12.evaluate(xy))
        diff = abs(original-observed)
        check(diff <= .001, f'{key} 12-sector AEP reproduction error {diff:.12f} exceeds .001 GWh.')
        results.append({'layout': key, 'archived_12sector_aep_gwh': original,
                        'recomputed_12sector_aep_gwh': observed, 'abs_error_gwh': diff})
    model2a = risk_model(mods, '2a', scenario12)
    model2b = risk_model(mods, '2b', scenario12)
    check(model2a.target_index == 0 and model2a.mode == 'interior_risky'
          and model2a.sigma_m == 960.0 and model2a.risk_scale() == 11.,
          '2a target rule, sigma or risk objective has changed.')
    check(model2b.sigma_m == 1920.0 and model2b.risk_scale() == 66.,
          '2b sigma or pairwise risk objective has changed.')
    # Confirm the exact original objective supplied to NSGA-II, with all parameters.
    demo_xy = csv_xy(archive_csv('2a', 'layout_energy_optimal.csv'))
    P = mods['ValidationLayoutProblem'](scenario=scenario12, aep_model=aep12,
         risk_model=model2a, aep_min=.5*AEP_REF_GWH, aep_max=AEP_REF_GWH,
         risk_min=0., risk_max=11.)
    out = {}
    P._evaluate(demo_xy.reshape(1, -1), out)
    aep = aep12.evaluate(demo_xy)
    q = model2a.collisions(demo_xy)
    check(np.allclose(out['F'][0], [2-2*aep/AEP_REF_GWH, q/11], rtol=0, atol=1e-12),
          'Original ValidationLayoutProblem objective differs from expected f1/f2.')
    # Check complete active optimizer configuration against original constructor.
    algo = mods['NSGA2'](pop_size=60,
           sampling=mods['GridSampling'](scenario12, seed=42, model_name='constant'),
           eliminate_duplicates=True)
    check(type(algo).__name__ == 'NSGA2', 'NSGA-II changed unexpectedly.')
    # The sampler delegates to the uploaded original: 50% random / 50% grid.
    sample = mods['GridSampling'](scenario12, seed=42, model_name='constant')._do(P, 60)
    check(np.asarray(sample).shape == (60, 24), 'Metric2 initial population has unexpected shape.')
    check(all(check_feasible(mods, xy.reshape(12, 2)) >= MIN_SPACING_M-1e-6 for xy in sample),
          'The initial population is not feasible.')
    # Performance estimate is machine-local; one evaluation ONLY.
    tic = time.perf_counter()
    aep360.evaluate(demo_xy)
    seconds = time.perf_counter()-tic
    projections = {'metric_2a_approx_seconds': seconds*60*60,
                   'metric_2b_approx_seconds': seconds*80*100,
                   'aep_only_approx_seconds': seconds*160*250,
                   'note': 'One AEP evaluation × population × generations; optimisation and post-processing overhead omitted.'}
    report = {
        'status': 'PASS', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'approved_only_change': {'wind_n_sectors_original': 12, 'wind_n_sectors_new': 360},
        'scientific_settings_diff': {'wind.n_sectors': [12, 360]},
        'site_and_Jensen_rules': 'verified from untouched project modules and scenario values',
        'metric_2a_target_selection': 'fixed array index 0, not most-interior',
        'metric_2a_sigma_m': 960.0, 'metric_2b_sigma_m': 1920.0,
        'f1': f'2 - 2*AEP/{AEP_REF_GWH}', 'f2_2a': 'Q_2a/11', 'f2_2b': 'Q_2b/66',
        'initialisation': 'GridSampling(model_name=constant): 50% random feasible / 50% jittered grid',
        'algorithm': 'original NSGA2/GA constructors and installed pymoo defaults',
        'version': EXPECTED_PYMOO, 'source_file_hashes': source_files,
        'aep_reproduction': results, 'wake_free_reference_gwh': AEP_REF_GWH,
        'seconds_per_360sector_aep_evaluation': seconds,
        'runtime_projections': projections,
    }
    if print_report:
        print('\nPRE-FLIGHT PASS (NO optimiser launched)')
        print('Designated 2a turbine: permanently array index 0 (not geometric selection)')
        print('Objective f1: 2-2*AEP/760.663033300808; f2_2a: Q/11; f2_2b: Q/66')
        print('Initial population: original 50% random feasible / 50% jittered grid')
        print('Only wind.n_sectors changes: 12 -> 360')
        print(pd.DataFrame(results).to_string(index=False, float_format=lambda x: f'{x:.10f}'))
        print(f'One local 360-sector AEP evaluation: {seconds:.4f} s')
        for key, value in projections.items():
            if key.endswith('_seconds'):
                print(f'Projected {key}: {value/60:.1f} minutes + overhead')
        print('pymoo:', package_version('pymoo'), '(new run version, historical version unknown)')
    return report


def create_output(name):
    check(RESULTS.is_dir(), f'Master results directory does not exist: {RESULTS}')
    output = RESULTS / NAMES[name]
    check(not output.exists(), f'Output already exists; WILL NOT OVERWRITE: {output}')
    output.mkdir(exist_ok=False)
    return output


def verify_previous_stage(metric):
    if metric == '2b':
        previous = '2a'
    elif metric == 'aep':
        previous = '2b'
    else:
        return
    prior = RESULTS / NAMES[previous] / 'run_settings.json'
    check(prior.is_file(), f'{previous} result is missing. Launch {previous} first.')
    settings = json.loads(prior.read_text(encoding='utf-8'))
    check(settings.get('run_status') == 'complete' and
          settings.get('post_run_endpoint_verification', {}).get('status') == 'pass',
          f'{previous} has NOT passed both 0.5-degree checks; refusing to launch {metric}.')


def op_snapshot(op):
    if op is None:
        return None
    info = {'class': f'{type(op).__module__}.{type(op).__qualname__}'}
    for key in ('prob', 'prob_var', 'eta', 'n_offsprings', 'n_parents', 'repair'):
        if not hasattr(op, key):
            continue
        v = getattr(op, key)
        if v is None or isinstance(v, (int, float, bool, str)):
            info[key] = v
        elif getattr(v, 'value', None) is not None and isinstance(v.value, (int, float, bool, str)):
            info[key] = {'wrapper': type(v).__name__, 'value': v.value}
        else:
            info[key] = {'class': type(v).__name__, 'repr': repr(v)}
    return info


def settings_template(metric, mods, scenario, aep_model, risk, algorithm, source_files, pre):
    spec = SPECS[metric]
    objrisk = None if metric == 'aep' else ('Q_2a/11' if metric == '2a' else 'Q_2b/66')
    return {
        'run_status': 'initialized', 'metric': metric,
        'previous_model': {'2a': 'Model 5', '2b': 'Model 6', 'aep': 'AEP-only reference'}[metric],
        'site': {'width_m': 6000., 'height_m': 6000., 'n_turbines': 12,
                 'min_spacing_D': 4., 'minimum_spacing_m': 960.},
        'turbine': {'D_m': 240., 'hub_m': 150., 'rated_MW': 15.,
                    'cut_in_ms': 3., 'rated_ms': 10.6, 'cut_out_ms': 25.},
        'wind': {'n_sectors': 360, 'sector_width_deg': 1., 'wind_from_deg': 225.,
                 'speed_bins_ms': list(map(float, scenario.wind.speed_bins)),
                 'weibull_k': 2., 'weibull_A_ms': 10.,
                 'direction_weights': 'exp(2 cos(delta FROM 225 degrees)), normalised',
                 'speed_weights': 'discrete Weibull PDF, normalised'},
        'wake': {'model': 'Jensen top-hat hub-centre test', 'kw': .04,
                 'combination': 'root-sum-square, deficit clipped to .95',
                 'CT': '0.8 for 3<=U<10.6, .25 for 10.6<=U<=25, else 0 (undisturbed U)',
                 'power_curve': 'cubic 3 to 10.6 m/s, then 15 MW through 25 m/s'},
        'metric_formulation': (
            {'designated_turbine_rule': 'permanently fixed index 0',
             'mode': 'interior_risky', 'sigma_D': 4., 'sigma_m': 960.,
             'Q': 'sum_{j != 0} exp(-d_0j^2/(2 sigma^2))', 'risk_multiplier': 1.}
            if metric == '2a' else
            {'permutation_invariant': True, 'sigma_D': 8., 'sigma_m': 1920.,
             'Q': 'sum_{i < j} exp(-d_ij^2/(2 sigma^2))', 'risk_multiplier': 1.,
             'number_of_unique_pairs': 66} if metric == '2b' else {'objective': 'maximise Jensen AEP'}),
        'objective_scaling': {'aep_ref_gwh': AEP_REF_GWH, 'aep_min_gwh': .5*AEP_REF_GWH,
                              'aep_max_gwh': AEP_REF_GWH,
                              'f1': '2 - 2*AEP/AEP_ref', 'f2': objrisk,
                              'archived_1a_to_1d_aep_ref_gwh': 754.4007727706155,
                              'note': 'Do not replace Metric 2a/2b AEP_ref with Metric 1a-1d AEP_ref.'},
        'optimizer': {'name': 'GA' if metric == 'aep' else 'NSGA-II',
                      'population': spec['pop'], 'generations': spec['gen'],
                      'seed': spec['seed'], 'eliminate_duplicates': True,
                      'sampling': 'original GridSampling(model_name=constant), 50/50 random/grid',
                      'operators': {
                          'crossover': op_snapshot(getattr(getattr(algorithm, 'mating', None),'crossover',None)),
                          'mutation': op_snapshot(getattr(getattr(algorithm, 'mating', None),'mutation',None)),
                          'selection': op_snapshot(getattr(getattr(algorithm, 'mating', None),'selection',None)),
                          'survival': op_snapshot(getattr(algorithm, 'survival',None))},
                      'hv_reference': None if metric=='aep' else [1.2,1.2]},
        'versions': {'python': platform.python_version(), 'pymoo': package_version('pymoo'),
                     'numpy':package_version('numpy'),'pandas':package_version('pandas'),
                     'matplotlib':package_version('matplotlib'),'platform':platform.platform()},
        'source_files_sha256': source_files,
        'source_archives_sha256': {p.name: sha256(p) for p in sorted(INPUTS.iterdir()) if p.is_file()},
        'runner_sha256': sha256(__file__),
        'preflight': {'status': pre['status'], 'all_6_saved_12sector_AEP_verified': True,
                      'only_approved_scientific_change': {'wind.n_sectors':[12,360]},
                      'local_eval_seconds': pre['seconds_per_360sector_aep_evaluation']},
        'reporting_aep': {'sectors': 360, 'status':'pending'},
        'post_run_endpoint_verification': {'sectors': 720, 'pass_rule':
             'abs(AEP_1degree - AEP_0p5degree)/abs(AEP_0p5degree)*100 < 0.2',
             'status':'pending'},
        'output_policy': 'new directory only, no overwrite of existing folders',
    }


def save_fig(fig, folder, basename):
    for ext in ('png','pdf'):
        fig.savefig(folder / f'{basename}.{ext}', dpi=220, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def endpoint_plot(xy, key, metric, folder):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    other = np.arange(N) if metric == '2b' else np.arange(1, N)
    shape = 'o' if key == 'energy_optimal' else 'D'
    ax.scatter(xy[other,0], xy[other,1], s=78, marker=shape,
               color='steelblue', edgecolor='black', linewidth=.7, zorder=4,
               label='Turbines' if metric == '2b' else 'Other turbines')
    if metric == '2a':
        ax.scatter([xy[0,0]], [xy[0,1]], s=195, marker='*',
                   facecolor='darkorange', edgecolor='black', linewidth=.9,
                   zorder=6, label='Designated turbine')
        ax.legend(frameon=False)
    ax.set(xlabel='x position (m)', ylabel='y position (m)',
           xlim=(-160,6160), ylim=(-160,6160))
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.17)
    fig.tight_layout()
    save_fig(fig, folder, f'layout_{key}')


def pareto_plot(df, anchor, metric, folder, refs=None, suffix=''):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(df['aep_gwh'], df['risk_Q'], s=30, color='slategrey',
               alpha=.8, label='NSGA-II Pareto layouts')
    for anchor_name, marker, label, color in [
        ('energy_optimal','o','Highest AEP','darkorange'),
        ('collision_optimal','D','Lowest risk','firebrick')]:
        row = df.loc[anchor[anchor_name]]
        ax.scatter([row.aep_gwh],[row.risk_Q],s=110, marker=marker,
                   color=color, edgecolor='black',linewidth=.7,label=label,zorder=5)
    if refs:
        for row in refs:
            ax.scatter([row['aep_gwh']],[row['risk_Q']],marker=row['marker'],
                       s=155, color=row['color'],edgecolor='black',linewidth=.7,
                       label=row['label'],zorder=5)
    ax.set(xlabel='AEP (GWh/year)', ylabel=f'Q_{metric} (dimensionless)')
    ax.grid(alpha=.2); ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(); save_fig(fig,folder,f'metric_{metric}_pareto{suffix}')


def hv_plot(history, folder):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(np.arange(1,len(history)+1),history)
    ax.set(xlabel='Generation',ylabel='Hypervolume')
    ax.grid(alpha=.2); fig.tight_layout(); save_fig(fig,folder,'hypervolume')


def aligned_pairs(xy):
    rows = []
    for i in range(N):
        for j in range(i+1,N):
            dx, dy = xy[j]-xy[i]
            dist = float(np.hypot(dx,dy))
            # Orientation modulo 180; difference from 45-225 degree axis.
            angle = math.degrees(math.atan2(dy,dx)) % 180
            axis_offset = abs((angle-45+90)%180 - 90)
            if dist <= 1920. + 1e-9 and axis_offset <= 5. + 1e-9:
                rows.append({'turbine_i_index0':i, 'turbine_j_index0':j,
                             'distance_m':dist, 'axis_offset_deg':axis_offset,
                             'line_angle_mod180_deg':angle})
    return pd.DataFrame(rows,columns=['turbine_i_index0','turbine_j_index0',
               'distance_m','axis_offset_deg','line_angle_mod180_deg'])


def target_only_samples(xy_energy, Q_joint, model, folder):
    # No optimiser. Neighbour positions are frozen from 2a highest-AEP layout.
    fixed = xy_energy[1:].copy()
    xs = np.linspace(0,6000,101)  # 60 m regular grid
    xx, yy = np.meshgrid(xs,xs)
    candidates = np.vstack([np.stack([xx.ravel(),yy.ravel()],axis=1), xy_energy[0:1]])
    # Include the current highest-AEP target explicitly; the sampled optimum
    # must be no worse than leaving the target where it originally was.
    distances = np.linalg.norm(candidates[:,None,:]-fixed[None,:,:],axis=2)
    feasible = (distances >= MIN_SPACING_M-1e-6).all(axis=1)
    positions = candidates[feasible]
    check(len(positions)>0,'No feasible target-only position on 60m grid.')
    q = np.exp(-np.sum((positions[:,None,:]-fixed[None,:,:])**2,axis=2)
             /(2*model.sigma_m**2)).sum(axis=1)
    best_idx = int(np.argmin(q))
    rows = pd.DataFrame({'x_m':positions[:,0], 'y_m':positions[:,1],
                         'Q_2a':q, 'min_distance_m':distances[feasible].min(axis=1)})
    rows.to_csv(folder / 'metric_2a_target_only_samples.csv',index=False)
    summary = {'method':'regular feasible 60m spatial grid; no optimisation',
               'other_11_turbines_frozen_from':'new highest-AEP Pareto layout',
               'candidate_grid_points':int(len(candidates)),
               'feasible_target_positions':int(len(positions)),
               'best_sampled_target_xy_m':positions[best_idx].tolist(),
               'best_sampled_Q_2a':float(q[best_idx]),
               'best_sampled_normalised_f2':float(q[best_idx]/11),
               'jointly_optimised_lowest_Q_2a':float(Q_joint),
               'target_only_minus_joint_Q':float(q[best_idx]-Q_joint),
               'warning':'grid-sampled minimum, not a proven continuous optimum'}
    atomic_json(folder/'metric_2a_target_only_summary.json',summary)
    return summary


def fresh_aep_report(mods, X, old_df, folder, selected):
    new_1 = mods['AEPModel'](get_scenario(mods,360))
    fresh=[]
    for i, vec in enumerate(X):
        reported=float(new_1.evaluate(vec.reshape(N,2)))
        optim=float(old_df.loc[i,'aep_gwh'])
        fresh.append({'layout_id':i,'aep_optimisation_1deg_gwh':optim,
           'aep_reported_1deg_gwh':reported,'delta_reported_minus_optimisation_gwh':reported-optim,
           'abs_delta_pct_of_optimisation':100*abs(reported-optim)/abs(optim),
           'risk_Q':float(old_df.loc[i,'risk_Q'])})
    pd.DataFrame(fresh).to_csv(folder/'pareto_aep_reported_1deg.csv',index=False)
    new_half=mods['AEPModel'](get_scenario(mods,720))
    checks=[]
    for name in ('energy_optimal','collision_optimal'):
        ix=selected[name]
        a1=fresh[ix]['aep_reported_1deg_gwh']
        a05=float(new_half.evaluate(X[ix].reshape(N,2)))
        delta=abs(a1-a05)
        pct=100*delta/abs(a05)
        checks.append({'anchor':name,'layout_id':ix,
            'aep_optimisation_1deg_gwh':fresh[ix]['aep_optimisation_1deg_gwh'],
            'aep_reported_1deg_gwh':a1, 'aep_check_0p5deg_gwh':a05,
            'delta_reported_minus_optimisation_gwh':fresh[ix]['delta_reported_minus_optimisation_gwh'],
            'abs_delta_1deg_vs_0p5deg_gwh':delta, 'abs_delta_pct_of_0p5deg':pct,
            'threshold_pct_strict':.2, 'convergence_pass':bool(pct < .2)})
    pd.DataFrame(checks).to_csv(folder/'endpoint_0p5deg_convergence.csv',index=False)
    return fresh, checks


def ref_eval(mods, xy, metric='2b'):
    data = {'risk_Q':float(risk_model(mods,metric,get_scenario(mods,360)).collisions(xy))}
    for sectors, key in [(12,'aep_12sector_gwh'),(360,'aep_1deg_gwh'),(720,'aep_0p5deg_gwh')]:
        data[key] = float(mods['AEPModel'](get_scenario(mods,sectors)).evaluate(xy))
    data['aep_difference_gwh']=abs(data['aep_1deg_gwh']-data['aep_0p5deg_gwh'])
    data['difference_pct_of_0p5deg']=100*data['aep_difference_gwh']/abs(data['aep_0p5deg_gwh'])
    return data


def legacy_refs(mods):
    return {
        'historical_aep_only':ref_eval(mods,csv_xy(pd.read_csv(INPUTS/'aep_only_layout_original.csv'))),
        'risk_only':ref_eval(mods,csv_xy(pd.read_csv(INPUTS/'risk_only_layout_original.csv'))),
    }


def run_metric(mods, metric, sources, pre):
    verify_previous_stage(metric)
    print(f'PREPARING METRIC {metric}; all archived 12-sector checks passed')
    spec=SPECS[metric]; scenario=get_scenario(mods,360)
    aep_model=mods['AEPModel'](scenario)
    model=risk_model(mods, metric, scenario)
    problem=mods['ValidationLayoutProblem'](scenario=scenario,aep_model=aep_model,
         risk_model=model,aep_min=.5*AEP_REF_GWH,aep_max=AEP_REF_GWH,
         risk_min=0.,risk_max=float(model.risk_scale()))
    sampling=mods['GridSampling'](scenario=scenario, seed=42, model_name='constant')
    # Intentionally identical to both original Model 5 and Model 6 NSGA-II constructors.
    algorithm=mods['NSGA2'](pop_size=spec['pop'],sampling=sampling,eliminate_duplicates=True)
    callback=mods['HVCallback'](ref_point=np.array([1.2,1.2]))
    folder=create_output(metric)
    settings=settings_template(metric,mods,scenario,aep_model,model,algorithm,sources,pre)
    settings['start_time_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(folder/'run_settings.json',settings)
    print('RUNNING:',folder,'\nMetric:',metric,'pop:',spec['pop'],'generations:',spec['gen'])
    result=mods['minimize'](problem,algorithm,termination=('n_gen',spec['gen']),
                            seed=spec['seed'],callback=callback,verbose=True)
    check(result.X is not None,'NSGA-II did not return feasible Pareto layouts.')
    X=np.atleast_2d(np.asarray(result.X,dtype=float)); F=np.atleast_2d(np.asarray(result.F,dtype=float))
    check(X.shape[1]==24 and F.shape==(len(X),2),'Unexpected optimizer X/F shapes.')
    rows=[]; long=[]
    for layout_id, vector in enumerate(X):
        xy=vector.reshape(N,2)
        separation=check_feasible(mods,xy)
        Q=float(model.collisions(xy))
        annual=float(aep_model.evaluate(xy))
        expectedF=[2-2*annual/AEP_REF_GWH,Q/(11 if metric=='2a' else 66)]
        check(np.allclose(F[layout_id],expectedF,rtol=0,atol=1e-10),
              f'Unexpected objective mismatch at layout {layout_id}.')
        row={'layout_id':layout_id,'aep_gwh':annual,'risk_Q':Q,
             'f1_normalised_minimise':float(F[layout_id,0]),
             'f2_normalised_risk':float(F[layout_id,1]),
             'min_spacing_m':separation,'spacing_violations':0,'boundary_violations':0}
        if metric=='2a':
            distances=np.linalg.norm(xy[1:]-xy[0],axis=1)
            row['designated_index0']=0
            row['designated_nearest_neighbour_m']=float(distances.min())
            row['designated_x_m']=float(xy[0,0]);row['designated_y_m']=float(xy[0,1])
        else:
            diagnostics=model.diagnostics(xy)
            row['minimum_pair_distance_m']=float(diagnostics['minimum_pair_distance_m'])
            row['mean_pair_distance_m']=float(diagnostics['mean_pair_distance_m'])
        rows.append(row)
        for tid,(x,y) in enumerate(xy):
            long.append({'layout_id':layout_id,'turbine_id_index0':tid,
                         'x_m':float(x),'y_m':float(y), 'is_designated_2a':metric=='2a' and tid==0})
    df=pd.DataFrame(rows)
    # Save all final result coordinates before convergence checks, even if a check fails.
    np.savez_compressed(folder/'pareto_full.npz',X=X,F=F)
    pd.DataFrame(long).to_csv(folder/'pareto_layouts_long.csv',index=False)
    df.to_csv(folder/'pareto_points.csv',index=False)
    selected={'energy_optimal':int(df['aep_gwh'].idxmax()),
              'collision_optimal':int(df['risk_Q'].idxmin())}
    anchors=[]
    for name,index in selected.items():
        row=dict(df.loc[index]);row['layout']=name
        row['aep_difference_from_1d_reference_gwh']=row['aep_gwh']-OLD_JENSEN_BEST_1D_GWH
        anchors.append(row)
        xy=X[index].reshape(N,2)
        pd.DataFrame({'turbine_id':np.arange(N),'x_m':xy[:,0],
                      'y_m':xy[:,1], 'is_designated':np.arange(N)==0 if metric=='2a' else False}).to_csv(
            folder/f'layout_{name}.csv',index=False)
        aligned_pairs(xy).to_csv(folder/f'aligned_pairs_{name}.csv',index=False)
    pd.DataFrame(anchors).to_csv(folder/'anchor_summary.csv',index=False)
    fresh, checks=fresh_aep_report(mods,X,df,folder,selected)
    all_pass=all(c['convergence_pass'] for c in checks)
    settings['reporting_aep']={'status':'complete','n_sectors':360,
         'max_abs_delta_pct_vs_optimisation':max(z['abs_delta_pct_of_optimisation'] for z in fresh)}
    settings['post_run_endpoint_verification']={'status':'pass' if all_pass else 'FAIL_STOP',
         'n_sectors':720,'results':checks}
    settings['non_dominated_layouts_saved']=len(X)
    pd.DataFrame({'generation':np.arange(1,len(callback.data['hv'])+1),
                  'hypervolume':callback.data['hv']}).to_csv(folder/'hypervolume.csv',index=False)
    for anchor_name, idx in selected.items():
        endpoint_plot(X[idx].reshape(N,2),anchor_name,metric,folder)
    ref_rows=None
    if metric=='2b':
        refs=legacy_refs(mods)
        pd.DataFrame([dict(reference=k,**v) for k,v in refs.items()]).to_csv(
             folder/'single_objective_references_rechecked.csv',index=False)
        ref_rows=[{'aep_gwh':refs['historical_aep_only']['aep_1deg_gwh'],
                   'risk_Q':refs['historical_aep_only']['risk_Q'],
                   'label':'Historical AEP-only (1° re-eval.)','marker':'^','color':'purple'},
                  {'aep_gwh':refs['risk_only']['aep_1deg_gwh'],
                   'risk_Q':refs['risk_only']['risk_Q'],
                   'label':'Risk-only (1° re-eval.)','marker':'P','color':'saddlebrown'}]
        settings['risk_only_reference'] = refs['risk_only']
        settings['original_aep_only_reference'] = refs['historical_aep_only']
    pareto_plot(df,selected,metric,folder,refs=ref_rows)
    hv_plot(callback.data['hv'],folder)
    riskEnd = df.loc[selected['collision_optimal']]
    energyEnd = df.loc[selected['energy_optimal']]
    settings['endpoint_loss_gwh']=float(energyEnd.aep_gwh-riskEnd.aep_gwh)
    settings['endpoint_loss_percent']=100*settings['endpoint_loss_gwh']/float(energyEnd.aep_gwh)
    settings['aligned_pairs_counts']={k:len(aligned_pairs(X[idx].reshape(N,2)))
            for k,idx in selected.items()}
    if metric=='2a':
        settings['target_only_diagnostic']=target_only_samples(
            X[selected['energy_optimal']].reshape(N,2),float(riskEnd.risk_Q),model,folder)
    settings['run_status']='complete' if all_pass else 'STOP_convergence_failed'
    settings['finish_time_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(folder/'run_settings.json',settings)
    print('\nNEW 1° ENDPOINT 0.5° CONVERGENCE:')
    print(pd.DataFrame(checks).to_string(index=False))
    print('\nSaved',len(X),'nondominated layouts to:',folder)
    print('Endpoint AEP loss:',f'{settings["endpoint_loss_gwh"]:.6f} GWh',
          f'({settings["endpoint_loss_percent"]:.4f}%)')
    if not all_pass:
        raise SystemExit('STOP: one or more endpoints failed STRICT <0.2%. DO NOT start the next job.')
    print('PASS: both endpoints meet STRICT <0.2%. Next stage now permitted.')


def run_aep_reference(mods,sources,pre):
    verify_previous_stage('aep')
    # The original source AEPOnlyLayoutProblem, GA, GridSampling, and original
    # convergence callback are used. We do NOT recreate the missing problem class.
    from aep_only_problem import AEPOnlyLayoutProblem
    from run_aep_only import AEPConvergenceCallback, save_final_population
    scenario=get_scenario(mods,360)
    aep_model=mods['AEPModel'](scenario)
    check(math.isclose(aep_model.ideal_aep(12),AEP_REF_GWH,abs_tol=1e-6),
          'AEP only: wake-free reference mismatch.')
    problem=AEPOnlyLayoutProblem(scenario=scenario,aep_model=aep_model,
            aep_min=.5*AEP_REF_GWH,aep_max=AEP_REF_GWH)
    sampling=mods['GridSampling'](scenario=scenario,seed=42,model_name='constant')
    algorithm=mods['GA'](pop_size=160,sampling=sampling,eliminate_duplicates=True)
    callback=AEPConvergenceCallback(problem)
    folder=create_output('aep')
    settings=settings_template('aep',mods,scenario,aep_model,None,algorithm,sources,pre)
    settings['start_time_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(folder/'run_settings.json',settings)
    print('RUNNING independent 360-sector AEP-only reference GA, pop=160 gen=250 seed=42')
    result=mods['minimize'](problem,algorithm,termination=('n_gen',250),
                            seed=42,callback=callback,verbose=True)
    check(result.X is not None,'AEP-only GA found no feasible layout.')
    X=np.atleast_2d(np.asarray(result.X,dtype=float))
    aep_values=np.array([aep_model.evaluate(x.reshape(N,2)) for x in X])
    best=int(np.argmax(aep_values)); xy=X[best].reshape(N,2)
    check_feasible(mods,xy)
    a1=float(aep_values[best])
    check(a1 <= AEP_REF_GWH+1e-6,'Optimised AEP exceeds wake-free ceiling.')
    a1_independent=float(mods['AEPModel'](get_scenario(mods,360)).evaluate(xy))
    a05=float(mods['AEPModel'](get_scenario(mods,720)).evaluate(xy))
    delta=abs(a1_independent-a05)
    pct=100*delta/abs(a05)
    converge={'anchor':'aep_only_best','aep_optimisation_1deg_gwh':a1,
              'aep_reported_1deg_gwh':a1_independent,
              'delta_reported_minus_optimisation_gwh':a1_independent-a1,
              'aep_check_0p5deg_gwh':a05,
              'abs_delta_1deg_vs_0p5deg_gwh':delta,
              'abs_delta_pct_of_0p5deg':pct,
              'threshold_pct_strict':.2,'convergence_pass':bool(pct < .2)}
    pd.DataFrame([converge]).to_csv(folder/'endpoint_0p5deg_convergence.csv',index=False)
    pd.DataFrame({'turbine_id':np.arange(1,N+1),'x_m':xy[:,0],
                  'y_m':xy[:,1]}).to_csv(folder/'aep_only_layout.csv',index=False)
    np.savez_compressed(folder/'aep_only_optimum.npz',X=xy,F=np.asarray(result.F))
    save_final_population(result,scenario,aep_model,folder/'final_population_summary.csv')
    pop=np.asarray(result.algorithm.pop.get('X'),dtype=float)
    np.savez_compressed(folder/'aep_only_final_population_full.npz',X=pop)
    pd.DataFrame({'generation':callback.data['generation'],
                  'best_aep_gwh':callback.data['best_aep_gwh'],
                  'best_objective':callback.data['best_objective']}).to_csv(
        folder/'aep_convergence.csv',index=False)
    # Original risk-only saved layout is evaluated, not re-optimised.
    riskxy=csv_xy(pd.read_csv(INPUTS/'risk_only_layout_original.csv'))
    risk_ref=ref_eval(mods,riskxy)
    pd.DataFrame([dict(reference='risk_only_not_rerun',**risk_ref)]).to_csv(
        folder/'risk_only_reference_recheck.csv',index=False)
    endpoint_plot(xy,'energy_optimal','2b',folder)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(callback.data['generation'],callback.data['best_aep_gwh'])
    ax.set(xlabel='Generation',ylabel='Best AEP (GWh/year)')
    ax.grid(alpha=.2); fig.tight_layout();save_fig(fig,folder,'aep_convergence')
    settings['reporting_aep']={'status':'complete','new_common_Jensen_energy_reference_gwh':a1_independent,
               'use_as_energy_reference_for':['1a','1b','1c','1d','2a','2b'],
               'note':'comparison benchmark only; does not alter the preserved objective normalisations'}
    settings['post_run_endpoint_verification']={'status':'pass' if pct<.2 else 'FAIL_STOP','results':[converge]}
    settings['risk_only_reference']=risk_ref
    settings['run_status']='complete' if pct<.2 else 'STOP_convergence_failed'
    settings['finish_time_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(folder/'run_settings.json',settings)
    print('\nAEP-only GA endpoint check:',pd.DataFrame([converge]).to_string(index=False))
    if pct>=.2:
        raise SystemExit('STOP: AEP-only reference fails the strict 0.2% check.')
    print(f'PASS. New common Jensen energy benchmark: {a1_independent:.12f} GWh/year')
    # Create NEW figure filenames within completed 2b results (never overwrite
    # existing figures), this time showing the newly optimised AEP-only reference.
    folder2b=RESULTS/NAMES['2b']
    df=pd.read_csv(folder2b/'pareto_points.csv')
    selected={'energy_optimal':int(df['aep_gwh'].idxmax()),
              'collision_optimal':int(df['risk_Q'].idxmin())}
    rows=[{'aep_gwh':a1_independent,
           'risk_Q':float(risk_model(mods,'2b',get_scenario(mods,360)).collisions(xy)),
           'label':'New AEP-only (360 sectors)','marker':'^','color':'purple'},
          {'aep_gwh':risk_ref['aep_1deg_gwh'],'risk_Q':risk_ref['risk_Q'],
           'label':'Risk-only reference (1° re-eval.)','marker':'P','color':'saddlebrown'}]
    # Save FINAL 2b-with-references plot in the NEW AEP-only folder. Do NOT
    # add or modify files in the already completed Metric 2b results directory.
    pareto_plot(df,selected,'2b',folder,refs=rows,suffix='_final_refs')
    pd.DataFrame([{'reference':z['label'],'aep_1deg_gwh':z['aep_gwh'],'risk_Q':z['risk_Q']}
                  for z in rows]).to_csv(folder/'final_reference_comparison.csv',index=False)
    print('Saved 2b Pareto figure with final references in NEW AEP-only folder (original 2b untouched).')
    aep_front=float(df['aep_gwh'].max());risk_front=float(df['risk_Q'].min())
    print(f'2b front reaches AEP-only reference? {aep_front >= a1_independent-1e-6} '
          f'(front={aep_front:.8f}; reference={a1_independent:.8f}; gap={a1_independent-aep_front:.8f} GWh)')
    print(f'2b front reaches risk-only reference? {risk_front <= risk_ref["risk_Q"]+1e-9} '
          f'(front={risk_front:.8f}; reference={risk_ref["risk_Q"]:.8f})')
    # Produce a unified benchmark table without changing previously saved models.
    benchmark=[]
    for m in ('1a','1b','1c','1d','2a','2b'):
        matches = sorted(RESULTS.glob(f'metric_{m}_360sectors_seed42_pop*_gen*'))
        if not matches:
            benchmark.append({'metric':m,'status':'result folder not found'})
            continue
        original_folder=matches[0]
        points=original_folder/'pareto_points.csv'
        if not points.is_file():
            benchmark.append({'metric':m,'status':'pareto_points.csv not found'})
            continue
        points_df=pd.read_csv(points)
        if 'aep_gwh' not in points_df.columns:
            benchmark.append({'metric':m,'status':'aep_gwh not available'})
            continue
        for endpoint, row in [('highest_aep',points_df.loc[points_df['aep_gwh'].idxmax()]),
                              ('lowest_risk',points_df.loc[points_df['risk_Q'].idxmin()]
                               if 'risk_Q' in points_df.columns else
                               points_df.loc[points_df['risk'].idxmin()])]:
            benchmark.append({'metric':m,'status':'verified saved 1° layout',
                    'endpoint':endpoint,'aep_1deg_gwh':float(row.aep_gwh),
                    'new_common_reference_gwh':a1_independent,
                    'aep_difference_from_reference_gwh':float(row.aep_gwh)-a1_independent,
                    'aep_difference_from_reference_percent':100*(float(row.aep_gwh)-a1_independent)/a1_independent})
    pd.DataFrame(benchmark).to_csv(folder/'jensen_energy_reference_vs_metrics_1a_to_2b.csv',index=False)
    print('Created common Jensen-reference comparison table for available Metrics 1a–2b.')
    print('The new common energy reference is a COMPARISON VALUE; it does not change f1 for 1a-2b.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('task',choices=['preflight','run-2a','run-2b','run-aep'])
    args=parser.parse_args()
    print('Project root:',ROOT)
    print('Approved master results:',RESULTS)
    check(RESULTS.is_dir(),'Create/verify original jensen_1deg_approved/results directory first.')
    # Every stage repeats all six 12-sector AEP checks; no launch if they fail.
    source_files=source_inventory()
    mods=imports()
    report=preflight(mods,source_files)
    if args.task=='preflight':
        filename=RESULTS/('metric2_preflight_report_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
        check(not filename.exists(), f'Preflight report exists; WILL NOT OVERWRITE {filename}')
        atomic_json(filename,report)
        print('Wrote:',filename)
    elif args.task=='run-2a':
        run_metric(mods,'2a',source_files,report)
    elif args.task=='run-2b':
        run_metric(mods,'2b',source_files,report)
    elif args.task=='run-aep':
        run_aep_reference(mods,source_files,report)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('\nSTOP: '+str(exc),file=sys.stderr)
        raise SystemExit(2) from exc
