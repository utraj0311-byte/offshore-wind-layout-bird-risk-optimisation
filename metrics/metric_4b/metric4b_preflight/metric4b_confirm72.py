#!/usr/bin/env python
"""Metric 4b: independent paired evaluation of two frozen 72-sector endpoints.

Commands (from bird_layout_opt/metric4b_preflight):
  python metric4b_confirm72.py verify
  python metric4b_confirm72.py smoke
  python metric4b_confirm72.py run
  python metric4b_confirm72.py resume  # only after interrupted run

The full job uses 15 original independent evaluation seed pairs, 100000 birds per
seed, common widened corridor, p=0.6/p=0, and baseline/fixed z=150 m.
Scientific source modules are imported unmodified from model9_code.
There is no AEP recomputation or optimisation in this script.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = HERE / 'model9_code'
RUN = ROOT / 'metric_4b_72sectors'
POST = ROOT / 'metric_4b_72sectors_postrun'
OUT = ROOT / 'metric_4b_72sectors_confirmation'

if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

HEADING = 225.0
D = 242.24
R = D / 2.0
SITE = 6000.0
N_BATCHES = 15
N_PER_BATCH = 100000
N_SUBSAMPLE = 10000
SEED_TRAJ = 30142
SEED_HEIGHT = 40142
EXPECTED_HASH = 'f6b788df9e38f006dfb6c68ddfb80ab090dad9cd555982a8d5c5e4b0c2f2d364'
SCENARIOS = ('baseline_p06', 'no_response_p0', 'fixed_height_p06', 'fixed_height_p0')
LAYOUTS = ('highest-AEP', 'lowest-risk')


def require(ok, message):
    if not ok:
        raise RuntimeError('STOP | ' + message)
    print('PASS | ' + message, flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialize(obj):
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def geometry(layouts):
    theta = np.deg2rad(HEADING)
    cross_vec = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    corners = np.array([[0., 0.], [SITE, 0.], [0., SITE], [SITE, SITE]])
    projected = corners @ cross_vec
    orig_min = float(projected.min())
    orig_max = float(projected.max())
    lower, upper = orig_min, orig_max
    margins = {}
    for name, xy in layouts.items():
        c = xy @ cross_vec
        margins[name] = float(min(np.min(c - R - orig_min), np.min(orig_max - c - R)))
        lower = min(lower, float(c.min() - R))
        upper = max(upper, float(c.max() + R))
    widened = min(margins.values()) < 0.0
    if widened:
        lower -= 1.0
        upper += 1.0
    return dict(original_min=orig_min, original_max=orig_max,
                original_width=orig_max-orig_min, common_min=lower,
                common_max=upper, common_width=upper-lower,
                margin_by_layout=margins, enlarged=widened)


def verify():
    for p in [RUN/'completion_status.json', RUN/'run_settings.json',
              RUN/'highest-AEP_layout.csv', RUN/'lowest-risk_layout.csv',
              POST/'geometry_report.json', POST/'endpoint_1degree_aep.json',
              CODE/'data'/'northern_gannet_Johnston2014_FHD.csv',
              CODE/'data'/'northern_gannet_Johnston2014_FHD_metadata.json',
              CODE/'floris_config'/'gch_iea15mw.yaml']:
        require(p.is_file(), f'File exists: {p.name}')
    st = json.loads((RUN/'run_settings.json').read_text())
    complete = json.loads((RUN/'completion_status.json').read_text())
    aept = json.loads((POST/'endpoint_1degree_aep.json').read_text())
    require(complete.get('status') == 'optimisation_finished', 'Completed optimisation archive')
    require(st['n_sectors'] == 72 and st['seed'] == 142, '72 sectors, optimiser seed 142')
    require(st['objective_trajectory_seeds'] == list(range(142,147)) and
            st['objective_flight_height_seeds'] == list(range(10142,10147)),
            'Original optimisation sample seeds, disjoint from independent evaluation')
    require(st['runtime_versions']['floris'] == '4.6.5' and
            st['runtime_versions']['pymoo'] == '0.6.1.5', 'Locked software versions')
    require(sha(CODE/'floris_config'/'gch_iea15mw.yaml') == EXPECTED_HASH,
            'FLORIS input YAML unchanged')
    fhd_meta = json.loads((CODE/'data'/'northern_gannet_Johnston2014_FHD_metadata.json').read_text())
    require(sha(CODE/'data'/'northern_gannet_Johnston2014_FHD.csv') == fhd_meta['sha256'],
            'Original northern-gannet FHD checksum matches')
    require(st['flight_height_distribution']['height_row_count'] == 300,
            'All 300 flight-height probability masses retained')
    # Ensure the 1-degree endpoint gate was recorded; accept differing JSON report structures.
    require(aept.get('all_endpoints_pass') is True and
            len(aept.get('endpoints', [])) == 2 and
            all(bool(q.get('passes_below_0p2_percent')) for q in aept['endpoints']),
            'Both 1-degree checks passed below 0.2%')
    layouts = {name: pd.read_csv(RUN/f'{name}_layout.csv')[['x_m','y_m']].to_numpy(dtype=float)
               for name in LAYOUTS}
    for name, xy in layouts.items():
        require(xy.shape == (12,2) and np.isfinite(xy).all() and
                np.all((xy >= 0.0) & (xy <= SITE)), f'Valid frozen {name} coordinates')
    g = geometry(layouts)
    saved = json.loads((POST/'geometry_report.json').read_text())
    require(abs(g['common_width'] - saved['common_corridor_width_m']) < 1e-6 and
            abs(g['common_min'] - saved['common_corridor_min_cross_m']) < 1e-6 and
            abs(g['common_max'] - saved['common_corridor_max_cross_m']) < 1e-6,
            'Widened common corridor matches geometry audit')
    require(g['enlarged'] and g['margin_by_layout']['highest-AEP'] < 0.0,
            'Negative original corridor margin triggers widening')
    require(N_BATCHES == 15 and N_PER_BATCH == 100000 and
            list(range(SEED_TRAJ, SEED_TRAJ+N_BATCHES)) == list(range(30142, 30157)) and
            list(range(SEED_HEIGHT, SEED_HEIGHT+N_BATCHES)) == list(range(40142, 40157)),
            'Original independent evaluation seeds and sample size')
    print(f"INFO | Original corridor W={g['original_width']:.9f} m; "
          f"independent evaluation corridor W={g['common_width']:.9f} m", flush=True)
    print('INFO | 4 scenarios per batch: baseline p=0.6, no-response p=0, fixed z=150 m '
          'at p=0.6 and p=0', flush=True)
    print('INFO | 10,000-bird subgroups estimate the requested one-sample SD directly', flush=True)
    print('VERIFY COMPLETE | NO BIRD SIMULATION OR OPTIMISATION STARTED', flush=True)
    return layouts, g, st, fhd_meta


def build_models(fhd, fixed):
    from config import Scenario
    from model8_meso_response import BehaviourScenario
    from model9_rotor_transit import RotorGeometry, RotorTransitModel
    s = Scenario()
    b = s.birds
    rotor = RotorGeometry(rotor_diameter_m=D, hub_height_m=150.0)
    base = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=D, response_probability=b.response_probability,
        response_radius_D=b.response_radius_D, encounter_radius_D=b.encounter_radius_D,
        deflection_angle_deg=b.deflection_angle_deg,
        return_length_D=b.return_length_D, step_size_D=b.step_size_D)
    require(abs(base.response_probability-0.6)<1e-12, 'Baseline response probability 0.6')
    none = replace(base, response_probability=0.0)
    return {
      'baseline_p06': RotorTransitModel(base,rotor,rotor_plane_heading_deg_toward=HEADING, flight_height_distribution=fhd),
      'no_response_p0': RotorTransitModel(none,rotor,rotor_plane_heading_deg_toward=HEADING, flight_height_distribution=fhd),
      'fixed_height_p06': RotorTransitModel(base,rotor,rotor_plane_heading_deg_toward=HEADING, flight_height_distribution=fixed),
      'fixed_height_p0': RotorTransitModel(none,rotor,rotor_plane_heading_deg_toward=HEADING, flight_height_distribution=fixed)
    }, base


def slice_population(pop, lo, hi):
    from model8_meso_response import TrajectoryPopulation
    from model9_rotor_transit import Model9Population
    p = pop.trajectories
    track = TrajectoryPopulation(
        start_positions_m=p.start_positions_m[lo:hi],
        headings_deg_toward=p.headings_deg_toward[lo:hi],
        downstream_distance_m=p.downstream_distance_m[lo:hi],
        response_uniforms=p.response_uniforms[lo:hi],
        side_uniforms=p.side_uniforms[lo:hi],
        seed=p.seed,
        entry_cross_fractions=(None if p.entry_cross_fractions is None else p.entry_cross_fractions[lo:hi]))
    return Model9Population(trajectories=track, flight_heights_m=pop.flight_heights_m[lo:hi],
                height_uniforms=pop.height_uniforms[lo:hi],
                flight_height_seed=pop.flight_height_seed,
                flight_height_distribution_name=pop.flight_height_distribution_name)


def widened_population(fhd, g, batch_index, count, base):
    from model9_rotor_transit import Model9Population
    from model8_meso_response import TrajectoryPopulation, cross_unit_vectors
    seed = SEED_TRAJ + batch_index
    hseed = SEED_HEIGHT + batch_index
    pop = Model9Population.generate_parallel(
        flight_height_distribution=fhd, flight_height_seed=hseed,
        site_width_m=SITE, site_height_m=SITE, heading_deg_toward=HEADING,
        number_of_trajectories=count, maximum_response_events=__import__('config').Scenario().birds.maximum_response_events,
        longitudinal_margin_m=base.response_radius_m+base.return_length_m,
        trajectory_seed=seed)
    # Reuse the EXACT entry fractions and random draws generated by the original
    # factory. Only map cross-flight entry positions onto the common widened range.
    p = pop.trajectories
    fractions = p.entry_cross_fractions
    require(fractions is not None, 'Saved entry fractions available for corridor remapping')
    n = cross_unit_vectors(HEADING)
    old_cross = np.asarray(p.start_positions_m @ n, dtype=float)
    new_cross = g['common_min'] + fractions * g['common_width']
    new_starts = p.start_positions_m + (new_cross - old_cross)[:,None]*n[None,:]
    shifted = TrajectoryPopulation(
       start_positions_m=new_starts, headings_deg_toward=p.headings_deg_toward,
       downstream_distance_m=p.downstream_distance_m,
       response_uniforms=p.response_uniforms, side_uniforms=p.side_uniforms,
       seed=p.seed, entry_cross_fractions=fractions)
    return replace(pop, trajectories=shifted)


def chunk_result(model, xy, pop):
    from model9_batching import summarise_model9_batches
    res = summarise_model9_batches(model,xy,[pop])
    if res['N_failed']!=0 or res['N_completed']!=res['N_started']:
        raise RuntimeError('STOP | Failed bird trajectories in evaluation')
    return res['batch_summaries'][0]


def check_run_output_empty():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f'STOP | Output already contains files: {OUT}. Use resume only after interruption.')
    OUT.mkdir(parents=True,exist_ok=True)


def flush_csv(path, rows):
    tmp = path.with_name(path.name+'.tmp')
    pd.DataFrame(rows).to_csv(tmp,index=False)
    os.replace(tmp,path)


def gather_summary(batch_rows, chunk_rows, g):
    import math
    full = pd.DataFrame(batch_rows)
    chunks = pd.DataFrame(chunk_rows)
    require(len(full) == N_BATCHES*len(SCENARIOS)*len(LAYOUTS), 'All paired batch rows recorded')
    require(len(chunks) == N_BATCHES*(N_PER_BATCH//N_SUBSAMPLE)*len(SCENARIOS)*len(LAYOUTS),
            'All 10,000-bird groups recorded')
    summary = []
    paired = []
    original_front=pd.read_csv(RUN/'pareto_points.csv')
    training_q={name:float(original_front.loc[original_front['solution_index'] == idx,'Q_4b'].iloc[0])
        for name,idx in [('highest-AEP',18),('lowest-risk',12)]}
    for sc in SCENARIOS:
        bysc = full.loc[full.scenario==sc].copy()
        smallsc = chunks.loc[chunks.scenario==sc]
        for lay in LAYOUTS:
            grp = bysc.loc[bysc.layout==lay].sort_values('batch_index')
            sm = smallsc.loc[smallsc.layout==lay]
            x = grp.Q_4b.to_numpy(dtype=float)
            mean=float(x.mean()); sd=float(x.std(ddof=1))
            half=float(student_t.ppf(.975,len(x)-1)*sd/math.sqrt(len(x)))
            summary.append(dict(scenario=sc,layout=lay,
                independently_evaluated_Q_4b=mean,ci95_lower=mean-half,ci95_upper=mean+half,
                Q_4b_times_W=mean*g['common_width'],
                Q_4b_times_W_ci95_lower=(mean-half)*g['common_width'],
                Q_4b_times_W_ci95_upper=(mean+half)*g['common_width'],
                Q_4b_adjusted_to_original_W=mean*g['common_width']/g['original_width'],
                adjusted_Q_4b_ci95_lower=(mean-half)*g['common_width']/g['original_width'],
                adjusted_Q_4b_ci95_upper=(mean+half)*g['common_width']/g['original_width'],
                optimisation_sample_Q_4b=(training_q[lay] if sc=='baseline_p06' else float('nan')),
                adjusted_change_vs_optimisation_sample_percent=(
                    100*(mean*g['common_width']/g['original_width']/training_q[lay]-1)
                    if sc=='baseline_p06' else float('nan')),
                sd_one_10000_bird_sample=float(sm.Q_4b.std(ddof=1)),
                evaluation_batches=len(x),birds_per_batch=N_PER_BATCH,total_birds=len(x)*N_PER_BATCH,
                original_corridor_W_m=g['original_width'],common_corridor_W_m=g['common_width']))
        z = bysc.pivot(index='batch_index',columns='layout',values='Q_4b').sort_index()
        differences=(z['lowest-risk'] - z['highest-AEP']).to_numpy(dtype=float)
        mean=float(differences.mean()); sd=float(differences.std(ddof=1))
        hw=float(student_t.ppf(.975,len(differences)-1)*sd/math.sqrt(len(differences)))
        paired.append(dict(scenario=sc,difference_definition='lowest-risk minus highest-AEP',
             Q_4b_difference=mean,ci95_lower=mean-hw,ci95_upper=mean+hw,
             Q_4b_times_W_difference=mean*g['common_width'],
             Q_4b_times_W_ci95_lower=(mean-hw)*g['common_width'],
             Q_4b_times_W_ci95_upper=(mean+hw)*g['common_width'],
             relative_change_percent_vs_highest_AEP=100*mean/float(z['highest-AEP'].mean()),
             interval_excludes_zero=bool((mean-hw)>0 or (mean+hw)<0)))
    return pd.DataFrame(summary),pd.DataFrame(paired)


def evaluate(mode, *, quick=False):
    layouts, g, st, fhd_meta = verify()
    from model9_rotor_transit import DiscreteFlightHeightDistribution, Model9Population
    from model9_hub_height_sensitivity import fixed_hub_height_distribution
    src=CODE/'data'/'northern_gannet_Johnston2014_FHD.csv'
    fhd=DiscreteFlightHeightDistribution.from_csv(src,
        name='northern_gannet_Johnston2014_Gannet.est',source=fhd_meta['primary_citation'])
    fixed=fixed_hub_height_distribution(150.0)
    models, base=build_models(fhd,fixed)
    if quick:
        total_batches=1; n_per_batch=100; n_chunk=100
        print('SMOKE | 1 x 100 birds, no production output; original seeds',flush=True)
    else:
        total_batches=N_BATCHES; n_per_batch=N_PER_BATCH; n_chunk=N_SUBSAMPLE
        if mode=='run': check_run_output_empty()
        elif mode=='resume':
            require(OUT.is_dir() and (OUT/'independent_batches_checkpoint.csv').is_file(),
                    'Existing batch checkpoint found')
        else: raise ValueError(mode)
    batches=[];chunks=[];start_index=0
    if not quick:
        if mode=='resume':
            batches=pd.read_csv(OUT/'independent_batches_checkpoint.csv').to_dict('records')
            chunks=pd.read_csv(OUT/'sample_10000_checkpoint.csv').to_dict('records')
            completed=sorted(set(int(r['batch_index']) for r in batches))
            require(completed==list(range(len(completed))), 'Completed batches contiguous')
            require(len(batches)==len(completed)*len(SCENARIOS)*len(LAYOUTS),
                    'All four conditions and both layouts present per saved batch')
            require(len(chunks)==len(completed)*(N_PER_BATCH//N_SUBSAMPLE)*len(SCENARIOS)*len(LAYOUTS),
                    'All 10,000-bird subsamples present per saved batch')
            start_index=len(completed)
            print(f'RESUME | {start_index} complete batches recorded',flush=True)
        else:
            settings=dict(terminology='optimisation sample / independent evaluation sample',
                status='started',first_trajectory_seed=SEED_TRAJ,first_height_seed=SEED_HEIGHT,
                n_batches=N_BATCHES,n_per_batch=N_PER_BATCH,subsample_size=N_SUBSAMPLE,
                response_probabilities=[0.6,0.0],fixed_height_m=150.0,
                corridor=g,source_FHD_sha256=sha(src),
                source_run_settings_sha256=sha(RUN/'run_settings.json'),
                two_layouts={name:sha(RUN/f'{name}_layout.csv') for name in LAYOUTS},
                original_model9_source_unchanged=True,
                note='Conditional expected rotor transits per entering bird, not blade strikes. '
                     'Corridor correction affects evaluation only, not optimisation.')
            (OUT/'evaluation_settings.json').write_text(json.dumps(serialize(settings),indent=2))
    began=time.perf_counter()
    for index in range(start_index,total_batches):
        print(f'Batch {index+1}/{total_batches} | seeds {SEED_TRAJ+index} / {SEED_HEIGHT+index}',flush=True)
        pop = widened_population(fhd,g,index,n_per_batch,base)
        for block in range(n_per_batch//n_chunk):
            a,b=block*n_chunk,(block+1)*n_chunk
            sub=slice_population(pop,a,b)
            fixed_sub=replace(sub,flight_heights_m=np.full(n_chunk,150.,dtype=float),
                    flight_height_distribution_name=fixed.name)
            for sc in SCENARIOS:
                for lay in LAYOUTS:
                    actual_pop=fixed_sub if sc.startswith('fixed_height') else sub
                    row=chunk_result(models[sc],layouts[lay],actual_pop)
                    value=float(row['E_RT_CE'])
                    chunks.append(dict(batch_index=index,block_index=block,scenario=sc,layout=lay,
                        trajectory_seed=SEED_TRAJ+index,flight_height_seed=SEED_HEIGHT+index,
                        n_started=int(row['N_started']),n_completed=int(row['N_completed']),
                        Q_4b=value,N_RT_CE_farm=float(row['N_RT_CE_farm'])))
        for sc in SCENARIOS:
            for lay in LAYOUTS:
                select=[r for r in chunks if int(r['batch_index'])==index and r['scenario']==sc and r['layout']==lay]
                assert len(select)==n_per_batch//n_chunk
                exp_total=float(sum(r['N_RT_CE_farm'] for r in select))
                q=exp_total/n_per_batch
                batches.append(dict(batch_index=index,scenario=sc,layout=lay,
                    trajectory_seed=SEED_TRAJ+index,flight_height_seed=SEED_HEIGHT+index,
                    n_started=n_per_batch,n_completed=n_per_batch,Q_4b=q,
                    N_RT_CE_farm=exp_total,Q_4b_times_W=q*g['common_width']))
        if not quick:
            flush_csv(OUT/'independent_batches_checkpoint.csv',batches)
            flush_csv(OUT/'sample_10000_checkpoint.csv',chunks)
        print(f'Completed batch {index+1}/{total_batches}; elapsed {(time.perf_counter()-began)/60:.1f} min',flush=True)
    if quick:
        for row in batches:
            print(f"SMOKE | {row['scenario']} | {row['layout']} | Q_4b={row['Q_4b']:.9g}",flush=True)
        print('SMOKE COMPLETE | NO PRODUCTION OUTPUT WRITTEN',flush=True)
        return
    summary,paired=gather_summary(batches,chunks,g)
    flush_csv(OUT/'independent_evaluation_batches.csv',batches)
    flush_csv(OUT/'independent_evaluation_10000_subsamples.csv',chunks)
    flush_csv(OUT/'independent_evaluation_endpoint_summary.csv',summary.to_dict('records'))
    flush_csv(OUT/'independent_evaluation_paired_differences.csv',paired.to_dict('records'))
    settings=json.loads((OUT/'evaluation_settings.json').read_text());settings.update(status='completed',
            elapsed_minutes=(time.perf_counter()-began)/60)
    (OUT/'evaluation_settings.json').write_text(json.dumps(serialize(settings),indent=2))
    print('\nINDEPENDENT EVALUATION COMPLETE')
    print(summary[['scenario','layout','independently_evaluated_Q_4b','sd_one_10000_bird_sample']].to_string(index=False))
    print(paired[['scenario','Q_4b_difference','ci95_lower','ci95_upper']].to_string(index=False))
    print('Reports:',OUT)
    print('Original optimisation remains read-only.')


def main():
    a=argparse.ArgumentParser(description='Metric 4b independent evaluation')
    a.add_argument('command',choices=('verify','smoke','run','resume'))
    args=a.parse_args()
    if args.command=='verify': verify()
    elif args.command=='smoke': evaluate('smoke',quick=True)
    else: evaluate(args.command)

if __name__=='__main__':
    main()
