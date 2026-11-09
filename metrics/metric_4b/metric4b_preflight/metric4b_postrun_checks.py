#!/usr/bin/env python
"""Metric 4b: read-only post-optimisation reproducibility and sector-resolution tests.

Run inside the *existing* floris-env, with this file placed directly beside
metric4b_preflight.py and the model9_code folder:

    python metric4b_postrun_checks.py audit
    python metric4b_postrun_checks.py endpoints
    python metric4b_postrun_checks.py fullfront1deg   # ONLY if endpoints fail <0.2%

Reads bird_layout_opt/metric_4b_72sectors and original unchanged model9_code.
Writes reports only into bird_layout_opt/metric_4b_72sectors_postrun.
No NSGA-II optimisation, no bird simulations, no changes to source results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'model9_code'
RUN = ROOT.parent / 'metric_4b_72sectors'
OUT = ROOT.parent / 'metric_4b_72sectors_postrun'

D = 242.24
ROTOR_RADIUS = D / 2.0
HEADING_DEG_TOWARD = 225.0
SITE = 6000.0
AEP_CEILING = 854.4956585617186
AEP_LOSS_REF = 137.96914764866568
Q_REF = 8.704996418223796e-4
AEP_REPRO_TOL_GWH = 0.001
ONE_DEGREE_TOL_PERCENT = 0.2
HASH_TURBINE = '93d2bcfcb619b1819efe8149ce9e8231656e51a29f2ae5c1079a46e339f02feb'
HASH_CONFIG = 'f6b788df9e38f006dfb6c68ddfb80ab090dad9cd555982a8d5c5e4b0c2f2d364'


def check(ok, description):
    print(('PASS' if ok else 'FAIL') + ' | ' + description, flush=True)
    if not bool(ok):
        raise RuntimeError('STOP: ' + description)


def save_json(filename, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    return path


def sha256(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def load_archive():
    required = ['completion_status.json', 'training_summary.json', 'run_settings.json',
                'settings_diff.json', 'file_hashes.json', 'pareto_points.csv',
                'pareto_decision_vectors.npy', 'pareto_normalised_objectives.npy',
                'final_population.npz', 'all_evaluated_candidates.csv',
                'selected_initial_population.csv', 'hypervolume.csv',
                'highest-AEP_layout.csv', 'lowest-risk_layout.csv']
    check(RUN.is_dir(), 'Original completed Metric 4b output folder exists')
    check(all((RUN / fn).is_file() for fn in required), 'All required archive files present')

    completion = json.loads((RUN/'completion_status.json').read_text(encoding='utf-8'))
    settings = json.loads((RUN/'run_settings.json').read_text(encoding='utf-8'))
    diff = json.loads((RUN/'settings_diff.json').read_text(encoding='utf-8'))
    summary = json.loads((RUN/'training_summary.json').read_text(encoding='utf-8'))
    df = pd.read_csv(RUN/'pareto_points.csv')
    X = np.load(RUN/'pareto_decision_vectors.npy', allow_pickle=False)
    F = np.load(RUN/'pareto_normalised_objectives.npy', allow_pickle=False)
    with np.load(RUN/'final_population.npz', allow_pickle=False) as arr:
        pop = {name: arr[name].copy() for name in arr.files}
    candidate_data = pd.read_csv(RUN/'all_evaluated_candidates.csv')
    hv = pd.read_csv(RUN/'hypervolume.csv')
    initial = pd.read_csv(RUN/'selected_initial_population.csv')

    check(completion.get('status') == 'optimisation_finished' and settings.get('run_status') == 'completed',
          'Completed run marker and settings agree')
    check(int(completion.get('generations',0)) == 100 and int(settings['profile']['generations']) == 100,
          '100 generations recorded')
    check(int(completion.get('function_evaluations',0)) == 8000 and
          int(settings.get('function_evaluations',0)) == 8000 and len(candidate_data) == 8000 and
          candidate_data.evaluation_index.tolist() == list(range(8000)),
          'Exactly 8,000 candidate evaluation records (0..7999)')
    check(int(settings['seed']) == 142 and int(settings['profile']['population']) == 80,
          'Seed 142 and population 80')
    check(int(settings['profile']['objective_batches']) == 5 and
          int(settings['profile']['objective_trajectories_per_batch']) == 2000 and
          settings['objective_trajectory_seeds'] == [142,143,144,145,146] and
          settings['objective_flight_height_seeds'] == [10142,10143,10144,10145,10146],
          'Optimisation sample: 5 x 2,000 birds and original seeds')
    check(int(settings['n_sectors']) == 72 and int(settings['aep_model']['number_of_combined_conditions']) == 504,
          '72 direction sectors and 504 FLORIS conditions')
    check(abs(float(settings['problem']['rotor_transit_reference']) - Q_REF) < 1e-15 and
          abs(float(settings['problem']['aep_ceiling_gwh']) - AEP_CEILING) < 1e-10 and
          abs(float(settings['problem']['aep_loss_reference_gwh']) - AEP_LOSS_REF) < 1e-10,
          'Original normalisation references preserved')
    check(diff.get('independent_scientific_change') == {'n_sectors':[12,72]} and
          diff.get('other_scientific_differences') == [],
          'Only scientific-setting difference is 12 -> 72 sectors')
    check(settings['runtime_versions']['floris'] == '4.6.5' and
          settings['runtime_versions']['pymoo'] == '0.6.1.5',
          'Recorded FLORIS and pymoo versions')
    check(settings['aep_model']['config_sha256'] == HASH_CONFIG and
          settings['aep_model']['installed_turbine_sha256'] == HASH_TURBINE,
          'Recorded FLORIS YAML and turbine hashes unchanged')
    check(settings['flight_height_distribution']['height_row_count'] == 300 and
          settings['flight_height_distribution']['minimum_height_m'] == 0 and
          settings['flight_height_distribution']['maximum_height_m'] == 299,
          'Northern-gannet FHD has original 300 height masses')
    check(int(summary['non_dominated_layouts']) == 36 and len(df) == 36 and
          X.shape == (36,24) and F.shape == (36,2) and
          df.solution_index.tolist() == list(range(36)),
          '36 Pareto layouts: saved CSV and arrays have matching dimensions')

    coords = df[[f'{coord}_{i}_m' for i in range(12) for coord in ('x','y')]].to_numpy(dtype=float)
    objectives = df[['f0_normalised_aep_loss','f1_normalised_Q_4b']].to_numpy(dtype=float)
    check(np.all(np.isfinite(coords)) and np.allclose(coords, X, rtol=0, atol=1e-8) and
          np.allclose(objectives, F, rtol=0, atol=1e-12),
          'Pareto vectors/objectives match saved CSV')
    check(np.allclose(F[:,0], (AEP_CEILING-df.aep_gwh.to_numpy())/AEP_LOSS_REF, rtol=0, atol=1e-12) and
          np.allclose(F[:,1], df.Q_4b.to_numpy()/Q_REF, rtol=0, atol=1e-12),
          'All 36 Pareto objective values use the locked references')
    check(np.all(np.isfinite(X)) and np.all((X>=0)&(X<=SITE)),
          'All Pareto coordinates inside 0–6,000 m site')

    farms = X.reshape(-1,12,2)
    distances = np.linalg.norm(farms[:,:,None,:]-farms[:,None,:,:],axis=3)
    distances[:,np.arange(12),np.arange(12)] = np.inf
    check(float(distances.min()) >= 4*D-1e-6, 'All Pareto layouts satisfy 4D spacing')

    # Strictly verify mutual non-domination in raw objective values.
    objs = np.column_stack((-df.aep_gwh.to_numpy(),df.Q_4b.to_numpy()))
    dominated = []
    for i in range(len(objs)):
        if np.any(np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)):
            dominated.append(i)
    check(not dominated, 'All 36 saved Pareto layouts are mutually non-dominated')

    check(pop['X'].shape == (80,24) and pop['F'].shape == (80,2) and
          pop['G'].shape == (80,66) and pop['CV'].shape[0] == 80 and
          np.isfinite(pop['X']).all() and np.isfinite(pop['F']).all() and
          np.max(pop['CV']) <= 1e-6 and np.max(pop['G']) <= 1e-6 and
          ((pop['X']>=0)&(pop['X']<=SITE)).all(),
          'Final 80-layout NSGA-II population feasible and saved')
    check(len(initial)==80, 'All 80 initial layouts recorded')
    check(len(hv)==100 and hv.generation.tolist()==list(range(1,101)) and
          (np.diff(hv.hypervolume.to_numpy()) >= -1e-10).all(),
          '100 hypervolume entries with no recorded decreases')

    hi = df.loc[df.aep_gwh.idxmax()]
    lo = df.loc[df.Q_4b.idxmin()]
    for label, row in (('highest-AEP',hi),('lowest-risk',lo)):
        saved = pd.read_csv(RUN/f'{label}_layout.csv')[['x_m','y_m']].to_numpy()
        check(saved.shape == (12,2) and
              np.allclose(saved, X[int(row.solution_index)].reshape(12,2), rtol=0, atol=1e-8),
              label+' coordinate file matches saved front')

    manifest = json.loads((RUN/'file_hashes.json').read_text(encoding='utf-8'))
    for filename, original_hash in manifest.items():
        file = ROOT / filename
        check(file.is_file(), 'Code/input exists for hash check: '+filename)
        check(sha256(file)==original_hash, 'SHA-256 unchanged: '+filename)
    print(f'PASS | {len(manifest)} source hashes verified',flush=True)

    record={'status':'PASS','generation_count':100, 'evaluations':8000,
            'pareto_layouts':len(df), 'final_population_size':len(pop['X']),
            'hash_count_verified':len(manifest),
            'highest_aep_solution_index':int(hi.solution_index),
            'lowest_risk_solution_index':int(lo.solution_index),
            'highest_aep_gwh':float(hi.aep_gwh), 'highest_aep_Q_4b':float(hi.Q_4b),
            'lowest_risk_aep_gwh':float(lo.aep_gwh),'lowest_risk_Q_4b':float(lo.Q_4b),
            'pymoo_version':settings['pymoo_version'],
            'historical_v4_to_v7_code_equivalence_unverified':True}
    output=save_json('archive_audit.json',record)
    print('PASS | Archive audit report:',output,flush=True)
    return df,X,settings,hi,lo


def geometry(df,X,settings,hi,lo):
    theta = np.deg2rad(float(settings['problem']['rotor_plane_heading_deg_toward']))
    cross_vec=np.array([-np.sin(theta),np.cos(theta)])
    corners=np.array([[0,0],[0,SITE],[SITE,0],[SITE,SITE]],dtype=float)
    cross_min, cross_max=np.min(corners@cross_vec),np.max(corners@cross_vec)
    out_rows=[]
    pair_rows=[]
    for label,row in [('highest-AEP',hi),('lowest-risk',lo)]:
        xy=X[int(row.solution_index)].reshape(12,2)
        cross_coords=xy@cross_vec
        for i,c in enumerate(cross_coords):
            left=float(c-ROTOR_RADIUS-cross_min)
            right=float(cross_max-(c+ROTOR_RADIUS))
            out_rows.append({'layout':label, 'turbine_index_for_csv_only':i,
                             'centre_cross_m':float(c),'left_rotor_margin_m':left,
                             'right_rotor_margin_m':right,
                             'minimum_rotor_margin_m':min(left,right)})
        for i in range(12):
            for j in range(i+1,12):
                v=xy[j]-xy[i]; dist=float(np.linalg.norm(v))
                angle=(np.degrees(np.arctan2(v[1],v[0]))%180)
                diff=abs((angle-45+90)%180-90)
                if dist <= 8*D+1e-9 and diff <= 5+1e-9:
                    pair_rows.append({'layout':label,
                        'first_turbine_index_for_csv_only':i,
                        'second_turbine_index_for_csv_only':j,
                        'separation_m':dist,
                        'axis_angle_deg':angle,
                        'angle_difference_deg':diff})
    margins=pd.DataFrame(out_rows)
    pairs=pd.DataFrame(pair_rows,columns=['layout','first_turbine_index_for_csv_only',
        'second_turbine_index_for_csv_only','separation_m','axis_angle_deg','angle_difference_deg'])
    min_margins={label:float(margins.loc[margins.layout==label,'minimum_rotor_margin_m'].min())
                 for label in ('highest-AEP','lowest-risk')}
    negative_counts={label:int((margins.loc[margins.layout==label,'minimum_rotor_margin_m']<0).sum())
                     for label in ('highest-AEP','lowest-risk')}
    needs_widen=any(v<0 for v in min_margins.values())
    lower,upper=float(cross_min),float(cross_max)
    if needs_widen:
        for row in (hi,lo):
            vals=X[int(row.solution_index)].reshape(12,2)@cross_vec
            lower=min(lower,float(np.min(vals)-ROTOR_RADIUS))
            upper=max(upper,float(np.max(vals)+ROTOR_RADIUS))
        lower-=1.0
        upper+=1.0
    report={'heading_deg_toward':float(np.degrees(theta)),
            'original_corridor_min_cross_m':float(cross_min),
            'original_corridor_max_cross_m':float(cross_max),
            'original_corridor_width_m':float(cross_max-cross_min),
            'rotor_radius_m':ROTOR_RADIUS,
            'minimum_rotor_margin_m':min_margins,
            'negative_margin_turbines':negative_counts,
            'aligned_pairs_within_8D_and_5deg':{
                label:int((pairs.layout==label).sum()) for label in ('highest-AEP','lowest-risk')},
            'common_widened_corridor_required':needs_widen,
            'common_corridor_min_cross_m':lower,
            'common_corridor_max_cross_m':upper,
            'common_corridor_width_m':upper-lower,
            'one_metre_safety_margin_added_each_side_if_widened':needs_widen,
            'independent_evaluation_note':
             'When running the paired independent evaluation, use these same common corridor bounds for both endpoints. '
             'Do not change the original optimisation sample or source models. Report Q_4b and Q_4b times W.'}
    OUT.mkdir(parents=True,exist_ok=True)
    margins.to_csv(OUT/'corridor_rotor_margins.csv',index=False)
    pairs.to_csv(OUT/'aligned_pairs.csv',index=False)
    location=save_json('geometry_report.json',report)
    for label in ('highest-AEP','lowest-risk'):
        print(f'{label}: minimum rotor margin {min_margins[label]:+.3f} m; '
              f'negative rotor margins {negative_counts[label]}; '
              f'aligned pairs {report["aligned_pairs_within_8D_and_5deg"][label]}',flush=True)
    print(f'Common independent-evaluation corridor width {upper-lower:.6f} m',flush=True)
    print('INFO | Geometry report:',location,flush=True)
    return report


def make_aep_model(n_sectors):
    sys.path.insert(0,str(CODE))
    from config import Scenario, WindResource
    from floris_aep import FLORISAEPModel
    model=FLORISAEPModel(
        Scenario(wind=WindResource(n_sectors=int(n_sectors))),
        config_path=CODE/'floris_config'/'gch_iea15mw.yaml',
        turbine_reference_path=CODE/'floris_config'/'iea_15MW_REFERENCE.yaml')
    check(model.number_of_conditions==n_sectors*7,
          f'{n_sectors} sectors produce {n_sectors*7} wind conditions')
    return model


def one_aep(model,coordinates):
    aep=float(model.evaluate(coordinates))
    diagnostic={'available':False,'negative_rotor_points':None,'minimum_rotor_velocity_ms':None}
    try:
        vals=np.asarray(model.floris_model.core.flow_field.u,dtype=float)
        if vals.size:
            diagnostic={'available':True,
                'negative_rotor_points':int(np.count_nonzero(vals<0.0)),
                'minimum_rotor_velocity_ms':float(np.nanmin(vals))}
    except (AttributeError,ValueError,TypeError):
        pass
    return aep, diagnostic


def endpoints(df,X,settings,hi,lo):
    check(importlib.metadata.version('floris')=='4.6.5','Installed FLORIS version is 4.6.5')
    check(importlib.metadata.version('pymoo')=='0.6.1.5','Installed pymoo version is 0.6.1.5')
    geom=geometry(df,X,settings,hi,lo)
    model5=make_aep_model(72)
    model1=make_aep_model(360)
    ceiling5=float(model5.ideal_aep(12))
    ceiling1=float(model1.ideal_aep(12))
    check(abs(ceiling5-AEP_CEILING)<1e-7 and abs(ceiling1-AEP_CEILING)<1e-7,
          'Wake-free AEP invariant at 5 degrees and 1 degree')
    result=[]
    for label,row in [('highest-AEP',hi),('lowest-risk',lo)]:
        xy=X[int(row.solution_index)].reshape(12,2)
        aep5,vel5=one_aep(model5,xy)
        error=abs(aep5-float(row.aep_gwh))
        print(f'{label}: saved AEP 5 degree {row.aep_gwh:.9f}, reproduced {aep5:.9f}; '
              f'error {error:.9g} GWh/year',flush=True)
        check(error<=AEP_REPRO_TOL_GWH,f'{label}: saved 5-degree AEP reproduced <=0.001 GWh/year')
        aep1,vel1=one_aep(model1,xy)
        pct=100*abs(aep1-aep5)/abs(aep1)
        good=pct<ONE_DEGREE_TOL_PERCENT
        print(f'{label}: AEP 1 degree {aep1:.9f} GWh/year; deviation {pct:.6f}% '
              f'(pass below {ONE_DEGREE_TOL_PERCENT}%: {good})',flush=True)
        if vel5['available']:
            print(f'{label}: negative rotor grid points at 5 degrees: '
                  f'{vel5["negative_rotor_points"]}',flush=True)
        else:
            print(f'{label}: FLORIS rotor velocity inspection not available; '
                  'retain original warnings as provenance',flush=True)
        result.append({'layout':label,'solution_index':int(row.solution_index),
            'aep_5deg_saved_gwh':float(row.aep_gwh),
            'aep_5deg_reproduced_gwh':aep5,
            'aep_1deg_gwh':aep1,'one_degree_deviation_percent':pct,
            'passes_below_0p2_percent':bool(good),
            'optimisation_sample_Q_4b':float(row.Q_4b),
            'rotor_velocity_5deg':vel5,'rotor_velocity_1deg':vel1})
    allpass=all(x['passes_below_0p2_percent'] for x in result)
    report={'all_endpoints_pass':allpass,
            're_evaluate_entire_pareto_front_1degree':not allpass,
            'threshold_percent':ONE_DEGREE_TOL_PERCENT,
            'wake_free_aep_5deg_gwh':ceiling5,
            'wake_free_aep_1deg_gwh':ceiling1,
            'common_independent_evaluation_corridor_width_m':geom['common_corridor_width_m'],
            'endpoints':result,
            'independent_evaluation_performed':False}
    reportpath=save_json('endpoint_1degree_aep.json',report)
    print('RESULT:', 'PASS | both endpoint 1-degree checks below 0.2%' if allpass else
          'FOLLOW-UP REQUIRED | run fullfront1deg to reassess all 36 layouts at 1 degree',flush=True)
    print('INFO | AEP report:',reportpath,flush=True)


def fullfront1deg(df,X,settings,hi,lo):
    existing=OUT/'endpoint_1degree_aep.json'
    check(existing.is_file(),'First run endpoints check')
    status=json.loads(existing.read_text(encoding='utf8'))
    check(status.get('re_evaluate_entire_pareto_front_1degree') is True,
          'Full front 1-degree analysis is required by a failed endpoint check')
    model=make_aep_model(360)
    rows=[]
    for _,r in df.iterrows():
        xy=X[int(r.solution_index)].reshape(12,2)
        aep1,vel=one_aep(model,xy)
        pct=100*abs(aep1-float(r.aep_gwh))/abs(aep1)
        rows.append({'solution_index':int(r.solution_index),
                     'aep_5deg_gwh':float(r.aep_gwh),
                     'aep_1deg_gwh':aep1,
                     'aep_deviation_percent':pct,
                     'optimisation_sample_Q_4b_unchanged':float(r.Q_4b),
                     'negative_rotor_points_1deg':vel['negative_rotor_points']})
        print(f'Front {len(rows):02d}/36: 1-degree AEP {aep1:.6f} GWh/year; '
              f'deviation {pct:.5f}%',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    destination=OUT/'all_36_pareto_1deg_aep.csv'
    pd.DataFrame(rows).to_csv(destination,index=False)
    print('PASS | All 36 saved Pareto layouts evaluated at 1 degree; no NSGA-II run',flush=True)
    print('Report:',destination,flush=True)


def main():
    parser=argparse.ArgumentParser(description='Read-only Metric 4b 72-sector post-run checks')
    parser.add_argument('command',choices=['audit','endpoints','fullfront1deg'])
    args=parser.parse_args()
    print('READ-ONLY | Original optimisation files will not be modified',flush=True)
    df,X,settings,hi,lo=load_archive()
    print('AUDIT COMPLETE | 36 Pareto layouts, 8,000 evaluations, 100 generations',flush=True)
    if args.command=='audit':
        geometry(df,X,settings,hi,lo)
    elif args.command=='endpoints':
        endpoints(df,X,settings,hi,lo)
    elif args.command=='fullfront1deg':
        fullfront1deg(df,X,settings,hi,lo)
    print('Saved reports in:',OUT,flush=True)


if __name__=='__main__':
    main()
