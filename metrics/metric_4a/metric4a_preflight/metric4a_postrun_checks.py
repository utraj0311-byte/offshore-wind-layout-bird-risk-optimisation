"""Read-only Metric 4a post-run checks for the ORIGINAL floris-env on Windows.

Place alongside metric4a_run72.py in bird_layout_opt/metric4a_preflight.
All generated reports are written OUTSIDE the immutable optimisation folder.

    python metric4a_postrun_checks.py geometry
    python metric4a_postrun_checks.py endpoints
    python metric4a_postrun_checks.py fullfront1deg  # ONLY if endpoint 1-degree test fails
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
RUN = ROOT.parent / 'metric_4a_72sectors'
OUT = ROOT.parent / 'metric_4a_72sectors_postrun'
D = 242.24
HEADING = 225.0
SITE = 6000.0
MAX_REL_AEP_ERROR_PERCENT = 0.2


def require(value, description):
    if not bool(value):
        raise RuntimeError('STOP: ' + description)


def load_data():
    require(RUN.is_dir(), f'Optimisation folder missing: {RUN}')
    required = ['completion_status.json', 'run_settings.json', 'settings_diff.json',
                'file_hashes.json', 'pareto_points.csv', 'pareto_decision_vectors.npy',
                'pareto_normalised_objectives.npy', 'final_population.npz',
                'all_evaluated_candidates.csv', 'hypervolume.csv',
                'highest-AEP_layout.csv', 'lowest-risk_layout.csv']
    missing = [n for n in required if not (RUN / n).is_file()]
    require(not missing, 'Missing archive files: ' + ', '.join(missing))
    status = json.loads((RUN / 'completion_status.json').read_text(encoding='utf-8'))
    settings = json.loads((RUN / 'run_settings.json').read_text(encoding='utf-8'))
    diff = json.loads((RUN / 'settings_diff.json').read_text(encoding='utf-8'))
    df = pd.read_csv(RUN / 'pareto_points.csv')
    X = np.load(RUN / 'pareto_decision_vectors.npy', allow_pickle=False)
    F = np.load(RUN / 'pareto_normalised_objectives.npy', allow_pickle=False)
    pop = np.load(RUN / 'final_population.npz', allow_pickle=False)
    candidates = pd.read_csv(RUN / 'all_evaluated_candidates.csv')
    hv = pd.read_csv(RUN / 'hypervolume.csv')
    require(status.get('status') == 'optimisation_finished', 'Run is not marked finished')
    require(int(status['generations']) == 100, 'Not 100 generations')
    require(int(settings['n_sectors']) == 72 and int(settings['population']) == 80
            and int(settings['generations']) == 100 and int(settings['seed']) == 42,
            'Critical optimisation settings are inconsistent')
    require(settings['problem']['encounter_reference'] == 0.6854, 'Bird reference changed')
    require(settings['problem']['aep_ceiling_gwh'] == 854.4956585617186,
            'Wake-free AEP reference changed')
    require(diff.get('other_scientific_differences') == [], 'Unexpected settings difference')
    require(len(candidates) == 8000, 'Candidate evaluation count is not 8000')
    require(len(hv) == 100 and hv.generation.tolist() == list(range(1, 101)),
            'Hypervolume history does not contain generations 1..100')
    require(len(df) == len(X) == len(F) == 38, 'Saved Pareto arrays/CSV are inconsistent')
    require(X.shape == (38, 24) and F.shape == (38, 2), 'Pareto array shape mismatch')
    require(df.solution_index.tolist() == list(range(38)), 'Pareto solution indexes are not 0..37')
    require(np.all(np.isfinite(X)) and np.all((X >= 0) & (X <= SITE)),
            'Layout coordinates invalid')
    require(np.allclose(F, df[['f0_normalised_aep_loss', 'f1_normalised_encounters']],
            rtol=0, atol=1e-12), 'Pareto objective arrays disagree')
    require(pop['X'].shape == (80, 24) and np.max(pop['CV']) <= 1e-8,
            'Final population missing or infeasible')
    require((df.spacing_violations == 0).all() and df.minimum_spacing_m.min() >= 4*D-1e-6,
            'Pareto front has spacing violations')
    require((df.N_completed == 10000).all() and (df.N_failed == 0).all(),
            'Bird trajectory accounting is inconsistent')
    require((np.diff(hv.normalised_hypervolume.values) >= -1e-10).all(),
            'Hypervolume unexpectedly decreases')
    # Strict pairwise non-domination in (minimise -AEP, minimise Q_4a).
    obj = np.column_stack((-df.aep_gwh.to_numpy(), df.E_enc.to_numpy()))
    for i, v in enumerate(obj):
        dominating = np.all(obj <= v, axis=1) & np.any(obj < v, axis=1)
        dominating[i] = False
        require(not dominating.any(), f'Dominated saved Pareto solution {i}')
    hi = df.loc[df.aep_gwh.idxmax()]
    lo = df.loc[df.E_enc.idxmin()]
    for label, row in [('highest-AEP', hi), ('lowest-risk', lo)]:
        actual = pd.read_csv(RUN / (label + '_layout.csv'))[['x_m', 'y_m']].to_numpy()
        expected = X[int(row.solution_index)].reshape(12, 2)
        require(np.max(np.abs(actual-expected)) < 1e-8,
                f'{label} CSV disagrees with decision vectors')
    hashes_saved = json.loads((RUN / 'file_hashes.json').read_text(encoding='utf-8'))
    for name, wanted in hashes_saved.items():
        file = ROOT / name
        require(file.exists(), f'Input missing for hash verification: {file}')
        require(hashlib.sha256(file.read_bytes()).hexdigest() == wanted,
                f'Original input/code changed since run: {name}')
    return df, X, settings, hi, lo


def geometry(df, X, settings, hi, lo):
    heading = float(settings['bird_heading_deg_toward'])
    n = np.array([-np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading))])
    corners = np.array([(0,0), (SITE,0), (0,SITE), (SITE,SITE)], float)
    cross = corners @ n
    left, right = float(cross.min()), float(cross.max())
    radius = float(settings['problem']['behaviour']['encounter_radius_m'])
    rows, aligned = [], []
    for label, row in [('highest-AEP', hi), ('lowest-risk', lo)]:
        xy = X[int(row.solution_index)].reshape(12, 2)
        position_cross = xy @ n
        for i, c in enumerate(position_cross):
            lower = c-radius-left
            upper = right-(c+radius)
            rows.append({'layout': label, 'turbine_index_for_csv_only': i,
                         'cross_position_m':float(c), 'left_margin_m':float(lower),
                         'right_margin_m':float(upper), 'minimum_margin_m':float(min(lower,upper))})
        for i in range(len(xy)):
            for j in range(i+1,len(xy)):
                v = xy[j]-xy[i]
                length=float(np.linalg.norm(v))
                axis=float(np.rad2deg(np.arctan2(v[1],v[0]))%180)
                # 45 - 225 same undirected axis.
                angle_err=abs((axis-45+90)%180-90)
                if length <= 8*D+1e-9 and angle_err <= 5+1e-9:
                    aligned.append({'layout':label,'first_turbine_index_for_csv_only':i,
                                    'second_turbine_index_for_csv_only':j,
                                    'distance_m':length, 'axis_angle_deg':axis,
                                    'angle_from_45_225_axis_deg':angle_err})
    margins = pd.DataFrame(rows)
    pairs=pd.DataFrame(aligned,columns=['layout','first_turbine_index_for_csv_only',
                       'second_turbine_index_for_csv_only','distance_m','axis_angle_deg',
                       'angle_from_45_225_axis_deg'])
    minmargin=margins.groupby('layout')['minimum_margin_m'].min().to_dict()
    cross_left = min(left, min(float((X[int(r.solution_index)].reshape(12,2)@n).min()-radius) for r in [hi,lo]))
    cross_right = max(right, max(float((X[int(r.solution_index)].reshape(12,2)@n).max()+radius) for r in [hi,lo]))
    widened = any(v<0 for v in minmargin.values())
    if widened:
        # Reuse exactly the original RNG seeds, with one common corridor; 1m safety each side.
        cross_left-=1.0
        cross_right+=1.0
    out={'original_cross_min_m':left,'original_cross_max_m':right,
         'original_width_m':right-left,'encounter_radius_m':radius,
         'margin_m_by_layout':minmargin,
         'out_of_corridor_turbine_count':{
           label:int(((margins.layout==label)&(margins.minimum_margin_m<0)).sum())
           for label in ['highest-AEP','lowest-risk']},
         'aligned_pair_counts':{label:int((pairs.layout==label).sum())
           for label in ['highest-AEP','lowest-risk']},
         'need_common_widened_corridor':widened,
         'common_confirmation_cross_min_m':cross_left,
         'common_confirmation_cross_max_m':cross_right,
         'common_confirmation_width_m':cross_right-cross_left,
         'corridor_boundary_safety_margin_m':1.0 if widened else 0.0}
    OUT.mkdir(parents=True,exist_ok=True)
    margins.to_csv(OUT/'corridor_margins.csv',index=False)
    pairs.to_csv(OUT/'aligned_pairs.csv',index=False)
    (OUT/'geometry_report.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    for label in ['highest-AEP','lowest-risk']:
        print(f'{label}: minimum margin {minmargin[label]:+.3f} m; '
              f'outside zones {out["out_of_corridor_turbine_count"][label]}; '
              f'aligned pairs {out["aligned_pair_counts"][label]}')
    print('Common confirmation corridor width: %.3f m' % out['common_confirmation_width_m'])
    return out


def make_aep_model(sectors):
    sys.path.insert(0, str(ROOT))
    from config import Scenario
    from floris_aep import FLORISAEPModel
    scenario = Scenario()
    scenario.site.n_turbines = 12
    scenario.site.width_m = SITE
    scenario.site.height_m = SITE
    scenario.site.min_spacing_D = 4.0
    scenario.site.migration_axis_deg = HEADING
    scenario.wind.n_sectors = int(sectors)
    scenario.turbine.rotor_diameter_m = D
    aep = FLORISAEPModel(scenario, config_path=ROOT/'floris_config'/'gch_iea15mw.yaml')
    require(aep.number_of_conditions == 7*sectors, f'FLORIS wind-condition mismatch for {sectors}')
    return aep


def evaluate_aep(model, xy):
    result=float(model.evaluate(xy))
    # This is a diagnostic only: do not change FLORIS numerical treatment.
    velocity_info={'rotor_velocities_inspectable':False, 'negative_rotor_velocity_count':None,
                   'minimum_rotor_velocity_ms':None}
    try:
        rotor_vels=np.asarray(model.floris_model.core.flow_field.u,dtype=float)
        if rotor_vels.size:
            velocity_info={'rotor_velocities_inspectable':True,
                           'negative_rotor_velocity_count':int(np.count_nonzero(rotor_vels<0.0)),
                           'minimum_rotor_velocity_ms':float(np.nanmin(rotor_vels))}
    except (AttributeError,TypeError,ValueError):
        pass
    return result,velocity_info


def endpoints(df,X,settings,hi,lo):
    require(importlib.metadata.version('floris')=='4.6.5','Installed FLORIS version changed')
    geometry(df,X,settings,hi,lo)
    aep5=make_aep_model(72)
    aep1=make_aep_model(360)
    ideal5=float(aep5.ideal_aep(12))
    ideal1=float(aep1.ideal_aep(12))
    require(abs(ideal5-854.4956585617186)<1e-7,'Wake-free AEP 5-degree mismatch')
    require(abs(ideal1-854.4956585617186)<1e-7,'Wake-free AEP 1-degree mismatch')
    rows=[]
    for label, r in [('highest-AEP',hi),('lowest-risk',lo)]:
        xy=X[int(r.solution_index)].reshape(12,2)
        val5,diag5=evaluate_aep(aep5,xy)
        gap5=abs(val5-float(r.aep_gwh))
        print(f'{label}: saved 5-degree {r.aep_gwh:.9f}, reproduced {val5:.9f}, error {gap5:.9g}')
        require(gap5 <= .001, f'Original 5-degree AEP did not reproduce for {label}')
        val1,diag1=evaluate_aep(aep1,xy)
        diff=abs(val1-val5)/abs(val1)*100
        print(f'{label}: 1-degree {val1:.9f} GWh/year; deviation {diff:.5f}%; '
              f'pass (<0.2%): {diff < MAX_REL_AEP_ERROR_PERCENT}')
        if diag5['rotor_velocities_inspectable']:
            print(f'  Negative rotor points at 5-degree: {diag5["negative_rotor_velocity_count"]}')
        else:
            print('  Rotor-grid velocity diagnostic unavailable; preserve FLORIS warnings')
        rows.append({'layout':label, 'solution_index':int(r.solution_index),
                     'aep_5deg_saved_gwh':float(r.aep_gwh),
                     'aep_5deg_reproduced_gwh':val5,
                     'aep_1deg_gwh':val1,
                     'relative_difference_percent':diff,
                     'relative_aep_pass_lt_0p2pct':bool(diff < MAX_REL_AEP_ERROR_PERCENT),
                     'aep5_rotor_velocity_diagnostics':diag5,
                     'aep1_rotor_velocity_diagnostics':diag1})
    all_pass=all(row['relative_aep_pass_lt_0p2pct'] for row in rows)
    report={'threshold_percent':MAX_REL_AEP_ERROR_PERCENT,
            'all_endpoints_pass':all_pass,
            'entire_front_1degree_required':not all_pass,
            'wake_free_5degree_gwh':ideal5,'wake_free_1degree_gwh':ideal1,
            'endpoints':rows}
    (OUT/'endpoint_aep_1deg.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RESULT:', 'PASS; all 1-degree endpoint checks <0.2%' if all_pass else
          'FOLLOW-UP REQUIRED: run fullfront1deg on all 38 layouts before confirmation')


def fullfront1deg(df,X,settings,hi,lo):
    test=OUT/'endpoint_aep_1deg.json'
    require(test.is_file(),'First run endpoints')
    report=json.loads(test.read_text(encoding='utf-8'))
    require(report['entire_front_1degree_required'],
            'No 1-degree full-front re-evaluation required; endpoint checks passed')
    aep=make_aep_model(360)
    rows=[]
    for i, r in df.iterrows():
        val,diag=evaluate_aep(aep,X[int(r.solution_index)].reshape(12,2))
        pct=100*abs(val-float(r.aep_gwh))/abs(val)
        rows.append({'solution_index':int(r.solution_index),'aep_5deg_gwh':float(r.aep_gwh),
                     'aep_1deg_gwh':val,'relative_change_percent':pct,
                     'Q_4a_training_unchanged':float(r.E_enc),
                     'negative_rotor_velocity_count_1deg':diag['negative_rotor_velocity_count']})
        print(f'Front {i+1:02}/{len(df)}: 1-degree {val:.6f} GWh; change {pct:.4f}%',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'all_38_pareto_1deg_aep.csv',index=False)
    print('All 38 saved layouts re-evaluated at 1 degree; no optimisation executed.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['geometry','endpoints','fullfront1deg'])
    args=parser.parse_args()
    print('Read-only post-run audit. Original optimisation results will NOT be modified.',flush=True)
    df,X,settings,hi,lo=load_data()
    print('PASS | Full completion/archive checks; 38 non-dominated layouts, 8,000 evaluations, 100 generations')
    if args.command=='geometry':
        geometry(df,X,settings,hi,lo)
    elif args.command=='endpoints':
        endpoints(df,X,settings,hi,lo)
    else:
        fullfront1deg(df,X,settings,hi,lo)
    print('Report folder:', OUT)


if __name__ == '__main__':
    main()
