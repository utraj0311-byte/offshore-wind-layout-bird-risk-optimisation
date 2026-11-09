"""Independent Metric 4a confirmation (local Windows / Anaconda).

Keep this file next to metric4a_postrun_checks.py in metric4a_preflight.

    python metric4a_confirm72.py verify
    python metric4a_confirm72.py run

Read-only to the finished optimisation and post-run verification folders.
Uses the unchanged original simulation module. The common corridor is widened
ONLY for independent confirmation because the highest-AEP endpoint has an
encounter-zone boundary overflow. Not an NSGA-II re-run.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN_DIR = ROOT / 'metric_4a_72sectors'
POST_DIR = ROOT / 'metric_4a_72sectors_postrun'
OUT_DIR = ROOT / 'metric_4a_72sectors_confirmation'
NUM_BATCHES = 20
BIRDS_PER_BATCH = 5000
FIRST_SEED = 20042
P_VALUES = (0.6, 0.0)
LABELS = ('highest-AEP', 'lowest-risk')
SITE = 6000.0
HEADING = 225.0


def check(condition, msg):
    if not bool(condition):
        raise RuntimeError('STOP: ' + msg)


def read_json(path):
    check(path.is_file(), f'missing required file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    """Validate exact source and widths without executing any bird simulations."""
    check(RUN_DIR.is_dir(), f'original run folder missing: {RUN_DIR}')
    run = read_json(RUN_DIR/'run_settings.json')
    end = read_json(POST_DIR/'endpoint_aep_1deg.json')
    geom = read_json(POST_DIR/'geometry_report.json')
    finished = read_json(RUN_DIR/'completion_status.json')
    check(finished['status'] == 'optimisation_finished', 'original run incomplete')
    check(end['all_endpoints_pass'] is True, '1-degree endpoint AEP verification failed')
    check(end['entire_front_1degree_required'] is False, 'front needs 1-degree AEP re-evaluation')
    check(run['n_sectors']==72 and run['population']==80 and run['generations']==100 and run['seed']==42,
          'run settings mismatch')
    check(run['objective_random_number_seeds']==[42,43,44,45,46], 'training seeds changed')
    check(run['fresh_random_number_seeds']==[10042,10043,10044], 'validation seeds changed')
    check(run['problem']['encounter_reference']==0.6854, 'reference changed')
    check(run['bird_heading_deg_toward']==HEADING, 'bird direction changed')
    check(run['site_width_m']==SITE and run['site_height_m']==SITE, 'physical site changed')
    check(run['problem']['number_of_turbines']==12, 'turbine count changed')
    check(run['problem']['behaviour']['response_probability']==0.6, 'baseline response changed')
    check(importlib.metadata.version('numpy')==run['numpy_version'], 'NumPy version changed')
    check(importlib.metadata.version('pymoo')==run['pymoo_version'], 'pymoo version changed')
    check(importlib.metadata.version('floris')==run['floris_version'], 'FLORIS version changed')
    check(importlib.metadata.version('scipy'), 'SciPy is unavailable')
    check(geom['need_common_widened_corridor'] is True, 'corridor unexpectedly does not need widening')
    left=float(geom['common_confirmation_cross_min_m'])
    right=float(geom['common_confirmation_cross_max_m'])
    W=right-left
    check(right>left and math.isclose(W,float(geom['common_confirmation_width_m']),abs_tol=1e-8),
          'geometry corridor width inconsistent')
    check(abs(W-8787.58)<0.02, 'widened corridor does not match postrun verification')
    check(float(geom['margin_m_by_layout']['highest-AEP'])<0.0, 'no negative margin for highest-AEP')
    check(float(geom['margin_m_by_layout']['lowest-risk'])>=0.0, 'unexpected lowest-risk overflow')
    check(abs(float(geom['encounter_radius_m'])-242.24)<1e-10, 'encounter radius changed')
    for name in LABELS:
        check((RUN_DIR/f'{name}_layout.csv').is_file(), f'missing {name} coordinates')
    X=np.load(RUN_DIR/'pareto_decision_vectors.npy',allow_pickle=False)
    f=pd.read_csv(RUN_DIR/'pareto_points.csv')
    check(len(f)==38 and X.shape==(38,24), 'saved Pareto files changed')
    selected={}
    for name, row in [('highest-AEP',f.loc[f.aep_gwh.idxmax()]),
                      ('lowest-risk',f.loc[f.E_enc.idxmin()])]:
        xy=pd.read_csv(RUN_DIR/f'{name}_layout.csv')[['x_m','y_m']].to_numpy(dtype=float)
        check(xy.shape==(12,2) and np.max(np.abs(xy-X[int(row.solution_index)].reshape(12,2)))<1e-8,
              f'{name} file differs from saved Pareto coordinates')
        selected[name]={'solution_index':int(row.solution_index),'xy':xy,
                        'aep_5deg_gwh':float(row.aep_gwh),
                        'training_Q_4a':float(row.E_enc),
                        'training_P_ge_1':float(row.P_ge_1)}
    # Original code/input SHA-256 checks - stops rather than silently using edited originals.
    hashes = read_json(RUN_DIR/'file_hashes.json')
    for rel,wanted in hashes.items():
        path=HERE/rel
        check(path.is_file(), f'original run code missing: {path}')
        check(digest(path)==wanted, f'original run code changed: {rel}')
    sys.path.insert(0,str(HERE/'model8_code'))
    from model8_meso_response import BehaviourScenario, MesoResponseModel, TrajectoryPopulation, cross_unit_vectors
    b=run['problem']['behaviour']
    bs=BehaviourScenario(response_probability=0.6,
        response_radius_m=float(b['response_radius_m']),
        encounter_radius_m=float(b['encounter_radius_m']),
        deflection_angle_deg=float(b['deflection_angle_deg']),
        return_length_m=float(b['return_length_m']),
        step_size_m=float(b['step_size_m']),
        max_steps_factor=float(b['max_steps_factor']))
    check(abs(float(b['return_length_m'])-1937.92)<1e-9, 'return length changed')
    check(abs(float(b['response_radius_m'])-484.48)<1e-9, 'response radius changed')
    check(abs(float(b['deflection_angle_deg'])-45.0)<1e-9, 'deflection changed')
    margin=float(b['response_radius_m'])+float(b['return_length_m'])
    seeds=list(range(FIRST_SEED,FIRST_SEED+NUM_BATCHES))
    check(not set(seeds)&set(run['objective_random_number_seeds']), 'confirmation seeds overlap training')
    check(not set(seeds)&set(run['fresh_random_number_seeds']), 'confirmation seeds overlap fresh validation')
    print('PASS | finished source run, hashes and 1-degree AEP checks')
    print('PASS | identical original model, 20 batches x 5,000 birds, seeds 20042..20061')
    print('PASS | paired layouts and p=0.6/p=0 use common random trajectories in each batch')
    print(f'PASS | common widened corridor: [{left:.6f}, {right:.6f}] m; W={W:.3f} m')
    print(f'PASS | unchanged site 6000 x 6000 m, heading 225 TOWARD, margin {margin:.3f} m')
    print('PASS | original training and optimisation folders are read-only')
    return {'run':run,'endpoints':end,'geometry':geom,'selected':selected,
            'width_m':W,'left_m':left,'right_m':right,'behaviour':bs,
            'longitudinal_margin_m':margin, 'seeds':seeds,
            'TrajectoryPopulation':TrajectoryPopulation,'MesoResponseModel':MesoResponseModel,
            'cross_unit_vectors':cross_unit_vectors}


def sample_on_common_corridor(ctx, seed):
    """Use original RNG and entry fractions; alter only cross-coordinate bounds.

    Generating the population with the original generator preserves all
    response and side uniforms. Repositioning every start using its original
    stratified cross-flight fraction samples uniformly over the common W.
    Along-flight entry positions and travel distance remain unchanged.
    """
    T=ctx['TrajectoryPopulation']
    p=T.generate_parallel(site_width_m=SITE,site_height_m=SITE,
        heading_deg_toward=HEADING,number_of_trajectories=BIRDS_PER_BATCH,
        maximum_response_events=12,longitudinal_margin_m=ctx['longitudinal_margin_m'],
        seed=seed,stratified_entries=True)
    nvec=ctx['cross_unit_vectors'](HEADING)
    old_cross=p.start_positions_m@nvec
    fractions=p.entry_cross_fractions
    check(fractions is not None and fractions.shape==(BIRDS_PER_BATCH,),
          'entry fractions not preserved')
    new_cross=ctx['left_m']+fractions*ctx['width_m']
    new_starts=p.start_positions_m+(new_cross-old_cross)[:,None]*nvec.reshape(1,2)
    new_p=replace(p,start_positions_m=new_starts)
    check(np.max(np.abs(new_p.start_positions_m@nvec-new_cross))<1e-8,
          'cross-coordinate remapping failed')
    check(np.all(new_p.response_uniforms==p.response_uniforms), 'response RNG changed')
    check(np.all(new_p.side_uniforms==p.side_uniforms), 'side RNG changed')
    return new_p


def interval(values):
    vals=np.asarray(values,dtype=float)
    check(vals.shape==(NUM_BATCHES,),f'incorrect paired batch count: {vals.shape}')
    mean=float(vals.mean()); sd=float(vals.std(ddof=1))
    se=sd/math.sqrt(len(vals)); t=float(student_t.ppf(.975,df=len(vals)-1))
    return {'mean':mean,'batch_sd':sd,'batch_se':se,'ci95_low':mean-t*se,
            'ci95_high':mean+t*se,'t_critical_95':t}


def make_pair_contrast(df, response, metric, label_a, label_b, W):
    """Difference B - A, negative means B has fewer encounters."""
    a=df[(df.layout==label_a)&(df.response_probability==response)].sort_values('batch_index')
    b=df[(df.layout==label_b)&(df.response_probability==response)].sort_values('batch_index')
    check(a.seed.tolist()==b.seed.tolist()==list(range(FIRST_SEED,FIRST_SEED+NUM_BATCHES)),
          'paired seeds not matched')
    diffs=b[metric].to_numpy()-a[metric].to_numpy()
    stats=interval(diffs)
    return {'response_probability':response,'metric':metric,
            'comparison':f'{label_b} minus {label_a}',
            'common_width_m':W,
            **stats,'confirmed_decrease':bool(stats['ci95_high']<0),
            'confirmed_increase':bool(stats['ci95_low']>0)}


def run():
    ctx=verify()
    check(not OUT_DIR.exists(), f'confirmation output already exists; refusing overwrite: {OUT_DIR}')
    OUT_DIR.mkdir(parents=False,exist_ok=False)
    # A first file proves provenance if a later simulation fails.
    settings={
       'status':'running','source_optimisation_folder':str(RUN_DIR),
       'original_results_unmodified':True,
       'scientific_bird_module':'model8_code/model8_meso_response.py (SHA-256 verified)',
       'original_confirmation_procedure':'20 independent batches x 5000, seeds 20042..20061',
       'confirmation_batches':NUM_BATCHES,'trajectories_per_batch':BIRDS_PER_BATCH,
       'seeds':ctx['seeds'],'response_probabilities':[0.6,0.0],
       'same_trajectories_per_batch_used_for_both_layouts_and_probabilities':True,
       'physical_site_m':[SITE,SITE], 'bird_heading_deg_toward':HEADING,
       'corridor':'cross-flight widened uniformly for both endpoints, 1 m outer safety',
       'original_width_m':ctx['geometry']['original_width_m'],
       'common_width_m':ctx['width_m'],
       'common_cross_min_m':ctx['left_m'],'common_cross_max_m':ctx['right_m'],
       'longitudinal_margin_m':ctx['longitudinal_margin_m'],
       'SD_10k_method':'10 independent groups pairing successive 5000-bird batches',
       'uncertainty':'two-sided Student-t 95% interval on paired differences across 20 batches',
       'mean_Q_times_W_units':'encounters*m per entering bird (scaled corridor-normalised encounter measure)',
       'reference_for_training_objective_unchanged':0.6854,
       'software_versions':{p:importlib.metadata.version(p) for p in ('numpy','scipy','pymoo','floris')},
       'source_run_settings_sha256':digest(RUN_DIR/'run_settings.json'),
       'source_endpoint_verification_sha256':digest(POST_DIR/'endpoint_aep_1deg.json'),
       'original_simulator_sha256':digest(HERE/'model8_code'/'model8_meso_response.py')}
    (OUT_DIR/'confirmation_settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    simulators={}
    b=ctx['behaviour']
    for p in P_VALUES:
        scenario=replace(b,response_probability=p)
        simulators[p]=ctx['MesoResponseModel'](scenario)
    rows=[]; turbines=[]
    try:
        for batch_index,seed in enumerate(ctx['seeds']):
            population=sample_on_common_corridor(ctx,seed)
            for p in P_VALUES:
                for label in LABELS:
                    result=simulators[p].simulate(ctx['selected'][label]['xy'],population)
                    result.validate_accounting()
                    check(result.number_started==BIRDS_PER_BATCH, 'incorrect bird sample size')
                    check(result.number_failed==0 and result.number_completed==BIRDS_PER_BATCH,
                          f'not all birds completed for {label}, seed {seed}, p={p}')
                    q=float(result.encounter_intensity)
                    prob=float(result.encounter_prevalence)
                    rows.append({'batch_index':batch_index, 'seed':seed,'layout':label,
                        'response_probability':p,'N_started':result.number_started,
                        'N_completed':result.number_completed,'N_failed':result.number_failed,
                        'N_enc_farm':result.total_encounters,
                        'N_with_at_least_one_encounter':int(np.count_nonzero(result.per_trajectory_encounters>=1)),
                        'N_response_events':result.total_response_events,
                        'Q_4a':q,'P_ge_1':prob,'Q_4a_times_W':q*ctx['width_m'],
                        'mean_route_extension_m':float(result.mean_route_extension_m)})
                    for ti,count in enumerate(result.per_turbine_encounters):
                        turbines.append({'batch_index':batch_index,'seed':seed,'layout':label,
                            'response_probability':p,'turbine_index_for_csv_only':ti,
                            'N_encounters':int(count)})
            print(f'Batch {batch_index+1:02d}/{NUM_BATCHES} complete (seed {seed})',flush=True)
            # Atomic-ish progress reports, never written to source results.
            pd.DataFrame(rows).to_csv(OUT_DIR/'confirmation_batches_partial.csv',index=False)
        df=pd.DataFrame(rows)
        check(len(df)==NUM_BATCHES*len(LABELS)*len(P_VALUES), 'missing confirmation results')
        df.to_csv(OUT_DIR/'confirmation_batches.csv',index=False)
        pd.DataFrame(turbines).to_csv(OUT_DIR/'confirmation_per_turbine.csv',index=False)
        summary=[]
        for p in P_VALUES:
            for label in LABELS:
                part=df[(df.layout==label)&(df.response_probability==p)].sort_values('batch_index')
                n=int(part.N_started.sum())
                total=int(part.N_enc_farm.sum())
                with_enc=int(part.N_with_at_least_one_encounter.sum())
                check(n==100000 and total==int(part.N_enc_farm.sum()),'incorrect aggregate')
                q=total/n; prob=with_enc/n
                q_batches=part.Q_4a.to_numpy()
                p_batches=part.P_ge_1.to_numpy()
                # Independent 10000-bird samples from pairs of independent 5000-bird batches.
                check(len(q_batches)==20,'cannot construct 10000-bird SD')
                q_10k=(q_batches[0::2]+q_batches[1::2])/2.0
                summary.append({'layout':label,'response_probability':p,
                    'solution_index':ctx['selected'][label]['solution_index'],
                    'aep_5deg_gwh':ctx['selected'][label]['aep_5deg_gwh'],
                    'training_Q_4a':ctx['selected'][label]['training_Q_4a'],
                    'training_P_ge_1':ctx['selected'][label]['training_P_ge_1'],
                    'N_birds':n,'N_encounters':total,'N_birds_with_encounter':with_enc,
                    'confirmed_Q_4a':q,'confirmed_P_ge_1':prob,
                    'confirmed_Q_4a_times_W':q*ctx['width_m'],
                    'Q_4a_5000_batch_sd':float(np.std(q_batches,ddof=1)),
                    'P_ge_1_5000_batch_sd':float(np.std(p_batches,ddof=1)),
                    'Q_4a_one_10000_bird_sample_sd':float(np.std(q_10k,ddof=1)),
                    'Q_4a_one_10000_bird_sample_sd_replicates':len(q_10k)})
        pd.DataFrame(summary).to_csv(OUT_DIR/'confirmation_endpoint_summary.csv',index=False)
        contrasts=[]
        for p in P_VALUES:
            for metric in ('Q_4a','P_ge_1','Q_4a_times_W'):
                contrasts.append(make_pair_contrast(df,p,metric,'highest-AEP','lowest-risk',ctx['width_m']))
        # Response effect within each endpoint, p=0.6 minus p=0 on exactly same birds.
        for label in LABELS:
            for metric in ('Q_4a','P_ge_1','Q_4a_times_W'):
                a=df[(df.layout==label)&(df.response_probability==0.0)].sort_values('batch_index')
                b=df[(df.layout==label)&(df.response_probability==0.6)].sort_values('batch_index')
                check(a.seed.tolist()==b.seed.tolist(),'response-effect seeds mismatched')
                stats=interval(b[metric].to_numpy()-a[metric].to_numpy())
                contrasts.append({'response_probability':'0.6 minus 0', 'layout':label,'metric':metric,
                    'comparison':f'{label}: p=0.6 minus p=0',
                    'common_width_m':ctx['width_m'],**stats,
                    'confirmed_decrease':bool(stats['ci95_high']<0),
                    'confirmed_increase':bool(stats['ci95_low']>0)})
        pd.DataFrame(contrasts).to_csv(OUT_DIR/'confirmation_paired_contrasts.csv',index=False)
        metrics={(r['response_probability'],r['metric']):r for r in contrasts
                  if r.get('comparison')=='lowest-risk minus highest-AEP'}
        nq=metrics[(0.6,'Q_4a')]; np_=metrics[(0.6,'P_ge_1')]
        report={
            'status':'confirmation_complete',
            'common_corridor_width_m':ctx['width_m'],
            'results':summary,
            'all_paired_contrasts':contrasts,
            'lowest_risk_has_confirmed_Q_reduction_at_p_0p6':nq['confirmed_decrease'],
            'lowest_risk_has_confirmed_P_increase_at_p_0p6':np_['confirmed_increase'],
            'original_finding_repeats_under_common_widened_corridor':bool(
                not nq['confirmed_decrease'] and np_['confirmed_increase']),
            'interpretation_limitations':[
                'Q_4a counts horizontal encounter zones, not collisions or annual bird flux',
                'Fresh confirmation uses a common widened entry corridor, not the original training corridor',
                'Confidence intervals quantify Monte Carlo uncertainty for fixed endpoints, not optimiser-search uncertainty',
                'W-scaled values are layout-comparable only because the same uniform corridor and flight direction are used'],
        }
        (OUT_DIR/'confirmation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        settings['status']='confirmation_complete'
        settings['completed_utc']=datetime.now(timezone.utc).isoformat()
        (OUT_DIR/'confirmation_settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
        print('\nINDEPENDENT CONFIRMATION COMPLETE')
        print(pd.DataFrame(summary)[['layout','response_probability','confirmed_Q_4a','confirmed_P_ge_1',
              'confirmed_Q_4a_times_W','Q_4a_one_10000_bird_sample_sd']].to_string(index=False))
        for p in P_VALUES:
            print(f'Paired lowest-risk minus highest-AEP at p={p:g}:')
            for metric in ('Q_4a','P_ge_1','Q_4a_times_W'):
                r=metrics[(p,metric)]
                print(f'  {metric}: {r["mean"]:+.8f}, 95% CI '
                      f'[{r["ci95_low"]:+.8f}, {r["ci95_high"]:+.8f}]')
        print(f'Original finding repeats (common corridor): '
              f'{report["original_finding_repeats_under_common_widened_corridor"]}')
        print(f'Reports: {OUT_DIR}')
        print('The original optimisation and post-run audit remain unchanged.')
    except Exception as error:
        (OUT_DIR/'confirmation_error.txt').write_text(str(error),encoding='utf-8')
        raise


if __name__ == '__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('verify','run'):
        raise SystemExit('Usage: python metric4a_confirm72.py verify|run')
    {'verify':verify,'run':run}[sys.argv[1]]()
