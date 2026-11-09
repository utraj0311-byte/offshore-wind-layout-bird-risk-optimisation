#!/usr/bin/env python3
"""Guarded independent confirmation of final Model 10 knee and low-risk endpoint.

Recalculates all FOUR layouts in one PBS job, ten independent paired batches
of 50,000 birds each at response probabilities 0.6 and 0.0. For EACH batch,
reference and highest-AEP Q_lit must EXACTLY equal the corresponding values
from the archived earlier six-layout confirmation, at both probabilities.
A mismatch is fatal: write a detailed failure report, exit nonzero, and
NEVER calculate paired comparisons. Only on all 40 exact matches do we
write multiplicity-adjusted pairwise comparisons for the four layouts.

No optimisation, bird-model changes, or overwriting previous results.

Usage:
    python model10_confirm_final_selected_with_guards.py --check-inputs
    python -u model10_confirm_final_selected_with_guards.py --run
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import model10_fhd_robust_seeded_50k_72sectors as base
from model9_rotor_transit import RotorTransitModel
from model10_band_strike import BandCrossingAccumulator

ROOT=Path(__file__).resolve().parent
V3={seed:ROOT/f'hpc_full_run_v3_72sectors{("_seed"+str(seed)) if seed!=730101 else ""}'/'optimisation' for seed in (730101,730102,730103)}
OLD=ROOT/'hpc_v3_paired_confirmation_p0'
OUTPUT=ROOT/'hpc_v3_final_knee_confirmation_guarded'
TRAJ=[915101+i for i in range(10)]
HEIGHT=[916101+i for i in range(10)]
N=50_000
BATCHES=10
EXPECTED={'highest_aep':(730101,455),'knee':(730103,418),'lowest_risk':(730103,220)}
CONTROL_LAYOUTS=['reference','highest_aep']
NEW_LAYOUTS=['knee','lowest_risk']
LAYOUT_ORDER=CONTROL_LAYOUTS+NEW_LAYOUTS
PAIR_NAMES=[('highest_aep','knee'),('knee','lowest_risk'),('highest_aep','lowest_risk'),('highest_aep','reference'),('knee','reference'),('lowest_risk','reference')]
RADIUS=242.24/2

def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path:Path):
    with path.open('r',encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))

def atomic_csv(path,rows):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def exact_one(rows,key,value):
    match=[r for r in rows if str(r[key])==str(value)]
    if len(match)!=1:raise RuntimeError(f'Expected one {key}={value}; got {len(match)}')
    return match[0]

def xy(row):
    return np.asarray([[float(row[f'x_{i:02d}_m']),float(row[f'y_{i:02d}_m'])] for i in range(12)],dtype=float)

def check_and_load():
    settings={s:json.loads((V3[s]/'run_settings.json').read_text()) for s in V3}
    s0=settings[730101]
    for seed,s in settings.items():
        if s['status']!='completed' or s['number_of_direction_sectors']!=72 or s['evaluations']!=480 or s['optimiser_seed']!=seed:raise RuntimeError('Incomplete or wrong optimisation settings')
        if (s['training_trajectory_seeds']!=s0['training_trajectory_seeds'] or s['training_height_seeds']!=s0['training_height_seeds'] or s['training_birds_per_batch']!=N//5 or s['training_batches']!=5):raise RuntimeError('Optimisation bird samples differ across seeds')
        if {k:v for k,v in s['source_hashes'].items() if k!='runner'} != {k:v for k,v in s0['source_hashes'].items() if k!='runner'}:raise RuntimeError('Frozen scientific source/input hashes differ')
        if read_csv(V3[seed]/'reference_training_batches.csv')!=read_csv(V3[730101]/'reference_training_batches.csv'):raise RuntimeError('Reference samples disagree')
    if base.source_hashes()!=s0['source_hashes'] or base.N_DIRECTION_SECTORS!=72:
        raise RuntimeError('Current frozen source differs from completed seed 730101')

    allrows=[]
    for seed,p in V3.items():
        for row in read_csv(p/'all_evaluated_candidates.csv'):
            if float(row['max_spacing_constraint_m'])<=1e-8 and float(row['max_boundary_violation_m'])<=1e-8:
                a=float(row['aep_gwh']);q=float(row['risk_per_entrant'])
                if math.isfinite(a) and math.isfinite(q):allrows.append((seed,int(row['evaluation_index']),a,q))
    if len(allrows)!=513:raise RuntimeError('Unexpected feasible layout count')
    vals=np.asarray([(r[2],r[3]) for r in allrows],dtype=float)
    nd=[]
    for i,(a,q) in enumerate(vals):
        if not np.any((vals[:,0]>=a)&(vals[:,1]<=q)&((vals[:,0]>a)|(vals[:,1]<q))):nd.append(allrows[i])
    if len(nd)!=24:raise RuntimeError('Combined non-dominated count changed')
    high=max(nd,key=lambda r:r[2]); low=min(nd,key=lambda r:r[3])
    candidates=[]
    for r in nd:
        a=(r[2]-low[2])/(high[2]-low[2]);q=(r[3]-low[3])/(high[3]-low[3])
        candidates.append((r,(a-q)/math.sqrt(2)))
    knee=max(candidates,key=lambda r:r[1])[0]
    for label,r in [('highest_aep',high),('knee',knee),('lowest_risk',low)]:
        if r[:2] != EXPECTED[label]:raise RuntimeError(f'{label} changed: {r[:2]}')
    if len(set(TRAJ+HEIGHT))!=20:raise RuntimeError('Fresh seed collision')
    original=json.loads((OLD/'plan.json').read_text())
    if original['trajectory_seeds']!=TRAJ or original['height_seeds']!=HEIGHT or original['batch_count']!=BATCHES or original['birds_per_batch']!=N:raise RuntimeError('Original independent confirmation does not match fixed seeds')
    if original['source_hashes']!=s0['source_hashes']:raise RuntimeError('Original confirmation scientific sources differ')
    original_rows=read_csv(OLD/'fresh_batches_p06_p0.csv')
    if len(original_rows)!=BATCHES*6*2:
        raise RuntimeError('Original confirmation is incomplete or contains extra rows')
    existing={(int(r['batch_index']),r['layout'],float(r['response_probability'])):r for r in original_rows}
    if len(existing)!=len(original_rows):
        raise RuntimeError('Original confirmation has duplicate batch-layout-p combinations')
    for i in range(BATCHES):
        for label in ('highest_aep','reference'):
            for p in (.6,.0):
                try:r=existing[(i,label,p)]
                except KeyError:raise RuntimeError(f'Original confirmation row missing {i,label,p}')
                if int(r['trajectory_seed'])!=TRAJ[i] or int(r['height_seed'])!=HEIGHT[i] or int(r['N_started'])!=N:raise RuntimeError('Original confirmation row seed mismatch')
    check_aep=original['layouts']['highest_aep']['aep5']
    if abs(float(check_aep)-high[2])>1e-8:raise RuntimeError('Highest-AEP score differs from earlier confirmation')
    layouts={}
    # Load exact frozen arrays, not rounded plot coordinates. Original highest-AEP
    # confirmation used pareto_layouts_xy_m.npy; verify identical byte hashes.
    for label in ('highest_aep', 'knee', 'lowest_risk'):
        seed,index=EXPECTED[label]
        front=read_csv(V3[seed]/'pareto_training.csv')
        row=exact_one(front,'evaluation_index',index)
        selected=next(r for r in nd if r[:2]==(seed,index))
        if abs(float(row['aep_gwh'])-selected[2])>1e-8 or abs(float(row['risk_per_entrant'])-selected[3])>1e-12:
            raise RuntimeError('Selected front row does not match combined front')
        front_xy=np.load(V3[seed]/'pareto_layouts_xy_m.npy',allow_pickle=False)
        if front_xy.shape!=(len(front),12,2):
            raise RuntimeError(f'Pareto numpy and CSV row counts differ: {seed}')
        coords=np.array(front_xy[front.index(row)],dtype=float)
        if not np.array_equal(coords,xy(row)):
            raise RuntimeError(f'CSV and frozen numpy coordinates differ: {label}')
        base.check_layout(coords,968.96)
        layouts[label]={'xy':coords,'aep_gwh':selected[2],
            'optimisation_sample_q':selected[3],'seed':seed,'eval':index}
    reference_xy=np.array(base.load_layout(base.REFERENCE),dtype=float)
    base.check_layout(reference_xy,968.96)
    layouts['reference']={'xy':reference_xy,
        'aep_gwh':float(s0['reference_aep_gwh']),
        'optimisation_sample_q':float(s0['reference_training_risk_per_entrant']),
        'seed':None,'eval':None}
    for label in CONTROL_LAYOUTS:
        oldlayout=original['layouts'][label]
        if hashlib.sha256(layouts[label]['xy'].tobytes()).hexdigest()!=oldlayout['coordinates_sha256']:
            raise RuntimeError(f'EXACT frozen coordinate hash mismatch for control {label}')
        if float(oldlayout['aep5'])!=layouts[label]['aep_gwh']:
            raise RuntimeError(f'AEP mismatch for control {label}')
        if abs(float(oldlayout['training_q'])-layouts[label]['optimisation_sample_q'])>1e-14:
            raise RuntimeError(f'Optimisation-sample risk mismatch for control {label}')
    angle=math.radians(225)
    cross=np.array([-math.sin(angle),math.cos(angle)])
    corners=np.array([[0,0],[0,6000],[6000,0],[6000,6000]])
    lo,hi=np.min(corners@cross),np.max(corners@cross)
    for label,layout in layouts.items():
        bound=layout['xy']@cross
        layout['rotor_edge_margin_m']=float(min(bound.min()-lo,hi-bound.max())-RADIUS)
        if layout['rotor_edge_margin_m']<0:
            raise RuntimeError(f'Corridor clipping: {label}')
    plan={'purpose':'Guarded four-layout independent confirmation: recalculate two exact controls every batch; no statistics on mismatch',
          'optimisation_samples_identical_across_seeds':True,
          'combined_feasible_count':513,'combined_nondominated_count':24,
          'selected':{name:{k:v for k,v in val.items() if k!='xy'}|{'coordinates_sha256':hashlib.sha256(val['xy'].tobytes()).hexdigest()} for name,val in layouts.items()},
          'response_probabilities':[.6,.0], 'trajectory_seeds':TRAJ,'height_seeds':HEIGHT,'birds_per_batch':N,'batch_count':BATCHES,
          'original_data_sha256':sha(OLD/'fresh_batches_p06_p0.csv'),
          'original_plan_sha256':sha(OLD/'plan.json'),
          'layout_order':LAYOUT_ORDER,'control_layouts':CONTROL_LAYOUTS,
          'exact_reproduction_policy':'Each of 10 batches x 2 controls x 2 response probabilities: float(Q_lit_new)==float(Q_lit_archived), no tolerance. Stop on first mismatch before comparisons.',
          'optimisation_inputs_sha256':{f'{seed}_{filename}':sha(V3[seed]/filename) for seed in V3 for filename in ('run_settings.json','pareto_training.csv','all_evaluated_candidates.csv')},
          'code_sources_sha256':s0['source_hashes'],
          'primary_comparisons':[[a,b] for a,b in PAIR_NAMES],
          'statistics':'For each probability: 6 paired tests over 10 batches; Bonferroni 95%-simultaneous CIs and Holm-adjusted p-values (df=9).'}
    return plan,layouts,existing

def check_checkpoint(checkpoint, original):
    expected=[(i,name,p) for i in range(BATCHES) for name in LAYOUT_ORDER for p in (.6,.0)]
    got=[(int(r['batch_index']),r['layout'],float(r['response_probability'])) for r in checkpoint]
    if got!=expected[:len(got)] or len(checkpoint)%(2*len(LAYOUT_ORDER)):
        raise RuntimeError('Saved checkpoint is not a contiguous sequence of complete 4-layout batches')
    for row in checkpoint:
        i=int(row['batch_index']);label=row['layout'];p=float(row['response_probability'])
        if int(row['trajectory_seed'])!=TRAJ[i] or int(row['height_seed'])!=HEIGHT[i] or int(row['N_started'])!=N:
            raise RuntimeError('Checkpoint mismatched bird seed or count')
        if label in CONTROL_LAYOUTS and float(row['Q_lit'])!=float(original[(i,label,p)]['Q_lit']):
            raise RuntimeError(f'GUARD FAIL: saved control batch {i}, {label}, p={p} no longer matches original')

def write_json_atomic(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def fail_guard(batch,label,p,observed,original,number_already_matched):
    details={'status':'FAILED','message':'Exact control Q_lit reproduction failed. No paired comparisons computed.',
        'batch_index':batch,'layout':label,'response_probability':p,
        'expected_archived_Q_lit':float(original['Q_lit']),
        'reproduced_Q_lit':float(observed),
        'difference':float(observed)-float(original['Q_lit']),
        'strict_policy':'Binary float equality after reading archived CSV; no tolerance',
        'control_values_matched_before_failure':number_already_matched}
    write_json_atomic(OUTPUT/'consistency_report.json',details)
    print('EXACT REPRODUCTION FAILURE — STOPPING:',json.dumps(details,indent=2),flush=True)
    raise RuntimeError('Guard failed. No paired statistics calculated. Inspect consistency_report.json.')

def qonly(xy,pop,rotor,blade,band):
    acc=BandCrossingAccumulator(blade=blade,scenario=band,
         number_of_trajectories=pop.number_of_trajectories,number_of_turbines=12)
    result=rotor.simulate(xy,pop,crossing_callback=acc)
    result.validate_accounting()
    return float(np.sum(acc.contribution_matrix))/result.number_started, result.conditional_rotor_transit_intensity

def stats_pair(values):
    x=np.asarray(values,dtype=float)
    mean=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)))
    tval=abs(mean/se) if se else (math.inf if mean else 0.0)
    p=float(2*student_t.sf(tval,len(x)-1))
    return mean,se,p

def make_report(additional,original,layouts,plan):
    allscores={}
    # Analyses MUST use this job's recalculated values for all four layouts.
    # No reuse of archived numbers is permitted in the final comparison.
    for row in additional:
        key=(int(row['batch_index']),row['layout'],float(row['response_probability']))
        if key in allscores:raise RuntimeError(f'Duplicate comparison row: {key}')
        allscores[key]=float(row['Q_lit'])
    if len(allscores)!=4*2*BATCHES:raise RuntimeError(f'Expected 80 independent observations, got {len(allscores)}')
    means=[]; comparisons=[]
    for p in (.6,.0):
        for label in ['highest_aep','knee','lowest_risk','reference']:
            x=[allscores[(i,label,p)] for i in range(BATCHES)]
            means.append({'layout':label,'response_probability':p,'mean_Q_lit':float(np.mean(x)),
                         'sd_batches':float(np.std(x,ddof=1)),'n_batches':BATCHES,'birds_per_batch':N})
        tests=[]
        for la,lb in PAIR_NAMES:
            diffs=[allscores[(i,la,p)]-allscores[(i,lb,p)] for i in range(BATCHES)]
            m,se,pval=stats_pair(diffs)
            bound=float(student_t.ppf(1-.05/(2*len(PAIR_NAMES)),BATCHES-1)*se)
            q_high=np.mean([allscores[(i,'highest_aep',p)] for i in range(BATCHES)])
            tests.append({'family':'main_response_6' if p==.6 else 'no_response_control_6',
              'response_probability':p,'layout_A':la,'layout_B':lb,'mean_A_minus_B':m,
              'risk_delta_percent_of_highest_AEP':100*m/q_high,
              'bonferroni_simultaneous95_lower':m-bound,'bonferroni_simultaneous95_upper':m+bound,
              'raw_two_sided_p':pval,'holm_adjusted_p':None})
        # Holm step-down across all six predefined comparisons in THIS probability family.
        order=sorted(range(len(tests)),key=lambda i:tests[i]['raw_two_sided_p'])
        running=0
        for rank,idx in enumerate(order):
            running=max(running,(len(tests)-rank)*tests[idx]['raw_two_sided_p'])
            tests[idx]['holm_adjusted_p']=float(min(1.0,running))
        comparisons+=tests
    return {'status':'completed','plan':plan,'means':means,'paired_comparisons':comparisons}

def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-inputs',action='store_true')
    group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    plan,layouts,original=check_and_load()
    print('INPUT CHECK PASS — three identical optimisation bird samples, 24 ND; knee 730103/418; low-risk 730103/220; EXACT control coordinate hashes verified',flush=True)
    print('NOTE: 40 per-batch numerical control checks will run on HPC; preflight does not establish their result.',flush=True)
    print('New layout corridor margins (m):',{name:round(d['rotor_edge_margin_m'],3) for name,d in layouts.items()},flush=True)
    if args.check_inputs:return
    if (OUTPUT/'report.json').exists():
        raise FileExistsError('Completed result exists — refusing to overwrite')
    if (OUTPUT/'consistency_report.json').exists():
        prior_guard=json.loads((OUTPUT/'consistency_report.json').read_text())
        if prior_guard.get('status')=='FAILED':
            raise RuntimeError('Earlier consistency check FAILED; do not resume without investigating. See consistency_report.json')
    if OUTPUT.exists():
        if json.loads((OUTPUT/'plan.json').read_text())!=plan:
            raise RuntimeError('Saved plan differs; refusing resume')
    else:
        OUTPUT.mkdir()
        write_json_atomic(OUTPUT/'plan.json',plan)
    path=OUTPUT/'guarded_fresh_batches.csv'
    rows=read_csv(path) if path.exists() else []
    check_checkpoint(rows,original)
    completed=len(rows)//(2*len(LAYOUT_ORDER))
    components=base.model_components()
    if components[3].response_probability!=.6:
        raise RuntimeError('Expected frozen p=0.6')
    zero_response=replace(components[3],response_probability=0.0)
    zero_rotor=RotorTransitModel(zero_response,components[5].rotor_geometry,
       rotor_plane_heading_deg_toward=components[5].rotor_plane_heading_deg_toward,
       flight_height_distribution=components[4])
    if zero_rotor.behaviour.response_probability!=0.0:
        raise RuntimeError('No-response control is not p=0')
    start=time.monotonic()
    matched=completed*len(CONTROL_LAYOUTS)*2
    print(f'Resuming after {completed}/{BATCHES} fully guarded batches; {matched}/40 controls checked',flush=True)
    for i in range(completed,BATCHES):
        pop=base.make_birds(components,N,1,TRAJ[i],HEIGHT[i])[0]
        batch_rows=[]
        # Compute the controls FIRST, comparing each against the saved Q_lit.
        # Do not calculate the two new layouts for a batch if a guard fails.
        for name in LAYOUT_ORDER:
            for p,rotor in ((.6,components[5]),(.0,zero_rotor)):
                q,transit=qonly(layouts[name]['xy'],pop,rotor,components[7],components[6])
                if name in CONTROL_LAYOUTS:
                    old_row=original[(i,name,p)]
                    if q!=float(old_row['Q_lit']):
                        fail_guard(i,name,p,q,old_row,matched)
                    matched+=1
                batch_rows.append({'batch_index':i,'layout':name,'response_probability':p,
                  'trajectory_seed':TRAJ[i],'height_seed':HEIGHT[i],'N_started':N,
                  'Q_lit':q,'E_RT_CE':transit,'AEP_5_GWh':layouts[name]['aep_gwh']})
        rows.extend(batch_rows)
        atomic_csv(path,rows)
        print(f'BATCH {i+1}/{BATCHES} saved: 4 layouts x 2 p; guards {matched}/40 EXACT; elapsed {(time.monotonic()-start)/3600:.2f} h',flush=True)
    check_checkpoint(rows,original)
    if matched!=40 or len(rows)!=BATCHES*len(LAYOUT_ORDER)*2:
        raise RuntimeError('Incomplete consistency checks — refusing statistics')
    # This PASSED artifact is written BEFORE any statistical comparison.
    write_json_atomic(OUTPUT/'consistency_report.json',{
        'status':'PASSED','controls':['reference','highest_aep'],
        'matching_values':40,'total_expected':40,'batch_count':BATCHES,
        'response_probabilities':[.6,.0],
        'exact_equality_no_tolerance':True,'archived_confirmation_sha256':plan['original_data_sha256'],
        'message':'All 40 recomputed Q_lit observations exactly equal the previous confirmation.'})
    print('GUARD PASSED: 40/40 original Q_lit values reproduced exactly. Computing comparisons now.',flush=True)
    report=make_report(rows,original,layouts,plan)
    report['control_consistency_status']='PASSED_40_OF_40_EXACT'
    atomic_csv(OUTPUT/'layout_means.csv',report['means'])
    atomic_csv(OUTPUT/'paired_comparisons_multiplicity_adjusted.csv',report['paired_comparisons'])
    write_json_atomic(OUTPUT/'report.json',report)
    print('COMPLETE:',OUTPUT,flush=True)

if __name__=='__main__':main()
