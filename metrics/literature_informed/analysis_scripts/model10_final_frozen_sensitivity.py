#!/usr/bin/env python3
"""Frozen 2026 Model 10 bird-risk sensitivities of the final 4 layouts.

Scientific baseline, exact saved layouts, and guard checks are imported from
model10_confirm_final_selected_with_guards.py. Does not retrain/optimise, change
AEP or alter frozen source files. One PBS job per sensitivity family.

Preflight: python model10_final_frozen_sensitivity.py --dimension probability --check-inputs
Full run:  python -u model10_final_frozen_sensitivity.py --dimension probability --run

The eight-direction family is deliberately BLOCKED from running until the user
acknowledges original-square entry-corridor rotor clipping. Read README.
"""
from __future__ import annotations
import argparse, csv, dataclasses, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import model10_fhd_robust_seeded_50k_72sectors as base
import model10_confirm_final_selected_with_guards as frozen
from model9_rotor_transit import RotorGeometry, RotorTransitModel
from model10_band_strike import BandCrossingAccumulator

ROOT=Path(__file__).resolve().parent
N=50_000
TRAJ=tuple(915101+i for i in range(10))
HEIGHT=tuple(916101+i for i in range(10))
D=242.24
SCENARIOS={
 'direction':(0.,45.,90.,135.,180.,225.,270.,315.),
 'probability':(.3,.6,.9),
 'radius':(.5,1.,2.),
 'hub':(150.,160.,170.,180.),
 'speed':(10.79,14.01,14.9),
}
BASELINE={'direction':225.,'probability':.6,'radius':2.,'hub':150.,'speed':14.9}
ORDER=('highest_aep','knee','lowest_risk','reference')
PAIRS=(('highest_aep','knee'),('knee','lowest_risk'),('highest_aep','lowest_risk'))

def atomic_json(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def atomic_csv(path, rows):
    if not rows:return
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def load_inputs():
    plan, layouts, unused = frozen.check_and_load()
    check=json.loads((ROOT/'hpc_v3_final_knee_confirmation_guarded'/'consistency_report.json').read_text())
    if check.get('status')!='PASSED' or check.get('matching_values')!=40:
        raise RuntimeError('Earlier exact reproduction gate has not passed')
    previous=read_csv(ROOT/'hpc_v3_final_knee_confirmation_guarded'/'guarded_fresh_batches.csv')
    prev={(int(r['batch_index']),r['layout'],float(r['response_probability'])):float(r['Q_lit']) for r in previous}
    if len(prev)!=80 or len(previous)!=80:raise RuntimeError('Earlier four-layout confirmation not complete')
    if tuple(plan['trajectory_seeds'])!=TRAJ or tuple(plan['height_seeds'])!=HEIGHT or plan['birds_per_batch']!=N:
        raise RuntimeError('Independent batches changed')
    if set(layouts)!=set(ORDER):raise RuntimeError('Selected layout set changed')
    for label in ORDER:
        if not np.isfinite(layouts[label]['xy']).all():raise RuntimeError('Non-finite coordinates')
    return plan,layouts,prev

def corridor_margin(xy, heading):
    t=math.radians(heading)
    cross=np.array([-math.sin(t),math.cos(t)])
    corners=np.array([[0.,0.],[0.,6000.],[6000.,0.],[6000.,6000.]])
    ends=corners@cross; projection=xy@cross
    return float(min(projection.min()-ends.min(),ends.max()-projection.max())-D/2)

def corridor_report(layouts):
    return [{'direction_deg_toward':angle,'layout':label,'rotor_edge_margin_m':corridor_margin(layouts[label]['xy'],angle),
             'rotor_clipped_by_entry_corridor':bool(corridor_margin(layouts[label]['xy'],angle)<0)}
             for angle in SCENARIOS['direction'] for label in ORDER]

def rebuild_rotor(components, *, behaviour=None, height=None, plane=None):
    behaviour=components[3] if behaviour is None else behaviour
    height=components[5].rotor_geometry.hub_height_m if height is None else height
    plane=components[5].rotor_plane_heading_deg_toward if plane is None else plane
    return RotorTransitModel(behaviour,RotorGeometry(D,float(height)),
             rotor_plane_heading_deg_toward=float(plane),flight_height_distribution=components[4])

def change_components(base_c,dim,value):
    c=list(base_c)
    if dim=='probability':
        b=dataclasses.replace(c[3],response_probability=value)
        c[3]=b;c[5]=rebuild_rotor(c,behaviour=b)
        if not math.isclose(c[5].behaviour.response_probability,value):raise RuntimeError('Probability did not propagate')
    elif dim=='radius':
        b=dataclasses.replace(c[3],response_radius_m=value*D)
        c[3]=b;c[5]=rebuild_rotor(c,behaviour=b)
        if not math.isclose(c[5].behaviour.response_radius_m,value*D):raise RuntimeError('Radius did not propagate')
    elif dim=='hub':
        c[5]=rebuild_rotor(c,height=value)
        if not math.isclose(c[5].rotor_geometry.hub_height_m,value):raise RuntimeError('Hub height did not propagate')
    elif dim=='speed':
        c[6]=dataclasses.replace(c[6],bird_speed_m_s=value)
        if not math.isclose(c[6].bird_speed_m_s,value):raise RuntimeError('Band bird speed did not propagate')
    elif dim=='direction':
        # FROZEN OLD FLIGHT-AXIS CONVENTION: bird, rotor plane, and Band wind
        # are rotated together. Band wind points 180deg opposite the bird.
        c[5]=rebuild_rotor(c,plane=value)
        c[6]=dataclasses.replace(c[6],wind_heading_deg_toward=(value+180)%360)
        if not math.isclose(c[5].rotor_plane_heading_deg_toward,value):raise RuntimeError('Plane direction failed')
    else:raise RuntimeError(f'Unknown dimension {dim}')
    return tuple(c)

def birds_for(base_c,c,dim,value,i):
    if dim=='direction' and value!=225.:
        # Reproduce the original Model9Population.generate_parallel sampler;
        # only the initial heading changes. Same random-number seed per batch.
        model=base.populations.__globals__.get('Model9Population')
        if model is None:raise RuntimeError('Model9Population not present in frozen sampler')
        pop=model.generate_parallel(
              flight_height_distribution=c[4],flight_height_seed=HEIGHT[i],
              site_width_m=6000,site_height_m=6000,
              heading_deg_toward=float(value),number_of_trajectories=N,
              maximum_response_events=12,
              longitudinal_margin_m=c[3].response_radius_m+c[3].return_length_m,
              trajectory_seed=TRAJ[i])
        h=np.asarray(pop.trajectories.headings_deg_toward)
        if h.shape!=(N,) or not np.allclose(h,value,rtol=0,atol=1e-12):raise RuntimeError('Bird headings wrong')
        return pop
    return base.make_birds(c,N,1,TRAJ[i],HEIGHT[i])[0]

def qonly(xy,pop,rotor,blade,band):
    acc=BandCrossingAccumulator(blade=blade,scenario=band,
         number_of_trajectories=pop.number_of_trajectories,number_of_turbines=12)
    res=rotor.simulate(xy,pop,crossing_callback=acc)
    res.validate_accounting()
    return float(np.sum(acc.contribution_matrix))/res.number_started, float(res.conditional_rotor_transit_intensity)

def summarise(rows,dim):
    raw={(int(r['batch_index']),float(r['scenario_value']),r['layout']):float(r['Q_lit']) for r in rows}
    expected=10*len(SCENARIOS[dim])*len(ORDER)
    if len(raw)!=expected or len(rows)!=expected:raise RuntimeError(f'Incomplete data for stats: {len(raw)}/{expected}')
    means=[];tests=[];rankings=[]
    for val in SCENARIOS[dim]:
        vs={label:np.array([raw[(i,val,label)] for i in range(10)],dtype=float) for label in ORDER}
        for label in ORDER:
            x=vs[label];se=float(x.std(ddof=1)/math.sqrt(10));critical=float(student_t.ppf(.975,9))
            means.append({'dimension':dim,'scenario_value':val,'layout':label,'mean_Q_lit':float(x.mean()),
              'sd_per_50000_bird_batch':float(x.std(ddof=1)), 'n_batches':10,
              'ci95_lower':float(x.mean()-critical*se),'ci95_upper':float(x.mean()+critical*se)})
        outcome=[]
        for la,lb in PAIRS:
            diff=vs[la]-vs[lb];m=float(diff.mean());sd=float(diff.std(ddof=1));se=sd/math.sqrt(10)
            p=float(2*student_t.sf(abs(m/se),9)) if se>0 else (0. if m else 1.)
            width=float(student_t.ppf(1-.05/(2*len(PAIRS)),9)*se)
            outcome.append({'dimension':dim,'scenario_value':val,'layout_A':la,'layout_B':lb,
              'mean_A_minus_B':m,'mean_difference_percent_of_highest_AEP_risk':100*m/float(vs['highest_aep'].mean()),
              'paired_batch_sd':sd,'bonferroni_simultaneous95_lower':m-width,
              'bonferroni_simultaneous95_upper':m+width,'raw_p':p,'holm_adjusted_p':None,
              'ranking_reversal_for_pair':bool(m<0),
              'statistically_supported_reversal':bool(m<0 and m+width<0)})
        ix=sorted(range(len(outcome)),key=lambda i:outcome[i]['raw_p'])
        accum=0.
        for rank,j in enumerate(ix):
            accum=max(accum,(len(PAIRS)-rank)*outcome[j]['raw_p'])
            outcome[j]['holm_adjusted_p']=min(1.,float(accum))
        tests.extend(outcome)
        reversed_pairs=[f'{r["layout_A"]} vs {r["layout_B"]}' for r in outcome if r['ranking_reversal_for_pair']]
        rankings.append({'dimension':dim,'scenario_value':val,
          'risk_order_high_to_low':' > '.join(sorted(ORDER[:3],key=lambda label:vs[label].mean(),reverse=True)),
          'any_ranking_reversal':bool(reversed_pairs),'reversed_pairs':'; '.join(reversed_pairs),
          'any_adjusted_supported_reversal':any(r['statistically_supported_reversal'] for r in outcome)})
    return means,tests,rankings

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dimension',required=True,choices=tuple(SCENARIOS))
    z=ap.add_mutually_exclusive_group(required=True)
    z.add_argument('--check-inputs',action='store_true')
    z.add_argument('--run',action='store_true')
    ap.add_argument('--acknowledge-direction-corridor-clipping',action='store_true',
       help='ONLY after explicit scientific approval to retain the original square-site entry corridor in directional scenarios')
    args=ap.parse_args();dim=args.dimension
    plan,layouts,previous=load_inputs()
    b=base.model_components()
    if b[1].wind.n_sectors!=72:raise RuntimeError('72-sector FLORIS baseline changed')
    if abs(b[3].response_probability-.6)>1e-12 or abs(b[3].response_radius_m-2*D)>1e-8:
        raise RuntimeError('Baseline behaviour changed')
    if abs(b[5].rotor_geometry.hub_height_m-150)>1e-8 or abs(b[6].bird_speed_m_s-14.9)>1e-12:
        raise RuntimeError('Baseline hub/speed changed')
    for val in SCENARIOS[dim]:
        c=change_components(b,dim,val)
        if dim=='direction':
            # Confirm exactly perpendicular to FIXED geographical wind is not used:
            # here Band wind rotates by 180deg, so cosine=-1, approach=upwind.
            along=math.cos(math.radians(val-c[6].wind_heading_deg_toward))
            if abs(along+1.)>1e-12:raise RuntimeError(f'Wind-axis rotation violates prior convention at {val}')
        if c[5].rotor_geometry.hub_height_m not in SCENARIOS['hub']:
            raise RuntimeError('Invalid hub height')
    margins=corridor_report(layouts)
    clipped=[r for r in margins if r['rotor_clipped_by_entry_corridor']]
    print(f'INPUT CHECK PASS: {dim} values {SCENARIOS[dim]}; 4 exact final layouts; 10 paired batches x {N}',flush=True)
    if dim=='direction':
        print('BAND SIGN CONVENTION: wind TOWARD=(bird TOWARD+180) mod 360. For 135 and 315: cosine=-1 => UPWIND, NOT perpendicular.',flush=True)
        print('DIRECTION CORRIDOR CLIPPING:',len(clipped),'layout-direction cases:',[(r['direction_deg_toward'],r['layout'],round(r['rotor_edge_margin_m'],2)) for r in clipped],flush=True)
    if args.check_inputs:return
    if dim=='direction' and clipped and not args.acknowledge_direction_corridor_clipping:
        raise RuntimeError('DIRECTION HOLD: some rotors intersect the original entry-corridor boundary. Discuss this before submitting; no run started.')
    out=ROOT/f'hpc_model10_frozen_final_{dim}'
    if (out/'report.json').exists():
        raise FileExistsError('This sensitivity has a completed report; refusing to overwrite it')
    experiment={'status':'in_progress','dimension':dim,'values':list(SCENARIOS[dim]),
        'baseline_value':BASELINE[dim],'layout_order':list(ORDER),
        'sample_seeds':{'trajectory':TRAJ,'flight_height':HEIGHT},'N_per_batch':N,'n_batches':10,
        'aep_policy':'Fixed archived 72-sector AEP for every sensitivity, including altered hub height; NO AEP recomputation.',
        'flight_direction_policy':'Rotate bird initial heading, rotor normal, Band wind axis; wind TOWARD=(bird TOWARD+180) mod 360',
        'direction_corridor_clipping_count':len(clipped),
        'direction_corridor_clipping_acknowledged':bool(args.acknowledge_direction_corridor_clipping),
        'sensitivity_program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_guard_plan_hash':plan['original_data_sha256'],
        'statistical_family':'3 paired contrasts, Bonferroni simultaneous 95% intervals and Holm-adjusted p-values WITHIN each scenario. Not adjusted across scenarios.',
        'reference':'context, not part of the three pairwise tests',
        'exact_baseline_reproduction':'Each baseline setting re-evaluated for all 4 layouts and all 10 same batches, EXACT agreement with independent baseline Q_lit is required before statistics.'}
    if not out.exists():
        out.mkdir(exist_ok=False)
        atomic_json(out/'plan.json',experiment)
        if dim=='direction':atomic_csv(out/'direction_entry_corridor_margins.csv',margins)
    else:
        prior=json.loads((out/'plan.json').read_text())
        if prior!=experiment:raise RuntimeError('Existing output plan differs from this frozen run; refusing to mix')
    path=out/'batch_results.csv'
    existing=read_csv(path) if path.exists() else []
    expected_keys=[(i,float(val),label) for i in range(10) for val in SCENARIOS[dim] for label in ORDER]
    keys=[(int(r['batch_index']),float(r['scenario_value']),r['layout']) for r in existing]
    if keys!=expected_keys[:len(keys)] or len(existing)%len(ORDER)!=0:
        raise RuntimeError('Existing checkpoint not a contiguous set of complete scenario batches')
    for r in existing:
        i=int(r['batch_index']);val=float(r['scenario_value']);lab=r['layout']
        if int(r['trajectory_seed'])!=TRAJ[i] or int(r['height_seed'])!=HEIGHT[i]:raise RuntimeError('Checkpoint seed mismatch')
        if val==BASELINE[dim] and float(r['Q_lit'])!=previous[(i,lab,.6)]:
            raise RuntimeError('Existing checkpoint baseline does not EXACTLY reproduce prior confirmation')
    completed=set(keys);started=time.time()
    for i in range(10):
        for val in SCENARIOS[dim]:
            needed=[lab for lab in ORDER if (i,val,lab) not in completed]
            if not needed:continue
            if len(needed)!=4:raise RuntimeError('Partial scenario checkpoint not permitted')
            c=change_components(b,dim,val)
            birds=birds_for(b,c,dim,val,i)
            rows=[]
            for label in ORDER:
                q,transit=qonly(layouts[label]['xy'],birds,c[5],c[7],c[6])
                if val==BASELINE[dim] and q!=previous[(i,label,.6)]:
                    failure={'status':'FAILED','dimension':dim,'scenario_value':val,'batch':i,'layout':label,
                       'expected_Q_lit':previous[(i,label,.6)],'observed_Q_lit':q,
                       'message':'Exact baseline reproduction failed; statistics must not be computed.'}
                    atomic_json(out/'failure.json',failure)
                    raise RuntimeError(f'BASELINE GUARD FAILED: {failure}')
                rows.append({'batch_index':i,'dimension':dim,'scenario_value':val,'layout':label,
                   'trajectory_seed':TRAJ[i],'height_seed':HEIGHT[i],'N_started':N,
                   'Q_lit':q,'E_RT_CE':transit,'AEP_72_sector_GWh_FIXED':layouts[label]['aep_gwh']})
            existing.extend(rows)
            atomic_csv(path,existing)
            completed.update((i,val,lab) for lab in ORDER)
            print(f'BATCH {i+1}/10 {dim}={val} DONE, rows={len(existing)}/{len(expected_keys)}, elapsed_h={(time.time()-started)/3600:.2f}',flush=True)
    if len(existing)!=len(expected_keys):raise RuntimeError('Not all scenarios completed')
    means,tests,rankings=summarise(existing,dim)
    atomic_csv(out/'layout_means_by_scenario.csv',means)
    atomic_csv(out/'three_paired_comparisons_bonferroni_holm.csv',tests)
    atomic_csv(out/'risk_ranking_reversals.csv',rankings)
    experiment['status']='completed';atomic_json(out/'report.json',experiment)
    print(f'COMPLETE {dim}: {len(existing)} batch-layout-scenario rows. Statistical comparisons computed only after exact baseline guard.',flush=True)

if __name__=='__main__':main()
