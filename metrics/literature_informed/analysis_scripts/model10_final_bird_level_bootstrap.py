#!/usr/bin/env python3
"""Model 10 final 455/418/220: reproduce exact 50k optimisation birds; paired STRATIFIED bird bootstrap.

Usage (HPC12, /home/uraj/model10):
  python model10_final_bird_level_bootstrap.py --check-inputs
  python -u model10_final_bird_level_bootstrap.py --run
  python model10_final_bird_level_bootstrap.py --analyse-existing

Never optimises, changes bird code, or touches the completed sensitivity results.
The per-bird arrays are generated only if missing. At each bootstrap iteration
we draw five birds WITH REPLACEMENT within each of 10,000 entry strata,
preserving the original five-batches-of-10k stratified design. We apply the
IDENTICAL draw to every layout. This resamples individuals, never batches.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np
import model10_fhd_robust_seeded_50k_72sectors as base
import model10_confirm_final_selected_with_guards as frozen
from model10_band_strike import BandCrossingAccumulator

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'hpc_v3_final_optimisation_sample_bird_bootstrap'
LAYOUTS=('highest_aep','knee','lowest_risk')
DISPLAY={'highest_aep':'Highest AEP (455)', 'knee':'Knee (418)', 'lowest_risk':'Lowest risk (220)'}
TRAJ=tuple(731101+i for i in range(5))
HEIGHT=tuple(732101+i for i in range(5))
BIRDS_PER_BATCH=10_000
BATCHES=5
BOOT_REPLICATES=10_000
BOOT_SEED=982741
# Hard-coded independently confirmed per-50k-batch SDs at p=0.6;
# validation is done against the existing CSV, not these literal values.
EXPECTED_EVAL={"highest_aep":(730101,455),"knee":(730103,418),"lowest_risk":(730103,220)}


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,obj):
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def atomic_csv(p,rows):
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def readcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def preflight():
    # Reuse the FINAL combined-front and coordinate-hash verifier.
    plan,layouts,_=frozen.check_and_load()
    s0=json.loads((frozen.V3[730101]/'run_settings.json').read_text())
    if tuple(s0['training_trajectory_seeds'])!=TRAJ or tuple(s0['training_height_seeds'])!=HEIGHT:
        raise RuntimeError('Optimisation bird seeds differ from frozen five batches')
    if s0['training_birds_per_batch']!=BIRDS_PER_BATCH or s0['training_batches']!=BATCHES:
        raise RuntimeError('Optimisation bird count differs from frozen five 10k batches')
    if plan['optimisation_samples_identical_across_seeds'] is not True:
        raise RuntimeError('Three optimisation runs use different bird samples')
    if base.N_DIRECTION_SECTORS!=72:
        raise RuntimeError('72-sector scientific runner not loaded')
    for label in LAYOUTS:
        if (layouts[label]['seed'],layouts[label]['eval'])!=EXPECTED_EVAL[label]:
            raise RuntimeError(f'Final {label} selection changed')
    guard_path=ROOT/'hpc_v3_final_knee_confirmation_guarded'/'consistency_report.json'
    guard=json.loads(guard_path.read_text())
    if guard['status']!='PASSED' or guard['matching_values']!=40:
        raise RuntimeError('Prior exact independent confirmation not passed')
    identity={label:{'evaluation':list(EXPECTED_EVAL[label]),
                     'optimisation_sample_Q':layouts[label]['optimisation_sample_q'],
                     'xy_sha256':hashlib.sha256(layouts[label]['xy'].tobytes()).hexdigest()}
              for label in LAYOUTS}
    meta={
        'purpose':'Paired individual-bird bootstrap for Model 10 final three, optimisation sample',
        'source_sha256':s0['source_hashes'],
        'run_settings_sha256':digest(frozen.V3[730101]/'run_settings.json'),
        'layout_identity':identity,
        'bird_trajectory_seeds':list(TRAJ),'bird_height_seeds':list(HEIGHT),
        'birds_per_batch':BIRDS_PER_BATCH,'batches':BATCHES,
        'total_optimisation_birds':50000,
        'entry_strata_per_batch':10000,
        'bootstrap_replicates':BOOT_REPLICATES,'bootstrap_seed':BOOT_SEED,
        'resampling':'10,000 paired bird-level bootstrap replicates; five birds with replacement within each of 10,000 entry strata; identical resample indices for all three layouts; not a batch bootstrap',
        'expected_method':'One bird per 1/10000 entry stratum in each of five fixed 10,000-bird optimisation batches',
        'no_new_optims':True,'no_new_independent_bird_samples':True,
    }
    return meta,layouts


def perbird_result(components,xy,pop):
    acc=BandCrossingAccumulator(blade=components[7],scenario=components[6],
        number_of_trajectories=pop.number_of_trajectories,number_of_turbines=12)
    sim=components[5].simulate(xy,pop,crossing_callback=acc)
    sim.validate_accounting()
    arr=np.asarray(acc.contribution_matrix.sum(axis=1),dtype=np.float64)
    if arr.shape!=(BIRDS_PER_BATCH,) or not np.isfinite(arr).all() or np.any(arr<0):
        raise RuntimeError('Invalid per-bird Band strike contributions')
    q=float(np.mean(arr))
    if abs(q-float(acc.expected_potential_strikes_per_entrant))>1e-14:
        raise RuntimeError('Per-bird mean differs from accumulator mean')
    return arr


def load_batch_checkpoint(i,expected_seed=None):
    path=OUT/f'optimisation_batch_{i:02d}_per_bird.npz'
    if not path.exists():return None
    with np.load(path,allow_pickle=False) as z:
        needed={'entry_fraction','trajectory_seed','height_seed',*LAYOUTS}
        if set(z.files)!=needed:raise RuntimeError(f'Incomplete checkpoint: {path}')
        if int(z['trajectory_seed'])!=TRAJ[i] or int(z['height_seed'])!=HEIGHT[i]:
            raise RuntimeError(f'Checkpoint seeds differ: {path}')
        data={k:np.array(z[k]) for k in z.files}
    for label in LAYOUTS:
        if data[label].shape!=(BIRDS_PER_BATCH,):raise RuntimeError('Checkpoint wrong bird count')
    entry=np.asarray(data['entry_fraction'])
    strata=np.floor(entry*BIRDS_PER_BATCH).astype(int)
    if entry.shape!=(BIRDS_PER_BATCH,) or not np.array_equal(np.sort(strata),np.arange(BIRDS_PER_BATCH)):
        raise RuntimeError('One-per-stratum design failed on saved checkpoint')
    return data


def extract_birds(meta,layouts):
    OUT.mkdir(exist_ok=True)
    p=OUT/'plan.json'
    if p.exists():
        if json.loads(p.read_text())!=meta:raise RuntimeError('Existing bootstrap output plan does not match; refusing overwrite')
    else:atomic_json(p,meta)
    components=base.model_components()
    for i in range(BATCHES):
        if load_batch_checkpoint(i) is not None:
            print(f'BATCH {i+1}/5 already verified; resuming',flush=True)
            continue
        birds=base.make_birds(components,BIRDS_PER_BATCH,1,TRAJ[i],HEIGHT[i])[0]
        e=np.asarray(birds.trajectories.entry_cross_fractions,dtype=np.float64)
        strata=np.floor(e*BIRDS_PER_BATCH).astype(int)
        if not np.array_equal(np.sort(strata),np.arange(BIRDS_PER_BATCH)):
            raise RuntimeError(f'Entry-stratum assumption failed for optimisation batch {i}')
        arrays={'entry_fraction':e,'trajectory_seed':np.array(TRAJ[i],dtype='int64'),
                'height_seed':np.array(HEIGHT[i],dtype='int64')}
        for label in LAYOUTS:
            arrays[label]=perbird_result(components,layouts[label]['xy'],birds)
        target=OUT/f'optimisation_batch_{i:02d}_per_bird.npz'
        temp=target.with_suffix('.npz.tmp')
        with temp.open('wb') as fh:
            np.savez_compressed(fh,**arrays)
            fh.flush();os.fsync(fh.fileno())
        os.replace(temp,target)
        print(f'BATCH {i+1}/5 saved; '+', '.join(f'{name}: {np.mean(arrays[name]):.12g}' for name in LAYOUTS),flush=True)


def analyse_existing(meta,layouts):
    p=OUT/'plan.json'
    if not p.exists() or json.loads(p.read_text())!=meta:
        raise RuntimeError('Completed per-bird extraction plan is missing/mismatched')
    # Original design: one bird from every 1/10000-width entry stratum PER batch.
    # Stack five actual birds for each fine stratum; preserve common-bird pairing.
    strat=np.empty((BIRDS_PER_BATCH,BATCHES,len(LAYOUTS)),dtype=np.float64)
    for batch in range(BATCHES):
        d=load_batch_checkpoint(batch)
        if d is None:raise RuntimeError('Cannot bootstrap without ALL five per-bird batches')
        s=np.floor(d['entry_fraction']*BIRDS_PER_BATCH).astype(int)
        ix=np.argsort(s)
        if not np.array_equal(s[ix],np.arange(BIRDS_PER_BATCH)):
            raise RuntimeError('Entry stratum assignment corrupted')
        for k,label in enumerate(LAYOUTS):
            strat[:,batch,k]=d[label][ix]
    observed=np.mean(strat,axis=(0,1))
    checked=[]
    for k,label in enumerate(LAYOUTS):
        expected=float(layouts[label]['optimisation_sample_q'])
        diff=float(observed[k]-expected)
        checked.append({'layout':label,'q_recomputed_from_individual_birds':float(observed[k]),
                       'q_archived_optimisation':expected,'difference':diff,
                       'passes_1e-12_absolute':abs(diff)<=1e-12})
    atomic_csv(OUT/'exact_optimisation_score_checks.csv',checked)
    if not all(r['passes_1e-12_absolute'] for r in checked):
        atomic_json(OUT/'guard_failure.json',{'status':'FAILED','details':checked,
          'message':'Individual contributions do not recover the frozen optimisation-sample Q. No bootstrap computed.'})
        raise RuntimeError('Optimisation score recreation FAILED; no statistics produced')
    flat=strat.reshape(BIRDS_PER_BATCH*BATCHES,3)
    rng=np.random.default_rng(BOOT_SEED)
    bs=np.empty((BOOT_REPLICATES,3))
    pos=0
    strata_base=(np.arange(BIRDS_PER_BATCH,dtype=np.int32)*BATCHES)[None,:,None]
    while pos<BOOT_REPLICATES:
        nb=min(16,BOOT_REPLICATES-pos)
        draw=rng.integers(0,BATCHES,size=(nb,BIRDS_PER_BATCH,BATCHES),dtype=np.int32)
        indices=(strata_base+draw).reshape(nb,-1)
        bs[pos:pos+nb]=flat[indices].mean(axis=1)
        pos+=nb
        if pos%1000==0:print(f'PAIRED STRATIFIED BIRD BOOTSTRAP {pos}/{BOOT_REPLICATES}',flush=True)
    labels_summary=[]
    with (ROOT/'hpc_v3_final_knee_confirmation_guarded'/'guarded_fresh_batches.csv').open(newline='') as f:
        eval_rows=list(csv.DictReader(f))
    eval_lookup={label:np.array([float(r['Q_lit']) for r in eval_rows
                      if r['layout']==label and float(r['response_probability'])==0.6]) for label in LAYOUTS}
    for j,label in enumerate(LAYOUTS):
        t=bs[:,j];ci=np.quantile(t,[.025,.975]); v=eval_lookup[label]
        if len(v)!=10:raise RuntimeError('Incomplete independent evaluation SD source')
        labels_summary.append({'layout':DISPLAY[label],'N_optimisation_sample':50000,
          'optimisation_sample_Q':float(observed[j]),'bootstrap_mean_Q':float(t.mean()),
          'bootstrap_sd_Q':float(t.std(ddof=1)),'bootstrap_sd_pct_of_bootstrap_mean':float(t.std(ddof=1)/t.mean()*100),
          'bootstrap_95percentile_lower':float(ci[0]),'bootstrap_95percentile_upper':float(ci[1]),
          'independent_50000_batch_mean_Q':float(v.mean()),
          'independent_50000_batch_sd_Q':float(v.std(ddof=1)),
          'independent_50000_batch_relative_sd_pct':float(v.std(ddof=1)/v.mean()*100)})
    atomic_csv(OUT/'bird_bootstrap_three_layout_summary.csv',labels_summary)
    diffs=[]
    for a,b in [('highest_aep','knee'),('knee','lowest_risk'),('highest_aep','lowest_risk')]:
        k,j=LAYOUTS.index(a),LAYOUTS.index(b)
        d=bs[:,k]-bs[:,j];q=np.quantile(d,[.025,.975])
        diffs.append({'comparison':f'{DISPLAY[a]} − {DISPLAY[b]}',
          'optimisation_sample_difference':float(observed[k]-observed[j]),
          'bootstrap_mean_difference':float(d.mean()),'bootstrap_sd_difference':float(d.std(ddof=1)),
          'bootstrap_95percentile_lower':float(q[0]),'bootstrap_95percentile_upper':float(q[1])})
    atomic_csv(OUT/'bird_bootstrap_three_paired_differences.csv',diffs)
    np.savez_compressed(OUT/'paired_10000_bird_bootstrap_replicates.npz',
                        highest_aep=bs[:,0],knee=bs[:,1],lowest_risk=bs[:,2])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    fig,ax=plt.subplots(figsize=(10.2,5.3))
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k,label in enumerate(LAYOUTS):
        d=bs[:,k]*1e5
        x=np.linspace(d.min()-3*d.std(),d.max()+3*d.std(),400)
        density=gaussian_kde(d)(x)
        ax.fill_between(x,density,alpha=.20,color=colors[k])
        ax.plot(x,density,lw=1.9,color=colors[k],label=DISPLAY[label])
        ax.axvline(observed[k]*1e5,color=colors[k],ls=':',lw=1.25)
    ax.set_xlabel(r'$Q_{\mathrm{lit}}$ — expected potential strikes per entering bird ($\times10^{-5}$)')
    ax.set_ylabel('Bootstrap probability density')
    ax.set_title('Model 10 — paired bird-level bootstrap',loc='left',fontsize=14)
    ax.grid(alpha=.16)
    ax.legend(loc='upper center',ncol=3,fontsize=9)
    fig.text(.12,.025,'N = 50,000 birds (optimisation sample); 10,000 paired resamples preserving entry strata.',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,1])
    fig.savefig(OUT/'model10_final_three_bird_bootstrap_distributions.png',dpi=230,facecolor='white')
    plt.close(fig)
    atomic_json(OUT/'report.json',{'status':'PASSED','number_of_birds':50000,
      'bootstrap_replicates':BOOT_REPLICATES,'sampling':'paired stratified individual-bird bootstrap',
      'optimisation_score_guards':checked,'layout_summary':labels_summary,
      'paired_differences':diffs,'caveat':'Conditional on the fixed optimisation sample and its 5 birds per entry stratum; SD may differ from 10 independent 50000-bird batches.'})
    print('COMPLETE: optimisation samples reproduced; 10,000 paired individual-bird bootstrap draws; statistics and figure saved.',flush=True)


def main():
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-inputs',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--analyse-existing',action='store_true')
    args=ap.parse_args()
    meta,layouts=preflight()
    print('INPUT CHECK PASS: final layouts and all 5 fixed 10k-bird optimisation seeds verified; no archived per-bird contribution arrays in supplied completed v3 archives.',flush=True)
    if args.check_inputs:return
    if args.run:
        if (OUT/'report.json').exists():
            raise FileExistsError('Completed results already exist; refusing overwrite')
        start=time.time();extract_birds(meta,layouts)
        print(f'EXTRACTION COMPLETE: {(time.time()-start)/60:.1f} min. Starting statistics.',flush=True)
    analyse_existing(meta,layouts)

if __name__=='__main__':main()
