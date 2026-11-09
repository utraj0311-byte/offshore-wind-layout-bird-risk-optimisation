#!/usr/bin/env python3
"""Model 10: controlled eight-heading sensitivity, fixed entry corridor.

This is a separate post-optimisation evaluation. It does not alter the
frozen meso-response, rotor-transit, Band or FLORIS source files.

Steps (checkpoint-resumable if interrupted; scientific gate failure is terminal):
  1. Geometry, source, sample-seed and Band-rotation preflight (no birds).
  2. Eight-direction experiment's 225-deg baseline against the earlier
     confirmation in the common width-weighted units Q*W.
  3. No-response controls at 45 and 90 deg (same width); compare every
     pair of layouts and headings with simultaneous confidence intervals.
  4. Remaining seven headings at p=0.6, with pairwise adjusted statistics.

All intermediate batches are checkpointed in a new directory. ANY failed
precondition stops the job BEFORE main direction-dependent comparisons.
Scientific interpretation: the wind axis rotates with bird heading and is
always 180 degrees opposite to the INITIAL bird heading. No annual flux.

Usage:
  python model10_direction_fixed_width_8way.py --check-inputs
  python -u model10_direction_fixed_width_8way.py --run
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

import model10_fhd_robust_seeded_50k_72sectors as base
import model10_final_frozen_sensitivity as sens

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'hpc_model10_frozen_final_direction_fixed_width'
D = 242.24
R = D / 2
SITE = 6000.0
SITE_CENTER = np.array([SITE / 2, SITE / 2])
W_BASE = SITE * math.sqrt(2.0)
W_FIXED = W_BASE + 2 * (R + 2 * D)
N_BASE = 50_000
N = int(math.floor(N_BASE * W_FIXED / W_BASE + 0.5))
BATCHES = 10
HEADINGS = tuple(float(v) for v in range(0, 360, 45))
LAYOUTS = ('highest_aep', 'knee', 'lowest_risk', 'reference')
PAIRS = (('highest_aep','knee'), ('knee','lowest_risk'), ('highest_aep','lowest_risk'))
TRAJ = tuple(915101 + i for i in range(BATCHES))
HEIGHT = tuple(916101 + i for i in range(BATCHES))
P0_HEADINGS = (45.0, 90.0)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def cross_unit(heading):
    a = math.radians(heading)
    return np.array([-math.sin(a), math.cos(a)], dtype=float)


def geometry(layouts):
    result = []
    for heading in HEADINGS:
        cross = cross_unit(heading)
        for label in LAYOUTS:
            coords = layouts[label]['xy']
            signed = (coords - SITE_CENTER) @ cross
            max_abs = float(np.max(np.abs(signed)))
            margin = W_FIXED / 2 - max_abs - R
            result.append({'heading_deg_toward':heading, 'layout':label,
                'fixed_entry_width_m':W_FIXED, 'max_abs_cross_offset_m':max_abs,
                'rotor_edge_clearance_m':margin,
                'full_rotor_coverage':bool(margin > 0)})
    return result


def check_inputs():
    original_plan, layouts, previous = sens.load_inputs()
    if tuple(original_plan['trajectory_seeds']) != TRAJ or tuple(original_plan['height_seeds']) != HEIGHT:
        raise RuntimeError('Independent sample seeds do not match the guarded confirmation')
    if set(layouts) != set(LAYOUTS):
        raise RuntimeError('Unexpected layout set')
    if sens.N != N_BASE or sens.D != D:
        raise RuntimeError('Frozen 50k bird sample or rotor diameter changed')
    if abs(W_BASE - 8485.281374238572) > 1e-7:
        raise RuntimeError('Incorrect original diagonal projection width')
    if N != 57_137:
        raise RuntimeError(f'Unexpected density-scaled bird count: {N}')
    comp = base.model_components()
    assert comp[1].wind.n_sectors == 72
    assert comp[3].response_probability == .6
    assert math.isclose(comp[3].response_radius_m, 2*D, abs_tol=1e-10)
    assert comp[5].rotor_plane_heading_deg_toward == 225.0
    assert comp[6].wind_heading_deg_toward == 45.0
    for angle in HEADINGS:
        changed = sens.change_components(comp, 'direction', angle)
        assert changed[5].rotor_plane_heading_deg_toward == angle
        assert changed[6].wind_heading_deg_toward == (angle + 180) % 360
        c = math.cos(math.radians(angle - changed[6].wind_heading_deg_toward))
        if abs(c + 1.0) > 1e-12:
            raise RuntimeError(f'Band sign convention failed for {angle}')
    margins = geometry(layouts)
    failing = [r for r in margins if not r['full_rotor_coverage']]
    if len(margins) != 32 or failing:
        raise RuntimeError(f'CLIPPING SAFEGUARD: {len(failing)} uncovered discs: {failing}')
    if len(previous) != 80:
        raise RuntimeError('Guarded original independent confirmation is incomplete')
    print(f'INPUT CHECK PASS: 32/32 positive margins; minimum {min(r["rotor_edge_clearance_m"] for r in margins):.3f} m',flush=True)
    print(f'Fixed corridor W={W_FIXED:.9f} m; 10 x {N} birds, N density ratio={N/N_BASE:.6f}',flush=True)
    print('Frozen: heading+rotor-normal+Band-wind rotate together; Band wind 180 degrees opposed (upwind).',flush=True)
    print('Original 72-sector baseline and four frozen layouts unchanged. No bird simulation in preflight.',flush=True)
    return comp, layouts, previous, margins


def fixed_width_population(comp, heading, batch):
    """Generate all frozen random streams; replace ONLY cross-flight entry coordinate.

    Retains the original stratified entry fractions, RNG response uniforms,
    side uniforms, height uniforms, upstream and downstream length. It does
    not modify the frozen Model 8/9 source or sampling algorithms.
    """
    from model9_rotor_transit import Model9Population
    changed = sens.change_components(comp, 'direction', heading)
    pop = Model9Population.generate_parallel(
        flight_height_distribution=changed[4], flight_height_seed=HEIGHT[batch],
        site_width_m=SITE, site_height_m=SITE, heading_deg_toward=heading,
        number_of_trajectories=N, maximum_response_events=12,
        longitudinal_margin_m=changed[3].response_radius_m + changed[3].return_length_m,
        trajectory_seed=TRAJ[batch], stratified_entries=True)
    original = pop.trajectories
    fractions = original.entry_cross_fractions
    if fractions is None or len(fractions)!=N or not np.all((fractions>=0)&(fractions<1)):
        raise RuntimeError('Frozen stratified entry fractions missing or invalid')
    cross = cross_unit(heading)
    new_across = (SITE_CENTER @ cross) + (fractions - 0.5) * W_FIXED
    original_across = original.start_positions_m @ cross
    shift = (new_across - original_across)[:, None] * cross[None, :]
    shifted_positions = original.start_positions_m + shift
    if np.max(np.abs(shifted_positions @ cross - new_across)) > 1e-8:
        raise RuntimeError('Fixed width entry positions not reproduced')
    replacement = dataclasses.replace(original, start_positions_m=shifted_positions)
    fixed = dataclasses.replace(pop, trajectories=replacement)
    if fixed.number_of_trajectories != N or fixed.trajectory_seed != TRAJ[batch] or fixed.flight_height_seed != HEIGHT[batch]:
        raise RuntimeError('Sample identity or size changed during corridor transformation')
    if not np.array_equal(fixed.trajectories.response_uniforms, original.response_uniforms):
        raise RuntimeError('Behaviour uniforms changed during entry transform')
    return fixed


def q_for(xy, birds, comp, probability):
    if probability not in (.0,.6):
        raise RuntimeError('Only 0.0 or 0.6 supported')
    if probability == .6:
        return sens.qonly(xy, birds, comp[5], comp[7], comp[6])[0]
    b = dataclasses.replace(comp[3], response_probability=0.0)
    rotor = sens.rebuild_rotor(comp, behaviour=b)
    return sens.qonly(xy, birds, rotor, comp[7], comp[6])[0]


def build_plan(margins):
    return {'experiment':'eight_direction_common_fixed_width',
        'status':'in_progress',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'frozen_sensitivity_script_sha256':hashlib.sha256(Path(sens.__file__).read_bytes()).hexdigest(),
        'heading_deg_toward':HEADINGS,
        'layouts':LAYOUTS,'bird_counts_per_batch':N,'batches':BATCHES,
        'trajectory_seeds':TRAJ,'height_seeds':HEIGHT,
        'rotor_diameter_m':D,'site_side_m':SITE,
        'original_baseline_corridor_width_m':W_BASE,'new_fixed_corridor_width_m':W_FIXED,
        'entry_density_birds_per_m':N_BASE/W_BASE,
        'input_population_method':'same frozen Model9Population.generate_parallel RNG; replace only cross-projection of entry starts using preserved stratified entry fractions',
        'rotor_coverage_checks':32,'minimum_rotor_clearance_m':min(x['rotor_edge_clearance_m'] for x in margins),
        'band_convention':'bird heading, rotor normal and Band wind rotate; Band wind TOWARD=(bird heading+180)%360; initially upwind',
        'bird_response_p':.6,'p0_controls':P0_HEADINGS,
        'aep_policy':'Fixed 72-sector original optimisation AEP; no new AEP evaluation',
        'baseline_consistency_gate':'Four two-sided 95% simultaneous Bonferroni paired Student-t intervals (alpha/4) on differences of W*Q between 225-deg fixed-width and original independent evaluation; all include zero; log old/new means, relative differences and intervals; do not retry automatically if failed',
        'no_response_gate':'At both 45/90 degrees, all six layout pair contrasts and four across-angle same-layout contrasts use 16 simultaneous Bonferroni 95% CIs; all must include zero',
        'three_internal_pairs':PAIRS,
        'statistical_family':'3 paired differences per direction; within-heading Bonferroni 95% simultaneous intervals, Holm p-values. No across-heading family correction.',
        'reference_policy':'Context only, not among 3 main contrasts.'}


def setup_outputs(margins):
    plan=build_plan(margins)
    OUTPUT.mkdir(parents=False,exist_ok=True)
    planfile=OUTPUT/'plan.json'
    if planfile.exists():
        existing=json.loads(planfile.read_text())
        if existing != json.loads(json.dumps(plan)):
            raise RuntimeError('Existing direction output plan differs; cannot safely resume')
    else:
        write_json(planfile, plan)
        write_csv(OUTPUT/'fixed_width_rotor_clearances.csv',margins)
    if (OUTPUT/'report.json').exists():
        raise FileExistsError('Direction run is already completed; refusing to overwrite')
    if (OUTPUT/'failure.json').exists():
        raise RuntimeError('A prior scientific gate FAILED. Refusing automatic retry/resubmission. Inspect failure.json and saved batch results before any new experiment.')


def restore_prefix(name, expected):
    f=OUTPUT/name
    existing=read_csv(f) if f.exists() else []
    keys=[(int(r['batch_index']),float(r['heading_deg_toward']),r['layout']) for r in existing]
    if keys != expected[:len(keys)] or len(existing)%4!=0:
        raise RuntimeError(f'Noncontiguous or incomplete checkpoint: {name}')
    for r in existing:
        i=int(r['batch_index']); ang=float(r['heading_deg_toward'])
        if int(r['trajectory_seed'])!=TRAJ[i] or int(r['height_seed'])!=HEIGHT[i]:
            raise RuntimeError(f'Checkpoint seed mismatch: {name}')
        if int(r['birds'])!=N or float(r['corridor_width_m'])!=W_FIXED:
            raise RuntimeError(f'Checkpoint corridor mismatch: {name}')
        if abs(float(r['Q_times_W'])-float(r['Q_lit'])*W_FIXED)>1e-12:
            raise RuntimeError(f'Checkpoint Q*W mismatch: {name}')
    return existing


def run_stage(comp,layouts,headings,prob,filename,caption):
    expected=[(i,h,lab) for i in range(BATCHES) for h in headings for lab in LAYOUTS]
    rows=restore_prefix(filename,expected)
    done=set((int(r['batch_index']),float(r['heading_deg_toward']),r['layout']) for r in rows)
    t0=time.monotonic()
    for i in range(BATCHES):
        for h in headings:
            if (i,h,LAYOUTS[0]) in done:
                continue
            if any((i,h,lab) in done for lab in LAYOUTS):
                raise RuntimeError('Partially completed population checkpoint')
            c=sens.change_components(comp,'direction',h)
            birds=fixed_width_population(comp,h,i)
            to_save=[]
            for lab in LAYOUTS:
                q=q_for(layouts[lab]['xy'],birds,c,prob)
                to_save.append({'batch_index':i,'heading_deg_toward':h,'layout':lab,
                    'response_probability':prob,'trajectory_seed':TRAJ[i],
                    'height_seed':HEIGHT[i], 'birds':N, 'corridor_width_m':W_FIXED,
                    'Q_lit':q,'Q_times_W':q*W_FIXED})
            rows.extend(to_save)
            write_csv(OUTPUT/filename, rows)
            done.update((i,h,lab) for lab in LAYOUTS)
            print(f'{caption}: batch {i+1}/{BATCHES}, heading {h:g}, saved {len(rows)}/{len(expected)}, elapsed {((time.monotonic()-t0)/3600):.2f} h',flush=True)
    return rows


def paired_t(rows, lhs, rhs, number_of_tests, name):
    lhs=np.asarray(lhs,float)
    rhs=np.asarray(rhs,float)
    if lhs.shape!=(BATCHES,) or rhs.shape!=(BATCHES,):
        raise RuntimeError('Paired arrays must be length 10')
    diffs=lhs-rhs
    mean=float(diffs.mean());sd=float(diffs.std(ddof=1));se=sd/math.sqrt(BATCHES)
    crit=float(student_t.ppf(1-.05/(2*number_of_tests),BATCHES-1))
    width=float(crit*se)
    p=(float(2*student_t.sf(abs(mean/se),BATCHES-1)) if se>0 else (1.0 if mean==0 else 0.0))
    return {'comparison':name,'difference_mean':mean,
       'difference_sd_across_10_paired_batches':sd,'ci_simultaneous95_lower':mean-width,
       'ci_simultaneous95_upper':mean+width,'raw_p':p,'bonferroni_tests_in_family':number_of_tests,
       'ci_includes_zero':bool(mean-width<=0<=mean+width)}


def input_series(rows,lab,h,key='Q_lit'):
    matching={(int(r['batch_index'])):float(r[key]) for r in rows if r['layout']==lab and float(r['heading_deg_toward'])==h}
    if len(matching)!=BATCHES:
        raise RuntimeError(f'Missing batch series for {lab} {h} {key}')
    return np.asarray([matching[i] for i in range(BATCHES)])


def baseline_gate(rows,old):
    """Four pre-specified 225-degree consistency contrasts, one Bonferroni family.

    We compare W * Q across the new and original independent evaluation
    batches, matched by their batch indices / RNG seed identifiers. The two
    samples differ in size and are not identical bird-for-bird. Differences
    are not required to be exactly zero; the Bonferroni simultaneous paired
    Student-t 95% CIs must each include zero.
    """
    checked=[]
    for lab in LAYOUTS:
        v=input_series(rows,lab,225.0,'Q_times_W')
        oldq=np.asarray([old[(i,lab,.6)] for i in range(BATCHES)])*W_BASE
        stat=paired_t(rows,v,oldq,4,f'newWQ_225_minus_originalWQ_225:{lab}')
        oldmean=float(oldq.mean())
        if not np.isfinite(oldmean) or oldmean <= 0:
            raise RuntimeError(f'Baseline Q*W denominator missing or nonpositive for {lab}')
        stat.update({
            'layout':lab,
            'mean_new_Q':float(np.mean(v/W_FIXED)),
            'mean_new_Q_times_W':float(v.mean()),
            'mean_old_Q_times_W':oldmean,
            'relative_WQ_change_pct':float(100*stat['difference_mean']/oldmean),
            'ci_lower_as_pct_of_old_mean':float(100*stat['ci_simultaneous95_lower']/oldmean),
            'ci_upper_as_pct_of_old_mean':float(100*stat['ci_simultaneous95_upper']/oldmean),
            'statistical_method':'paired Student-t, two-sided Bonferroni 95% simultaneous CI, 4 comparisons, df=9',
        })
        checked.append(stat)
    write_csv(OUTPUT/'baseline_225_paired_width_scaled_checks.csv',checked)
    print('GATE 1: four-comparison Bonferroni 95% simultaneous paired intervals (W*Q)',flush=True)
    for c in checked:
        verdict='PASS' if c['ci_includes_zero'] else 'FAIL'
        print(f"  {c['layout']}: new WQ={c['mean_new_Q_times_W']:.10g}; "
              f"old WQ={c['mean_old_Q_times_W']:.10g}; "
              f"relative change={c['relative_WQ_change_pct']:+.5f}%; "
              f"paired diff={c['difference_mean']:+.10g}; "
              f"Bonf 95% CI=[{c['ci_simultaneous95_lower']:+.10g}, "
              f"{c['ci_simultaneous95_upper']:+.10g}]; {verdict}",flush=True)
    failing=[c for c in checked if not c['ci_includes_zero']]
    if failing:
        failure={
            'status':'FAILED_BASELINE_WIDTH_SCALED',
            'gate':'225_degree_four_layout_Bonferroni_family',
            'bonferroni_comparisons':4,
            'all_comparisons':checked,
            'failed_comparisons':failing,
            'do_not_retry_automatically':True,
            'explanation':'Stop before p=0 controls and main directional comparisons. Bonferroni simultaneous 95% paired W*Q intervals reject zero for at least one layout. Review ALL four absolute and relative differences; do not automatically rerun.'
        }
        write_json(OUTPUT/'failure.json',failure)
        raise RuntimeError('GATE 1 FAILED. Four comparisons and relative WQ differences saved in failure.json and baseline_225_paired_width_scaled_checks.csv; do not retry.')
    write_json(OUTPUT/'gate_225_passed.json',{
        'status':'PASSED',
        'comparisons':4,
        'all_comparisons':checked,
        'method':'Four Bonferroni simultaneous 95% Student paired-t CIs across 10 batch-matched estimates of WQ; all include zero',
        'note':'Compatible with baseline within sampling uncertainty; not a proof of equality.'
    })
    print('GATE 225 PASSED: 4/4 Bonferroni-adjusted width-scaled paired intervals contain zero.',flush=True)
    return checked


def p0_gate(rows):
    comparisons=[]
    for ang in P0_HEADINGS:
        for j,a in enumerate(LAYOUTS):
            for b in LAYOUTS[j+1:]:
                comparisons.append(paired_t(rows,input_series(rows,a,ang),input_series(rows,b,ang),16,f'{ang:g}°:{a}-{b}'))
    for lab in LAYOUTS:
        comparisons.append(paired_t(rows,input_series(rows,lab,45.),input_series(rows,lab,90.),16,f'{lab}:45°-90°'))
    if len(comparisons)!=16:raise RuntimeError('Unexpected p=0 comparison count')
    # Overall Holm family of 16 tests for supporting information.
    ordered=sorted(range(16),key=lambda j:comparisons[j]['raw_p']);adjusted=0.0
    for k,j in enumerate(ordered):
        adjusted=max(adjusted,comparisons[j]['raw_p']*(16-k))
        comparisons[j]['holm_adjusted_p']=min(1.,adjusted)
    write_csv(OUTPUT/'p0_45_90_equality_checks.csv',comparisons)
    if any(not c['ci_includes_zero'] for c in comparisons):
        write_json(OUTPUT/'failure.json',{'status':'FAILED_P0_EQUALITY',
           'failed_comparisons':[r for r in comparisons if not r['ci_includes_zero']],
           'all_comparisons':comparisons, 'do_not_retry_automatically':True,
           'explanation':'A statistically detectable no-response difference was observed. Inspect before interpreting the direction sensitivity; failure is not necessarily a material difference. Do not automatically retry.'})
        raise RuntimeError('P=0 control failed Bonferroni simultaneous 95% consistency test. Inspect failure.json')
    write_json(OUTPUT/'gate_p0_passed.json',{'status':'PASSED','comparisons':16,
      'method':'All sixteen within/between direction no-response paired Bonferroni simultaneous 95% CIs include zero. Consistency check, not proof of exact equality.'})
    print('GATE p=0 PASSED: 16/16 simultaneous paired intervals contain zero.',flush=True)
    return comparisons


def final_stats(rows,layouts):
    means=[];paired=[];rank=[]
    for h in HEADINGS:
        v={lab:input_series(rows,lab,h) for lab in LAYOUTS}
        for lab in LAYOUTS:
            x=v[lab];mean=float(x.mean());sd=float(x.std(ddof=1));sem=sd/math.sqrt(BATCHES)
            c=float(student_t.ppf(.975,9))
            means.append({'heading_deg_toward':h,'layout':lab,
                'mean_Q_lit':mean,'mean_Q_times_W':mean*W_FIXED,
                'sd_Q_per_batch':sd,'N_birds_per_batch':N,'N_batches':BATCHES,
                'ci95_Q_lower':mean-c*sem,'ci95_Q_upper':mean+c*sem,
                'AEP_72sector_GWh_FIXED':layouts[lab]['aep_gwh']})
        local=[]
        for a,b in PAIRS:
            s=paired_t(rows,v[a],v[b],3,f'{a}_minus_{b}')
            s.update({'heading_deg_toward':h,'layout_A':a,'layout_B':b,
                'QW_difference_mean':s['difference_mean']*W_FIXED,
                'QW_bonferroni95_lower':s['ci_simultaneous95_lower']*W_FIXED,
                'QW_bonferroni95_upper':s['ci_simultaneous95_upper']*W_FIXED,
                'difference_as_percent_highest_AEP_risk':100*s['difference_mean']/float(v['highest_aep'].mean()),
                'reversal':s['difference_mean']<0,
                'bonferroni_supported_reversal':s['ci_simultaneous95_upper']<0})
            local.append(s)
        inds=sorted(range(3),key=lambda k:local[k]['raw_p']);acc=0.0
        for k,j in enumerate(inds):
            acc=max(acc,(3-k)*local[j]['raw_p']);local[j]['holm_adjusted_p']=min(1.,acc)
        paired.extend(local)
        ordering=sorted(LAYOUTS[:3],key=lambda lab:v[lab].mean(),reverse=True)
        rank.append({'heading_deg_toward':h,'rank_from_highest_to_lowest_Q':' > '.join(ordering),
           'baseline_rank_preserved':ordering==list(LAYOUTS[:3]),
           'any_mean_ranking_reversal':any(s['reversal'] for s in local),
           'any_supported_reversal':any(s['bonferroni_supported_reversal'] for s in local)})
    return means,paired,rank


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-inputs',action='store_true')
    group.add_argument('--run',action='store_true')
    args=p.parse_args()
    comp,layouts,old,margins=check_inputs()
    if args.check_inputs:return
    setup_outputs(margins)
    # Same direction always evaluated first, full 10 batches, to detect
    # corridor-definition inconsistencies before the main expensive scan.
    baseline=run_stage(comp,layouts,(225.,),.6,'baseline_225_batches.csv','BASELINE CONTROL')
    baseline_gate(baseline,old)
    zero=run_stage(comp,layouts,P0_HEADINGS,.0,'p0_45_90_batches.csv','NO RESPONSE CONTROL')
    p0_gate(zero)
    angles=tuple(a for a in HEADINGS if a!=225.)
    direction=run_stage(comp,layouts,angles,.6,'remaining_seven_direction_batches.csv','DIRECTION')
    # The eight-direction table keeps the baseline entries, unmodified.
    allrows=baseline+direction
    if len(allrows)!=BATCHES*8*4:raise RuntimeError('Not all directional evaluations saved')
    write_csv(OUTPUT/'all_8_directions_batches.csv',allrows)
    means,paired,rank=final_stats(allrows,layouts)
    write_csv(OUTPUT/'layout_means_8_directions.csv',means)
    write_csv(OUTPUT/'three_paired_comparisons_bonferroni_holm.csv',paired)
    write_csv(OUTPUT/'ranking_reversals_by_direction.csv',rank)
    report={'status':'completed','all_32_rotor_clearances_positive':True,
        'baseline_WQ_guard':'PASSED','p0_directional_no_response_guard':'PASSED',
        'fixed_width_m':W_FIXED,'N_per_batch':N,'batches':BATCHES,
        'headings':HEADINGS,'rankings_with_reversals':[x for x in rank if x['any_mean_ranking_reversal']],
        'mean_signed_per_entrant_strikes_and_W_scaled_exported':True,
        'baseline_225_reference_is_previous_50000_bird_independent_evaluation_sample':True,
        'AEP_recomputed':False}
    write_json(OUTPUT/'report.json',report)
    print('COMPLETE: 8 directions, 4 layouts, ten independent batches; all gates passed. No optimisation or AEP rerun.',flush=True)


if __name__=='__main__': main()
