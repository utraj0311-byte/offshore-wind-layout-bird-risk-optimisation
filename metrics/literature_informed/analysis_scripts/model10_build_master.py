#!/usr/bin/env python3
"""Create a reproducible, non-destructive master archive of FINAL Model 10 v3 runs.

Usage: python model10_build_master.py --check
       python model10_build_master.py --build
The script never moves/deletes/overwrites scientific source results.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, shutil, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SEEDS = (730101, 730102, 730103)
SOURCE_DIRS = {
    '01_optimisation/seed_730101': 'hpc_full_run_v3_72sectors',
    '01_optimisation/seed_730102': 'hpc_full_run_v3_72sectors_seed730102',
    '01_optimisation/seed_730103': 'hpc_full_run_v3_72sectors_seed730103',
    '03_independent_evaluation/final_four_guarded': 'hpc_v3_final_knee_confirmation_guarded',
    '04_sensitivities/probability': 'hpc_model10_frozen_final_probability',
    '04_sensitivities/radius': 'hpc_model10_frozen_final_radius',
    '04_sensitivities/hub_height': 'hpc_model10_frozen_final_hub',
    '04_sensitivities/bird_speed': 'hpc_model10_frozen_final_speed',
    '04_sensitivities/direction_fixed_width': 'hpc_model10_frozen_final_direction_fixed_width',
    '05_uncertainty/bird_level_bootstrap': 'hpc_v3_final_optimisation_sample_bird_bootstrap',
    '06_numerical_resolution/aep_360sector': 'hpc_model10_final_selected_360sector_aep',
}
OPTIONAL_DIRS = {
    '03_independent_evaluation/earlier_six_guarded': 'hpc_v3_paired_confirmation_p0',
}
ROOT_SOURCE_NAMES = [
    'model10_direction_fixed_width_8way.py', 'hpc_model10_direction_fixed_width_8way.pbs',
    'model10_confirm_final_selected_with_guards.py', 'hpc_model10_confirm_final_selected_with_guards.pbs',
    'model10_final_bird_level_bootstrap.py', 'hpc_model10_final_bird_level_bootstrap.pbs',
    'model10_final_frozen_sensitivity.py', 'model10_final_three_1degree_aep.py',
    'hpc_model10_final_three_1degree_aep.pbs',
    'hpc_model10_frozen_final_probability.pbs', 'hpc_model10_frozen_final_radius.pbs',
    'hpc_model10_frozen_final_hub.pbs', 'hpc_model10_frozen_final_speed.pbs',
]
FROZEN_EXPECTED = {'highest_aep': (730101,455), 'knee': (730103,418), 'lowest_risk': (730103,220)}


def err(x):
    raise RuntimeError(x)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def audit(root):
    missing=[]; problems=[]; settings={}; originals={}
    for destination, source in SOURCE_DIRS.items():
        path=root/source
        if not path.is_dir(): missing.append(source);continue
        if destination.startswith('01_'):
            sett_path=path/'optimisation/run_settings.json'
            all_path=path/'optimisation/all_evaluated_candidates.csv'
            for mandatory in (sett_path,all_path,path/'optimisation/pareto_layouts_xy_m.npy'):
                if not mandatory.is_file(): problems.append('Missing '+str(mandatory))
            if sett_path.is_file():
                s=json.loads(sett_path.read_text());sd=int(s['optimiser_seed']);settings[sd]=s
                if s.get('status')!='completed' or s.get('number_of_direction_sectors')!=72 or s.get('intended_evaluations')!=480:
                    problems.append('Unexpected run settings '+source)
                if s['training_batches']!=5 or s['training_birds_per_batch']!=10000:
                    problems.append('Unexpected optimisation-sample batch size '+source)
        else:
            report=path/'report.json'
            if not report.is_file():problems.append('Missing report '+str(report))
            else:
                r=json.loads(report.read_text()); status=r.get('status')
                if status not in ('PASSED','PASS','completed'):
                    problems.append('Result not completed: '+source+' ('+str(status)+')')
                if source.endswith('direction_fixed_width'):
                    if not (r.get('all_32_rotor_clearances_positive') is True and r.get('baseline_WQ_guard')=='PASSED' and r.get('p0_directional_no_response_guard')=='PASSED'):
                        problems.append('Directional scientific gate incomplete')
    if set(settings)!=set(SEEDS):problems.append('Three optimisation seed settings not available')
    else:
        a=settings[730101]
        for seed in SEEDS:
            b=settings[seed]
            if b.get('optimiser_seed')!=seed:problems.append('Optimiser seed inconsistent: '+str(seed))
            for key in ['training_trajectory_seeds','training_height_seeds','initial_population_seed','reference_aep_gwh','reference_training_risk_per_entrant','bird_direction_deg_toward','FHD','band','behaviour','turbines','rotor_diameter_m','minimum_spacing_m']:
                if b.get(key)!=a.get(key):problems.append('Optimisation samples or parameters differ: '+key+', seed '+str(seed))
            if seed!=SEEDS[0]:
                aa={k:v for k,v in a.get('source_hashes',{}).items() if k!='runner'}
                bb={k:v for k,v in b.get('source_hashes',{}).items() if k!='runner'}
                if aa!=bb:problems.append('Frozen scientific source hashes differ, seed '+str(seed))
    if (root/'hpc_v3_final_knee_confirmation_guarded/report.json').exists():
        r=json.loads((root/'hpc_v3_final_knee_confirmation_guarded/report.json').read_text())
        cr=root/'hpc_v3_final_knee_confirmation_guarded/consistency_report.json'
        if cr.exists():
            guard=json.loads(cr.read_text())
            if guard.get('status') not in ('PASSED','PASS'):
                problems.append('Guarded confirmation numerical guard did not pass')
    return missing,problems,settings


def combined_front(master):
    pieces=[]
    for seed in SEEDS:
        f=master/f'01_optimisation/seed_{seed}/optimisation/all_evaluated_candidates.csv'
        df=pd.read_csv(f)
        good=(df.max_spacing_constraint_m<=1e-7)&(df.max_boundary_violation_m<=1e-7)&(df.minimum_spacing_m>=968.96-1e-7)
        good &= np.isfinite(df.aep_gwh)&np.isfinite(df.risk_per_entrant)
        df=df.loc[good].copy();df.insert(0,'seed',seed)
        pieces.append(df)
    allfeasible=pd.concat(pieces,ignore_index=True)
    xy_cols=[c for c in allfeasible.columns if c.startswith('x_') or c.startswith('y_')]
    aeps=allfeasible.aep_gwh.to_numpy(dtype=float);risks=allfeasible.risk_per_entrant.to_numpy(dtype=float)
    dominated=np.zeros(len(allfeasible),dtype=bool)
    for i in range(len(allfeasible)):
        dominated[i]=np.any((aeps>=aeps[i])&(risks<=risks[i])&((aeps>aeps[i])|(risks<risks[i])))
    nd=allfeasible.loc[~dominated,['seed','evaluation_index','aep_gwh','risk_per_entrant',*xy_cols]].copy()
    nd=nd.sort_values('aep_gwh',ascending=False).reset_index(drop=True)
    ehi=nd.iloc[nd.aep_gwh.argmax()];rlo=nd.iloc[nd.risk_per_entrant.argmin()]
    diff_a=float(ehi.aep_gwh-rlo.aep_gwh);diff_q=float(ehi.risk_per_entrant-rlo.risk_per_entrant)
    if diff_a<=0 or diff_q<=0:err('Endpoints have nonpositive objective ranges')
    nd['a']=(nd.aep_gwh-rlo.aep_gwh)/diff_a
    nd['r']=(nd.risk_per_entrant-rlo.risk_per_entrant)/diff_q
    nd['d']=(nd.a-nd.r)/math.sqrt(2)
    knee=nd.iloc[nd.d.argmax()]
    ehi=nd.iloc[nd.aep_gwh.argmax()]
    rlo=nd.iloc[nd.risk_per_entrant.argmin()]
    selected={'highest_aep':ehi,'knee':knee,'lowest_risk':rlo}
    for name,row in selected.items():
        got=(int(row.seed),int(row.evaluation_index))
        if got!=FROZEN_EXPECTED[name]:err(f'FAILED selection check {name}: {got} != {FROZEN_EXPECTED[name]}')
    if len(allfeasible)!=513 or len(nd)!=24:err(f'FAILED combined-front count: feasible={len(allfeasible)} (expected 513), front={len(nd)} (expected 24)')
    dest=master/'02_combined_pareto';dest.mkdir(exist_ok=True)
    nd=nd.rename(columns={'risk_per_entrant':'optimisation_sample_Q_lit'})
    nd.to_csv(dest/'combined_non_dominated_24.csv',index=False,float_format='%.16g')
    allfeasible[['seed','evaluation_index','aep_gwh','risk_per_entrant']].to_csv(dest/'all_feasible_candidate_scores_513.csv',index=False,float_format='%.16g')
    table=[]
    for label,row in selected.items():
        table.append({'selection':label,'seed':int(row.seed),'evaluation_index':int(row.evaluation_index),
                      'aep_gwh':float(row.aep_gwh),'optimisation_sample_Q_lit':float(row.risk_per_entrant),'d':float(row.d),
                      'aep_delta_pct_of_highest':100*(float(row.aep_gwh)-float(ehi.aep_gwh))/float(ehi.aep_gwh),
                      'risk_delta_pct_of_highest':100*(float(row.risk_per_entrant)-float(ehi.risk_per_entrant))/float(ehi.risk_per_entrant)})
        xy=np.array([[row[f'x_{i:02d}_m'],row[f'y_{i:02d}_m']] for i in range(12)],dtype=float)
        np.save(dest/f'{label}_coordinates_xy_m.npy',xy)
        pd.DataFrame({'turbine_id':np.arange(12),'x_m':xy[:,0],'y_m':xy[:,1]}).to_csv(dest/f'{label}_coordinates_xy_m.csv',index=False)
    pd.DataFrame(table).to_csv(dest/'final_selected_three.csv',index=False,float_format='%.16g')
    (dest/'KNEE_SELECTION_RULE.txt').write_text('Final three-seed 72-sector non-dominated set. AEP maximised, optimisation-sample Q_lit minimised.\n'
      'a=(AEP-AEP_lowrisk)/(AEP_highAEP-AEP_lowrisk); r=(Q-Q_lowrisk)/(Q_highAEP-Q_lowrisk); d=(a-r)/sqrt(2). Max positive d selects knee.\n'
      'Reference is not part of combined Pareto front. 730101/465 is intermediate (diagnostic), not knee.\n')
    return {'n_total_evaluations':1440,'n_feasible':len(allfeasible),'n_non_dominated':len(nd),'selected':table}


def make_manifest(master, root, srcmap, stats):
    rows=[]
    for file in sorted(master.rglob('*')):
        if not file.is_file():continue
        if file.name in ('archive_manifest.csv','archive_audit.json'):continue
        rows.append({'relative_path':str(file.relative_to(master)),'size_bytes':file.stat().st_size,'sha256':sha256(file),'original_source':srcmap.get(str(file.relative_to(master)),'derived or provided script')})
    with (master/'archive_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['relative_path','size_bytes','sha256','original_source']);w.writeheader();w.writerows(rows)
    (master/'archive_audit.json').write_text(json.dumps({'status':'PASSED','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_hpc_root':str(root),'n_files_in_manifest':len(rows),'counts_and_selection':stats,
        'policy':'copy only; never move/delete or edit original results',
        'provenance':'manifest lists original source and SHA-256 of archived bytes'},indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    grp=ap.add_mutually_exclusive_group(required=True);grp.add_argument('--check',action='store_true');grp.add_argument('--build',action='store_true')
    ap.add_argument('--root',type=Path,default=Path.home()/'model10',help='Original HPC Model 10 directory')
    ap.add_argument('--dest',type=Path,default=None,help='New archive directory; defaults to ROOT/MODEL10_FINAL_72SECTOR_MASTER')
    args=ap.parse_args(); root=args.root.expanduser().resolve();dest=(args.dest or root/'MODEL10_FINAL_72SECTOR_MASTER').expanduser().absolute()
    missing,problems,_=audit(root)
    print('Original source root:',root)
    for destrel,src in SOURCE_DIRS.items(): print(('FOUND' if src not in missing else 'MISSING')+'  '+src+'  -> '+destrel)
    if missing or problems:
        print('Cannot build. Missing folders:',missing,'\nProblems:',problems)
        return 2
    print('SOURCE AUDIT PASS: all 11 required directories; 3 matched optimisation-sample settings.')
    if args.check:return 0
    if dest.exists():err('Destination ALREADY EXISTS (will not merge/overwrite): '+str(dest))
    if root==dest or root in dest.parents and dest.name=='model10':err('Refusing unsafe destination')
    final_dest=dest
    dest=final_dest.with_name(final_dest.name+'.BUILDING_INCOMPLETE')
    if dest.exists():err('A previous incomplete staging directory exists. Inspect it before proceeding: '+str(dest))
    dest.mkdir(parents=True)
    provenance={}
    for rel,src in {**SOURCE_DIRS,**OPTIONAL_DIRS}.items():
        source=root/src
        if not source.is_dir():print('OPTIONAL MISSING:',src);continue
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        print('Copying:',source.name,'->',rel,flush=True)
        shutil.copytree(source,target,symlinks=False)
        for f in target.rglob('*'):
            if f.is_file():provenance[str(f.relative_to(dest))]=str((source/f.relative_to(target)).resolve())
    srcdest=dest/'07_scientific_code_and_job_scripts';srcdest.mkdir(exist_ok=True)
    for name in ROOT_SOURCE_NAMES:
        source=root/name
        if source.is_file():
            shutil.copy2(source,srcdest/name);provenance[str((srcdest/name).relative_to(dest))]=str(source)
        else:print('Optional root script not found:',name)
    src=dest/'01_optimisation/seed_730101/optimisation/source'
    if src.is_dir():shutil.copytree(src,srcdest/'frozen_source_snapshot_seed730101')
    stats=combined_front(dest)
    for f in ['model10_plot_master.py','model10_build_master.py','hpc_model10_master_plots.pbs','README_HPC_MASTER.md']:
        this=Path(__file__).resolve().parent/f
        if this.is_file():shutil.copy2(this,srcdest/f)
    (dest/'08_figures').mkdir(exist_ok=True)
    (dest/'09_derived_tables').mkdir(exist_ok=True)
    (dest/'README_START_HERE.txt').write_text(
        'MODEL 10 FINAL 72-SECTOR MASTER ARCHIVE\n\n'
        'This folder is a COPY of completed HPC source results. Original files were NOT moved, renamed, overwritten or removed.\n'
        '01_optimisation/       Original complete v3 runs for 730101, 730102, 730103, with source snapshots.\n'
        '02_combined_pareto/     Reconstructed 24-layout final front, 513 feasible candidate scores, selected turbine XY.\n'
        '03_independent_evaluation/  Final four guarded, optionally original six-layout confirmation.\n'
        '04_sensitivities/      Corrected probability, radius, hub, speed and fixed-width 8-direction results.\n'
        '05_uncertainty/        Optimisation-sample individual-bird bootstrap, 10000 paired resamples.\n'
        '06_numerical_resolution/  Saved 360-sector AEP verification.\n'
        '07_scientific_code_and_job_scripts/  Frozen source snapshot, runner/PBS copies, this plotter.\n'
        '08_figures/           Generated PDF, SVG and PNG plots (create on HPC via plot script).\n'
        '09_derived_tables/    Tables generated from archived data by plotting script.\n'
        'archive_manifest.csv  Per-file SHA-256 and provenance.\n'
        'archive_audit.json    Verified completion, sample comparability and layout selection.\n\n'
        'IMPORTANT: pre-correction 12-sector experiments are NOT part of this final corrected-results archive. Keep separate.\n'
        'Never use the internal labels from old per-seed runner (e.g. training knee) for final selection.\n'
        'Final selections: highest-AEP 730101/455; knee 730103/418; lowest-risk 730103/220.\n'
        'Optimisation sample: 5x10000 birds. Independent evaluation sample: 10x50000 birds.\n'
        'Directional sensitivity: fixed W=9696.481 m, 10x57137 birds; bird heading, rotor and Band wind rotate together, AEP fixed.\n'
        'Plot: python 07_scientific_code_and_job_scripts/model10_plot_master.py --master .\n')
    make_manifest(dest,root,provenance,stats)
    dest.rename(final_dest)
    print('MASTER ARCHIVE BUILT:',final_dest)
    print('Selected layouts:',[(r['selection'],r['seed'],r['evaluation_index']) for r in stats['selected']]);print('No original source data modified.')
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print('ERROR:',str(e),file=sys.stderr);sys.exit(1)
