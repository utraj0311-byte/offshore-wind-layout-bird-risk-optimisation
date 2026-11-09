#!/usr/bin/env python3
"""AEP-only 360-sector verification of the three FINAL 72-sector Model 10 layouts.

No NSGA-II, no bird calculations, no original results overwritten.
Usage: python model10_final_three_1degree_aep.py --check-inputs
       python -u model10_final_three_1degree_aep.py --run
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import model10_fhd_robust_seeded_50k_72sectors as base
ROOT=Path('/home/uraj/model10')
OUTPUT=ROOT/'hpc_model10_final_selected_360sector_aep'
SELECT={'highest_aep':(730101,455),'knee':(730103,418),'lowest_risk':(730103,220)}

def directory(seed):
    stem='hpc_full_run_v3_72sectors'+('_seed'+str(seed) if seed!=730101 else '')
    return ROOT/stem/'optimisation'

def get_cases():
    settings={s:json.loads((directory(s)/'run_settings.json').read_text()) for s in [730101,730102,730103]}
    for s,obj in settings.items():
        assert obj['status']=='completed' and obj['number_of_direction_sectors']==72 and obj['optimiser_seed']==s
    if base.source_hashes()!=settings[730101]['source_hashes']:
        raise RuntimeError('Frozen runner/source hash mismatch')
    cases={}
    for label,(seed,eid) in SELECT.items():
        with (directory(seed)/'pareto_training.csv').open(newline='') as f: rr=list(csv.DictReader(f))
        with (directory(seed)/'pareto_layouts_xy_m.npy').open('rb') as f: layouts=np.load(f,allow_pickle=False)
        if layouts.shape!=(len(rr),12,2):raise RuntimeError('Frozen coordinates shape mismatch')
        ids=[i for i,r in enumerate(rr) if int(r['evaluation_index'])==eid]
        if len(ids)!=1:raise RuntimeError(f'Expected one {label} seed={seed}, eval={eid}')
        idx=ids[0]; r=rr[idx];coords=layouts[idx]
        csvcoords=np.array([[float(r[f'x_{i:02d}_m']),float(r[f'y_{i:02d}_m'])] for i in range(12)])
        if not np.array_equal(coords,csvcoords):raise RuntimeError(f'Frozen coordinate mismatch {label}')
        base.check_layout(coords,968.96)
        cases[label]={'xy':coords,'saved_aep5':float(r['aep_gwh']),'seed':seed,'evaluation_index':eid}
    return cases

def main():
    ap=argparse.ArgumentParser();z=ap.add_mutually_exclusive_group(required=True)
    z.add_argument('--check-inputs',action='store_true');z.add_argument('--run',action='store_true')
    args=ap.parse_args();cases=get_cases()
    print('INPUT CHECK PASSED: exact final selections, frozen runner hashes and arrays, 72-sector saved AEP',flush=True)
    for name,c in cases.items():print(f"{name}: seed {c['seed']}, eval {c['evaluation_index']}, 5deg={c['saved_aep5']:.9f}",flush=True)
    if args.check_inputs:return
    if OUTPUT.exists():raise FileExistsError(f'Output already exists; do not overwrite: {OUTPUT}')
    def make(n):
        sc=base.Scenario();sc.wind.n_sectors=n
        m=base.FLORISAEPModel(sc,base.CONFIG)
        ss=m.settings_summary()
        if int(ss['number_of_direction_sectors'])!=n or int(ss['number_of_speed_bins'])!=7:raise RuntimeError(f'Bad wind configuration: {ss}')
        ceiling=m.ideal_aep(12)
        if abs(ceiling-854.4956585617186)>1e-6:raise RuntimeError(f'Wake-free changed at sectors={n}: {ceiling}')
        return m
    m5=make(72);m1=make(360)
    rows=[]
    for name,c in cases.items():
        a5=float(m5.evaluate(c['xy']))
        if abs(a5-c['saved_aep5'])>0.001:raise RuntimeError(f'{name} 5deg AEP failed reproduction')
        a1=float(m1.evaluate(c['xy']))
        rec={'layout':name,'seed':c['seed'],'eval':c['evaluation_index'],'AEP_5_saved_GWh':c['saved_aep5'],
          'AEP_5_recomputed_GWh':a5,'AEP_1degree_GWh':a1,'difference_1minus5_GWh':a1-a5,
          'difference_1minus5_percent_of_5degree':100*(a1-a5)/a5}
        rows.append(rec);print(f'{name}: 5deg={a5:.9f} | 1deg={a1:.9f} | difference={a1-a5:+.9f} GWh',flush=True)
    r={x['layout']:x for x in rows}
    gap5=r['highest_aep']['AEP_5_saved_GWh']-r['knee']['AEP_5_saved_GWh']
    gap1=r['highest_aep']['AEP_1degree_GWh']-r['knee']['AEP_1degree_GWh']
    result={'status':'PASS','source':'frozen Model 10 72-sector FLORIS runner; AEP-only',
            'wind_sectors_compared':[72,360], 'AEP5_gap_455_minus_418_GWh':gap5,
            'AEP5_gap_percent_of_455':100*gap5/r['highest_aep']['AEP_5_saved_GWh'],
            'AEP1_gap_455_minus_418_GWh':gap1,
            'AEP1_gap_percent_of_455':100*gap1/r['highest_aep']['AEP_1degree_GWh'],
            'gap_sign_preserved':gap1>0,'gap_change_percentage_points':100*gap1/r['highest_aep']['AEP_1degree_GWh']-100*gap5/r['highest_aep']['AEP_5_saved_GWh'],
            'rows':rows}
    OUTPUT.mkdir(parents=False,exist_ok=False)
    with (OUTPUT/'three_layout_aep_1deg.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUTPUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE; gap at 1degree:',f'{gap1:.9f} GWh, {result["AEP1_gap_percent_of_455"]:.6f}%',flush=True)

if __name__=='__main__':main()
