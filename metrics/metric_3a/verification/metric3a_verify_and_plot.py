"""Metric 3a audit, archive and figures. Does not optimise or calculate AEP.

Run in a folder containing source_code/ and input_archives/:
 python metric3a_verify_and_plot.py --output ../metric3a_local_recheck

All original source inputs are read-only. The output directory must not exist.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import math
import platform
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CODE_NAMES = (
    'config.py', 'directional_absorbing_risk.py', 'risk_only_problem.py',
    'run_directional_absorbing_risk_only.py', 'run_directional_absorbing_seeded.py',
    'run_validation.py',
)
ARCHIVE_NAMES = ('random_start_original.zip', 'aligned_start_original.zip')
EXPECTED = {'bird_travel_deg':225.0, 'downstream_scale_D':8.0,
            'cross_scale_D':1.0,'maximum_shielding':0.8,
            'seed':42,'population':100,'generations':120,
            'rotor_diameter_m':240.0, 'width_m':6000.0,'height_m':6000.0,
            'n_turbines':12,'min_spacing_D':4.0}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):
            h.update(buf)
    return h.hexdigest()

def check_code(src):
    # Protect against accidentally evaluating energy code or changing the objective.
    code={name:(src/name).read_text(encoding='utf-8') for name in CODE_NAMES}
    risk=code['directional_absorbing_risk.py']
    prob=code['risk_only_problem.py']
    random=code['run_directional_absorbing_risk_only.py']
    aligned=code['run_directional_absorbing_seeded.py']
    banned=('AEPModel','FLORIS','n_sectors','sector_frequencies','aep_model','wake.')
    for fname,body in [('directional_absorbing_risk.py',risk),('risk_only_problem.py',prob),
                       ('run_directional_absorbing_risk_only.py',random),('run_directional_absorbing_seeded.py',aligned)]:
        for term in banned:
            if term in body:
                raise AssertionError(f'CHECK 1 FAIL: {fname} contains {term}. Review manually; stopped.')
    if 'self.scenario.wind' in risk or 'scenario.wind' in random or 'scenario.wind' in aligned:
        raise AssertionError('CHECK 1 FAIL: wind resource access found')
    if 'n_obj=1' not in prob or 'self.risk_model.collisions(xy)' not in prob:
        raise AssertionError('CHECK 1 FAIL: single risk objective not found')
    if 'GA(' not in random or 'GA(' not in aligned or 'RiskOnlyLayoutProblem(' not in random or 'RiskOnlyLayoutProblem(' not in aligned:
        raise AssertionError('CHECK 1 FAIL: expected single-objective GA not found')
    if 'bird_travel_deg: float = 225.0' not in risk or 'float(bird_travel_deg)' not in risk:
        raise AssertionError('CHECK 1 FAIL: fixed bird-heading input missing')
    if 'self.bird_travel_deg' not in risk or 'np.cos(angle_rad)' not in risk or 'np.sin(angle_rad)' not in risk:
        raise AssertionError('CHECK 1 FAIL: heading projection changed')
    # Confirm the runner's arguments, rather than relying only on model defaults.
    for fn, body in [('random', random), ('aligned',aligned)]:
        t=ast.parse(body)
        found={}
        for x in ast.walk(t):
            if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='add_argument' and x.args:
                try: arg=ast.literal_eval(x.args[0])
                except (ValueError,TypeError): continue
                for kw in x.keywords:
                    if kw.arg=='default':
                        try: found[arg]=ast.literal_eval(kw.value)
                        except (ValueError,TypeError): pass
        for key,value in [('--bird_deg',225.),('--downstream_scale_D',8.),('--cross_scale_D',1.),('--shielding',0.8),('--seed',42),('--pop',100),('--gen',120)]:
            if found.get(key) != value:raise AssertionError(f'CHECK 1 FAIL: {fn} runner arg {key} is {found.get(key)}')
    # run_validation may import a wake module for OTHER experiments. Its imported utilities are separate.
    sampling=code['run_validation.py']
    for term in ('def min_spacing(', 'def count_spacing_violations(', 'def count_boundary_violations(', 'class GridSampling('):
        if term not in sampling:raise AssertionError(f'CHECK 1 FAIL: missing sampling dependency {term}')
    return code

def load_local_module(name,source):
    spec=importlib.util.spec_from_file_location(name,source)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def read_member(zf, suffix):
    paths=[n for n in zf.namelist() if n.endswith('/'+suffix) or n==suffix]
    if len(paths)!=1:raise AssertionError(f'Archive member {suffix}: found {len(paths)}, expected 1')
    return zf.read(paths[0])

def csv_layout(buf):
    rows=list(csv.DictReader(io.StringIO(buf.decode('utf-8-sig'))))
    if len(rows)!=12:raise AssertionError(f'Expected 12 turbine rows, got {len(rows)}')
    xy=np.array([[float(r['x_m']),float(r['y_m'])] for r in rows],dtype=float)
    return xy

def reference_function(source):
    tree=ast.parse(source)
    node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_reference_layout']
    if len(node)!=1:raise AssertionError('Cannot identify original reference constructor')
    sub=ast.Module(body=node,type_ignores=[])
    scope={'np':np}
    exec(compile(ast.fix_missing_locations(sub),'<reference_from_original_runner>','exec'),scope)
    return scope['create_reference_layout']

def upstream_counts(xy,model):
    along,cross=model.bird_frame_coordinates(xy)
    ds=along[:,None]-along[None,:]
    dn=np.abs(cross[:,None]-cross[None,:])
    mask=(ds>0)&(ds<=model.downstream_scale_m)&(dn<=model.cross_scale_m)
    c=mask.sum(axis=1)
    return c,int(np.count_nonzero(c)), int(c.sum())

def write_csv(path,rows,fields=None):
    rows=list(rows)
    if fields is None:fields=list(rows[0].keys())
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def make_layout_rows(xy,model):
    along,cross=model.bird_frame_coordinates(xy)
    scores=model.exposure_per_turbine(xy)
    counts,n,t=upstream_counts(xy,model)
    return [dict(turbine_id=i,x_m=format(xy[i,0],'.17g'),y_m=format(xy[i,1],'.17g'),
                 along_flight_m=format(along[i],'.17g'),cross_flight_m=format(cross[i],'.17g'),
                 remaining_exposure=format(scores[i],'.17g'),
                 qualifying_upstream_count=int(counts[i])) for i in range(len(xy))]

def plot_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
       'axes.labelsize':11,'legend.fontsize':9.5,'xtick.labelsize':9,
       'ytick.labelsize':9,'figure.dpi':150,'savefig.dpi':250,
       'pdf.fonttype':42})

DISPLAY_ORDER=[('random_start','random start','#396b98','o'),
               ('aligned_start','aligned start','#b36d39','o'),
               ('aligned_reference','aligned reference','#36806a','s'),
               ('cross_flow_reference','cross-flow reference','#946a96','D')]

def plot_layout(ax,xy,score,label,color,marker,scen,model,shared_limits):
    # Plot projected square perimeter and allow padding outside the full domain.
    w=float(scen.site.width_m);h=float(scen.site.height_m)
    corner=np.array([[0,0],[w,0],[w,h],[0,h],[0,0]],dtype=float)
    sa,na=model.bird_frame_coordinates(corner)
    ax.plot(sa,na,ls='--',lw=0.85,c='#969ba4',alpha=0.55,zorder=1)
    along,cross=model.bird_frame_coordinates(xy)
    ax.scatter(along,cross,marker=marker,s=95,c=color,edgecolor='#282b31',linewidth=0.6,
               zorder=4,label=label+r' ($Q_{3a}$ = '+f'{score:.3f})')
    x0,x1,y0,y1=shared_limits
    # In bird coordinates, bird travel is strictly toward positive along-flight axis.
    ax.annotate('',xy=(x0+0.17*(x1-x0),y1-0.115*(y1-y0)),
                xytext=(x0+0.073*(x1-x0),y1-0.115*(y1-y0)),
                arrowprops=dict(arrowstyle='-|>',lw=1.6,color='#313a46',mutation_scale=17))
    ax.text(x0+0.073*(x1-x0),y1-0.085*(y1-y0),'bird flight',ha='left',va='bottom',fontsize=9,color='#313a46')
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
    ax.grid(alpha=0.17,linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('along-flight position (m)')
    ax.set_ylabel('cross-flight position (m)')
    ax.legend(loc='lower left',framealpha=0.95,edgecolor='#dadde2',borderpad=0.55)

def save_plot(fig,pathbase):
    fig.savefig(str(pathbase)+'.pdf',bbox_inches='tight',pad_inches=0.20)
    fig.savefig(str(pathbase)+'.png',bbox_inches='tight',pad_inches=0.20)
    plt.close(fig)

def make_plots(out,layouts,scores,scenario,model,convergence):
    plot_style()
    figdir=out/'figures';figdir.mkdir()
    corners=np.array([[0,0],[6000,0],[6000,6000],[0,6000]],float)
    sa,na=model.bird_frame_coordinates(corners)
    margin=300.0
    lims=(sa.min()-margin,sa.max()+margin,na.min()-margin,na.max()+margin)
    for key,label,col,marker in DISPLAY_ORDER:
        fig,ax=plt.subplots(figsize=(8,6.5))
        plot_layout(ax,layouts[key],scores[key],label,col,marker,scenario,model,lims)
        save_plot(fig,figdir/('metric3a_layout_'+key))
    fig,axes=plt.subplots(2,2,figsize=(14.5,12),constrained_layout=True)
    for ax,(key,label,col,marker) in zip(axes.flat,DISPLAY_ORDER):
        plot_layout(ax,layouts[key],scores[key],label,col,marker,scenario,model,lims)
    save_plot(fig,figdir/'metric3a_layouts_comparison')
    fig,ax=plt.subplots(figsize=(8,5))
    for key,label,col,_ in DISPLAY_ORDER[:2]:
        data=convergence[key]
        g=[int(v['generation']) for v in data]
        q=[float(v['best_risk']) for v in data]
        ax.plot(g,q,lw=1.8,c=col,label=label+r' ($Q_{3a}$ = '+f'{scores[key]:.3f})')
    ax.set_xlabel('generation');ax.set_ylabel(r'best $Q_{3a}$')
    ax.set_xlim(left=1);ax.grid(alpha=0.2);ax.legend(frameon=True)
    save_plot(fig,figdir/'metric3a_convergence_comparison')
    return sorted([p.name for p in figdir.iterdir()])

def main():
    parser=argparse.ArgumentParser(description='Verify saved Metric 3a; no optimisation or energy calculations')
    base=Path(__file__).resolve().parent
    parser.add_argument('--source-dir',type=Path,default=base/'source_code')
    parser.add_argument('--archive-dir',type=Path,default=base/'input_archives')
    parser.add_argument('--output',type=Path,default=base/'verification_output')
    args=parser.parse_args()
    source=args.source_dir.resolve();archive=args.archive_dir.resolve();out=args.output.resolve()
    print('Check 1: auditing code and heading...')
    code=check_code(source)
    cfg=load_local_module('metric3a_snapshot_config',source/'config.py')
    mod=load_local_module('metric3a_snapshot_model',source/'directional_absorbing_risk.py')
    scenario=cfg.Scenario()
    model=mod.DirectionalAbsorbingRiskModel(scenario=scenario)
    assert float(model.bird_travel_deg)==225.0 and float(model.downstream_scale_m)==1920.0 and float(model.cross_scale_m)==240.0
    for key,val in [('rotor_diameter_m',scenario.turbine.rotor_diameter_m),('width_m',scenario.site.width_m),('height_m',scenario.site.height_m),('n_turbines',scenario.site.n_turbines),('min_spacing_D',scenario.site.min_spacing_D)]:
        assert val==EXPECTED[key],f'CHECK 1 FAIL: {key} changed: {val}'
    print('PASS: no AEP/wake evaluation; bird heading fixed 225 degrees TOWARD')
    print('Check 2: recalculating four archived layouts...')
    with zipfile.ZipFile(archive/ARCHIVE_NAMES[0]) as zr,zipfile.ZipFile(archive/ARCHIVE_NAMES[1]) as za:
        layouts={
           'random_start':csv_layout(read_member(zr,'best_layout.csv')),
           'aligned_start':csv_layout(read_member(za,'best_layout.csv')),
           'aligned_reference':csv_layout(read_member(za,'reference_layout.csv'))}
        rnd=json.loads(read_member(zr,'summary.json'))
        alg=json.loads(read_member(za,'summary.json'))
        ref_constructor=reference_function(code['run_directional_absorbing_risk_only.py'])
        reference_from_code=ref_constructor(scenario,model,True)
        if not np.allclose(reference_from_code,layouts['aligned_reference'],rtol=0,atol=1e-9):
            raise AssertionError('CHECK 2 FAIL: saved reference differs from constructor')
        layouts['cross_flow_reference']=ref_constructor(scenario,model,False)
        vals={'random_start':rnd['best_total_risk'], 'aligned_start':alg['final_best_risk'],
              'aligned_reference':alg['reference_risk'],
              'cross_flow_reference':rnd['cross_flow_reference_risk']}
        convergence={'random_start':list(csv.DictReader(io.StringIO(read_member(zr,'convergence.csv').decode('utf-8-sig')))),
                     'aligned_start':list(csv.DictReader(io.StringIO(read_member(za,'convergence.csv').decode('utf-8-sig'))))}
    score={}
    for key,xy in layouts.items():
        score[key]=float(model.collisions(xy))
        delta=abs(score[key]-float(vals[key]))
        print(f'{key:22s} saved={vals[key]:.15f} recalculated={score[key]:.15f} |difference|={delta:.3g}')
        if delta>2e-12: raise AssertionError('CHECK 2 FAIL: '+key+' differs from original; stop')
        bounds=(xy[:,0]>=-1e-8).all() and (xy[:,0]<=6000+1e-8).all() and (xy[:,1]>=-1e-8).all() and (xy[:,1]<=6000+1e-8).all()
        mindist=min(np.linalg.norm(xy[i]-xy[j]) for i in range(12) for j in range(i+1,12))
        if not bounds or mindist<960-1e-6:raise AssertionError('CHECK 2 FAIL: geometry infeasible '+key)
    print('PASS: all four values match stored precision; all layouts feasible.')
    if out.exists():
        raise FileExistsError(f'Output path already exists: {out}. Choose a NEW path; never overwrite originals.')
    # Both verification checks have passed; the only write activity begins here.
    out.mkdir(parents=True)
    laydir=out/'layouts';laydir.mkdir()
    srcdir=out/'source_code';srcdir.mkdir()
    ardir=out/'input_archives';ardir.mkdir()
    for name in CODE_NAMES:shutil.copy2(source/name,srcdir/name)
    for name in ARCHIVE_NAMES:shutil.copy2(archive/name,ardir/name)
    counts_table=[];eval_table=[]
    for key,xy in layouts.items():
        write_csv(laydir/f'{key}.csv',make_layout_rows(xy,model))
        cnt,n,edges=upstream_counts(xy,model)
        counts_table.append({'layout':key,'turbines_with_upstream_neighbour':n,'qualifying_pairs':edges,
              'L_s_m':model.downstream_scale_m,'sigma_n_m':model.cross_scale_m})
        eval_table.append({'layout':key,'saved_Q_3a':format(vals[key],'.17g'),
            'recomputed_Q_3a':format(score[key],'.17g'),
            'absolute_difference':format(abs(vals[key]-score[key]),'.17g'),
            'passed':True})
    write_csv(out/'metric3a_verification.csv',eval_table)
    write_csv(out/'upstream_neighbour_counts.csv',counts_table)
    best=layouts['aligned_start'];ref=layouts['aligned_reference']
    dij=np.linalg.norm(best[:,None,:]-ref[None,:,:],axis=2)
    row,col=linear_sum_assignment(dij)
    moves=dij[row,col]
    write_csv(out/'aligned_layout_displacements.csv',[
       {'aligned_start_turbine_id':int(i),'reference_turbine_id':int(j),
        'displacement_m':format(dij[i,j],'.17g')} for i,j in zip(row,col)])
    settings={**EXPECTED,'L_s_m':1920.0,'sigma_n_m':240.0,
         'optimiser':'single-objective GA (historical only; no new optimisation performed)',
         'risk_objective':'Q_3a = sum of remaining turbine exposures',
         'bird_heading_convention':'225 deg TOWARD, mathematical angle (southwest)',
         'upstream_diagnostic':'0 < delta_along <= 1920 m AND abs(delta_cross) <= 240 m',
         'displacement_matching':'one-to-one minimum total Euclidean distance (Hungarian algorithm)',
         'no_AEP_or_wake_calculation':True,'source':'two original saved run archives',
         'environment':{'python':platform.python_version(),'numpy':np.__version__,
                        'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}}
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    figs=make_plots(out,layouts,score,scenario,model,convergence)
    notes='''# Metric 3a verification: code excerpts and decision

CHECK 1a: PASS (single-objective GA, no energy or wake evaluation).
- risk_only_problem.py:57-62: `n_obj=1`.
- risk_only_problem.py:82-84: `F[population_index, 0] = float(self.risk_model.collisions(xy))`.
- run_directional_absorbing_risk_only.py:644-658: risk-only problem and GA.
- run_directional_absorbing_seeded.py:247-268: the same risk-only problem and GA.
- run_validation.py:14-16 imports `AEPModel`, but is imported by these runners ONLY for GridSampling and geometry utilities; its AEP paths are not executed by these runners.

CHECK 1b: PASS (fixed bird travel direction, not derived from wind).
- directional_absorbing_risk.py:43-49: `bird_travel_deg: float = 225.0`, downstream 8D, cross 1D, alpha 0.8.
- directional_absorbing_risk.py:143-149: heading is a TOWARD angle, 0 toward +x, 90 toward +y; radians of `self.bird_travel_deg`.
- directional_absorbing_risk.py:152-168: along=(cos(theta),sin(theta)), cross=(-sin(theta),cos(theta)).
- The 225 deg numerical value is independent of the wind FROM convention and wind rose resolution.

CHECK 2: PASS. See metric3a_verification.csv for full saved and recalculated precision.

DECISION: Keep existing Metric 3a results. Do not rerun the optimisation.

Geometry counts apply hard cutoffs 1920 m along and 240 m across only for diagnostic interpretation; the model's exponential kernel itself has no hard downstream/cross-flight cutoff.

Original ZIPs and input code were copied without alteration. Output figures have no titles or turbine labels, use bird-flight coordinates with flight arrow, and put Q_3a in legend entries.
'''
    (out/'code_audit.md').write_text(notes,encoding='utf-8')
    summary={'check_1_code_audit':'PASS','check_2_recalculation':'PASS','decision':'keep_metric_3a_no_rerun',
       'recalculated_Q_3a':score,'saved_Q_3a':vals,
       'mean_displacement_m':float(moves.mean()),'max_displacement_m':float(moves.max()),
       'match_pairs':[[int(i),int(j)] for i,j in zip(row,col)],
       'upstream_counts':{x['layout']:x['turbines_with_upstream_neighbour'] for x in counts_table},
       'figures':figs}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    input_hashes=[]
    for name in CODE_NAMES:input_hashes.append({'type':'original_code','filename':name,'sha256':sha256(source/name)})
    for name in ARCHIVE_NAMES:input_hashes.append({'type':'original_archive','filename':name,'sha256':sha256(archive/name)})
    write_csv(out/'source_input_hashes.csv',input_hashes)
    generated=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='output_file_hashes.csv':
            generated.append({'path':str(p.relative_to(out)).replace('\\','/'),'sha256':sha256(p)})
    write_csv(out/'output_file_hashes.csv',generated)
    print('Diagnostics: displacement mean={:.6f} m max={:.6f} m'.format(moves.mean(),moves.max()))
    print('Upstream counts:',{x['layout']:x['turbines_with_upstream_neighbour'] for x in counts_table})
    print('Created:',out)
    print('Figures:',', '.join(figs))

if __name__=='__main__':
    try: main()
    except Exception as err:
        print('STOP:',err,file=sys.stderr)
        sys.exit(1)
