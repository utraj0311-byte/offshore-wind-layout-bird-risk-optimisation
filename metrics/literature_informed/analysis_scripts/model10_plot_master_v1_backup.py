#!/usr/bin/env python3
"""Generate Model 10 thesis figures FROM SAVED HPC RESULTS on HPC12.

No FLORIS or bird simulations, no optimiser, no file changes to originals.

Run (in the model10 Conda environment):
 python 07_scientific_code_and_job_scripts/model10_plot_master.py --master .
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde, binomtest

LAYOUTS=['highest_aep','knee','lowest_risk']
LABEL={'highest_aep':'Highest-AEP','knee':'Knee','lowest_risk':'Lowest-risk','reference':'Reference'}
COL={'highest_aep':'#2166AC','knee':'#E18D29','lowest_risk':'#27964B','reference':'#777777'}
SEED_COL={730101:'#3E7A9A',730102:'#9871AC',730103:'#BE7950'}
MARK={'highest_aep':'o','knee':'D','lowest_risk':'s','reference':'X'}
DIRECTIONS=[0,45,90,135,180,225,270,315]

plt.rcParams.update({'figure.dpi':120,'savefig.dpi':260,'font.family':'DejaVu Sans',
'font.size':10,'axes.labelsize':11,'axes.titlesize':12,'legend.fontsize':9,
'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
'grid.color':'#D4DCE3','grid.alpha':.55,'axes.axisbelow':True,'pdf.fonttype':42,'svg.fonttype':'none'})


def read(path):
    if not Path(path).is_file():raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def save(fig,outdir,name,record):
    fig.tight_layout(pad=1.4)
    for ext in ('png','pdf','svg'):
        fig.savefig(outdir/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    record.append(name)
    print('PLOTTED',name,flush=True)


def setup(master):
    out=master/'08_figures';out.mkdir(exist_ok=True)
    tables=master/'09_derived_tables';tables.mkdir(exist_ok=True)
    return out,tables


def opt_front(master,out,records):
    nd=read(master/'02_combined_pareto/combined_non_dominated_24.csv')
    prim=read(master/'02_combined_pareto/final_selected_three.csv')
    fig,ax=plt.subplots(figsize=(8.2,5.2))
    for seed in [730101,730102,730103]:
        ds=nd[nd.seed==seed]
        ax.scatter(ds.aep_gwh,ds.optimisation_sample_Q_lit*1e5,s=28,color=SEED_COL[seed],alpha=.76,label=f'Seed {seed}',zorder=3)
    for _,s in prim.iterrows():
        k=s['selection'];ax.scatter(s.aep_gwh,s.optimisation_sample_Q_lit*1e5,s=132,facecolor=COL[k],edgecolor='black',linewidth=1.15,marker=MARK[k],zorder=8,label=LABEL[k])
        dy={'highest_aep':.21,'knee':-.35,'lowest_risk':.24}[k]
        ax.annotate(LABEL[k],(s.aep_gwh,s.optimisation_sample_Q_lit*1e5),xytext=(8,dy*48),textcoords='offset points',fontsize=9)
    baseline=804.3901377275079;qr=3.20618950134584e-5
    ax.scatter([baseline],[qr*1e5],s=90,color=COL['reference'],marker='X',label='Reference (context)',zorder=6)
    ax.set_xlabel('72-sector AEP (GWh/year)');ax.set_ylabel(r'Optimisation-sample $Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
    ax.legend(loc='upper left',ncol=2,frameon=True);save(fig,out,'01_final_combined_24_layout_pareto',records)


def layouts(master,out,records):
    import matplotlib.patches as patches
    ref=read(master/'01_optimisation/seed_730101/optimisation/source/verification/runtime_inputs/energy_baseline.csv')
    points={k:read(master/f'02_combined_pareto/{k}_coordinates_xy_m.csv') for k in LAYOUTS}
    points['reference']=ref
    fig,axs=plt.subplots(2,2,figsize=(9,9),sharex=True,sharey=True)
    for ax,k in zip(axs.flat,[*LAYOUTS,'reference']):
        d=points[k]
        ax.set_xlim(-175,6175);ax.set_ylim(-175,6175);ax.set_aspect('equal')
        ax.add_patch(patches.Rectangle((0,0),6000,6000,fill=False,lw=1.2,ec='#616B76',ls='--'))
        for _,row in d.iterrows():
            ax.add_patch(Circle((row.x_m,row.y_m),radius=121.12,facecolor=COL[k],edgecolor=COL[k],alpha=.23,lw=.7))
        ax.scatter(d.x_m,d.y_m,color=COL[k],s=14,zorder=4)
        ax.set_title(LABEL[k],fontsize=12)
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    save(fig,out,'02_four_layout_maps_rotor_discs',records)


def confirm(master,out,tables,records):
    nd=read(master/'03_independent_evaluation/final_four_guarded/layout_means.csv')
    q=nd[np.isclose(nd.response_probability,.6)].set_index('layout')
    base=read(master/'02_combined_pareto/final_selected_three.csv').set_index('selection')
    meta={k:float(base.loc[k,'aep_gwh']) for k in LAYOUTS};meta['reference']=804.3901377275079
    full=[]
    for k in [*LAYOUTS,'reference']:
        full.append({'layout':LABEL[k],'aep_72sector_gwh':meta[k],
          'independent_evaluation_Q_lit':float(q.loc[k,'mean_Q_lit']),
          'independent_batch_sd':float(q.loc[k,'sd_batches'])})
    pd.DataFrame(full).to_csv(tables/'independent_selected_layout_means.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.8,5.0))
    for k in [*LAYOUTS,'reference']:
        a=meta[k];v=float(q.loc[k,'mean_Q_lit'])*1e5
        se=float(q.loc[k,'sd_batches'])*1e5/math.sqrt(10)
        ax.errorbar([a],[v],yerr=[2.26216*se],fmt=MARK[k],markersize=9,capsize=4,elinewidth=1.5,color=COL[k],label=LABEL[k])
        ax.annotate(LABEL[k],(a,v),xytext=(8,10 if k!='knee' else -18),textcoords='offset points')
    ax.set_xlabel('72-sector AEP (GWh/year)');ax.set_ylabel(r'Independent-evaluation $Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
    ax.legend(loc='upper left'); save(fig,out,'03_independent_evaluation_selected_AEP_vs_Q',records)
    fig,ax=plt.subplots(figsize=(7.4,4.8))
    x=np.arange(3); width=.34
    for i,k in enumerate(LAYOUTS):
        ax.bar(i-width/2,base.loc[k,'optimisation_sample_Q_lit']*1e5,width,color=COL[k],alpha=.65)
        ax.bar(i+width/2,q.loc[k,'mean_Q_lit']*1e5,width,color=COL[k],edgecolor='#111111',linewidth=1.0)
    ax.bar([],[],color='#777777',alpha=.65,label='Optimisation sample')
    ax.bar([],[],color='#777777',edgecolor='#111111',label='Independent evaluation sample')
    ax.set_xticks(x,[LABEL[k] for k in LAYOUTS]);ax.set_ylabel(r'$Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
    ax.legend(frameon=False);save(fig,out,'04_optimisation_vs_independent_evaluation',records)


def sensitivity_one(master,out,tables,records,dim):
    f=master/f'04_sensitivities/{dim}/layout_means_by_scenario.csv'
    df=read(f)
    names={'probability':('Response probability, p','Response probability'),
      'radius':('Response radius (D)','Response radius'),
      'hub_height':('Hub height in bird model (m)','Hub height'),
      'bird_speed':('Bird speed in Band calculation (m/s)','Bird speed')}
    xs=sorted(pd.to_numeric(df.scenario_value).unique())
    fig,ax=plt.subplots(figsize=(7.7,4.9))
    for k in [*LAYOUTS,'reference']:
        s=df[df.layout==k].copy();s['x']=pd.to_numeric(s.scenario_value);s=s.sort_values('x')
        ax.plot(s.x,s.mean_Q_lit*1e5,marker=MARK[k],markersize=6,lw=1.7,label=LABEL[k],color=COL[k])
    ax.set_xlabel(names[dim][0]);ax.set_ylabel(r'$Q_{\mathrm{lit}}$ ($\times10^{-5}$)')
    ax.set_xticks(xs);ax.legend(loc='best',frameon=False)
    if dim=='radius':ax.axvspan(.47,.54,color='#f5dbb2',alpha=.3)
    if dim=='hub_height':ax.text(.02,.95,'AEP fixed; bird model only',transform=ax.transAxes,va='top',fontsize=9,color='#56606B')
    save(fig,out,'05_sensitivity_'+dim,records)
    base=read(master/f'04_sensitivities/{dim}/three_paired_comparisons_bonferroni_holm.csv')
    base.to_csv(tables/f'sensitivity_{dim}_adjusted_differences.csv',index=False)


def direction(master,out,tables,records):
    df=read(master/'04_sensitivities/direction_fixed_width/layout_means_8_directions.csv')
    res=read(master/'04_sensitivities/direction_fixed_width/three_paired_comparisons_bonferroni_holm.csv')
    res.to_csv(tables/'direction_24_adjusted_paired_comparisons.csv',index=False)
    spread=[]
    for k in [*LAYOUTS,'reference']:
        v=df[df.layout==k]['mean_Q_lit']
        spread.append({'layout':LABEL[k],'max_over_min_heading':float(v.max()/v.min()),'min':float(v.min()),'max':float(v.max())})
    pd.DataFrame(spread).to_csv(tables/'direction_layout_range_ratios.csv',index=False)
    for scaled in [False, True]:
        fig,ax=plt.subplots(figsize=(8.3,5.0))
        ax.axvspan(220,230,color='#E8E6D1',alpha=.78,zorder=0,label='225° optimisation heading')
        for k in [*LAYOUTS,'reference']:
            s=df[df.layout==k].sort_values('heading_deg_toward')
            y=s['mean_Q_times_W'] if scaled else s['mean_Q_lit']*1e5
            ax.scatter(s.heading_deg_toward,y,marker=MARK[k],s=68,color=COL[k],label=LABEL[k],zorder=4)
        ax.set_xticks(DIRECTIONS,[f'{n}°' for n in DIRECTIONS]);ax.set_xlim(-17,334)
        ax.set_xlabel('Bird heading (degrees towards)')
        ax.set_ylabel(r'$W Q_{\mathrm{lit}}$ (m × expected strikes per entering bird)' if scaled else r'$Q_{\mathrm{lit}}$ ($\times10^{-5}$)')
        ax.legend(loc='best',ncol=2,frameon=True)
        save(fig,out,'06_direction_fixed_width_'+('Q_times_W' if scaled else 'Q'),records)


def bootstrap(master,out,tables,records):
    f=master/'05_uncertainty/bird_level_bootstrap/paired_10000_bird_bootstrap_replicates.npz'
    if not f.exists():raise FileNotFoundError('Missing exact bird-level bootstrap replicate file: '+str(f))
    a=np.load(f,allow_pickle=False)
    samples=[np.asarray(a[k],dtype=float)*1e5 for k in LAYOUTS]
    n=len(samples[0]);assert all(len(s)==n for s in samples) and n==10000
    fig,ax=plt.subplots(figsize=(8.3,4.9))
    for k,s in zip(LAYOUTS,samples):
        grid=np.linspace(s.min()-3*s.std(),s.max()+3*s.std(),450)
        den=gaussian_kde(s)(grid)
        ax.plot(grid,den,color=COL[k],lw=2,label=LABEL[k]);ax.fill_between(grid,den,alpha=.14,color=COL[k])
        ax.axvline(np.mean(s),color=COL[k],ls=':',lw=1.4)
    ax.set_xlabel(r'Optimisation-sample $Q_{\mathrm{lit}}$ ($\times10^{-5}$)');ax.set_ylabel('Bootstrap density')
    ax.legend(ncol=3,frameon=True,loc='upper center')
    save(fig,out,'07_bird_level_bootstrap_N50000_10000resamples',records)
    original=read(master/'05_uncertainty/bird_level_bootstrap/bird_bootstrap_three_layout_summary.csv')
    original.to_csv(tables/'bird_level_bootstrap_summary.csv',index=False)
    differences=read(master/'05_uncertainty/bird_level_bootstrap/bird_bootstrap_three_paired_differences.csv')
    differences.to_csv(tables/'bird_level_bootstrap_paired_differences.csv',index=False)


def montecarlo(master,out,tables,records):
    df=read(master/'03_independent_evaluation/final_four_guarded/guarded_fresh_batches.csv')
    df=df[np.isclose(df.response_probability,.6)].copy()
    rows=[]; seeds={}
    for k in [*LAYOUTS,'reference']:
        v=df[df.layout==k].sort_values('batch_index')['Q_lit'].to_numpy()
        if len(v)!=10:raise RuntimeError('Expected 10 independent batches for '+k)
        seeds[k]=v
        for b in [1,2,3,4,5]:
            blocks=np.array([np.mean(v[i*b:(i+1)*b]) for i in range(10//b)],dtype=float)
            if len(blocks)>=2:
                rows.append({'layout':LABEL[k],'N_birds':b*50000,'disjoint_groups':len(blocks),'empirical_SD':float(np.std(blocks,ddof=1))})
    pd.DataFrame(rows).to_csv(tables/'monte_carlo_disjoint_batch_aggregations.csv',index=False)
    # Exact paired sign tests on the ten independent batches (not a batch bootstrap).
    sign=[]
    for aa,bb in [('highest_aep','knee'),('knee','lowest_risk'),('highest_aep','lowest_risk')]:
        diff=seeds[aa]-seeds[bb]
        positive=int(np.sum(diff>0)); negative=int(np.sum(diff<0)); nonzero=positive+negative
        p=float(binomtest(positive,nonzero,p=.5,alternative='two-sided').pvalue) if nonzero else 1.0
        sign.append({'layout_A':LABEL[aa],'layout_B':LABEL[bb],'positive_batches':positive,
          'negative_batches':negative,'nonzero_batches':nonzero,'mean_A_minus_B_Q_lit':float(diff.mean()),
          'exact_two_sided_sign_p':p,'bonferroni_3_comparisons_p':min(1.,p*3)})
    pd.DataFrame(sign).to_csv(tables/'exact_three_independent_batch_sign_tests.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.1,5.3))
    ngrid=np.geomspace(1000,500000,120)
    for k in LAYOUTS:
        refsd=np.std(seeds[k],ddof=1)*1e5
        ax.plot(ngrid,refsd*np.sqrt(50000/ngrid),color=COL[k],lw=1.8,linestyle='--',alpha=.75)
        subset=[r for r in rows if r['layout']==LABEL[k]]
        ax.scatter([r['N_birds'] for r in subset],[r['empirical_SD']*1e5 for r in subset],
           marker=MARK[k],s=54,color=COL[k],label=LABEL[k],zorder=5)
    means={k:np.mean(seeds[k])*1e5 for k in LAYOUTS}
    for u,v in [('highest_aep','knee'),('knee','lowest_risk'),('highest_aep','lowest_risk')]:
        gap=means[u]-means[v]
        ax.axhline(gap,lw=1.1,color='#555555',alpha=.5)
        ax.text(1400,gap*1.06,f'{LABEL[u]} – {LABEL[v]}: {gap:.2f}',fontsize=7.5,color='#333333')
    ax.axvline(50000,color='#777777',ls=':',lw=1.2)
    ax.text(52000,.014,'Optimisation N = 50,000',fontsize=8,color='#666666',rotation=90,va='bottom')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(950,520000)
    ax.set_xlabel('Birds per Monte Carlo estimate (N)')
    ax.set_ylabel(r'SD of estimated $Q_{\mathrm{lit}}$ ($\times10^{-5}$)')
    ax.legend(loc='upper right',frameon=False)
    save(fig,out,'08_monte_carlo_convergence_empirical_and_sqrtN',records)


def seed_exploration(master,out,tables,records):
    details=[]
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    for seed in [730101,730102,730103]:
        f=master/f'01_optimisation/seed_{seed}/optimisation/all_evaluated_candidates.csv';df=read(f)
        feasible=(df.max_spacing_constraint_m<=1e-7)&(df.max_boundary_violation_m<=1e-7)&(df.minimum_spacing_m>=968.96-1e-7)
        ds=df.loc[feasible]
        row=ds.loc[ds.aep_gwh.idxmax()]
        details.append({'seed':seed,'evaluation_index':int(row.evaluation_index),'highest_feasible_AEP_gwh':float(row.aep_gwh),'feasible_candidates':len(ds)})
        ax.scatter(ds.aep_gwh,ds.risk_per_entrant*1e5,s=8,alpha=.23,color=SEED_COL[seed],label=f'Seed {seed}: feasible candidates')
        ax.scatter([row.aep_gwh],[row.risk_per_entrant*1e5],marker='x',s=80,lw=2.0,color=SEED_COL[seed])
    pd.DataFrame(details).to_csv(tables/'three_seed_best_energy_spread.csv',index=False)
    ax.set_xlabel('72-sector AEP (GWh/year)');ax.set_ylabel(r'Optimisation-sample $Q_{\mathrm{lit}}$ ($\times10^{-5}$)')
    ax.legend(frameon=False,markerscale=2)
    save(fig,out,'09_three_seed_feasible_candidate_exploration',records)


def numerical(master,out,tables,records):
    import json
    f=master/'06_numerical_resolution/aep_360sector/report.json'
    data=json.loads(f.read_text());rows=data.get('rows',[])
    if not rows:raise RuntimeError('AEP 360-sector report.json has no rows')
    df=pd.DataFrame(rows);df.to_csv(tables/'aep_72_vs_360_sectors.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    cats=[];deltas=[];cols=[]
    for row in rows:
        k=row['layout'];cats.append(LABEL.get(k,k));deltas.append(float(row['difference_1minus5_GWh']));cols.append(COL.get(k,'#888'))
    ax.bar(cats,deltas,color=cols,width=.58)
    ax.axhline(0,color='#4E5963',lw=.8)
    ax.set_ylabel('AEP(360 sectors) − AEP(72 sectors) (GWh/year)')
    save(fig,out,'10_1degree_5degree_AEP_difference',records)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--master',type=Path,default=Path.cwd())
    args=ap.parse_args(); master=args.master.resolve()
    if not (master/'archive_audit.json').is_file():raise RuntimeError('Not a verified master archive (missing archive_audit.json): '+str(master))
    out,tables=setup(master);record=[]
    opt_front(master,out,record);layouts(master,out,record);confirm(master,out,tables,record)
    for k in ('probability','radius','hub_height','bird_speed'):sensitivity_one(master,out,tables,record,k)
    direction(master,out,tables,record);bootstrap(master,out,tables,record);montecarlo(master,out,tables,record)
    seed_exploration(master,out,tables,record);numerical(master,out,tables,record)
    (out/'FIGURES_README.txt').write_text('Generated by model10_plot_master.py on HPC12 from saved final 72-sector Model 10 results.\n'
      'The original HPC source results are untouched. Each figure is exported to PDF, SVG and PNG.\n'
      'Directional plot uses markers only; between-heading values were NOT calculated.\n'
      'The 225 degree shaded band marks the baseline optimisation heading.\n'
      'Direction sensitivity uses fixed W=9696.481 m and 57137 birds x 10 independent batches; AEP held fixed.\n'
      'Monte Carlo plot: only empirical points at >=50000 birds from disjoint 50000-bird batches.\n'
      'Dashed 1/sqrt(N) curves outside observed sizes are theoretical extrapolations.\n'
      'SD is sampling variability, not uncertainty in ecological behaviour parameters.\n'
      'Bootstrap: 10000 stratified paired individual-bird resamples from fixed optimisation sample N=50000.\n'
      'Bootstrap curves use Gaussian KDE; dots in AEP confirmation indicate independent-sample means.\n'
      'Original 72-sector objectives and independent 360-sector AEP verification must remain separate.\n'
      'Final reported layout identifiers are highest AEP 455, knee 418, lowest risk 220, reference contextual.\n'
      'No plot displays or claims annual fatalities.\n\nGenerated figures:\n'+'\n'.join(record)+'\n')
    print('DONE: Generated',len(record),'figure sets (PDF, SVG, PNG) in',out)

if __name__=='__main__':
    try:main()
    except Exception as e:print('PLOTTING FAILED:',str(e),file=sys.stderr);sys.exit(1)
