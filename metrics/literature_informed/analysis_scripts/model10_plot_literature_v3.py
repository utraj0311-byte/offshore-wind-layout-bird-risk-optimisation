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
COL={'highest_aep':'#2166AC','knee':'#27964B','lowest_risk':'#E18D29','reference':'#777777'}
SEED_COL={730101:'#3E7A9A',730102:'#9871AC',730103:'#BE7950'}
MARK={'highest_aep':'o','knee':'s','lowest_risk':'D','reference':'X'}
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
    # Versioned folder protects the 14 original HPC figures from being overwritten.
    out=master/'08_figures_thesis_v3';out.mkdir(exist_ok=True)
    tables=master/'09_derived_tables_thesis_v3';tables.mkdir(exist_ok=True)
    return out,tables


def opt_front(master,out,records):
    nd=read(master/'02_combined_pareto/combined_non_dominated_24.csv')
    prim=read(master/'02_combined_pareto/final_selected_three.csv')
    fig,ax=plt.subplots(figsize=(8.2,5.2))
    for seed in [730101,730102,730103]:
        ds=nd[nd.seed==seed]
        ax.scatter(ds.aep_gwh,ds.optimisation_sample_Q_lit*1e5,s=28,color=SEED_COL[seed],alpha=.76,label=f'Seed {seed}',zorder=3)
    for _,s in prim.iterrows():
        k=s['selection'];ax.scatter(s.aep_gwh,s.optimisation_sample_Q_lit*1e5,s=132,facecolor=('none' if k=='lowest_risk' else COL[k]),edgecolor=COL[k],linewidth=1.6,marker=MARK[k],zorder=8,label=LABEL[k])
        dy={'highest_aep':.21,'knee':-.35,'lowest_risk':.24}[k]
        ax.annotate(LABEL[k],(s.aep_gwh,s.optimisation_sample_Q_lit*1e5),xytext=(8,dy*48),textcoords='offset points',fontsize=9)
    baseline=804.3901377275079;qr=3.20618950134584e-5
    ax.scatter([baseline],[qr*1e5],s=90,color=COL['reference'],marker='X',label='Reference (context)',zorder=6)
    ax.set_xlabel('AEP (GWh/year)');ax.set_ylabel(r'Optimisation-sample $Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
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
        ax.scatter(d.x_m,d.y_m,facecolor=('none' if k=='lowest_risk' else COL[k]),edgecolor=COL[k],linewidths=1.0,marker=MARK[k],s=21,zorder=4)
        ax.set_title(LABEL[k],fontsize=12)
        ax.set_xlabel('x position (m)');ax.set_ylabel('y position (m)')
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
        ax.errorbar([a],[v],yerr=[2.26216*se],fmt=MARK[k],markersize=9,capsize=4,elinewidth=1.5,color=COL[k],mfc=('none' if k=='lowest_risk' else COL[k]),mec=COL[k],mew=1.5,label=LABEL[k])
        ax.annotate(LABEL[k],(a,v),xytext=(8,10 if k!='knee' else -18),textcoords='offset points')
    ax.set_xlabel('AEP (GWh/year)');ax.set_ylabel(r'Independent-evaluation $Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
    ax.legend(loc='upper left',title='Means with individual 95% t intervals\n(not simultaneous paired intervals)',title_fontsize=8,frameon=True)
    save(fig,out,'03_independent_evaluation_selected_AEP_vs_Q',records)
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
        ax.scatter(s.x,s.mean_Q_lit*1e5,marker=MARK[k],s=50,label=LABEL[k],facecolor=('none' if k=='lowest_risk' else COL[k]),edgecolor=COL[k],linewidth=1.3)
    ax.set_xlabel(names[dim][0]);ax.set_ylabel(r'$Q_{\mathrm{lit}}$ ($\times10^{-5}$)')
    ax.set_xticks(xs);ax.legend(loc='best',frameon=False)
    # No shading of the 0.5D radius: 2D is the actual baseline.
    # Fixed AEP and bird-model-only hub height belong in the caption.
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
            ax.scatter(s.heading_deg_toward,y,marker=MARK[k],s=68,facecolor=('none' if k=='lowest_risk' else COL[k]),edgecolor=COL[k],linewidth=1.5,label=LABEL[k],zorder=4)
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
    ax.text(.985,.94,'Dotted lines: bootstrap means', transform=ax.transAxes,
            ha='right',va='top',fontsize=8,color='#4D5962')
    ax.text(.26,.70,'N = 50,000 birds (optimisation sample)\n10,000 paired, entry-stratified resamples',
            transform=ax.transAxes,ha='left',va='bottom',fontsize=8,color='#53616A',
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
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
           marker=MARK[k],s=54,facecolor=('none' if k=='lowest_risk' else COL[k]),edgecolor=COL[k],linewidth=1.5,label=LABEL[k],zorder=5)
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
    ax.legend(loc='upper right',frameon=False,title='Markers: empirical disjoint-batch SD',title_fontsize=8)
    ax.text(.015,.025,'Dashed curves: $1/\\sqrt{N}$ scaling anchored at 50,000 birds\n'
            'Below 50,000: extrapolated; at 250,000: only 2 disjoint groups',
            transform=ax.transAxes,ha='left',va='bottom',fontsize=8,color='#53616A',
            bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=2))
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
    ax.set_xlabel('AEP (GWh/year)');ax.set_ylabel(r'Optimisation-sample $Q_{\mathrm{lit}}$ ($\times10^{-5}$)')
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


# Recommended thesis placement. Original scientific filenames remain unchanged.
MAIN_FIGURES = [
 '01_final_combined_24_layout_pareto',
 '02_four_layout_maps_rotor_discs',
 '03_independent_evaluation_selected_AEP_vs_Q',
 '05_sensitivity_radius',
 '06_direction_fixed_width_Q',
 '07_bird_level_bootstrap_N50000_10000resamples',
 '08_monte_carlo_convergence_empirical_and_sqrtN',
]

FIGURE_CAPTIONS = {
 '01_final_combined_24_layout_pareto': (
  'Final Model 10 non-dominated set pooled from three independent NSGA-II searches '
  '(seeds 730101, 730102, 730103), evaluated using 72 wind-direction sectors and '
  'a fixed optimisation sample of 50,000 simulated birds per candidate. '
  'The knee was selected from the optimisation-sample Q_lit and 72-sector AEP '
  'by the pre-specified maximum positive normalised signed-distance rule; '
  'the reference is shown separately for context, not as a member of the 24-point front.'),
 '02_four_layout_maps_rotor_discs': (
  'Turbine-centre coordinates and 121.12 m-radius rotor-disc footprints for the highest-AEP, '
  'knee, baseline lowest-risk and reference layouts. The dashed 6 by 6 km site boundary '
  'constrains turbine centres; rotor discs can extend beyond this boundary.'),
 '03_independent_evaluation_selected_AEP_vs_Q': (
  'Seventy-two-sector AEP versus independently evaluated expected Band potential strikes '
  'per entering bird for the three selected Pareto layouts and reference. Independent evaluation '
  'uses 10 batches of 50,000 birds per layout at baseline p = 0.6; error bars indicate individual '
  'unadjusted 95% t intervals for the ten-batch mean (not paired simultaneous intervals). '
  'Paired Bonferroni confidence intervals and Holm-adjusted tests are reported in the text/table.'),
 '04_optimisation_vs_independent_evaluation': (
  'Optimisation-sample Q_lit (5 batches of 10,000 birds) compared with the independently '
  'evaluated Q_lit (10 batches of 50,000 birds) for the final three layouts. '
  'The same optimisation sample was used across all three NSGA-II searches; '
  'the independent evaluation samples were not used for layout selection.'),
 '05_sensitivity_probability': (
  'Independent evaluation sensitivity to bird-response probability (0.3, 0.6, 0.9). '
  'Results use four fixed turbine layouts and 10 paired batches per scenario; '
  'other baseline parameters are held constant.'),
 '05_sensitivity_radius': (
  'Independent evaluation sensitivity to response radius (0.5D, 1D, 2D), where D = 242.24 m. '
  'The highest-AEP and knee layouts reverse risk ordering at 0.5D; this is a simulation-model '
  'sensitivity, not a re-optimisation. The 2D case is the baseline.'),
 '05_sensitivity_hub_height': (
  'Bird-model-only hub-height sensitivity at 150, 160, 170 and 180 m. '
  'AEP and the turbine layout are held fixed; no FLORIS AEP recalculation '
  'or turbine re-optimisation is included.'),
 '05_sensitivity_bird_speed': (
  'Band strike-weighting sensitivity to bird speed (10.79, 14.01 and 14.9 m/s), '
  'evaluated on the same four fixed layouts using paired independent samples. '
  'The bird-speed change affects the Band weighting, not the simulated rotor-transit events.'),
 '06_direction_fixed_width_Q': (
  'Independently evaluated Q_lit for eight discrete bird headings (0 to 315 degrees in 45-degree steps). '
  'A fixed 9,696.481 m cross-flight corridor and 10 paired independent batches of 57,137 birds '
  'per heading were used; all 32 layout-direction rotor-coverage checks passed. '
  'The shaded band marks the 225-degree optimisation heading. Bird heading, rotor-plane normal '
  'and Band wind rotate together; AEP remains fixed. Markers only: intervening headings were not tested. '
  'The Lowest-risk legend name refers to selection under the 225-degree baseline, not all headings.'),
 '06_direction_fixed_width_Q_times_W': (
  'Same eight-direction sensitivity as the unscaled Q_lit figure, multiplied by the common fixed '
  'corridor width W = 9,696.481 m. Because W is constant, rankings are unchanged. '
  'The scaled quantity is for comparing against the original width-dependent baseline convention '
  'and is not an annual mortality estimate.'),
 '07_bird_level_bootstrap_N50000_10000resamples': (
  'Paired bird-level bootstrap distributions of optimisation-sample Q_lit for the highest-AEP '
  '(455), knee (418) and baseline lowest-risk (220) layouts. Distributions derive from 10,000 '
  'paired, entry-stratified resamples of the fixed N = 50,000 optimisation-sample birds. '
  'Dotted lines mark bootstrap means; curves are smoothed kernel-density estimates. '
  'These distributions quantify finite-sample uncertainty conditional on the optimisation sample; '
  'the independent evaluation is the primary confirmation of layout differences.'),
 '08_monte_carlo_convergence_empirical_and_sqrtN': (
  'Monte Carlo sampling variability for highest-AEP (455), knee (418) and baseline lowest-risk (220) '
  'layouts. Markers represent empirical sample SD estimated from disjoint aggregates of independent '
  '50,000-bird batches; at N = 250,000 only two independent aggregates are available. '
  'Dashed lines show theoretical 1/sqrt(N) scaling anchored at the observed N = 50,000 SD, '
  'and are extrapolations below N = 50,000. Grey horizontal lines denote the three independently '
  'evaluated paired mean differences; these are comparisons of magnitude, not confidence intervals. '
  'The dotted vertical line marks the optimisation-sample size N = 50,000.'),
 '09_three_seed_feasible_candidate_exploration': (
  'Feasible candidates evaluated in each of three NSGA-II runs using identical fixed bird samples '
  "and 72-sector AEP; crosses mark each seed's maximum feasible AEP. "
  'The three searches do not demonstrate global optimality.'),
 '10_1degree_5degree_AEP_difference': (
  'Recomputed 360-sector (1-degree) minus 72-sector (5-degree) AEP for highest-AEP, knee and '
  'baseline lowest-risk layouts, without re-optimisation or additional bird simulations. '
  'The highest-AEP-to-knee gap is 1.6869 GWh/year (0.2049% of the highest-AEP layout) at 1-degree resolution, '
  'versus 1.7380 GWh/year (0.2112%) at 5-degree resolution.'),
}


def write_figure_guide(out, generated):
    if set(generated) != set(FIGURE_CAPTIONS):
        raise RuntimeError('Caption coverage mismatch. Missing: '+str(set(generated)-set(FIGURE_CAPTIONS)))
    lines = [
      '# Model 10 — figure selection and thesis captions',
      '',
      'Generated on HPC12 from the verified master archive. This is a proposed placement, not a thesis requirement.',
      'The numerical source files are unchanged. Original first-generation figures remain in `08_figures/`;',
      'these revised exports are in `08_figures_thesis_v2/`.',
      '',
      '## Suggested main Results figures',
      '',
    ]
    for name in MAIN_FIGURES:
        lines += [f'### {name}', '', FIGURE_CAPTIONS[name], '']
    lines += ['## Suggested appendix / supplementary figures', '']
    for name in generated:
        if name not in MAIN_FIGURES:
            lines += [f'### {name}', '', FIGURE_CAPTIONS[name], '']
    lines += ['## Notes for interpretation', '',
      '- The main-text selection deliberately prioritises the combined front, selected layouts, independent evaluation, response-radius reversal, direction dependence and two uncertainty checks.',
      '- Full adjusted pairwise comparisons belong in tables; single-layout figure error bars must not be described as paired Bonferroni intervals.',
      '- AEP is the frozen 72-sector optimisation objective; 360-sector AEP is a numerical-resolution verification, not a re-optimisation.',
      '- "Lowest-risk" names the baseline-selected layout, not a claim of superiority under every directional sensitivity.',
      '- Monte Carlo and bootstrap uncertainty do not cover uncertainty in avoidance assumptions or flight-direction distributions.',
      '- Use consistent thesis wording: "optimisation sample", "independent evaluation sample".',
      '- This directional sensitivity rotates initial bird heading, rotor-plane normal and Band wind together; it does not model a site-specific bird-direction distribution.',
      '- Reference layout results provide context and are not part of the non-dominated Pareto set.',
      '',
    ]
    (out/'FIGURE_SELECTION_AND_CAPTIONS.md').write_text('\n'.join(lines),encoding='utf-8')


# ----- FINAL FIVE MAIN-TEXT FIGURES (figure paths match locked Results .tex) -----
# Scientific input files are read-only; the functions below only plot saved HPC results.

def _plot_selected(ax, k, x, y, *, size=95, label=None, zorder=7):
    """Chapter-wide marker convention: open orange diamond=lowest-risk."""
    kws = dict(marker=MARK[k], s=size, edgecolors=COL[k], linewidths=1.7,
               label=LABEL[k] if label is None else label, zorder=zorder)
    kws['facecolors'] = 'none' if k == 'lowest_risk' else COL[k]
    return ax.scatter(x, y, **kws)


def main_front(master, out, record):
    nd=read(master/'02_combined_pareto/combined_non_dominated_24.csv')
    sel=read(master/'02_combined_pareto/final_selected_three.csv')
    if len(nd)!=24: raise RuntimeError('Final front does not have 24 points.')
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    for seed in (730101,730102,730103):
        subset=nd[nd.seed==seed]
        ax.scatter(subset.aep_gwh,subset.optimisation_sample_Q_lit*1e5,
                   s=30,color=SEED_COL[seed],alpha=.60,label=f'Seed {seed}',zorder=2)
    for _,r in sel.iterrows():
        k=r.selection
        _plot_selected(ax,k,[r.aep_gwh],[r.optimisation_sample_Q_lit*1e5],size=130)
    # A fixed contextual reference: not a member of the 24-point front.
    _plot_selected(ax,'reference',[804.3901377275079],[3.20618950134584],size=110,
                   label='Reference (context)')
    ax.set_xlabel('AEP (GWh/year)')
    ax.set_ylabel(r'Optimisation-sample $Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
    ax.legend(loc='upper left',ncol=2,frameon=True,fontsize=8)
    ax.set_xlim(803.1,824.6)
    ax.set_ylim(2.25,9.20)
    save(fig,out,'lit_front',record)


def main_layouts(master,out,record):
    from matplotlib.patches import Rectangle
    ref=read(master/'01_optimisation/seed_730101/optimisation/source/verification/runtime_inputs/energy_baseline.csv')
    pts={k:read(master/f'02_combined_pareto/{k}_coordinates_xy_m.csv') for k in LAYOUTS}
    pts['reference']=ref
    fig,axes=plt.subplots(2,2,figsize=(8.3,8.35),sharex=True,sharey=True)
    for i,(ax,k) in enumerate(zip(axes.flat,[*LAYOUTS,'reference'])):
        ds=pts[k]
        if len(ds)!=12: raise RuntimeError('Expected 12 turbines: '+k)
        ax.set_xlim(-200,6200);ax.set_ylim(-200,6200)
        ax.set_aspect('equal', adjustable='box')
        ax.add_patch(Rectangle((0,0),6000,6000,fill=False,edgecolor='#66717C',
                               linestyle='--',linewidth=1.2,zorder=1))
        for r in ds.itertuples():
            ax.add_patch(Circle((r.x_m,r.y_m),121.12,facecolor=COL[k],
                                edgecolor=COL[k],lw=.85,alpha=.21,zorder=2))
        _plot_selected(ax,k,ds.x_m,ds.y_m,size=20,label='_nolegend_',zorder=4)
        ax.text(.025,.98,f'({"abcd"[i]}) {LABEL[k]}',ha='left',va='top',
                transform=ax.transAxes,fontsize=11,fontweight='semibold',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=1.3))
        ax.set_xlabel('x position (m)')
        ax.set_ylabel('y position (m)')
        ax.set_xticks([0,2000,4000,6000]);ax.set_yticks([0,2000,4000,6000])
    # Bird heading is 225° (towards SW); arrow in a consistent site position.
    a=axes.flat[0]
    a.annotate('',xy=(4560,4300),xytext=(5580,5320),
               arrowprops=dict(arrowstyle='-|>',color='#3F5365',lw=2.0,mutation_scale=16))
    a.text(5680,5450,'Bird flight\n225°',color='#3F5365',fontsize=9,ha='right',va='bottom',
           bbox=dict(facecolor='white',edgecolor='none',alpha=.78,pad=1.3))
    fig.subplots_adjust(hspace=.19,wspace=.18)
    save(fig,out,'lit_layouts',record)


def main_uncertainty(master,out,record):
    from scipy.stats import gaussian_kde
    df=read(master/'03_independent_evaluation/final_four_guarded/guarded_fresh_batches.csv')
    df=df[np.isclose(df.response_probability,.6)]
    vals={k:df[df.layout==k].sort_values('batch_index')['Q_lit'].to_numpy()*1e5 for k in LAYOUTS}
    if any(len(v)!=10 for v in vals.values()): raise RuntimeError('Expected 10 independent batches per layout.')
    data=np.load(master/'05_uncertainty/bird_level_bootstrap/paired_10000_bird_bootstrap_replicates.npz')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.5,4.8),gridspec_kw=dict(width_ratios=[1,1.05]))
    ns=np.geomspace(1000,500000,180)
    for k in LAYOUTS:
        v=vals[k]
        sd=np.std(v,ddof=1)
        ax.plot(ns,sd*np.sqrt(50000/ns),color=COL[k],ls='--',lw=1.65,alpha=.79)
        points=[]
        for b in (1,2,3,4,5):
            aggregates=np.array([v[j*b:(j+1)*b].mean() for j in range(10//b)])
            if len(aggregates)>=2:points.append((b*50000,np.std(aggregates,ddof=1)))
        _plot_selected(ax,k,[x for x,y in points],[y for x,y in points],size=46,zorder=5)
    diff_labels=[('highest_aep','knee','H–K'),('knee','lowest_risk','K–L'),('highest_aep','lowest_risk','H–L')]
    for u,v,short in diff_labels:
        gap=(vals[u]-vals[v]).mean()
        ax.axhline(gap,color='#666666',alpha=.60,lw=.9)
        ax.text(1300,gap*1.06,f'{short}: {gap:.2f}',color='#444444',fontsize=10)
    ax.axvline(50000,color='#777777',ls=':',lw=1.2)
    # Optimisation-sample size is marked by the vertical dotted line and described below.
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(900,550000);ax.set_ylim(.016,7.2)
    ax.set_xlabel('Birds per estimate (N)')
    ax.set_ylabel(r'SD of estimated $Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
    ax.set_title('(a) Monte Carlo convergence',loc='left',fontsize=13)
    ax.text(.02,.02,'Markers: disjoint-batch SD\nDashed: $1/\\sqrt{N}$ scaling\nVertical: N = 50,000\nBelow 50,000: extrapolated',
            transform=ax.transAxes,ha='left',va='bottom',fontsize=9,color='#53606B',
            bbox=dict(facecolor='white',edgecolor='none',alpha=.91,pad=1.5))
    # Avoid duplicate legend entries in panel a: use a compact combined legend in b.
    for k in LAYOUTS:
        s=np.asarray(data[k],dtype=float)*1e5
        if len(s)!=10000:raise RuntimeError('Expected 10,000 resamples for '+k)
        grid=np.linspace(s.min()-3*np.std(s),s.max()+3*np.std(s),350)
        density=gaussian_kde(s)(grid)
        bx.plot(grid,density,color=COL[k],lw=1.9,label=LABEL[k])
        bx.fill_between(grid,density,color=COL[k],alpha=.10)
        bx.axvline(s.mean(),ls=':',lw=1.1,color=COL[k])
    bx.set_xlabel(r'Optimisation-sample $Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
    bx.set_ylabel('Bootstrap density')
    bx.set_title('(b) Bird-level paired bootstrap',loc='left',fontsize=13)
    bx.legend(loc='upper center',frameon=False,ncol=3,fontsize=10)
    # The N and resampling detail is in the locked LaTeX caption; avoid overprinting densities.
    for target in (ax,bx):
        target.tick_params(axis='both',labelsize=11)
        target.xaxis.label.set_size(12)
        target.yaxis.label.set_size(12)
    fig.subplots_adjust(wspace=.31)
    save(fig,out,'lit_uncertainty',record)


def main_sensitivity(master,out,record):
    specs=[('probability','Response probability',r'Response probability $p$',0.6),
           ('radius','Response radius','Response radius',2.0),
           ('hub_height','Hub height in bird model','Hub height (m)',150.0),
           ('bird_speed','Bird speed','Bird speed (m/s)',14.9)]
    fig,axs=plt.subplots(2,2,figsize=(10.8,7.6))
    for i,(ax,(dim,title,xlabel,baseline)) in enumerate(zip(axs.flat,specs)):
        df=read(master/f'04_sensitivities/{dim}/layout_means_by_scenario.csv')
        xs=sorted(float(x) for x in df.scenario_value.unique())
        for k in [*LAYOUTS,'reference']:
            ds=df[df.layout==k].copy()
            x=ds.scenario_value.astype(float).to_numpy()
            y=ds.mean_Q_lit.to_numpy()*1e5
            _plot_selected(ax,k,x,y,size=43,label=LABEL[k])
        ax.axvline(baseline,color='#8C9196',lw=.9,ls=':',zorder=0)
        ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
        ax.set_title(f'({"abcd"[i]}) {title}',loc='left',fontsize=11)
        if dim=='radius':ax.set_xticklabels(['0.5D','1D','2D'])
        if dim=='bird_speed':ax.set_xticklabels(['10.79','14.01','14.9'])
        # No connecting lines: only the discrete sensitivity levels were simulated.
    handles,labels=axs.flat[0].get_legend_handles_labels()
    uniq={label:handle for handle,label in zip(handles,labels)}
    fig.legend(uniq.values(),uniq.keys(),loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.005))
    fig.subplots_adjust(hspace=.38,wspace=.30,bottom=.12)
    save(fig,out,'lit_sensitivity',record)


def main_direction(master,out,record):
    df=read(master/'04_sensitivities/direction_fixed_width/layout_means_8_directions.csv')
    fig,ax=plt.subplots(figsize=(8.0,4.8))
    ax.axvspan(220,230,color='#ECE7C9',alpha=.83,zorder=0)
    for k in [*LAYOUTS,'reference']:
        ds=df[df.layout==k].sort_values('heading_deg_toward')
        if len(ds)!=8:raise RuntimeError('Expected eight headings for '+k)
        _plot_selected(ax,k,ds.heading_deg_toward.to_numpy(),
                       ds.mean_Q_lit.to_numpy()*1e5,size=78)
    ax.text(225,.975,'225° optimisation',rotation=90,fontsize=8.5,ha='right',va='top',
            color='#6B684F',transform=ax.get_xaxis_transform())
    ax.set_xticks(DIRECTIONS,[f'{i}°' for i in DIRECTIONS]);ax.set_xlim(-16,332)
    ax.set_xlabel('Bird-flight heading (towards)')
    ax.set_ylabel(r'$Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
    ax.legend(ncol=2,loc='upper center',frameon=True)
    save(fig,out,'lit_direction',record)


def write_final_guide(out, generated, previous):
    main_names=['lit_front','lit_layouts','lit_uncertainty','lit_sensitivity','lit_direction']
    if generated != main_names:raise RuntimeError('Unexpected main figure set: '+repr(generated))
    lines=['# Literature-informed metric — final figure exports','',
      'Figures were generated from verified archived HPC data; no optimiser, FLORIS or bird simulations were run.',
      'Master archive source and the earlier `08_figures` / `08_figures_thesis_v2` folders remain unchanged.','',
      '## Main-text figures — names match the locked Results LaTeX','',
      '- `lit_front.pdf`: 24-point pooled optimisation-sample front; contextual reference, no independent evaluation markers.',
      '- `lit_layouts.pdf`: four turbine-centre layouts, scaled illustrative rotor radii, dashed site border, 225° flight arrow.',
      '- `lit_uncertainty.pdf`: (a) empirical Monte Carlo SD and 1/sqrt(N) curves; (b) paired optimisation-sample bird-level bootstrap.',
      '- `lit_sensitivity.pdf`: four 2×2 parameter panels; markers only, dotted baseline settings, no misleading 0.5D shading.',
      '- `lit_direction.pdf`: fixed-width corridor (9696.481 m); eight observed headings; no interpolation.',
      '', '## Visual conventions', '',
      '- Highest-AEP: blue filled circle; knee: green filled square; lowest-risk: orange **open diamond**; reference: grey X.',
      '- AEP axis uses `AEP (GWh/year)`; x-coordinate axes use `x position (m)`.',
      '- No AEP recomputation in parameter or directional sensitivities.',
      '- In direction plot the values remain Q per entrant of the larger fixed-width corridor; these are not the original baseline-corridor Q values.',
      '- Density curves in the bootstrap panel are smoothed visualisations of 10,000 replicates.',
      '- Monte Carlo SD at N=250,000 uses two disjoint aggregates; curves below N=50,000 are theoretical extrapolation.',
      '', '## Ancillary plots (regenerated with matching palette)', '',
      *[f'- `{x}`' for x in previous], '',
      'All plots are saved as PDF, PNG and SVG in this new versioned directory.',
      'The original plotting source, original figures and scientific result files remain untouched.', '']
    (out/'FIGURES_README_FINAL.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--master',type=Path,default=Path.cwd())
    ap.add_argument('--check',action='store_true',help='Verify master inputs without making plots.')
    args=ap.parse_args(); master=args.master.resolve()
    if not (master/'archive_audit.json').is_file():raise RuntimeError('Not a verified master archive (missing archive_audit.json): '+str(master))
    expected=[
      '02_combined_pareto/combined_non_dominated_24.csv',
      '02_combined_pareto/final_selected_three.csv',
      '03_independent_evaluation/final_four_guarded/layout_means.csv',
      '03_independent_evaluation/final_four_guarded/guarded_fresh_batches.csv',
      '04_sensitivities/direction_fixed_width/layout_means_8_directions.csv',
      '05_uncertainty/bird_level_bootstrap/paired_10000_bird_bootstrap_replicates.npz',
      '06_numerical_resolution/aep_360sector/report.json',
    ]
    missing=[p for p in expected if not (master/p).is_file()]
    if missing:raise FileNotFoundError('Missing master inputs: '+repr(missing))
    selected=read(master/'02_combined_pareto/final_selected_three.csv')
    actual={(r.selection,int(r.seed),int(r.evaluation_index)) for r in selected.itertuples()}
    expected_selected={('highest_aep',730101,455),('knee',730103,418),('lowest_risk',730103,220)}
    if actual!=expected_selected:raise RuntimeError('Master selected layouts mismatch: '+repr(actual))
    nd=read(master/'02_combined_pareto/combined_non_dominated_24.csv')
    if len(nd)!=24:raise RuntimeError('Expected 24-point non-dominated set, got '+str(len(nd)))
    if args.check:
        print('PLOT PREFLIGHT PASS: verified master, 24 non-dominated layouts, '
              '455/418/220 selections and required inputs; original 08_figures is preserved.')
        return
    out,tables=setup(master);record=[]
    opt_front(master,out,record);layouts(master,out,record);confirm(master,out,tables,record)
    for k in ('probability','radius','hub_height','bird_speed'):sensitivity_one(master,out,tables,record,k)
    direction(master,out,tables,record);bootstrap(master,out,tables,record);montecarlo(master,out,tables,record)
    seed_exploration(master,out,tables,record);numerical(master,out,tables,record)
    prior_record=list(record)
    final_record=[]
    main_front(master,out,final_record)
    main_layouts(master,out,final_record)
    main_uncertainty(master,out,final_record)
    main_sensitivity(master,out,final_record)
    main_direction(master,out,final_record)
    write_final_guide(out,final_record,prior_record)
    (out/'FIGURES_README.txt').write_text(
      'Literature-informed figures v3 generated on HPC12 from saved final 72-sector data.\n'
      'This set is versioned in 08_figures_thesis_v3; original 08_figures and v2 remain untouched.\n'
      'All 14 figures are exported to PDF, SVG and PNG. See FIGURE_SELECTION_AND_CAPTIONS.md for placement and captions.\n'
      'Figure 03 error bars: ordinary individual 95% t intervals, not adjusted paired comparisons.\n'
      'Figure 06: eight observed directions only, plotted as separate markers; W=9696.481 m fixed.\n'
      'Figure 07: paired individual-bird bootstrap, N=50000, B=10000, entry strata preserved.\n'
      'Figure 08: empirical SD uses independent batches; dashed 1/sqrt(N) lines include extrapolation.\n'
      'AEP frozen at 72-sector objective; 360-sector AEP is only a verification.\n\n'
      'Figure sets:\n'+'\n'.join(record)+'\n')
    print('DONE: Generated',len(record),'auxiliary and',len(final_record),'main-text figure sets (PDF, SVG, PNG) in',out)

if __name__=='__main__':
    try:main()
    except Exception as e:print('PLOTTING FAILED:',str(e),file=sys.stderr);sys.exit(1)
