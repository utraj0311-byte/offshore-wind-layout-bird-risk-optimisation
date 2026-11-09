"""Metric 4b | final thesis figures and independent-evaluation audit.

Run on user's laptop:
  conda activate floris-env
  cd /d C:\\Users\\Deepak\\Desktop\\bird_layout_opt\\metric4b_preflight
  python metric4b_plot72_final.py

Reads sibling folders metric_4b_72sectors/ and
metric_4b_72sectors_confirmation/. NO simulations/optimisation.
Creates a NEW sibling folder metric_4b_72sectors_figures_final.

Scientific note: independent-evaluation Q_4b, confidence intervals, and paired
contrasts are multiplied by W_ind / W_opt for FIGURE comparison only.
This is width-normalisation, NOT a fresh re-simulation on the original corridor.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OPT = ROOT / 'metric_4b_72sectors'
CONF = ROOT / 'metric_4b_72sectors_confirmation'
OUT = ROOT / 'metric_4b_72sectors_figures_final'

INPUTS = [
    OPT/'pareto_points.csv', OPT/'highest-AEP_layout.csv',
    OPT/'lowest-risk_layout.csv', OPT/'hypervolume.csv',
    OPT/'completion_status.json',
    CONF/'independent_evaluation_endpoint_summary.csv',
    CONF/'independent_evaluation_paired_differences.csv',
    CONF/'independent_evaluation_batches.csv',
    CONF/'independent_evaluation_10000_subsamples.csv',
    CONF/'evaluation_settings.json',
]
for f in INPUTS:
    if not f.is_file():
        raise FileNotFoundError(f'Missing input: {f}')
if OUT.exists():
    raise FileExistsError(
        f'Refusing to overwrite previous figures: {OUT}. Rename/move it first.')

front = pd.read_csv(OPT/'pareto_points.csv')
hv = pd.read_csv(OPT/'hypervolume.csv')
summary = pd.read_csv(CONF/'independent_evaluation_endpoint_summary.csv')
contrasts = pd.read_csv(CONF/'independent_evaluation_paired_differences.csv')
batches = pd.read_csv(CONF/'independent_evaluation_batches.csv')
subsamples = pd.read_csv(CONF/'independent_evaluation_10000_subsamples.csv')
settings = json.loads((CONF/'evaluation_settings.json').read_text(encoding='utf-8'))
complete = json.loads((OPT/'completion_status.json').read_text(encoding='utf-8'))
W0 = float(settings['corridor']['original_width'])
W1 = float(settings['corridor']['common_width'])
SCALE = W1/W0
LAYOUTS = ('highest-AEP', 'lowest-risk')
SCENARIOS = ('baseline_p06','no_response_p0','fixed_height_p06','fixed_height_p0')
assert settings['status'] == 'completed'
assert len(front)==36 and len(hv)==100
assert len(batches)==15*4*2 and len(subsamples)==150*4*2
assert len(summary)==8 and len(contrasts)==4
assert sorted(batches.trajectory_seed.unique().tolist())==list(range(30142,30157))
assert sorted(batches.flight_height_seed.unique().tolist())==list(range(40142,40157))
assert abs(W0-8485.28137423857)<1e-6
assert abs(W1-8577.079595624595)<1e-6
assert batches.n_started.eq(100000).all() and batches.n_completed.eq(100000).all()
assert subsamples.n_started.eq(10000).all() and subsamples.n_completed.eq(10000).all()
for name in LAYOUTS:
    file = OPT/(name+'_layout.csv')
    assert hashlib.sha256(file.read_bytes()).hexdigest() == settings['two_layouts'][name]
assert np.isclose(front.aep_gwh.max(), 828.668218644, atol=1e-6)
assert np.isclose(front.Q_4b.min(), 0.0007365695079398, atol=1e-14)

# Recompute paired CIs from the 15 *independent evaluation* batches.
recomputed = {}
for sc in SCENARIOS:
    for name in LAYOUTS:
        b = batches[(batches.scenario==sc)&(batches.layout==name)].sort_values('batch_index')
        s = subsamples[(subsamples.scenario==sc)&(subsamples.layout==name)]
        assert len(b)==15 and len(s)==150
        vals = b.Q_4b.to_numpy(float)
        ci = t.ppf(.975,14)*vals.std(ddof=1)/math.sqrt(15)
        row = summary[(summary.scenario==sc)&(summary.layout==name)]
        assert len(row)==1
        row = row.iloc[0]
        avg=float(vals.mean())
        assert np.isclose(avg,row.independently_evaluated_Q_4b,atol=1e-12,rtol=0)
        assert np.isclose(avg-ci,row.ci95_lower,atol=1e-12,rtol=0)
        assert np.isclose(avg+ci,row.ci95_upper,atol=1e-12,rtol=0)
        sd10k=float(s.Q_4b.std(ddof=1))
        assert np.isclose(sd10k,row.sd_one_10000_bird_sample,atol=1e-12,rtol=0)
        recomputed[(sc,name)]={'mean':avg,'ci':float(ci),'sd10k':sd10k,'row':row}
    h=batches[(batches.scenario==sc)&(batches.layout=='highest-AEP')].sort_values('batch_index').Q_4b.to_numpy(float)
    l=batches[(batches.scenario==sc)&(batches.layout=='lowest-risk')].sort_values('batch_index').Q_4b.to_numpy(float)
    dif=l-h
    centre=float(dif.mean()); ci=float(t.ppf(.975,14)*dif.std(ddof=1)/math.sqrt(15))
    row=contrasts[contrasts.scenario==sc].iloc[0]
    assert np.isclose(centre,row.Q_4b_difference,atol=1e-12,rtol=0)
    assert np.isclose(centre-ci,row.ci95_lower,atol=1e-12,rtol=0)
    assert np.isclose(centre+ci,row.ci95_upper,atol=1e-12,rtol=0)
    recomputed[(sc,'contrast')]={'mean':centre,'ci':ci}

# Read the 5-degree optimisation endpoints. The optimisation-sample points are not rescaled.
eps={
    'highest-AEP': front.loc[front.aep_gwh.idxmax()],
    'lowest-risk': front.loc[front.Q_4b.idxmin()],
}
for name in LAYOUTS:
    r=recomputed[('baseline_p06',name)]['row']
    assert np.isclose(float(eps[name].Q_4b),float(r.optimisation_sample_Q_4b),atol=1e-12,rtol=0)

COL={'front':'#326CAB','energy':'#E58A1F','risk':'#C6443B','outline':'#243449',
     'gray':'#64748B','light':'#DFE6ED','purple':'#5B58A6'}
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':10.3,'mathtext.fontset':'dejavusans',
    'axes.labelsize':11.4, 'legend.fontsize':8.6,
    'xtick.labelsize':9.4,'ytick.labelsize':9.4,
    'axes.edgecolor':'#CBD3DB', 'axes.linewidth':.8,
    'xtick.color':'#334155','ytick.color':'#334155',
    'figure.facecolor':'white','savefig.facecolor':'white',
    'pdf.fonttype':42,
})
OUT.mkdir(parents=True,exist_ok=False)

def save(fig,stem):
    fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight',pad_inches=.24)
    fig.savefig(OUT/(stem+'.png'),dpi=230,bbox_inches='tight',pad_inches=.24)
    plt.close(fig)

# Figure 1: Pareto front (Q horizontal, AEP vertical). 5-degree AEP, 72 sectors.
fig,ax=plt.subplots(figsize=(8.25,5.75))
ax.scatter(front.Q_4b,front.aep_gwh,s=42,color=COL['front'],edgecolor='white',
           linewidth=.6,alpha=.83,zorder=2,label='Pareto layouts (optimisation sample)')
for name,marker,color,sz in [('highest-AEP','o',COL['energy'],115),
                             ('lowest-risk','*',COL['risk'],205)]:
    p=eps[name]
    d=recomputed[('baseline_p06',name)]
    x=d['mean']*SCALE; xc=d['ci']*SCALE
    ax.plot([p.Q_4b,x],[p.aep_gwh,p.aep_gwh],ls=':',lw=1.1,color=color,alpha=.7,zorder=3)
    ax.scatter([p.Q_4b],[p.aep_gwh],s=sz,marker=marker,color=color,
               edgecolor='white',linewidth=1.1,zorder=5,label=f'{name}: optimisation sample')
    ax.errorbar([x],[p.aep_gwh],xerr=[xc],fmt='D',color=color,
                mfc='white',mec=color,mew=1.65,ms=8,capsize=4,elinewidth=1.4,
                zorder=6,label=f'{name}: independent evaluation (95% CI)')
ax.set_xlabel(r'$Q_{4\mathrm{b}}$ (rotor transits per entering bird)')
ax.set_ylabel('AEP (GWh/year)')
ax.grid(alpha=.18,linewidth=.7); ax.set_axisbelow(True)
ax.margins(x=.15,y=.13)
ax.legend(loc='lower right',frameon=True,facecolor='white',edgecolor=COL['light'],
          fontsize=8.1,labelspacing=.65)
fig.tight_layout(pad=1.4);save(fig,'metric4b_pareto_independent_evaluation')

# Figure 2: Each endpoint optimisation sample vs corridor-width-adjusted independent evaluation.
fig,axes=plt.subplots(1,2,figsize=(9.65,5.15))
for ax,name,panel,color in zip(axes,LAYOUTS,'ab',(COL['energy'],COL['risk'])):
    opt=float(eps[name].Q_4b)
    d=recomputed[('baseline_p06',name)]; adj=d['mean']*SCALE; ci=d['ci']*SCALE
    ax.plot([0,1],[opt,adj],color=color,lw=1.35,alpha=.6,zorder=2)
    ax.scatter([0],[opt],s=95,marker='o',facecolor=color,edgecolor='white',zorder=4)
    ax.errorbar([1],[adj],yerr=[ci],fmt='D',color=color,mfc='white',mec=color,
                mew=1.7,ms=8.7,elinewidth=1.5,capsize=5,zorder=5)
    ax.set_xlim(-.33,1.33);ax.margins(y=.3)
    ax.set_xticks([0,1],labels=['Optimisation\nsample','Independent\nevaluation'])
    ax.set_ylabel(r'$Q_{4\mathrm{b}}$ (rotor transits per entering bird)')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    ax.text(.5,-.2,f'({panel}) {name}',transform=ax.transAxes,ha='center',va='top',
            fontsize=10.5,weight='bold',color=COL['outline'])
fig.subplots_adjust(left=.11,right=.98,top=.95,bottom=.25,wspace=.38)
save(fig,'metric4b_optimisation_vs_independent')

# Figure 3: schematic turbine plan, actual centre positions and aligned pairs.
D=242.24;R=D/2;TOL=1e-9
rad=np.deg2rad(225.0);direction=np.array([np.cos(rad),np.sin(rad)])
def aligned_pairs(xy):
    found=[]
    for i in range(len(xy)):
        for j in range(i+1,len(xy)):
            dv=xy[j]-xy[i]; dist=np.linalg.norm(dv)
            if dist>8*D+TOL: continue
            cosang=abs(float(np.dot(dv,direction)/dist))
            if cosang>=np.cos(np.deg2rad(5)) - 1e-12:
                found.append((i,j))
    return found
fig,axes=plt.subplots(1,2,figsize=(10.8,5.95),sharex=True,sharey=True)
for ax,name,panel,color,expected in zip(axes,LAYOUTS,'ab',
                                   (COL['energy'],COL['risk']),(0,5)):
    xy=pd.read_csv(OPT/(name+'_layout.csv'))[['x_m','y_m']].to_numpy(float)
    assert xy.shape==(12,2)
    ax.add_patch(Rectangle((0,0),6000,6000,facecolor='#F8FAFC',edgecolor=COL['gray'],linewidth=1.15,zorder=0))
    for x,y in xy:
        ax.add_patch(Circle((x,y),R,facecolor=COL['front'],alpha=.09,
                            edgecolor=COL['front'],linewidth=.8,zorder=1))
    pairs=aligned_pairs(xy)
    assert len(pairs)==expected,(name,len(pairs),expected)
    for i,j in pairs:
        ax.plot(xy[[i,j],0],xy[[i,j],1],ls='--',color='#8B94A3',lw=1.5,zorder=2)
    ax.scatter(xy[:,0],xy[:,1],s=60,c=color,edgecolor='white',lw=.8,zorder=3)
    ax.annotate('',xy=(5180,5120),xytext=(5760,5700),
                arrowprops={'arrowstyle':'-|>','lw':2.0,'color':COL['outline']})
    ax.text(5860,6090,'Bird flight 225°',ha='right',va='bottom',fontsize=9.4,color=COL['outline'])
    ax.set_xlim(-425,6440);ax.set_ylim(-425,6460)
    ax.set_xticks([0,2000,4000,6000]);ax.set_yticks([0,2000,4000,6000])
    ax.set_xlabel('x position (m)');ax.set_ylabel('y position (m)')
    ax.grid(alpha=.14);ax.set_aspect('equal',adjustable='box')
    ax.text(.5,-.18,f'({panel}) {name}',transform=ax.transAxes,ha='center',va='top',
            fontsize=11,weight='bold',color=COL['outline'])
fig.subplots_adjust(left=.08,right=.985,top=.965,bottom=.23,wspace=.2)
save(fig,'metric4b_endpoint_layouts')

# Figure 4: convergence history, not modified by confirmation or width normalisation.
fig,ax=plt.subplots(figsize=(7.7,4.5))
ax.plot(hv.generation,hv.hypervolume,color=COL['front'],lw=2.1)
ax.scatter([100],[float(hv.hypervolume.iloc[-1])],s=52,c=COL['energy'],zorder=4)
ax.set_xlabel('Generation');ax.set_ylabel('Hypervolume')
ax.set_xlim(1,101);ax.margins(y=.14)
ax.grid(alpha=.2);ax.set_axisbelow(True)
fig.tight_layout(pad=1.3);save(fig,'metric4b_hypervolume')

# Figure 5: comparison across two controls (95% confidence intervals per layout).
fig,axes=plt.subplots(1,2,figsize=(10.0,4.9))
for ax,fhd,scenarios,panel in [
    (axes[0],'Northern-gannet FHD',('no_response_p0','baseline_p06'),'a'),
    (axes[1],'Fixed height, z = 150 m',('fixed_height_p0','fixed_height_p06'),'b')]:
    for name,color,marker in [('highest-AEP',COL['energy'],'o'),('lowest-risk',COL['risk'],'D')]:
        ys=[recomputed[(sc,name)]['mean'] for sc in scenarios]
        errs=[recomputed[(sc,name)]['ci'] for sc in scenarios]
        ax.errorbar([0,0.6],ys,yerr=errs,fmt=marker+'-',color=color,
                    lw=1.55,ms=6.8,capsize=4,elinewidth=1.35,label=name)
    ax.set_xticks([0,.6]);ax.set_xlim(-.09,.69)
    ax.set_xlabel('Response probability, p')
    ax.set_ylabel(r'$Q_{4\mathrm{b}}$ (rotor transits per entering bird)')
    ax.grid(alpha=.18);ax.set_axisbelow(True);ax.margins(y=.12)
    ax.text(.5,-.23,f'({panel}) {fhd}',transform=ax.transAxes,ha='center',va='top',
            fontsize=10.3,weight='bold',color=COL['outline'])
axes[0].legend(loc='upper right',frameon=True,edgecolor=COL['light'],fontsize=8.3)
fig.subplots_adjust(left=.10,right=.985,top=.96,bottom=.29,wspace=.3)
save(fig,'metric4b_response_controls')

# Full precision machine-readable result tables and captions.
rows=[]
for sc in SCENARIOS:
    for name in LAYOUTS:
        d=recomputed[(sc,name)]
        opt_q=float(eps[name].Q_4b) if sc=='baseline_p06' else float('nan')
        rows.append({
            'scenario':sc,'layout':name,
            'optimisation_sample_Q_4b':opt_q,
            'independently_evaluated_Q_4b_original_widened_corridor':d['mean'],
            'independent_95CI_lower':d['mean']-d['ci'],
            'independent_95CI_upper':d['mean']+d['ci'],
            'independent_Q_4b_adjusted_to_original_corridor_width':d['mean']*SCALE,
            'adjusted_95CI_lower':(d['mean']-d['ci'])*SCALE,
            'adjusted_95CI_upper':(d['mean']+d['ci'])*SCALE,
            'adjusted_percent_change_from_optimisation_sample':
                100*(d['mean']*SCALE/opt_q-1) if sc=='baseline_p06' else float('nan'),
            'independently_evaluated_Q_4b_times_W':d['mean']*W1,
            'sd_single_10000_bird_sample':d['sd10k'],
        })
pd.DataFrame(rows).to_csv(OUT/'metric4b_full_precision_endpoint_results.csv',index=False)
paired=[]
for sc in SCENARIOS:
    d=recomputed[(sc,'contrast')]
    paired.append({'scenario':sc,'paired_difference_lowest_risk_minus_highest_AEP':d['mean'],
                   '95CI_lower':d['mean']-d['ci'],'95CI_upper':d['mean']+d['ci'],
                   'adjusted_difference_W_ind_over_W_opt':d['mean']*SCALE,
                   'adjusted_95CI_lower':(d['mean']-d['ci'])*SCALE,
                   'adjusted_95CI_upper':(d['mean']+d['ci'])*SCALE})
pd.DataFrame(paired).to_csv(OUT/'metric4b_paired_differences.csv',index=False)

n=len(front);a1=float(eps['highest-AEP'].aep_gwh);a2=float(eps['lowest-risk'].aep_gwh)
mainhi=recomputed[('baseline_p06','highest-AEP')]['mean']
mainlo=recomputed[('baseline_p06','lowest-risk')]['mean']
red=100*(mainhi-mainlo)/mainhi
txt=f'''# Metric 4b - final figure captions and thesis results\n\n## Figure captions\n\n**metric4b_pareto_independent_evaluation.pdf.** Pareto layouts obtained with NSGA-II using 72 wind-direction sectors (5-degree resolution). Horizontal positions of the blue points and filled endpoints show optimisation-sample $Q_{{4\\mathrm{{b}}}}$ (10,000 birds per candidate). Open diamonds show the independently evaluated endpoints (15 batches of 100,000 birds per layout), with horizontal 95% confidence intervals, at the baseline response probability $p=0.6$. The independent-evaluation values and intervals are multiplied by the ratio of the independent-evaluation corridor width ({W1:.6f} m) to the optimisation corridor width ({W0:.6f} m), a factor of {SCALE:.9f}, to put the scores on a common width basis. This is a width normalisation, not a re-simulation within the original corridor. AEP corresponds to the 5-degree FLORIS calculations. Only the endpoints were independently evaluated.\n\n**metric4b_optimisation_vs_independent.pdf.** Optimisation-sample and corridor-width-adjusted independent-evaluation $Q_{{4\\mathrm{{b}}}}$ for (a) highest-AEP and (b) lowest-risk layouts. Whiskers show 95% confidence intervals on the independent-evaluation sample of 15 paired batches of 100,000 birds per layout. Optimisation samples contain 10,000 birds per layout. See previous caption for corridor-width normalisation.\n\n**metric4b_endpoint_layouts.pdf.** Turbine-centre plans for (a) highest-AEP and (b) lowest-risk endpoints; bird travel is towards 225 degrees (south-west). Light circles indicate rotor-diameter scale schematically, not horizontal rotor planes. Dashed segments highlight turbine-centre pairs separated by at most 8D and aligned within +/-5 degrees of the 45-225-degree bird-travel axis. There are 0 and 5 such pairs, respectively. Turbines are unnumbered.\n\n**metric4b_hypervolume.pdf.** Normalised multi-objective hypervolume across the saved 100 generations at 5-degree wind-direction resolution; displayed values are taken from the optimisation archive.\n\n**metric4b_response_controls.pdf.** Paired independent-evaluation $Q_{{4\\mathrm{{b}}}}$ by response probability, for (a) the northern-gannet flight-height distribution and (b) fixed $z=150$ m. Both layouts use the same widened entry corridor of {W1:.6f} m and paired seeds. Whiskers denote 95% confidence intervals across the 15 independent evaluation batches; values are shown unadjusted to preserve each control's original reported metric.\n\n## Results text (ready to edit for thesis)\n\nThe 72-sector Metric 4b optimisation yielded {n} mutually non-dominated layouts. The highest-AEP endpoint produced {a1:.4f} GWh/year and the lowest-risk endpoint {a2:.4f} GWh/year at 5-degree wind-direction resolution. The lowest-risk layout therefore sacrificed {a1-a2:.4f} GWh/year ({100*(a1-a2)/a1:.2f}%) relative to the highest-AEP layout. Both endpoints met the 1-degree AEP convergence criterion, with deviations of 0.01820% and 0.01165%, respectively.\n\nOn the optimisation sample of 10,000 birds per layout, $Q_{{4\\mathrm{{b}}}}$ was {float(eps['highest-AEP'].Q_4b):.8f} for highest-AEP and {float(eps['lowest-risk'].Q_4b):.8f} for lowest-risk. The independently evaluated values on a shared widened entry corridor (15 batches of 100,000 birds per layout) were {mainhi:.8f} and {mainlo:.8f}, respectively. The paired difference (lowest-risk minus highest-AEP) was {recomputed[('baseline_p06','contrast')]['mean']:.8f} with a 95% confidence interval [{recomputed[('baseline_p06','contrast')]['mean']-recomputed[('baseline_p06','contrast')]['ci']:.8f}, {recomputed[('baseline_p06','contrast')]['mean']+recomputed[('baseline_p06','contrast')]['ci']:.8f}]. This represents {red:.2f}% fewer independently evaluated expected rotor transits per entering bird at the lowest-risk endpoint under the specified behavioural assumptions.\n\nThe highest-AEP layout extended 89.798 m beyond the original rotor-margin corridor, motivating a common {W1:.3f} m corridor for independent evaluation. For direct visual comparison with optimisation-sample values, the independent-evaluation scores were width-normalised to the original {W0:.3f} m corridor; the adjusted values are {mainhi*SCALE:.8f} and {mainlo*SCALE:.8f}. They exceeded their respective optimisation-sample values by {100*(mainhi*SCALE/float(eps['highest-AEP'].Q_4b)-1):.2f}% and {100*(mainlo*SCALE/float(eps['lowest-risk'].Q_4b)-1):.2f}%. Width-normalisation should not be interpreted as a new independent evaluation within the narrower corridor.\n\nThe no-response control ($p=0$) produced practically identical means ({recomputed[('no_response_p0','highest-AEP')]['mean']:.8f} and {recomputed[('no_response_p0','lowest-risk')]['mean']:.8f}), with a paired difference whose 95% interval includes zero. Thus, when all rotor zones are covered, expected total rotor transits of straight-flying birds are approximately layout-independent. At $p=0.6$, the layout difference emerges, consistent with the interaction between turbine arrangement and the prescribed meso-response rule. The lowest-risk layout had five closely aligned turbine pairs along the bird-flight axis compared with none in the highest-AEP layout. The fixed-height $z=150$ m experiment also retained a reduction at $p=0.6$ (paired difference {recomputed[('fixed_height_p06','contrast')]['mean']:.6f}), whereas at $p=0$ the difference was indistinguishable from zero.\n\nThe new endpoint trade-off survived independent evaluation; therefore the earlier observation that the apparent rotor-transit trade-off disappeared does not repeat for these two selected endpoints. However, only two of the {n} non-dominated layouts received this independent evaluation, and $Q_{{4\\mathrm{{b}}}}$ measures expected rotor-disc transits rather than blade strikes or collision mortality. The Monte Carlo intervals quantify sampling uncertainty for the frozen model and chosen layouts, not uncertainty in the biological response assumptions.\n'''
(OUT/'metric4b_thesis_captions_and_results.md').write_text(txt,encoding='utf-8')
report={
    'source':{'opt':str(OPT),'confirmation':str(CONF)},
    'n_pareto':n,'n_evaluations':8000,'sector_count':72,
    'original_corridor_width_m':W0,'independent_corridor_width_m':W1,
    'width_normalisation_factor':SCALE,'paired_CI_method':'Student t interval on 15 batch-wise paired differences',
    'one_sample_SD_method':'sample SD across 150 disjoint groups of 10,000 birds',
    'independent_batches':15,'birds_per_batch':100000,
    'baseline_reduction_percent':red,
    'aep_highest_GWh':a1,'aep_lowest_GWh':a2,
    'result_qualification':'Only endpoints independently evaluated; width normalisation is not a new simulation',
}
(OUT/'figure_generation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('PASS | source audit and 15-batch paired statistics')
print(f'PASS | 36 Pareto layouts; corridor width factor {SCALE:.9f}')
for n in LAYOUTS:
 d=recomputed[('baseline_p06',n)]; optq=float(eps[n].Q_4b)
 print(f'   {n}: optimisation sample={optq:.10f}; independent adjusted={d["mean"]*SCALE:.10f}; '
       f'change={100*(d["mean"]*SCALE/optq-1):+.2f}%')
print('PASS | 5 figure PDFs and 5 PNGs, plus tables and thesis captions')
print('Output:',OUT)
