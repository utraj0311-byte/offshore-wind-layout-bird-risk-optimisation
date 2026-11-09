"""Reproducible Metric 4a thesis figures with standardised sample terminology.

Read-only on scientific archives. Plot outputs only in a new separate figures folder.
Run: python metric4a_plot72_thesis_terms.py
Expected: this script in bird_layout_opt/metric4a_preflight/, alongside
bird_layout_opt/metric_4a_72sectors/ and ..._confirmation/.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OPT = ROOT / "metric_4a_72sectors"
CONF = ROOT / "metric_4a_72sectors_confirmation"
OUT = ROOT / "metric_4a_72sectors_figures_thesis_terms"
for required in [OPT/"pareto_points.csv", OPT/"highest-AEP_layout.csv", OPT/"lowest-risk_layout.csv", OPT/"hypervolume.csv", CONF/"confirmation_batches.csv", CONF/"confirmation_endpoint_summary.csv", CONF/"confirmation_paired_contrasts.csv", CONF/"confirmation_settings.json"]:
    if not required.is_file():
        raise FileNotFoundError(f"Missing required input: {required}")
if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite existing figures folder: {OUT}")
OUT.mkdir(parents=True)

FRONT = pd.read_csv(OPT/"pareto_points.csv")
BATCHES = pd.read_csv(CONF/"confirmation_batches.csv")
SUMMARY = pd.read_csv(CONF/"confirmation_endpoint_summary.csv")
CONTRAST = pd.read_csv(CONF/"confirmation_paired_contrasts.csv")
HV = pd.read_csv(OPT/"hypervolume.csv")
SETTINGS = json.loads((CONF/"confirmation_settings.json").read_text(encoding="utf-8"))
assert len(FRONT)==38, f"Expected 38 saved front rows; got {len(FRONT)}"
assert len(HV)==100 and HV.generation.iloc[-1]==100, "Missing convergence history"
assert list(sorted(BATCHES.seed.unique())) == list(range(20042,20062))
assert len(BATCHES)==80  # 20 batches x 2 layouts x 2 probabilities
assert (BATCHES.N_failed==0).all()
W = float(SETTINGS['common_width_m'])
assert np.isclose(W, 8787.57961404495, atol=1e-5)

COL = {'front':'#326CAB','energy':'#E58A1F','risk':'#C6443B','confirm':'#1B2635','gray':'#7A8694'}
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':10.5, 'axes.labelsize':11,
    'legend.fontsize':9.0, 'xtick.labelsize':9.0, 'ytick.labelsize':9.0,
    'axes.edgecolor':'#CBD3DB','axes.linewidth':0.8,
    'xtick.color':'#334155','ytick.color':'#334155',
    'savefig.facecolor':'white','figure.facecolor':'white'
})

def endpoint(name):
    row = SUMMARY[(SUMMARY.layout==name)&(SUMMARY.response_probability==0.6)].iloc[0]
    vals = BATCHES[(BATCHES.layout==name)&(BATCHES.response_probability==0.6)].sort_values('seed').Q_4a.to_numpy()
    mean=float(vals.mean()); hw=float(t.ppf(.975, len(vals)-1)*np.std(vals,ddof=1)/math.sqrt(len(vals)))
    assert abs(row.confirmed_Q_4a-mean)<1e-12
    return row,mean,hw

h,hm,hh=endpoint('highest-AEP'); l,lm,lh=endpoint('lowest-risk')
assert abs(h.aep_5deg_gwh-831.4654021246256)<1e-7
assert abs(l.aep_5deg_gwh-810.9579417748998)<1e-7

# Figure 1: original optimisation-sample Pareto front with independently
# evaluated endpoint Q_4a values (and per-endpoint 95% Monte Carlo CIs).
# Archive column names are retained only for reading legacy CSV files:
# 'E_enc' is the stored optimisation-sample Q_4a; 'training_Q_4a'
# is also an archived CSV column. Neither appears as a plotted label.
fig, ax=plt.subplots(figsize=(8.0,5.5))
ax.scatter(FRONT['E_enc'], FRONT.aep_gwh, s=40, color=COL['front'], alpha=.76,
           edgecolor='white',linewidth=.45,zorder=2,label='Pareto layouts — optimisation sample (n=38)')
for name,row,mean,hw,face,marker in [
    ('highest-AEP',h,hm,hh,COL['energy'],'o'),
    ('lowest-risk',l,lm,lh,COL['risk'],'*')
]:
    ax.scatter(row.training_Q_4a,row.aep_5deg_gwh,s=175 if marker=='*' else 105,
               c=face,marker=marker,edgecolor='white',linewidth=1.2,zorder=5,
               label=f'{name} — optimisation-sample Q_4a')
    ax.errorbar(mean,row.aep_5deg_gwh,xerr=hw,fmt='D',color=face,
                markerfacecolor='white',markeredgewidth=1.7,markersize=8,
                capsize=4,elinewidth=1.6,zorder=6,
                label=f'{name} — independently evaluated Q_4a (95% CI)')
ax.set_xlabel('Q_4a (encounters per entering bird)')
ax.set_ylabel('AEP (GWh/year)')
ax.grid(alpha=.2,linewidth=.7)
ax.set_axisbelow(True)
ax.margins(x=.10,y=.12)
ax.legend(loc='lower right',frameon=True,facecolor='white',edgecolor='#E1E6EC', fontsize=8.4)
fig.tight_layout(pad=1.6)
# Axis scores are conditioned on different cross-flight corridor widths.
fig.text(.5, .015, 'Independent evaluation sample uses a shared widened corridor (W = 8,787.58 m).',
         ha='center', va='bottom', fontsize=8.2, color=COL['gray'])
for ext in ('png','pdf'):
    fig.savefig(OUT/f'metric4a_front_independent_evaluation.{ext}',dpi=220,bbox_inches='tight',pad_inches=.2)
plt.close(fig)

# Figure 2: endpoint layouts; turbine numbers deliberately omitted.
theta=np.deg2rad(225)
d=np.array([np.cos(theta),np.sin(theta)])
def aligned_pairs(xy):
    pairs=[]
    for i in range(len(xy)):
        for j in range(i+1,len(xy)):
            vec=xy[j]-xy[i]
            dist=np.linalg.norm(vec)
            if dist<=8*242.24+1e-9:
                offaxis=np.abs(np.arctan2((d[0]*vec[1]-d[1]*vec[0]),np.dot(d,vec)))
                offaxis=min(offaxis, np.abs(np.pi-offaxis))
                if offaxis<=np.deg2rad(5)+1e-12:
                    pairs.append((i,j))
    return pairs

fig,axs=plt.subplots(1,2,figsize=(10.3,5.25),sharex=True,sharey=True)
for ax,name in zip(axs,['highest-AEP','lowest-risk']):
    xy=pd.read_csv(OPT/f'{name}_layout.csv')[['x_m','y_m']].to_numpy(float)
    assert xy.shape==(12,2)
    ax.add_patch(Rectangle((0,0),6000,6000,facecolor='#F8FAFC',edgecolor='#607186',linewidth=1.3,zorder=0))
    for x,y in xy:
        ax.add_patch(Circle((x,y),radius=242.24,facecolor=COL['front'],alpha=.055,
                            edgecolor=COL['front'],linewidth=.9,zorder=1))
    pairs=aligned_pairs(xy)
    for i,j in pairs:
        ax.plot(xy[[i,j],0],xy[[i,j],1],ls='--',c='#8795A5',lw=1.3,zorder=2)
    ax.scatter(xy[:,0],xy[:,1],s=58,c=COL['risk'] if name=='lowest-risk' else COL['energy'],
               edgecolor='white',linewidth=.8,zorder=3)
    ax.annotate('',xy=(5320,5280),xytext=(5890,5850),
                arrowprops=dict(arrowstyle='-|>',lw=2.1,color='#1F2937'))
    ax.text(5900,6110,'Bird flight 225°',fontsize=9.1,ha='right',color='#1F2937')
    ax.set_xlim(-400,6400); ax.set_ylim(-400,6400)
    ax.set_xticks([0,2000,4000,6000]); ax.set_yticks([0,2000,4000,6000])
    ax.set_xlabel('x (m)')
    ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.15)
    # Label figures without subplot titles.
    ax.text(.5,-.19,name,transform=ax.transAxes,ha='center',va='top',
            color='#273549',weight='bold',fontsize=11)
    print(f'{name}: {len(pairs)} aligned pairs')
    assert (len(pairs)==0 if name=='highest-AEP' else len(pairs)==7)
axs[0].set_ylabel('y (m)')
fig.subplots_adjust(left=.075,right=.985,top=.975,bottom=.23,wspace=.12)
for ext in ('png','pdf'):
    fig.savefig(OUT/f'metric4a_endpoint_layouts.{ext}',dpi=220,bbox_inches='tight',pad_inches=.18)
plt.close(fig)

# Figure 3: NSGA-II convergence.
fig,ax=plt.subplots(figsize=(7.5,4.4))
ax.plot(HV.generation,HV.normalised_hypervolume,c=COL['front'],lw=2.1)
ax.scatter([100],[HV.normalised_hypervolume.iloc[-1]],c=COL['energy'],s=65,zorder=4)
ax.set_xlabel('Generation')
ax.set_ylabel('Normalised hypervolume')
ax.set_xlim(1,102)
ax.grid(alpha=.2)
fig.tight_layout(pad=1.6)
for ext in ('png','pdf'):
    fig.savefig(OUT/f'metric4a_hypervolume.{ext}',dpi=220,bbox_inches='tight',pad_inches=.2)
plt.close(fig)

# Plain-text complete numeric summary with automatic CI check.
def contrast(p, metric):
    row=CONTRAST[(CONTRAST.response_probability.astype(str)==str(p)) & (CONTRAST.metric==metric)]
    assert len(row)==1, (p,metric)
    return row.iloc[0]
baseq=contrast('0.6','Q_4a'); basep=contrast('0.6','P_ge_1'); zeroq=contrast('0.0','Q_4a'); zerop=contrast('0.0','P_ge_1')
for p,cq,cp in [(0.6,baseq,basep),(0.,zeroq,zerop)]:
    v=BATCHES[BATCHES.response_probability==p].pivot(index='seed',columns='layout',values='Q_4a')
    delta=v['lowest-risk']-v['highest-AEP']
    assert np.isclose(delta.mean(),cq['mean'])

summary_lines=[
'Metric 4a | 72-sector run | Optimisation and independent evaluation samples',
'Input: archived NSGA-II optimisation and independent evaluation results',
'WARNING: horizontal encounter score, not rotor transit or bird mortality.',
'',
'Wind sectors: 72 (5 degrees); Pareto layouts: 38',
'Optimisation sample: 5 x 2,000 = 10,000 simulated birds per layout; seeds 42-46.' ,
'Independent evaluation sample: 20 x 5,000 = 100,000 birds per layout; paired seeds 20042-20061; p=0.6 and p=0.',
f'Widened common corridor width W={W:.6f} m for independent evaluation. Optimisation used the original narrower corridor.',
'',
'Endpoint | highest-AEP | lowest-risk',
'5-degree AEP (GWh/year): 831.465402 | 810.957942',
'1-degree AEP (GWh/year): 831.411575 | 810.999640',
'Optimisation-sample Q_4a: 0.2369 | 0.1058',
f'Independently evaluated Q_4a (p=0.6): {h.confirmed_Q_4a:.5f} | {l.confirmed_Q_4a:.5f}',
f'Independently evaluated P(>=1) (p=0.6): {h.confirmed_P_ge_1:.5f} | {l.confirmed_P_ge_1:.5f}',
f'Independently evaluated Q_4a x W (p=0.6): {h.confirmed_Q_4a_times_W:.3f} | {l.confirmed_Q_4a_times_W:.3f}',
'',
f'Independently evaluated paired Q_4a (lowest-risk - highest-AEP, p=.6): {baseq["mean"]:.5f}, 95% paired CI [{baseq["ci95_low"]:.6f}, {baseq["ci95_high"]:.6f}]',
f'Independently evaluated paired P(>=1) (lowest-risk - highest-AEP, p=.6): {basep["mean"]:.5f}, 95% paired CI [{basep["ci95_low"]:.6f}, {basep["ci95_high"]:.6f}]',
f'Independently evaluated paired Q_4a (lowest-risk - highest-AEP, p=0): {zeroq["mean"]:.5f}, 95% paired CI [{zeroq["ci95_low"]:.6f}, {zeroq["ci95_high"]:.6f}]',
f'Independently evaluated paired P(>=1) (lowest-risk - highest-AEP, p=0): {zerop["mean"]:.5f}, 95% paired CI [{zerop["ci95_low"]:.6f}, {zerop["ci95_high"]:.6f}]',
'',
'Conclusion: the original lack of independently evaluated Q_4a reduction does not repeat at 5 degrees in the common widened corridor.',
'Limits: only two endpoints independently evaluated. Confidence intervals describe sampling uncertainty, not optimiser coverage or ecological assumptions.',
'Use only highest-AEP and lowest-risk: endpoints are extrema of the optimisation-sample Pareto front.',
]
(OUT/'metric4a_results_summary.txt').write_text('\n'.join(summary_lines)+'\n',encoding='utf-8')
print('PASS | plots and summary saved in',OUT)
