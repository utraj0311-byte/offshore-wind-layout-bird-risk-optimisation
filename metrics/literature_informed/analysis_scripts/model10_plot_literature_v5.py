#!/usr/bin/env python3
"""Two readability updates to final literature-informed metric figures, from frozen HPC data.

Creates a new 08_figures_thesis_v5 folder. Plots only lit_front and
lit_uncertainty. Copies lit_layouts, lit_sensitivity and lit_direction
from verified v4 output WITHOUT modifying any source files or scientific data.
Run: python model10_plot_literature_v5.py --master MODEL10_FINAL_72SECTOR_MASTER --check
Then: python model10_plot_literature_v5.py --master MODEL10_FINAL_72SECTOR_MASTER
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAYOUTS=['highest_aep','knee','lowest_risk']
LABEL={'highest_aep':'Highest-AEP','knee':'Knee','lowest_risk':'Lowest-risk','reference':'Reference'}
COL={'highest_aep':'#2166AC','knee':'#27964B','lowest_risk':'#E18D29','reference':'#777777'}
SEED_COL={730101:'#3E7A9A',730102:'#9871AC',730103:'#BE7950'}
MARK={'highest_aep':'o','knee':'s','lowest_risk':'D','reference':'X'}
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':260,'font.family':'DejaVu Sans',
'font.size':10,'axes.labelsize':11,'axes.titlesize':12,'legend.fontsize':9,
'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
'grid.color':'#D4DCE3','grid.alpha':.55,'axes.axisbelow':True,
'pdf.fonttype':42,'svg.fonttype':'none'})

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
    fig,ax=plt.subplots(figsize=(7.6,4.7))
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
    ax.tick_params(axis='both',labelsize=12)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.legend(loc='upper left',ncol=2,frameon=True,fontsize=11,markerscale=.78, labelspacing=.55, handletextpad=.7)
    ax.set_xlim(803.1,824.6)
    ax.set_ylim(2.25,9.20)
    save(fig,out,'lit_front',record)


def main_uncertainty(master,out,record):
    from scipy.stats import gaussian_kde
    df=read(master/'03_independent_evaluation/final_four_guarded/guarded_fresh_batches.csv')
    df=df[np.isclose(df.response_probability,.6)]
    vals={k:df[df.layout==k].sort_values('batch_index')['Q_lit'].to_numpy()*1e5 for k in LAYOUTS}
    if any(len(v)!=10 for v in vals.values()): raise RuntimeError('Expected 10 independent batches per layout.')
    data=np.load(master/'05_uncertainty/bird_level_bootstrap/paired_10000_bird_bootstrap_replicates.npz')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.5,5.15),gridspec_kw=dict(width_ratios=[1,1.05]))
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
        ax.text(1300,gap*1.06,f'{short}: {gap:.2f}',color='#444444',fontsize=12)
    ax.axvline(50000,color='#777777',ls=':',lw=1.2)
    # Optimisation-sample size is marked by the vertical dotted line and described below.
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(900,550000);ax.set_ylim(.016,7.2)
    ax.set_xlabel('Birds per estimate (N)')
    ax.set_ylabel(r'SD of estimated $Q_{\mathrm{lit}}$ ($\times 10^{-5}$)')
    ax.set_title('(a) Monte Carlo convergence',loc='left',fontsize=15)
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
    bx.set_title('(b) Bird-level paired bootstrap',loc='left',fontsize=15)
    bx.legend(loc='upper center',frameon=False,ncol=3,fontsize=11)
    # The N and resampling detail is in the locked LaTeX caption; avoid overprinting densities.
    for target in (ax,bx):
        target.tick_params(axis='both',labelsize=13)
        target.xaxis.label.set_size(14)
        target.yaxis.label.set_size(14)
    fig.subplots_adjust(wspace=.36)
    save(fig,out,'lit_uncertainty',record)


def main():
    import shutil
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--master',type=Path,default=Path.cwd())
    ap.add_argument('--check',action='store_true',help='Verify master and v4 figure inputs without plotting.')
    args=ap.parse_args(); master=args.master.resolve()
    if not (master/'archive_audit.json').is_file():
        raise RuntimeError('Not a verified master archive: '+str(master))
    expected=[
      '02_combined_pareto/combined_non_dominated_24.csv',
      '02_combined_pareto/final_selected_three.csv',
      '03_independent_evaluation/final_four_guarded/guarded_fresh_batches.csv',
      '05_uncertainty/bird_level_bootstrap/paired_10000_bird_bootstrap_replicates.npz',
    ]
    missing=[n for n in expected if not (master/n).is_file()]
    if missing:raise FileNotFoundError('Missing scientific inputs: '+repr(missing))
    nd=read(master/'02_combined_pareto/combined_non_dominated_24.csv')
    if len(nd)!=24:raise RuntimeError('Expected exactly 24 non-dominated layouts')
    selected=read(master/'02_combined_pareto/final_selected_three.csv')
    actual={(r.selection,int(r.seed),int(r.evaluation_index)) for r in selected.itertuples()}
    if actual!={('highest_aep',730101,455),('knee',730103,418),('lowest_risk',730103,220)}:
        raise RuntimeError('Selected-layout identities differ from frozen results')
    v4=master/'08_figures_thesis_v4'
    copied=['lit_layouts','lit_sensitivity','lit_direction']
    required=[v4/(f'{name}.{ext}') for name in copied for ext in ('pdf','png','svg')]
    if any(not f.is_file() for f in required):
        raise FileNotFoundError('v4 unchanged figures not found: '+repr([str(f) for f in required if not f.is_file()]))
    if args.check:
        print('PLOT PREFLIGHT PASS: 24 points, 455/418/220 selections, independent batches, bootstrap replicates, '
              'and unchanged v4 figure assets verified. No numerical results modified.')
        return
    out=master/'08_figures_thesis_v5';out.mkdir(exist_ok=True)
    # Only these two plots change: enlarge readable labels, remove duplicate note.
    changed=[]
    main_front(master,out,changed)
    main_uncertainty(master,out,changed)
    if changed!=['lit_front','lit_uncertainty']:
        raise RuntimeError('Unexpected changed plot set: '+str(changed))
    for src in required:
        dst=out/src.name
        shutil.copy2(src,dst)
        if dst.read_bytes()!=src.read_bytes():
            raise RuntimeError('Unchanged v4 asset did not copy byte-for-byte: '+str(src))
    readme=('Literature-informed v5: PRINT READABILITY ONLY\n'
      'Newly plotted: lit_front; lit_uncertainty.\n'
      'Unchanged byte-for-byte from v4: lit_layouts; lit_sensitivity; lit_direction.\n'
      'Each set includes PDF, PNG and SVG. v4 and original HPC source files are untouched.\n'
      'No optimisation, bird trajectories, FLORIS, or scientific calculations were rerun.\n'
      'The document caption defines H, K and L in uncertainty panel (a).\n'
      'The LaTeX front width is increased to 0.90\\linewidth and the uncertainty width to 0.98\\linewidth.\n'
      'The new folder has five main-text sets, not the 14 supplementary plot sets stored in v4.\n')
    (out/'README_V5.txt').write_text(readme,encoding='utf-8')
    print('DONE: 5 main figure sets in '+str(out)+'; 2 revised, 3 unchanged.')

if __name__=='__main__':
    try:main()
    except Exception as e:print('PLOTTING FAILED:',str(e),file=sys.stderr);sys.exit(1)
