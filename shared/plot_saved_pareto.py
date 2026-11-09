"""Create thesis-style Pareto figures from existing Jensen result folders.

POST-PROCESSING ONLY; never imports or launches the optimiser.

Usage, inside floris-env:
    python plot_saved_pareto.py "results/metric_1a_360sectors_seed42_pop80_gen100"

Works for 1a–1d. Saves PNG + PDF inside the supplied folder.
Does not overwrite existing figures unless --overwrite is supplied.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot(directory: Path, overwrite=False):
    table = pd.read_csv(directory / 'pareto_points.csv')
    anchor = pd.read_csv(directory / 'anchor_summary.csv')
    if not {'aep_gwh','risk'}.issubset(table.columns):
        raise ValueError('pareto_points.csv must have aep_gwh and risk columns')
    if not {'anchor','aep_gwh','risk'}.issubset(anchor.columns):
        raise ValueError('anchor_summary.csv must have anchor, aep_gwh and risk columns')
    table = table.sort_values(['aep_gwh','risk'])
    metric = directory.name.split('_')[1] if directory.name.startswith('metric_') else 'unknown'
    base = directory / f'metric_{metric}_pareto'
    dests=[base.with_suffix('.png'),base.with_suffix('.pdf')]
    if not overwrite and any(p.exists() for p in dests):
        raise FileExistsError(f'Pareto figure already exists: {[str(p) for p in dests if p.exists()]}')
    energy=anchor.loc[anchor.anchor.eq('energy_optimal')].iloc[0]
    lowrisk=anchor.loc[anchor.anchor.eq('collision_optimal')].iloc[0]
    coincide=(np.isclose(energy.aep_gwh,lowrisk.aep_gwh,atol=1e-9,rtol=0)
              and np.isclose(energy.risk,lowrisk.risk,atol=1e-9,rtol=0))

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(table.aep_gwh,table.risk, s=34, marker='o', color='#74818b',
               alpha=.75, label='Non-dominated layouts', zorder=2)
    # When both endpoint labels belong to one layout, superimpose a hollow
    # orange diamond and a smaller filled blue circle at the EXACT same point.
    ax.scatter([lowrisk.aep_gwh],[lowrisk.risk],s=148, marker='D',
               facecolor='none',edgecolor='#bb6a28',linewidth=1.7,zorder=4)
    ax.scatter([energy.aep_gwh],[energy.risk],s=74,marker='o',
               facecolor='#276aa0',edgecolor='black',linewidth=.65,zorder=5)
    handles=[
        Line2D([0],[0],marker='o',color='none',markerfacecolor='#74818b',
               markeredgecolor='#74818b',markersize=6,label='Non-dominated layouts'),
        Line2D([0],[0],marker='o',color='none',markerfacecolor='#276aa0',
               markeredgecolor='black',markeredgewidth=.65,markersize=8,label='Highest AEP'),
        Line2D([0],[0],marker='D',color='none',markerfacecolor='none',
               markeredgecolor='#bb6a28',markeredgewidth=1.5,markersize=9,label='Lowest risk'),
    ]
    ax.legend(handles=handles, frameon=False, loc='best', fontsize=9)
    ax.set_xlabel('AEP (GWh/year)')
    ax.set_ylabel('Artificial bird-activity score, Q')
    ax.grid(alpha=.18,linewidth=.7)
    ax.set_axisbelow(True)

    xs=table.aep_gwh.to_numpy(dtype=float)
    ys=table.risk.to_numpy(dtype=float)
    dx=np.ptp(xs); dy=np.ptp(ys)
    if dx < 1e-8: dx=max(abs(xs[0])*.012, 5)
    if dy < 1e-8: dy=max(abs(ys[0])*.08, 1)
    ax.set_xlim(float(np.min(xs)-dx*.25),float(np.max(xs)+dx*.25))
    ax.set_ylim(float(np.min(ys)-dy*.25),float(np.max(ys)+dy*.25))

    if coincide:
        ax.annotate(f'Both endpoints coincide\nAEP = {energy.aep_gwh:.3f} GWh/year\nQ = {energy.risk:,.0f}',
                    (energy.aep_gwh,energy.risk), xytext=(28,-47),
                    textcoords='offset points',fontsize=9,
                    arrowprops={'arrowstyle':'-','color':'#606060','lw':.7})
        ax.legend(handles=handles,frameon=False,loc='upper left',fontsize=9)
    fig.tight_layout()
    for target in dests:
        fig.savefig(target,dpi=240,bbox_inches='tight')
        print('Saved',target)
    plt.close(fig)
    return dests


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('results_folder',type=Path)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    plot(args.results_folder,args.overwrite)
