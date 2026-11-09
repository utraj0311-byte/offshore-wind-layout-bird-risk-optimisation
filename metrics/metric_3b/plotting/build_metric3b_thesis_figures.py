#!/usr/bin/env python3
"""Figure-only exports of frozen 72-sector Metric 3b results.

Inputs: final non-dominated front and saved decision vectors from the existing\n72-sector results directory, or its frozen ZIP archive.
Does not run FLORIS, optimise, or change saved coordinates.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

PREFIX='heading225_W6000_seed48_pop160_gen250/'
D=242.24
SITE=6000.0
RAD=np.deg2rad(225.0)
ALONG=np.array([np.cos(RAD),np.sin(RAD)])
CROSS=np.array([-np.sin(RAD),np.cos(RAD)])

C_POINTS='#8e9ba8'
C_ENERGY='#2476af'
C_RISK='#d47c24'
BANDS=['#4e87ae','#67a485','#f2af63','#b492c6','#dc8684','#a4bd72','#c4a067','#719ec4']


def style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':9.0,'axes.labelsize':9.5,
        'xtick.labelsize':8.4,'ytick.labelsize':8.4,'axes.linewidth':0.8,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})


def read_front(source: Path):
    """Read the completed Metric 3b front from an existing run folder or frozen ZIP."""
    if source.is_dir():
        p = pd.read_csv(source / 'pareto_points.csv')
        v = np.load(source / 'pareto_decision_vectors.npy')
    elif source.is_file() and source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as z:
            p = pd.read_csv(io.BytesIO(z.read(PREFIX + 'pareto_points.csv')))
            v = np.load(io.BytesIO(z.read(PREFIX + 'pareto_decision_vectors.npy')))
    else:
        raise FileNotFoundError(
            f'Metric 3b source not found: {source}\n'
            'Supply the existing run directory or the frozen run ZIP using --archive.'
        )
    if len(p)!=126 or v.shape!=(126,24):
        raise RuntimeError('Unexpected Metric 3b archive dimensions')
    if not np.all(np.diff(p['aep_gwh'].to_numpy())>=-1e-8):
        raise RuntimeError('Archive AEP order unexpected')
    h=int(p['aep_gwh'].idxmax()); l=int(p['N_eff'].idxmin())
    checks=[(h,832.3277511113436,9.530964530757,8),
            (l,775.0639685653775,2.0,2)]
    for i,a,q,b in checks:
        if abs(float(p.loc[i,'aep_gwh'])-a)>1e-5:raise RuntimeError('Endpoint AEP mismatch')
        if abs(float(p.loc[i,'N_eff'])-q)>1e-7:raise RuntimeError('Endpoint metric mismatch')
        if int(p.loc[i,'number_of_merged_intervals'])!=b:raise RuntimeError('Endpoint band mismatch')
    return p,v,h,l


def save(fig,out: Path,stem: str):
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/(stem+'.pdf'),bbox_inches='tight',pad_inches=0.06)
    fig.savefig(out/(stem+'.png'),dpi=260,bbox_inches='tight',pad_inches=0.06)
    plt.close(fig)


def draw_front(front,idx_energy,idx_risk,out):
    fig,ax=plt.subplots(figsize=(4.55,3.55),dpi=170)
    ax.scatter(front.aep_gwh,front.N_eff, s=19,color=C_POINTS,
               edgecolor='none',alpha=.77,zorder=2)
    h=front.iloc[idx_energy];l=front.iloc[idx_risk]
    ax.scatter([h.aep_gwh],[h.N_eff],s=95,c=C_ENERGY,
               marker='o',edgecolors='#183f55',linewidths=.8,zorder=5)
    ax.scatter([l.aep_gwh],[l.N_eff],s=105,facecolors='none',
               edgecolors=C_RISK,marker='D',linewidths=2.0,zorder=6)
    ax.set_xlabel('AEP (GWh/year)')
    ax.set_ylabel(r'$Q_{3b}$ (rotor diameters)')
    ax.set_xlim(772.0,835.5)
    ax.set_ylim(1.65,9.95)
    ax.set_yticks([2,3,4,5,6,7,8,9])
    ax.grid(lw=.62,color='#dddddd',alpha=.65)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    # Per Section 3.3, endpoint marker meanings are provided only in the caption.
    fig.subplots_adjust(left=.13,right=.985,bottom=.15,top=.98)
    save(fig,out,'metric3b_front_final')


def bands(xy):
    cr=xy@CROSS
    intervals=sorted((float(k-D/2),float(k+D/2)) for k in cr)
    merged=[]
    for lo,hi in intervals:
        if not merged or lo>merged[-1][1]+1e-10:merged.append([lo,hi])
        else:merged[-1][1]=max(hi,merged[-1][1])
    return merged


def draw_layout(xy: np.ndarray, target_q: float, target_bands: int, stem: str, out: Path):
    # Site-centred physical x/y, not the rotated along/cross flight frame.
    xy=xy-np.array([SITE/2,SITE/2])
    merged=bands(xy)
    q=sum(b-a for a,b in merged)/D
    if abs(q-target_q)>1e-7 or len(merged)!=target_bands:
        raise RuntimeError('Band integrity check failed: '+stem)
    fig,ax=plt.subplots(figsize=(4.25,4.12),dpi=170)
    # A shaded strip is the full union rotor projection along the 225-degree axis.
    # It must not be clipped to the site because the rotor may extend beyond it.
    L=10000.0
    for j,(lo,hi) in enumerate(merged):
        verts=np.array([a*ALONG+b*CROSS for a,b in
                        [(-L,lo),(L,lo),(L,hi),(-L,hi)]])
        ax.add_patch(Polygon(verts,closed=True,facecolor=BANDS[j%len(BANDS)],
                             edgecolor='none',alpha=.17,zorder=0.8))
    ax.add_patch(Rectangle((-3000,-3000),6000,6000,fill=False,
                           linestyle=(0,(4,2)),lw=1.2,ec='#555555',zorder=2))
    face=C_ENERGY if target_bands==8 else C_RISK
    ax.scatter(xy[:,0],xy[:,1],s=52,c=face,edgecolors='#292929',
               linewidths=.7,zorder=4)
    ax.set_xlim(-3440,3440)
    ax.set_ylim(-3440,3440)
    ax.set_xticks([-3000,-1500,0,1500,3000])
    ax.set_yticks([-3000,-1500,0,1500,3000])
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position (m)')
    ax.grid(True,color='#aaaaaa',lw=.55,alpha=.28)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    # heading 225 degrees TOWARD southwest in physical site coordinates.
    ax.annotate('',xy=(2270,2790),xytext=(2770,3290),
                arrowprops={'arrowstyle':'-|>','lw':1.35,'color':'#292929'},zorder=6)
    ax.text(1180,3290,r'Bird flight ($225^\circ$)',fontsize=8.1,
            ha='center',va='bottom',color='#292929')
    fig.subplots_adjust(left=.20,bottom=.155,right=.98,top=.98)
    save(fig,out,stem)


def review_side_by_side(en_xy, risk_xy, out: Path):
    fig, axes=plt.subplots(1,2,figsize=(9.0,4.8),dpi=150)
    for ax,xy,target_q,bb,color in zip(axes,[en_xy,risk_xy],[9.530964530757,2.0],
                                        [8,2],[C_ENERGY,C_RISK]):
        xy=xy-np.array([3000,3000])
        L=10000.0
        for j,(lo,hi) in enumerate(bands(xy)):
            verts=np.array([a*ALONG+b*CROSS for a,b in [(-L,lo),(L,lo),(L,hi),(-L,hi)]])
            ax.add_patch(Polygon(verts,closed=True,color=BANDS[j%8],alpha=.17,lw=0,zorder=.8))
        ax.add_patch(Rectangle((-3000,-3000),6000,6000,fill=False,
                     ec='#555555',lw=1.2,ls=(0,(4,2)),zorder=2))
        ax.scatter(xy[:,0],xy[:,1],s=60,c=color,edgecolors='#292929',lw=.75,zorder=4)
        ax.set_xlim(-3440,3440);ax.set_ylim(-3440,3440)
        ax.set_xticks([-3000,-1500,0,1500,3000]);ax.set_yticks([-3000,-1500,0,1500,3000])
        ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x position (m)');ax.set_ylabel('y position (m)')
        ax.grid(alpha=.28,lw=.55);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.annotate('',xy=(2270,2790),xytext=(2770,3290),
                    arrowprops={'arrowstyle':'-|>','lw':1.35,'color':'#292929'},zorder=6)
        ax.text(1180,3290,r'Bird flight ($225^\circ$)',fontsize=8.0,ha='center',va='bottom')
    axes[0].text(.01,1.05,'(a) Highest-AEP layout',transform=axes[0].transAxes,fontsize=10)
    axes[1].text(.01,1.05,'(b) Lowest-risk layout',transform=axes[1].transAxes,fontsize=10)
    fig.subplots_adjust(left=.105,right=.975,bottom=.14,top=.88,wspace=.33)
    save(fig,out,'metric3b_layouts_review_only')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,default=Path('results/metric_3b_72sectors/heading225_W6000_seed48_pop160_gen250'), help='Existing run directory or frozen run ZIP')
    p.add_argument('--output-dir',type=Path,default=Path('metric3b_thesis_figures'))
    a=p.parse_args()
    style()
    f,v,h,l=read_front(a.archive)
    out=a.output_dir
    draw_front(f,h,l,out)
    draw_layout(v[h].reshape(12,2),9.530964530757,8,'metric3b_layout_highest_aep',out)
    draw_layout(v[l].reshape(12,2),2.0,2,'metric3b_layout_lowest_risk',out)
    review_side_by_side(v[h].reshape(12,2),v[l].reshape(12,2),out)
    print('Metric 3b figure exports complete: no FLORIS or optimisation.')
    print('Verified endpoints:',h,l)
    print('Output:',out.absolute())
    for item in sorted(out.iterdir()):print(' ',item.name)

if __name__=='__main__':main()
