#!/usr/bin/env python3
"""Re-export final Metric 3a figures from an extracted verification folder.

No FLORIS, optimisation or risk-model calculation. Pure plotting of saved records.
Usage:
    python build_metric3a_thesis_figures.py
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

RELPREFIX = 'metric3a_verification_20261017/'
CHECK_VALUES = {
    'random_start': 7.688414314122593,
    'aligned_start': 5.460555118557852,
    'aligned_reference': 5.525407012391223,
    'cross_flow_reference': 9.560725671350504,
}
C_RANDOM = '#3378a6'
C_ALIGNED = '#c77d2b'
C_REFERENCE = '#178078'
C_CROSS = '#8b5a9f'


def plot_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9.2,
        'axes.labelsize': 9.6,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.0,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.facecolor': 'white',
    })


def output_pair(fig, directory: Path, stem: str):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / (stem + '.pdf'), bbox_inches='tight', pad_inches=0.06)
    fig.savefig(directory / (stem + '.png'), dpi=260, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


def read_csv_from_folder(directory: Path, relative: str):
    return pd.read_csv(directory / relative)


def load_convergence(directory: Path, which: str):
    by_name = {'random_start': 'random_start_original.zip', 'aligned_start': 'aligned_start_original.zip'}
    with zipfile.ZipFile(directory / 'input_archives' / by_name[which]) as original:
        paths = [x for x in original.namelist() if x.endswith('/convergence.csv')]
        if len(paths) != 1:
            raise RuntimeError('Expected one convergence.csv in '+by_name[which])
        df = pd.read_csv(io.BytesIO(original.read(paths[0])))
    if len(df) != 120 or int(df.generation.iloc[-1]) != 120:
        raise RuntimeError('Original convergence history is not 120 generations')
    if abs(float(df.best_risk.iloc[-1]) - CHECK_VALUES[which]) > 1e-10:
        raise RuntimeError('Convergence endpoint does not match verified result')
    return df


def draw_convergence(random: pd.DataFrame, aligned: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(4.45, 3.35), dpi=160)
    ax.plot(random.generation, random.best_risk, color=C_RANDOM, lw=1.9,
            label='Without a prescribed starting layout', zorder=3)
    ax.plot(aligned.generation, aligned.best_risk, color=C_ALIGNED, lw=1.9,
            label='With the aligned starting layout', zorder=4)
    ax.axhline(CHECK_VALUES['aligned_reference'], color=C_REFERENCE, lw=1.35,
               ls=(0, (4, 2)), zorder=2)
    ax.axhline(CHECK_VALUES['cross_flow_reference'], color=C_CROSS, lw=1.35,
               ls=(0, (4, 2)), zorder=2)

    ax.set_xlim(0, 121)
    ax.set_ylim(5.2, 10.75)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_xlabel('Generation')
    ax.set_ylabel(r'$Q_{3a}$ (a.u.)')
    ax.grid(color='#dddddd', alpha=0.65, lw=0.65)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(118.0, CHECK_VALUES['cross_flow_reference']+0.09,
            'Cross-flow reference (9.561)', color=C_CROSS,
            fontsize=8.0, va='bottom', ha='right')
    ax.text(118.0, CHECK_VALUES['aligned_reference']+0.09,
            'Aligned reference (5.525)', color=C_REFERENCE,
            fontsize=8.0, va='bottom', ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False,
              labelspacing=0.45, handlelength=2.6)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.98, bottom=0.145)
    output_pair(fig, out, 'metric3a_convergence_final')


def draw_frame(ax):
    ax.add_patch(Rectangle((-3000, -3000), 6000, 6000,
                           fill=False, lw=1.15, ls=(0,(4,2)),
                           ec='#555555', zorder=1))
    ax.set_xlim(-3440, 3440)
    ax.set_ylim(-3440, 3440)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-3000, -1500, 0, 1500, 3000])
    ax.set_yticks([-3000, -1500, 0, 1500, 3000])
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position (m)')
    ax.grid(lw=0.55, alpha=0.28, color='#9aa0a6')
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    # Bird direction 225 degrees TOWARD, mathematical convention: southwest.
    ax.annotate('', xy=(2270, 2790), xytext=(2770, 3290),
                arrowprops=dict(arrowstyle='-|>', lw=1.35, color='#292929'), zorder=7)
    ax.text(1180, 3290, r'Bird flight ($225^\circ$)', fontsize=8.1,
            ha='center', va='bottom', color='#292929')


def coords(df):
    required={'x_m','y_m'}
    if not required.issubset(df.columns) or len(df) != 12:
        raise RuntimeError('Expected 12 original turbine coordinates with x_m,y_m')
    xy=df[['x_m','y_m']].to_numpy(dtype=float)-3000.0
    if (xy < -3000 - 1e-8).any() or (xy > 3000+1e-8).any():
        raise RuntimeError('Coordinates outside centre-bounded site')
    return xy


def draw_single(xy: np.ndarray, kind: str, out: Path,
                reference_xy: np.ndarray|None=None):
    fig,ax=plt.subplots(figsize=(4.25, 4.12),dpi=170)
    draw_frame(ax)
    main_color = C_RANDOM if kind=='without_start' else C_ALIGNED
    ax.scatter(xy[:,0],xy[:,1], s=52, color=main_color,
               edgecolor='#252525',lw=0.65, zorder=4)
    if reference_xy is not None:
        # Keep close-but-different points visible: hollow outline behind filled optimised markers.
        ax.scatter(reference_xy[:,0],reference_xy[:,1], s=155, marker='o',
                   facecolors='none', edgecolors=C_REFERENCE, lw=1.55,
                   zorder=5)
    fig.subplots_adjust(left=0.20, bottom=0.155, right=0.98, top=0.98)
    stem = ('metric3a_layout_without_start' if kind=='without_start'
            else 'metric3a_layout_with_start')
    output_pair(fig, out, stem)


def draw_comparison(xy_random, xy_aligned, xy_reference, out):
    # This is only a convenient preview; thesis inserts the two PDF panels separately.
    fig, axes = plt.subplots(1,2,figsize=(9.0,4.8),dpi=150)
    for ax in axes: draw_frame(ax)
    axes[0].scatter(xy_random[:,0],xy_random[:,1],s=60,color=C_RANDOM,
                    edgecolors='#222222', lw=0.65,zorder=4)
    axes[1].scatter(xy_aligned[:,0],xy_aligned[:,1],s=60,color=C_ALIGNED,
                    edgecolors='#222222',lw=0.65,zorder=4)
    axes[1].scatter(xy_reference[:,0],xy_reference[:,1],s=165,
                    facecolors='none',edgecolors=C_REFERENCE,lw=1.5,zorder=5)
    axes[0].text(0.01,1.05,'(a) Without a prescribed starting layout',
                 fontsize=10,transform=axes[0].transAxes,ha='left')
    axes[1].text(0.01,1.05,'(b) With the aligned starting layout',
                 fontsize=10,transform=axes[1].transAxes,ha='left')
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.14, top=0.88,wspace=0.33)
    output_pair(fig,out,'metric3a_layouts_review_only')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('metric3a_verification_20261017'),
                   help='Extracted verification folder containing summary.json, layouts/ and input_archives/')
    p.add_argument('--output-dir', type=Path, default=Path('figures_3_3'))
    a = p.parse_args()
    directory = a.data_dir
    required = [directory / 'summary.json', directory / 'layouts/random_start.csv',
                directory / 'layouts/aligned_start.csv', directory / 'layouts/aligned_reference.csv',
                directory / 'input_archives/random_start_original.zip',
                directory / 'input_archives/aligned_start_original.zip']
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise FileNotFoundError('Missing extracted source files:\n  ' + '\n  '.join(missing) +
                                '\nKeep the source folder hierarchy unchanged.')
    plot_style()
    check = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    for k,v in CHECK_VALUES.items():
        if abs(float(check['recalculated_Q_3a'][k])-v)>1e-11:
            raise RuntimeError('Verification value mismatch: '+k)
    random = load_convergence(directory, 'random_start')
    aligned = load_convergence(directory, 'aligned_start')
    xy_random = coords(read_csv_from_folder(directory,'layouts/random_start.csv'))
    xy_aligned = coords(read_csv_from_folder(directory,'layouts/aligned_start.csv'))
    xy_reference = coords(read_csv_from_folder(directory,'layouts/aligned_reference.csv'))
    if not np.allclose((xy_aligned+3000),
                       read_csv_from_folder(directory,'layouts/aligned_start.csv')[['x_m','y_m']].to_numpy(),
                       atol=1e-10):
        raise RuntimeError('Coordinate transform invalid')
    out=a.output_dir
    draw_convergence(random,aligned,out)
    draw_single(xy_random,'without_start',out)
    draw_single(xy_aligned,'with_start',out,reference_xy=xy_reference)
    draw_comparison(xy_random,xy_aligned,xy_reference,out)
    print('Metric 3a exports complete. No optimisation performed.')
    print('Source folder: '+str(directory.resolve()))
    print('Output folder: '+str(out.resolve()))
    for f in sorted(out.iterdir()):
        if f.name.startswith('metric3a_'): print(' ',f.name)

if __name__=='__main__':
    main()
