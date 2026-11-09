"""Create print-sized vector PDFs for Section 3.4 from YOUR saved Metric 4 archives.

Run inside Anaconda Prompt (project root):
    conda activate floris-env
    cd /d C:\\Users\\Deepak\\Desktop\\bird_layout_opt
    python plot_metric4_print_ready_local.py

Reads the original completed 72-sector optimisation and independent evaluation
folders, or their ZIPs with the same names, without modifying them.
No FLORIS, NSGA-II, trajectory simulation or image generation.

PDF physical widths: Pareto fronts 3.70 in, individual layouts 2.95 in.
Fonts are set in points for their final physical size and PDFs must NOT be
shrunk a second time in LaTeX. Filenames match the locked Section 3.4.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import t

COLOUR_AEP = '#3577AD'
COLOUR_RISK = '#D27A20'
COLOUR_OTHER = '#9DABB5'
COLOUR_DASH = '#6B8193'
COLOUR_DARK = '#25384C'
SITE_M = 6000.
ROTOR_DIAMETER_M = 242.24
ALIGNED_MAX_M = 8 * ROTOR_DIAMETER_M
ALIGNMENT_AXIS_DEG = 45.
ALIGNMENT_TOL_DEG = 5.
PLOT_NAMES = {
    ('4a', 'highest-AEP'): 'metric4a_layout_highest_aep_section34',
    ('4a', 'lowest-risk'): 'metric4a_layout_lowest_risk_section34',
    ('4b', 'highest-AEP'): 'metric4b_layout_highest_aep_section34',
    ('4b', 'lowest-risk'): 'metric4b_layout_lowest_risk_section34',
}
EXPECTED_PAIR_COUNTS = {('4a','highest-AEP'):0, ('4a','lowest-risk'):7,
                        ('4b','highest-AEP'):0, ('4b','lowest-risk'):5}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'font.size': 8.5,
    'axes.labelsize': 8.7,
    'xtick.labelsize': 8.1,
    'ytick.labelsize': 8.1,
    'axes.linewidth': 0.7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})


def original_bytes(root: Path, folder: str, filename: str) -> bytes:
    direct = root / folder / filename
    if direct.is_file():
        print('READ:', direct)
        return direct.read_bytes()
    zfile = root / (folder + '.zip')
    if zfile.is_file():
        with zipfile.ZipFile(zfile) as z:
            item = folder + '/' + filename
            if item not in z.namelist():
                raise FileNotFoundError(f'{item} is absent in {zfile}')
            print('READ:', zfile, '->', item)
            return z.read(item)
    raise FileNotFoundError(f'Cannot find {direct} or {zfile}')


def load_csv(root: Path, folder: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(original_bytes(root,folder,filename)))


def load_json(root: Path, folder: str, filename: str) -> dict:
    return json.loads(original_bytes(root,folder,filename).decode('utf-8'))


def save_pdf_png(fig, out: Path, stem: str):
    # IMPORTANT: no bbox_inches='tight'; retain the EXACT physical PDF width.
    pdf = out / (stem+'.pdf')
    png = out / (stem+'.png')
    fig.savefig(pdf, facecolor='white',
                metadata={'Creator': 'Matplotlib from archived Metric 4 data'})
    fig.savefig(png, dpi=300, facecolor='white')
    print(f'SAVED: {pdf.name}  ({fig.get_size_inches()[0]:.2f} inches wide)')
    plt.close(fig)


def confirmation_row(table: pd.DataFrame, label: str, metric: str):
    if metric=='4a':
        rows=table[(table['layout']==label)&np.isclose(table['response_probability'],0.6)]
    else:
        rows=table[(table['layout']==label)&(table['scenario']=='baseline_p06')]
    if len(rows)!=1:
        raise ValueError(f'Expected one {metric} confirmation row for {label}; found {len(rows)}')
    return rows.iloc[0]


def front_figure(root: Path, out: Path, metric: str) -> list[dict]:
    folder=f'metric_{metric}_72sectors'
    conf_folder=f'metric_{metric}_72sectors_confirmation'
    front=load_csv(root,folder,'pareto_points.csv')
    q_col='E_enc' if metric=='4a' else 'Q_4b'  # Original archive headers
    expected_size=38 if metric=='4a' else 36
    if len(front)!=expected_size:
        raise ValueError(f'Expected {expected_size} points for {metric}, got {len(front)}')
    mult=1.0 if metric=='4a' else 1000.0
    if metric=='4a':
        summary=load_csv(root,conf_folder,'confirmation_endpoint_summary.csv')
        batches=load_csv(root,conf_folder,'confirmation_batches.csv')
        settings=load_json(root,conf_folder,'confirmation_settings.json')
        original=float(settings['original_width_m'])
        widened=float(settings['common_width_m'])
    else:
        summary=load_csv(root,conf_folder,'independent_evaluation_endpoint_summary.csv')
        settings=load_json(root,conf_folder,'evaluation_settings.json')
        original=float(settings['corridor']['original_width'])
        widened=float(settings['corridor']['common_width'])
    scale=widened/original

    # Figure is authored at actual print width. This replaces the earlier
    # figsize=(5.6,4.15), includegraphics[width=0.6\linewidth] approach.
    fig=plt.figure(figsize=(3.70,3.25))
    ax=fig.add_axes([0.245,0.185,0.735,0.785])
    ax.scatter(front['aep_gwh'],front[q_col]*mult,
               s=15, c=COLOUR_OTHER, linewidths=0.2,
               edgecolors='white',alpha=0.88,zorder=2)
    records=[]
    endpoints={'highest-AEP':front.loc[front['aep_gwh'].idxmax()],
               'lowest-risk':front.loc[front[q_col].idxmin()]}
    for label,entry in endpoints.items():
        x=float(entry['aep_gwh'])
        y_opt=float(entry[q_col])*mult
        saved=confirmation_row(summary,label,metric)
        if metric=='4a':
            batch=batches[(batches['layout']==label)&np.isclose(batches['response_probability'],0.6)]
            if len(batch)!=20:
                raise ValueError(f'Metric 4a: expected 20 batches, got {len(batch)}')
            raw=float(saved['confirmed_Q_4a'])
            half=float(t.ppf(0.975,19))*float(batch['Q_4a'].std(ddof=1))/np.sqrt(20)
            lower=(raw-half)*scale*mult
            upper=(raw+half)*scale*mult
        else:
            raw=float(saved['independently_evaluated_Q_4b'])
            lower=float(saved['ci95_lower'])*scale*mult
            upper=float(saved['ci95_upper'])*scale*mult
        y_ind=raw*scale*mult
        color=COLOUR_AEP if label=='highest-AEP' else COLOUR_RISK
        marker='o' if label=='highest-AEP' else 'D'
        ax.plot([x,x],[y_opt,y_ind],ls=':',lw=1.0,c=color,alpha=.9,zorder=3)
        ax.scatter([x],[y_opt],s=55,marker=marker,
                   facecolors=(color if label=='highest-AEP' else 'white'),
                   edgecolors=color,linewidths=1.35,zorder=4)
        ax.errorbar([x],[y_ind],yerr=[[y_ind-lower],[upper-y_ind]],fmt='x',
                    color=color,mew=1.8,ms=7.1,elinewidth=1.05,
                    capsize=2.5,zorder=6,clip_on=False)
        records.append({'metric':metric,'endpoint':label,'AEP_GWh_per_year':x,
            'optimisation_sample_Q':y_opt/mult,'independent_adjusted_Q':y_ind/mult,
            'independent_adjusted_lower95':lower/mult,
            'independent_adjusted_upper95':upper/mult,'original_corridor_m':original,
            'evaluation_corridor_m':widened,'corridor_scale_factor':scale})

    ax.set_xlabel('AEP (GWh/year)',labelpad=3)
    if metric=='4a':
        ax.set_ylabel(r'$Q_{4a}$ (encounters per entering bird)',labelpad=3)
    else:
        # Two lines prevent tiny lettering on the long y-axis label.
        ax.set_ylabel('$Q_{4b}$ ($10^{-3}$ rotor transits\nper entering bird)',labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(length=2.4,width=.65,pad=2.4)
    ax.grid(alpha=0.14,lw=.42)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.margins(x=0.10,y=0.11)
    save_pdf_png(fig,out,f'metric{metric}_front_section34')
    return records


def load_layout(root:Path, metric:str, endpoint:str)->np.ndarray:
    folder=f'metric_{metric}_72sectors'
    content=original_bytes(root,folder,f'{endpoint}_layout.csv')
    rows=list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
    xy=np.asarray([[float(row['x_m']),float(row['y_m'])] for row in rows])
    if xy.shape!=(12,2) or not np.isfinite(xy).all() or (xy < 0).any() or (xy > SITE_M).any():
        raise ValueError(f'Invalid 12-turbine saved coordinates: Metric {metric}, {endpoint}')
    return xy


def aligned_pairs(xy:np.ndarray)->list[tuple[int,int]]:
    pairs=[]
    for i in range(len(xy)):
        for j in range(i+1,len(xy)):
            delta=xy[j]-xy[i]
            dist=float(np.linalg.norm(delta))
            if dist>ALIGNED_MAX_M+1e-9:
                continue
            ang=np.degrees(np.arctan2(delta[1],delta[0]))%180
            diff=abs((ang-ALIGNMENT_AXIS_DEG+90)%180-90)
            if diff<=ALIGNMENT_TOL_DEG:
                pairs.append((i,j))
    return pairs


def layout_figure(root:Path,out:Path,metric:str,endpoint:str):
    xy=load_layout(root,metric,endpoint)
    pairs=aligned_pairs(xy)
    exp=EXPECTED_PAIR_COUNTS[(metric,endpoint)]
    if len(pairs)!=exp:
        raise ValueError(f'{metric}/{endpoint}: expected {exp} aligned pairs, got {len(pairs)}')
    color=COLOUR_AEP if endpoint=='highest-AEP' else COLOUR_RISK
    fig=plt.figure(figsize=(2.95,3.15))
    ax=fig.add_axes([0.235,0.19,0.745,0.745])
    ax.set_aspect('equal',adjustable='box')
    ax.add_patch(Rectangle((0,0),SITE_M,SITE_M,fill=False,
                           ec='#455A6C',lw=.9,zorder=2))
    for i,j in pairs:
        ax.plot(xy[[i,j],0],xy[[i,j],1],ls='--',lw=1.0,
                color=COLOUR_DASH,alpha=.94,zorder=3)
    ax.scatter(xy[:,0],xy[:,1],s=39,c=color,edgecolors='white',lw=.65,zorder=4)
    ax.annotate('',xy=(5110,5060),xytext=(5780,5750),
                arrowprops=dict(arrowstyle='-|>',color=COLOUR_DARK,lw=1.3),zorder=5)
    ax.set_xlim(-330,6330)
    ax.set_ylim(-330,6330)
    ax.set_xticks([0,2000,4000,6000])
    ax.set_yticks([0,2000,4000,6000])
    ax.set_xlabel('x position (m)',labelpad=2)
    ax.set_ylabel('y position (m)',labelpad=2)
    ax.tick_params(length=2.3,width=.65,pad=2.2)
    ax.grid(lw=.45,alpha=.14,zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(f'Metric {metric} {endpoint}: {len(pairs)} aligned pairs (verified)')
    save_pdf_png(fig,out,PLOT_NAMES[(metric,endpoint)])


def make_tex(root:Path,out:Path):
    # Optional copy for users who want natural PDF size without scaling.
    # Does not modify the live Overleaf document or any original files.
    source=root/'section34_final_checked'/'Section_3_4.tex'
    if not source.exists():
        return
    text=source.read_text(encoding='utf-8')
    text=text.replace(r'\includegraphics[width=0.6\linewidth]{figures/metric4a_front_section34.pdf}',
                      r'\includegraphics{figures/metric4a_front_section34.pdf}')
    text=text.replace(r'\includegraphics[width=0.6\linewidth]{figures/metric4b_front_section34.pdf}',
                      r'\includegraphics{figures/metric4b_front_section34.pdf}')
    for m,e in PLOT_NAMES:
        name=PLOT_NAMES[(m,e)]
        text=text.replace(r'\includegraphics[width=\linewidth]{figures/'+name+'.pdf}',
                          r'\includegraphics{figures/'+name+'.pdf}')
    (out/'Section_3_4_Print_Ready.tex').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent,
                        help='Folder containing your four original result ZIPs or folders')
    args=parser.parse_args()
    root=args.root.expanduser().resolve()
    out=root/'metric4_print_ready_local'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for metric in ('4a','4b'):
        rows.extend(front_figure(root,out,metric))
    for metric in ('4a','4b'):
        for endpoint in ('highest-AEP','lowest-risk'):
            layout_figure(root,out,metric,endpoint)
    pd.DataFrame(rows).to_csv(out/'plotted_values_from_original_archives.csv',index=False)
    make_tex(root,out)
    print('\nCOMPLETE: 2 print-sized fronts (3.70 in) and 4 layouts (2.95 in), PDFs + PNG previews')
    print('Output folder:',out)
    print('Only existing archive data used. Never reran FLORIS, bird simulations, or NSGA-II.')

if __name__=='__main__':
    main()
