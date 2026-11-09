"""Metric 1a-1d controlled Jensen rerunner: revised approval 2026-10-16.

- Fixed 360 sectors (1 degree), Jensen turbine D=240m, seed 42, archived AEP reference.
- Original NSGA2 constructor and initial-population code are reused unchanged.
- Run-specific settings and complete nondominated coordinates saved.
- Independent 0.5-degree (720-sector) endpoint convergence check after each run.
- AEP reported at 1 degree; independent 1-degree re-evaluation and delta saved.
- STOP with exit status 2 if either endpoint fails strict <0.2% convergence.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from config import Scenario
from wake import AEPModel
from hypothetical_risk import HypotheticalCollisionModel
from validation_problem import ValidationLayoutProblem
from run_validation import (GridSampling, HVCallback, raw_objectives_dataframe,
                            find_anchor_indices, save_anchor_layouts,
                            plot_hypervolume)

# Unrounded exact historical value implied by the archived objective columns.
# Historical calculation / layout provenance has NOT been recovered.
AEP_REF_GWH = 754.4007727706155
EXPECTED_PYMOO_VERSION = "0.6.1.5"
N_SECTORS = 360
DIRECTION_STEP_DEG = 1.0
EXPERIMENTS = {
    '1a': ('constant', 80, 100),
    '1b': ('half_site_east_risk', 120, 150),
    '1c': ('central_corridor', 120, 150),
    '1d': ('point_hotspot', 120, 150),
}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not available'


def class_description(value):
    if value is None:
        return None
    return f'{type(value).__module__}.{type(value).__qualname__}'


def operator_snapshot(op):
    """Record resolved operator defaults without altering any pymoo operators."""
    if op is None:
        return None
    result = {'class': class_description(op)}
    for attr in ('prob', 'prob_var', 'eta', 'n_offsprings', 'n_parents',
                 'repair', 'elementwise'):
        if hasattr(op, attr):
            value = getattr(op, attr)
            if value is None or isinstance(value, (bool, int, float, str)):
                result[attr] = value
            else:
                # Pymoo uses variable wrappers (e.g. Real) for parameters.
                inner = getattr(value, 'value', None)
                if inner is not None and isinstance(inner, (bool, int, float, str)):
                    result[attr] = {'wrapper': class_description(value), 'value': inner}
                else:
                    result[attr] = {'class': class_description(value), 'repr': repr(value)}
    return result


def hash_sources():
    paths=['config.py','wake.py','hypothetical_risk.py','validation_problem.py',
           'run_validation.py','run_jensen_1deg.py']
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def save_complete_layouts(X,F,metric,output_dir):
    np.savez_compressed(output_dir/'pareto_full.npz', X=X, F=F)
    long=[]
    for ix,x in enumerate(X):
        for i,(tx,ty) in enumerate(x.reshape(12,2)):
            long.append({'layout_id':ix,'turbine':i+1,'x_m':float(tx),'y_m':float(ty)})
    pd.DataFrame(long).to_csv(output_dir/'pareto_layouts_long.csv',index=False)


def plot_endpoint_layouts(X,anchors,scenario,risk_model,output_dir):
    # Plotting is post-processing only; this function never feeds the optimizer.
    xx=np.linspace(0,scenario.site.width_m,301)
    yy=np.linspace(0,scenario.site.height_m,301)
    gx,gy=np.meshgrid(xx,yy)
    field=risk_model.collisions_per_turbine(np.column_stack([gx.ravel(),gy.ravel()])).reshape(gx.shape)
    for key,marker in [('energy_optimal','o'),('collision_optimal','D')]:
        xy=X[anchors[key]].reshape(12,2)
        fig,ax=plt.subplots(figsize=(6,6))
        ax.imshow(field,origin='lower',extent=[0,6000,0,6000],cmap='Oranges',
                  vmin=0,vmax=1000,alpha=0.35,interpolation='nearest')
        ax.scatter(xy[:,0],xy[:,1],s=84,marker=marker,facecolor='steelblue',
                   edgecolor='black',linewidth=0.7,zorder=5)
        ax.set(xlim=(-160,6160),ylim=(-160,6160),xlabel='x position (m)',
               ylabel='y position (m)')
        ax.set_aspect('equal',adjustable='box')
        ax.grid(alpha=0.16)
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(output_dir/f'layout_{key}.{ext}',dpi=220)
        plt.close(fig)


def plot_pareto_if_1d(df,anchors,output_dir):
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(df['aep_gwh'],df['risk'],s=30,alpha=0.8,color='slategrey',
               label='Non-dominated layouts')
    for key, marker, label in [('energy_optimal','o','Highest AEP'),
                               ('collision_optimal','D','Lowest risk')]:
        row=df.loc[anchors[key]]
        ax.scatter([row.aep_gwh],[row.risk],s=100,marker=marker,
                   edgecolor='black',linewidth=0.7,label=label,zorder=5)
    ax.set(xlabel='AEP (GWh/year)',ylabel='Artificial bird-activity score, Q')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(output_dir/f'metric_1d_pareto.{ext}',dpi=220)
    plt.close(fig)


def independently_reevaluate_pareto_at_1degree(X, df, output_dir):
    """Fresh, independent AEP evaluation of ALL final layouts at reporting grid (1 degree).

    This is postprocessing-only; does not reorder/select/change optimised layouts.
    We keep both the optimisation-resolution AEP and reporting-resolution AEP,
    although both use 360 sectors for Jensen and should match within roundoff.
    """
    independent_scenario = Scenario()
    independent_scenario.wind.n_sectors = 360
    independent_model = AEPModel(independent_scenario)
    records = []
    for ix, coords in enumerate(X):
        optimised = float(df.loc[ix, 'aep_gwh'])
        reported = float(independent_model.evaluate(coords.reshape(12, 2)))
        delta = reported-optimised
        records.append({
            'layout_id': ix,
            'aep_optimisation_1deg_gwh': optimised,
            'aep_reported_1deg_gwh': reported,
            'delta_reported_minus_optimisation_gwh': delta,
            'abs_delta_pct_of_optimisation': 100.0*abs(delta)/abs(optimised),
            'risk_Q': float(df.loc[ix,'risk']),
        })
    report = pd.DataFrame(records)
    report.to_csv(output_dir/'pareto_aep_reported_1deg.csv',index=False,float_format='%.12g')
    return report


def recheck_endpoints_at_halfdegree(X, reported_aep_df, original_df, anchors, output_dir):
    """Recalculate both new 1-degree-optimised endpoint layouts at 0.5 degree.

    Strict pass: |AEP_1deg - AEP_0p5deg|/|AEP_0p5deg| * 100 < 0.2.
    Never feeds into optimisation, modifies endpoints or changes Pareto dominance.
    """
    fine_scenario = Scenario()
    fine_scenario.wind.n_sectors = 720
    assert fine_scenario.turbine.rotor_diameter_m == 240.0
    assert fine_scenario.site.n_turbines == 12
    fine_model = AEPModel(fine_scenario)
    records = []
    for anchor_name in ('energy_optimal','collision_optimal'):
        ix = int(anchors[anchor_name])
        xy = np.asarray(X[ix],dtype=float).reshape(12,2)
        aep_1deg = float(reported_aep_df.loc[ix,'aep_reported_1deg_gwh'])
        aep_halfdeg = float(fine_model.evaluate(xy))
        absolute_delta = abs(aep_1deg-aep_halfdeg)
        delta_pct = 100.0*absolute_delta/abs(aep_halfdeg)
        records.append({
            'anchor': anchor_name,
            'layout_id': ix,
            'aep_optimisation_1deg_gwh':float(reported_aep_df.loc[ix,'aep_optimisation_1deg_gwh']),
            'aep_reported_1deg_gwh':aep_1deg,
            'aep_check_0p5deg_gwh':aep_halfdeg,
            'delta_reported_minus_optimisation_gwh':float(reported_aep_df.loc[ix,'delta_reported_minus_optimisation_gwh']),
            'abs_delta_1deg_vs_0p5deg_gwh':absolute_delta,
            'abs_delta_pct_of_0p5deg':delta_pct,
            'threshold_pct_strict':0.2,
            'convergence_pass':bool(delta_pct<0.2),
            'risk_Q':float(original_df.loc[ix,'risk']),
        })
    report=pd.DataFrame(records)
    report.to_csv(output_dir/'endpoint_0p5deg_convergence.csv',index=False,float_format='%.12g')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--metric',required=True,choices=sorted(EXPERIMENTS))
    args=parser.parse_args()
    installed_pymoo = package_version("pymoo")
    if installed_pymoo != EXPECTED_PYMOO_VERSION:
        raise RuntimeError(
            f"pymoo version mismatch: expected {EXPECTED_PYMOO_VERSION}, "
            f"found {installed_pymoo}. Stopping before creating any run output."
        )
    model_name,pop,gen=EXPERIMENTS[args.metric]
    seed=42
    scenario=Scenario()
    assert scenario.wind.n_sectors==N_SECTORS
    assert scenario.turbine.rotor_diameter_m==240.0
    assert scenario.site.n_turbines==12
    assert scenario.site.min_spacing_D==4.0
    assert scenario.site.width_m==6000.0 and scenario.site.height_m==6000.0

    output_dir=Path('results')/f'metric_{args.metric}_360sectors_seed42_pop{pop}_gen{gen}'
    if output_dir.exists():
        raise FileExistsError(f'Output directory exists; refusing to overwrite: {output_dir}')
    output_dir.mkdir(parents=True,exist_ok=False)

    aep_model=AEPModel(scenario)
    risk_model=HypotheticalCollisionModel(scenario,model_name=model_name,
            risk_high=1000.0,risk_low=0.0,corridor_width_m=3000.0,
            hotspot_radius_m=2000.0,risk_multiplier=1.0)
    aep_min=.5*AEP_REF_GWH
    aep_max=AEP_REF_GWH
    problem=ValidationLayoutProblem(scenario=scenario,aep_model=aep_model,
        risk_model=risk_model,aep_min=aep_min,aep_max=aep_max,
        risk_min=0.0,risk_max=risk_model.risk_scale())
    sampling=GridSampling(scenario,seed=seed,model_name=model_name)

    # KEEP THIS CALL IDENTICAL TO THE UPLOADED ORIGINAL RUNNER.
    algorithm=NSGA2(
        pop_size=pop,
        sampling=sampling,
        eliminate_duplicates=True,
    )
    callback=HVCallback(ref_point=np.array([1.2,1.2]))

    settings={
        'run_status':'initialized',
        'metric':args.metric,
        'model_name':model_name,
        'energy_model':'Jensen top-hat; hub-centre wake overlap',
        'site':{'width_m':6000.0,'height_m':6000.0,'n_turbines':12,
                'minimum_spacing_D':4.0,'minimum_spacing_m':960.0},
        'turbine':{'rotor_diameter_m':240.0,'rotor_radius_m':120.0,
                   'rated_power_mw':15.0,'hub_height_m':150.0,
                   'cut_in_ms':3.0,'rated_ms':10.6,'cut_out_ms':25.0},
        'wind':{'n_sectors':scenario.wind.n_sectors,'sector_width_deg':DIRECTION_STEP_DEG,
                'wind_from_deg':scenario.site.wind_from_deg,
                'speed_bins_ms':scenario.wind.speed_bins.tolist(),
                'weibull_k':scenario.wind.weibull_k,
                'weibull_A_ms':scenario.wind.weibull_A,
                'direction_weights_formula':'exp(2*cos(delta_from_225deg))/sum_over_sectors',
                'speed_weights_formula':'discrete Weibull PDF, renormalized'},
        'wake':{'kw':aep_model.wake_decay_k,'wake_radius':'D/2 + kw*downstream_distance',
                'wake_interaction':'hub-centre falls within top-hat cone',
                'deficit_combination':'root-sum-square, clipped at 0.95',
                'ct_from_undisturbed_wind_ms':'0.8 for 3<=U<10.6, 0.25 for 10.6<=U<=25, otherwise 0'},
        'bird':{'risk_high':1000.,'risk_low':0.,'risk_multiplier':1.,
                'central_corridor_width_m':3000.,'gaussian_sigma_m':2000.,
                'risk_normalization_reference':12000.},
        'objective_scaling':{'aep_ref_gwh':AEP_REF_GWH,
                'aep_ref_origin':'reconstructed from archived objective columns; physical source unknown',
                'aep_min_gwh':aep_min,'aep_max_gwh':aep_max,
                'f1':'1-(AEP-0.5*AEP_ref)/(0.5*AEP_ref) = 2-2*AEP/AEP_ref',
                'f2':'Q/12000'},
        'optimizer':{'name':'NSGA-II','library':'pymoo','expected_pymoo_version':EXPECTED_PYMOO_VERSION,'population':pop,
                'generations':gen,'seed':seed,'eliminate_duplicates':True,
                'sampling_class':class_description(sampling),
                'sampling_rule':'1a: 50% random/50% jittered grid; 1b-1d: 40% risk-aware/30% random/30% jittered grid',
                'sampler_corridor_width_m':1200.,'sampler_hotspot_exclusion_radius_m':900.,
                'crossover_class':class_description(getattr(getattr(algorithm,'mating',None),'crossover',None)),
                'mutation_class':class_description(getattr(getattr(algorithm,'mating',None),'mutation',None)),
                'selection_class':class_description(getattr(getattr(algorithm,'mating',None),'selection',None)),
                'survival_class':class_description(getattr(algorithm,'survival',None)),
                'operator_parameters':'Unmodified installed-pymoo NSGA-II defaults; inspect snapshots and installed version',
                'operator_snapshots':{
                    'selection':operator_snapshot(getattr(getattr(algorithm,'mating',None),'selection',None)),
                    'crossover':operator_snapshot(getattr(getattr(algorithm,'mating',None),'crossover',None)),
                    'mutation':operator_snapshot(getattr(getattr(algorithm,'mating',None),'mutation',None)),
                    'survival':operator_snapshot(getattr(algorithm,'survival',None)),
                },
                'hv_reference':[1.2,1.2]},
        'environment':{'python':platform.python_version(),
            'pymoo':package_version('pymoo'),'numpy':package_version('numpy'),
            'pandas':package_version('pandas'),'matplotlib':package_version('matplotlib'),
            'platform':platform.platform()},
        'file_sha256':hash_sources(),
        'approval':{'approved_on':'2026-10-16',
                    'previous_2degree_1a':'retained as diagnostic only; NOT final',
                    'authorised_metrics':['1a','1b','1c','1d'],
                    'selected_direction_resolution_deg':1.0,
                    'selected_direction_sectors':360,
                    'selected_by':'2-degree 1a freshly optimised endpoint failed 2-vs-1-degree 0.2% convergence; user approved 1-degree reruns',
                    'floris_direction_sectors_unchanged':72},
        'reporting_aep':{'resolution_deg':1.0,'n_sectors':360,
                    'source':'independent Jensen 1-degree evaluations of ALL final Pareto layouts',
                    'optimisation_resolution_deg':1.0,
                    'delta_vs_optimisation':'reported minus optimisation (GWh)',
                    'status':'pending'},
        'post_run_endpoint_verification':{
                    'resolution_deg':0.5,'n_sectors':720,
                    'anchors':['energy_optimal','collision_optimal'],
                    'relative_difference_denominator':'0.5-degree AEP',
                    'pass_rule':'abs(AEP_1deg-AEP_0p5deg)/abs(AEP_0p5deg)*100 < 0.2',
                    'status':'pending'},
        'output_policy':'unique 360-sector output directory, no overwrite; STOP on convergence failure',
        'output_files':['run_settings.json','pareto_points.csv','pareto_full.npz',
            'pareto_layouts_long.csv','anchor_summary.csv','layout_*.csv',
            'hypervolume.csv','pareto_aep_reported_1deg.csv','endpoint_0p5deg_convergence.csv',
            'layout_*.png/pdf','metric_1{a,b,c,d}_pareto.png/pdf'],
    }
    (output_dir/'run_settings.json').write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    print(f'Running {args.metric}, sectors={N_SECTORS}, seed={seed}, pop={pop}, gens={gen}')
    print('pymoo version:',settings['environment']['pymoo'])
    print('Output directory:',output_dir)

    # KEEP THIS CALL IDENTICAL TO THE UPLOADED ORIGINAL RUNNER.
    result=minimize(
        problem,
        algorithm,
        termination=('n_gen',gen),
        seed=seed,
        callback=callback,
        verbose=True,
    )
    if result.X is None:
        settings['run_status']='finished_no_feasible_solutions'
        (output_dir/'run_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
        raise RuntimeError('No feasible Pareto solutions. Original results remain untouched.')
    X=np.asarray(result.X)
    F=np.asarray(result.F)
    if X.ndim==1:
        X=X.reshape(1,-1)
        F=F.reshape(1,-1)

    save_complete_layouts(X,F,args.metric,output_dir)
    df=raw_objectives_dataframe(X,F,scenario,aep_model,risk_model)
    df.to_csv(output_dir/'pareto_points.csv',index=False)
    anchors=find_anchor_indices(df)
    save_anchor_layouts(X,df,anchors,scenario,output_dir)
    # Independent, reporting-grid 1-degree evaluation of every final Pareto layout.
    # Since optimisation is also at 1 degree, this should agree to numerical roundoff.
    final_report = independently_reevaluate_pareto_at_1degree(X,df,output_dir)
    # Required independent 0.5-degree check on the NEW optima (not historical layouts).
    verification = recheck_endpoints_at_halfdegree(X,final_report,df,anchors,output_dir)
    check_passed = bool(verification['convergence_pass'].all())
    settings['reporting_aep'].update({
        'status':'complete',
        'results_file':'pareto_aep_reported_1deg.csv',
        'max_abs_delta_pct_of_optimisation':float(final_report['abs_delta_pct_of_optimisation'].max()),
    })
    settings['post_run_endpoint_verification'].update({
        'status':'pass' if check_passed else 'FAIL_STOP',
        'max_abs_delta_pct_of_0p5deg':float(verification['abs_delta_pct_of_0p5deg'].max()),
        'results_file':'endpoint_0p5deg_convergence.csv',
        'results':verification.to_dict(orient='records'),
    })
    print('\nIndependent 0.5-degree convergence re-check (new 1-degree endpoints):')
    print(verification.to_string(index=False))
    pd.DataFrame({'generation':np.arange(1,len(callback.data['hv'])+1),
                  'hypervolume':callback.data['hv']}).to_csv(output_dir/'hypervolume.csv',index=False)
    plot_endpoint_layouts(X,anchors,scenario,risk_model,output_dir)
    # Generate Pareto diagrams for 1a, 1b, 1c and 1d, including 1a's single point.
    from plot_saved_pareto import plot as plot_saved_pareto
    plot_saved_pareto(output_dir)
    settings['non_dominated_layouts_saved']=int(len(X))
    settings['run_status']='complete' if check_passed else 'STOP_convergence_failed'
    (output_dir/'run_settings.json').write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    print('\nSaved ALL Pareto coordinates, reported 1-degree AEP, and settings:',output_dir)
    if not check_passed:
        print('\nSTOP: newly optimised 1-degree endpoint differs by >=0.2% from 0.5-degree.')
        print('Do NOT launch the next metric. Send endpoint_0p5deg_convergence.csv for review.')
        raise SystemExit(2)
    print('PASS: both new endpoints agree with 0.5-degree to strictly less than 0.2%.')

if __name__=='__main__':
    main()
