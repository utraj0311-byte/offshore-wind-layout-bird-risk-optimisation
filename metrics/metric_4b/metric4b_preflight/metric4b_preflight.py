#!/usr/bin/env python
"""Read-only, on-laptop pre-flight checks for Metric 4b (72-sector proposal).

Commands: python metric4b_preflight.py audit|bird|aep|benchmark
No NSGA-II runs or source edits are performed here.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'model9_code'
ORIGINAL = ROOT / 'original_results'
REPORT = ROOT / 'preflight_reports'
sys.path.insert(0, str(CODE))
SETTINGS = ORIGINAL / 'run_settings.json'
FHD = CODE / 'data' / 'northern_gannet_Johnston2014_FHD.csv'
CONFIG = CODE / 'floris_config' / 'gch_iea15mw.yaml'
EXPECTED_CONFIG_HASH = 'f6b788df9e38f006dfb6c68ddfb80ab090dad9cd555982a8d5c5e4b0c2f2d364'
REFERENCE_Q = 8.704996418223796e-4
REFERENCE_AEP = 854.4956585617186
AEP_LOSS_REF = 137.96914764866568
SAVED_CASES = {'highest-AEP': 32, 'lowest-risk': 10}


def require(ok, label, observed=None, expected=None):
    rendered = f'PASS | {label}' if ok else f'FAIL | {label}'
    if observed is not None or expected is not None:
        rendered += f': observed={observed!r}; expected={expected!r}'
    print(rendered, flush=True)
    if not ok:
        raise RuntimeError(f'Pre-flight STOP: {label}')


def info(label, obj):
    print(f'INFO | {label}: {obj}', flush=True)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_archive():
    require(SETTINGS.is_file(), 'original run_settings.json is present')
    require((ORIGINAL / 'pareto_decision_vectors.npy').is_file(), 'original Pareto coordinates present')
    require(FHD.is_file(), 'frozen northern-gannet FHD is present')
    require(CONFIG.is_file(), 'Model 9 FLORIS YAML is present')
    return json.loads(SETTINGS.read_text(encoding='utf-8'))


def report_json(name, record):
    REPORT.mkdir(exist_ok=True)
    path=REPORT/name
    path.write_text(json.dumps(record,indent=2,default=str),encoding='utf-8')
    info('report',str(path))


def package_versions():
    out={}
    for package in ('pymoo','floris','numpy','pandas','scipy'):
        try: out[package] = metadata.version(package)
        except metadata.PackageNotFoundError: out[package]='NOT INSTALLED'
    return out


def scientific_fields(data, n_sectors):
    problem=data['problem']
    return {
      'n_sectors': n_sectors,
      'speed_bins': [4.,6.,8.,10.,12.,14.,16.],
      'weibull_k':2., 'weibull_A':10., 'dominant_wind_from_deg':225.,
      'population':data['profile']['population'],
      'generations':data['profile']['generations'],
      'optimizer_seed':data['seed'],
      'optimisation_sample_batches':data['profile']['objective_batches'],
      'birds_per_batch':data['profile']['objective_trajectories_per_batch'],
      'trajectory_seeds':data['objective_trajectory_seeds'],
      'height_seeds':data['objective_flight_height_seeds'],
      'f0_ceiling':problem['aep_ceiling_gwh'],
      'f0_loss_reference':problem['aep_loss_reference_gwh'],
      'f1_reference':problem['rotor_transit_reference'],
      'f1_metric':problem['primary_bird_objective'],
      'FHD_sha256':data['flight_height_distribution']['input_sha256'],
      'floris_config_sha256':data['aep_model']['config_sha256'],
      'turbine_sha256':data['aep_model']['installed_turbine_sha256'],
      'min_spacing_m':problem['minimum_spacing_m'],
      'site_width_m':6000., 'site_height_m':6000.,
      'behaviour':problem['behaviour'],
      'rotor_geometry':problem['rotor_geometry'],
      'rotor_plane_heading_deg_toward':problem['rotor_plane_heading_deg_toward'],
      'initial_population_source_rows':data['initial_population_source_rows'],
      'algorithm':'NSGA2(pop_size=80, sampling=saved_80_vectors, eliminate_duplicates=True); inherited pymoo defaults',
    }


def audit():
    print('--- METRIC 4b ORIGINAL ARCHIVE / CODE AUDIT ---',flush=True)
    data=load_archive()
    p=data['problem']; a=data['aep_model']; prof=data['profile']; versions=package_versions()
    require(data['run_status']=='completed', 'original run completed')
    require(data['function_evaluations']==8000,'original evaluations',data['function_evaluations'],8000)
    require(data['training_pareto_size']==51,'original Pareto size',data['training_pareto_size'],51)
    require(prof['population']==80 and prof['generations']==100,'80 individuals / 100 generations')
    require(data['seed']==142,'original seed',data['seed'],142)
    require(prof['objective_batches']==5 and prof['objective_trajectories_per_batch']==2000,'optimisation sample: 5 x 2000 birds')
    require(data['objective_trajectory_seeds']==list(range(142,147)),'trajectory seeds 142..146')
    require(data['objective_flight_height_seeds']==list(range(10142,10147)),'height seeds 10142..10146')
    require(a['number_of_direction_sectors']==12 and a['number_of_combined_conditions']==84,'12 sectors / 84 conditions')
    require(p['primary_bird_objective']=='E_RT_CE','conditional-expectation rotor-transit objective')
    require(abs(p['rotor_transit_reference']-REFERENCE_Q)<1e-18,'bird objective fixed reference',p['rotor_transit_reference'],REFERENCE_Q)
    require(abs(p['aep_loss_reference_gwh']-AEP_LOSS_REF)<1e-10,'AEP loss reference')
    require(abs(p['aep_ceiling_gwh']-REFERENCE_AEP)<1e-10,'AEP ceiling')
    require(sha(CONFIG)==a['config_sha256']==EXPECTED_CONFIG_HASH,'FLORIS YAML checksum matches original archive and Metric 4a YAML')
    require(sha(FHD)==data['flight_height_distribution']['input_sha256'],'300-point northern-gannet FHD checksum')
    require(data['flight_height_distribution']['height_row_count']==300,'FHD 300 rows')
    require(data['flight_height_distribution']['minimum_height_m']==0 and data['flight_height_distribution']['maximum_height_m']==299, 'FHD heights 0..299')
    require(p['rotor_geometry']['rotor_diameter_m']==242.24 and p['rotor_geometry']['hub_height_m']==150.0,'turbine geometry 242.24 m / 150 m')
    require(p['minimum_spacing_m']==968.96 and p['spacing_constraint_count']==66,'spacing 4D / 66 constraints')
    require(p['behaviour']['response_probability']==0.6 and p['behaviour']['response_radius_D']==2.0 and p['behaviour']['encounter_radius_D']==1.0 and p['behaviour']['return_length_D']==8.0,'behaviour unchanged')
    require(data['runtime_versions']['pymoo']==versions['pymoo'],'pymoo same as original run',versions['pymoo'],data['runtime_versions']['pymoo'])
    require(data['runtime_versions']['floris']==versions['floris'],'FLORIS same as original run',versions['floris'],data['runtime_versions']['floris'])
    # Geometry and entrance sampling are inherited from Metric 4a exactly.
    from config import Scenario
    from model8_meso_response import BehaviourScenario, cross_unit_vectors
    scenario=Scenario()
    behaviour=BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=242.24,response_probability=scenario.birds.response_probability,
        response_radius_D=scenario.birds.response_radius_D,
        encounter_radius_D=scenario.birds.encounter_radius_D,
        deflection_angle_deg=scenario.birds.deflection_angle_deg,
        return_length_D=scenario.birds.return_length_D,
        step_size_D=scenario.birds.step_size_D)
    from run_model9_optimisation import make_populations
    import inspect
    source=inspect.getsource(make_populations)
    require('Model9Population.generate_parallel' in source and 'scenario.site.width_m' in source and 'scenario.site.height_m' in source, 'original corridor source uses same 6 km site extent')
    require('behaviour.response_radius_m + behaviour.return_length_m' in source, 'original corridor longitudinal margin is (2D+8D)')
    span=6000.*math.sqrt(2.)
    info('original Metric 4a / 4b cross-flight corridor width (m)',span)
    info('longitudinal margin (m)',behaviour.response_radius_m+behaviour.return_length_m)
    from model9_optimisation_utils import coordinate_columns, load_initial_population
    initial=ORIGINAL/'selected_initial_population.csv'
    require(initial.is_file(),'original initial population file present')
    stored=pd.read_csv(initial)
    coords=coordinate_columns(12)
    require(stored.shape[0]==80 and len(coords)==24,'saved 80 initial layouts / 24 coordinates')
    reconstructed, labels, rows=load_initial_population(CODE/'data/model8_initial_population.csv',population_size=80,seed=142,number_of_turbines=12,site_width_m=6000.,site_height_m=6000.,minimum_spacing_m=968.96)
    err=float(np.max(np.abs(reconstructed-stored[coords].to_numpy(dtype=float))))
    require(err<=1e-9,'initial population matches saved source',err,'<=1e-9 m')
    require(rows==list(range(80)),'initial source row order 0..79')
    counts=dict(Counter(labels))
    require(sum(counts.values())==80 and sum(v for k,v in counts.items() if k.startswith('exact_'))==4 and sum(v for k,v in counts.items() if k.startswith('jitter_'))==40 and counts.get('random_feasible')==36,'composition 4 exact / 40 jitter / 36 random', counts, '4/40/36')
    text=(CODE/'run_model9_optimisation.py').read_text(encoding='utf-8')
    require('sampling=initial_x' in text and 'eliminate_duplicates=True' in text, 'original NSGA-II inherited default operators')
    info('original runner wind guard','number_of_conditions != 84 (must be adjusted for 72-sector run)')
    info('original runner Model8 reference guard','expects matching wind-direction count (must allow the documented 12->72 change)')
    info('original runner checkpoint','saves population CSV, NOT full resumable optimiser state')
    info('original code provenance','zip includes v7; original run documented as v4: cannot prove all source files bit-for-bit identical without v4 source/hash; numeric reproduction remains required')
    old=scientific_fields(data,12); new=scientific_fields(data,72)
    diff={key:[old[key],new[key]] for key in old if old[key]!=new[key]}
    require(diff=={'n_sectors':[12,72]},'planned scientific settings diff is sector count only',diff,{'n_sectors':[12,72]})
    report_json('audit_report.json',{'status':'PASS','original_versions':data['runtime_versions'],'laptop_versions':versions,'fhd_hash':sha(FHD),'yaml_hash':sha(CONFIG),'initial_counts':counts,'initial_max_coord_error_m':err,'cross_flight_corridor_width_m':span,'longitudinal_margin_m':behaviour.response_radius_m+behaviour.return_length_m,'source_verification_limitation':'v7 code, original v4 code unverified','planned_scientific_diff':diff,'original_objectives':{'f0':f'({REFERENCE_AEP} - AEP)/{AEP_LOSS_REF}','f1':f'Q_4b/{REFERENCE_Q}'}})
    print('PASS | ARCHIVE/CODE AUDIT COMPLETE; NO OPTIMISATION STARTED')


def setup_simulations():
    from config import Scenario
    from model8_meso_response import BehaviourScenario
    from model9_rotor_transit import RotorGeometry, DiscreteFlightHeightDistribution, RotorTransitModel
    from run_model9_optimisation import make_populations
    data=load_archive()
    scenario=Scenario()
    rotor=RotorGeometry(rotor_diameter_m=242.24,hub_height_m=150.)
    birds=scenario.birds
    behaviour=BehaviourScenario.from_rotor_diameters(rotor_diameter_m=rotor.rotor_diameter_m,response_probability=birds.response_probability,response_radius_D=birds.response_radius_D,encounter_radius_D=birds.encounter_radius_D,deflection_angle_deg=birds.deflection_angle_deg,return_length_D=birds.return_length_D,step_size_D=birds.step_size_D)
    fhd=DiscreteFlightHeightDistribution.from_csv(FHD,name='northern_gannet_Johnston2014_Gannet.est',source=json.loads((CODE/'data/northern_gannet_Johnston2014_FHD_metadata.json').read_text())['primary_citation'])
    require(fhd.input_sha256==data['flight_height_distribution']['input_sha256'],'flight-height distribution matches archive')
    model=RotorTransitModel(behaviour,rotor,rotor_plane_heading_deg_toward=birds.heading_deg_toward,flight_height_distribution=fhd)
    populations=make_populations(distribution=fhd,scenario=scenario,behaviour=behaviour,batch_count=5,trajectories_per_batch=2000,first_trajectory_seed=142,first_height_seed=10142)
    xy=np.load(ORIGINAL/'pareto_decision_vectors.npy')
    require(xy.shape==(51,24),'51 stored layouts each with 24 coordinates',xy.shape,(51,24))
    return scenario,model,populations,xy


def bird():
    print('--- METRIC 4b EXACT OPTIMISATION-SAMPLE REPRODUCTION ---',flush=True)
    data=load_archive()
    from model9_batching import summarise_model9_batches
    scenario,model,populations,xy=setup_simulations()
    layout=CODE/'data/energy_endpoint_layout.csv'
    baseline=pd.read_csv(layout)[['x_m','y_m']].to_numpy(dtype=float)
    t=time.perf_counter(); bas=summarise_model9_batches(model,baseline,populations)
    require(bas['N_failed']==0,'baseline bird simulations completed')
    require(abs(bas['E_RT_CE']-REFERENCE_Q)<=5e-16,'recomputed fixed reference Q_4b',bas['E_RT_CE'],REFERENCE_Q)
    data_frame=pd.read_csv(ORIGINAL/'pareto_fresh_validation.csv').set_index('candidate_id')
    results={}
    for name,i in SAVED_CASES.items():
        expect=data_frame.loc[f'pareto_{i:03d}']
        t0=time.perf_counter(); raw=summarise_model9_batches(model,xy[i].reshape(12,2),populations)
        value=raw['E_RT_CE']
        err=abs(float(value)-float(expect['training_E_RT_CE']))
        require(raw['N_failed']==0 and raw['N_completed']==10000,f'{name} all 10000 birds completed')
        require(err<=5e-16,f'{name} exact archived optimisation-sample Q_4b',f'{value:.17g}',f"{expect['training_E_RT_CE']:.17g} (error {err:.2g})")
        info(f'{name} optimisation-sample Q_4b',format(value,'.17g'))
        results[name]={'pareto_index':i,'Q_4b':value,'archived_Q_4b':float(expect['training_E_RT_CE']),'absolute_error':err,'simulation_seconds':time.perf_counter()-t0}
    report_json('bird_report.json',{'status':'PASS','original_fixed_reference':REFERENCE_Q,'recomputed_reference':bas['E_RT_CE'],'baseline_seconds':time.perf_counter()-t,'results':results})
    print('PASS | BIRD REPRODUCTION COMPLETE; NO OPTIMISATION STARTED')


def aep():
    print('--- METRIC 4b LOCAL 12- AND 72-SECTOR FLORIS REPRODUCTION ---',flush=True)
    from config import Scenario, WindResource
    from floris_aep import FLORISAEPModel
    data=load_archive()
    vectors=np.load(ORIGINAL/'pareto_decision_vectors.npy')
    require(vectors.shape==(51,24),'saved coordinate array shape')
    originals=pd.read_csv(ORIGINAL/'pareto_fresh_validation.csv').set_index('candidate_id')
    record={}
    for sectors in (12,72):
        model=FLORISAEPModel(Scenario(wind=WindResource(n_sectors=sectors)),CONFIG,turbine_reference_path=CODE/'floris_config/iea_15MW_REFERENCE.yaml')
        require(model.number_of_conditions==sectors*7,f'{sectors} sectors / {sectors*7} conditions',model.number_of_conditions,sectors*7)
        ceiling=model.ideal_aep(12)
        require(abs(ceiling-REFERENCE_AEP)<=1e-7,f'{sectors}-sector wake-free AEP invariant',f'{ceiling:.12f}',f'{REFERENCE_AEP:.12f}')
        record[str(sectors)]={'wake_free_aep_gwh':ceiling}
        for name,i in SAVED_CASES.items():
            t=time.perf_counter()
            computed=model.evaluate(vectors[i].reshape(12,2))
            dur=time.perf_counter()-t
            info(f'{name} {sectors}-sector AEP (GWh/year)',f'{computed:.12f} ({dur:.2f}s)')
            record[str(sectors)][name]={'aep_gwh':computed,'seconds':dur}
            if sectors==12:
                archived=float(originals.loc[f'pareto_{i:03d}','training_aep_gwh'])
                require(abs(computed-archived)<=0.001,f'{name} original 12-sector AEP reproduced',f'{computed:.12f}',f'{archived:.12f}; abs error <=0.001')
    require((REPORT/'bird_report.json').is_file(),'Step 2 bird reproduction passed before AEP invariance claim')
    saved=json.loads((REPORT/'bird_report.json').read_text())
    require(saved['status']=='PASS','previous bird report says PASS')
    for name in SAVED_CASES:
        info(f'{name} Q_4b unaffected by wind sectors (wind-independent bird code)', saved['results'][name]['Q_4b'])
    report_json('aep_report.json',{'status':'PASS','results':record,'bird_report_reused':saved['results']})
    print('PASS | FLORIS REPRODUCTION COMPLETE; NO OPTIMISATION STARTED')


def benchmark():
    print('--- METRIC 4b SINGLE OBJECTIVE EVALUATION BENCHMARK ---',flush=True)
    require((REPORT/'bird_report.json').is_file() and (REPORT/'aep_report.json').is_file(),'earlier bird and AEP reproduction reports exist')
    from config import Scenario, WindResource
    from floris_aep import FLORISAEPModel
    from model9_batching import summarise_model9_batches
    _,model,populations,vectors=setup_simulations()
    aep=FLORISAEPModel(Scenario(wind=WindResource(n_sectors=72)),CONFIG,turbine_reference_path=CODE/'floris_config/iea_15MW_REFERENCE.yaml')
    layout=vectors[32].reshape(12,2)
    # Exactly one objective evaluation includes energy and all 5x2000 birds.
    wall0=time.perf_counter(); cpu0=time.process_time()
    energy=aep.evaluate(layout)
    aep_wall=time.perf_counter()-wall0
    bird_wall0=time.perf_counter()
    result=summarise_model9_batches(model,layout,populations)
    bird_wall=time.perf_counter()-bird_wall0
    wall=time.perf_counter()-wall0; cpu=time.process_time()-cpu0
    require(result['N_completed']==10000 and result['N_failed']==0,'one combined objective evaluation completed')
    est_hours=8000*wall/3600.
    cores=os.cpu_count()
    info('complete evaluation wall / CPU seconds',{'wall_s':round(wall,3),'cpu_s':round(cpu,3)})
    info('AEP seconds / bird-only seconds',{'aep_s':round(aep_wall,3),'birds_s':round(bird_wall,3)})
    info('naive 8000-evaluation runtime (hours)',round(est_hours,2))
    info('logical CPU cores visible / NSGA-II process workers',{'logical_cores':cores,'worker_processes':1})
    # 15 x 100000 x 2 endpoints x 2 response probabilities x 2 height scenarios
    # Each birds-only evaluation is a separate 100k-bird simulation per scenario.
    confirmation_bird_simulations = 15 * 100000 * 2 * 2 * 2
    rough_confirm_hours = (confirmation_bird_simulations / 10000.)*bird_wall/3600
    info('very rough independent evaluation + controls/fixed-height (hours)',round(rough_confirm_hours,2))
    print('CAUTION | 100,000-bird batches are not benchmarked; memory and runtime may grow nonlinearly.')
    print('CAUTION | estimate excludes writing results, setup, failures, and separate re-evaluation of any other layouts.')
    print('CAUTION | do not run 4a and 4b simultaneously until CPU and memory contention are tested.')
    report_json('benchmark_report.json',{'status':'BENCHMARKED','wall_s':wall,'cpu_s':cpu,'aep_wall_s':aep_wall,'birds_wall_s':bird_wall,'naive_8000_eval_hours':est_hours,'logical_cpu_cores':cores,'NSGA2_worker_processes':1,'rough_15x100000_2_endpoint_4_scenario_confirmation_hours':rough_confirm_hours,'note':'100k batches not benchmarked and may scale nonlinearly'})
    print('BENCHMARK COMPLETE; NO OPTIMISATION STARTED')


if __name__=='__main__':
    arg=argparse.ArgumentParser(description='Metric 4b 72-sector pre-flight: read-only original archive verification')
    arg.add_argument('command',choices=['audit','bird','aep','benchmark'])
    args=arg.parse_args()
    {'audit':audit,'bird':bird,'aep':aep,'benchmark':benchmark}[args.command]()
