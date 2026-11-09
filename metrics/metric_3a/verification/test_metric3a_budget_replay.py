#!/usr/bin/env python3
"""Isolate Metric 3a generation-budget/replay behavior without changing saved runs.

Place in bird_layout_opt root; in floris-env run:
    python test_metric3a_budget_replay.py

Reruns the original 120-generation GA twice *in memory*, compares both
trajectories with archived original gen120 and newer gen300 first 120.
No output files written; no results overwritten.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from config import Scenario
from directional_absorbing_risk import DirectionalAbsorbingRiskModel
from risk_only_problem import RiskOnlyLayoutProblem
from run_validation import GridSampling
from run_directional_absorbing_risk_only import DirectionalRiskCallback

ROOT = Path(__file__).resolve().parent
NAME = 'angle225p0_down8p0D_cross1p0D_seed42_pop100_gen{}'
FOLDER = ROOT / 'results' / 'directional_absorbing_risk_only'

def load_history(gen):
    p=FOLDER/NAME.format(gen)/'convergence.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}')
    frame=pd.read_csv(p)
    indices=frame['generation'].to_numpy(dtype=int)
    if not np.array_equal(indices, np.arange(1,gen+1)):
        raise RuntimeError(f'Unexpected generation labels: {p}')
    return frame['best_risk'].to_numpy(dtype=float)

def run_120():
    scenario = Scenario()
    model = DirectionalAbsorbingRiskModel(
        scenario=scenario, bird_travel_deg=225.0,
        downstream_scale_D=8.0, cross_scale_D=1.0,
        maximum_shielding=0.8)
    problem = RiskOnlyLayoutProblem(scenario=scenario, risk_model=model)
    sampling = GridSampling(scenario=scenario, seed=42, model_name='constant')
    algorithm = GA(pop_size=100, sampling=sampling, eliminate_duplicates=True)
    callback = DirectionalRiskCallback(scenario=scenario, risk_model=model)
    minimize(problem, algorithm, termination=('n_gen',120),
             seed=42, callback=callback, verbose=False)
    gens=np.asarray(callback.data['generation'],dtype=int)
    values=np.asarray(callback.data['best_risk'],dtype=float)
    if not np.array_equal(gens,np.arange(1,121)):
        raise RuntimeError(f'Unexpected replay generations {gens}')
    return values

def compare(label,a,b):
    d=np.abs(a-b)
    bad=np.flatnonzero(d>1e-9)
    if len(bad)==0:
        print(f'{label}: MATCH within 1e-9; max |delta|={np.max(d):.3e}')
    else:
        i=int(bad[0]); print(f'{label}: DIFFERENT; first at generation {i+1}; '
            f'values={a[i]:.12f} vs {b[i]:.12f}; max |delta|={np.max(d):.6f}')
    return len(bad)==0

def main():
    print('Metric 3a: same-environment GA replay / budget diagnostic')
    print('Running 120 generations twice in memory. Saved originals are read-only.\n')
    old=load_history(120)
    new300=load_history(300)[:120]
    a=run_120()
    b=run_120()
    stable=compare('Replay A vs replay B',a,b)
    vs_new=compare('Replay A vs newer 300-gen run (first 120)',a,new300)
    vs_old=compare('Replay A vs original 120-gen run',a,old)
    print('\nINTERPRETATION')
    if stable and vs_new and not vs_old:
        print('Current environment reproduces the new 300-generation run through generation 120.')
        print('The different total-generation budget is NOT the cause in this environment.')
        print('The historical environment or another unrecorded runtime factor differs; the exact cause remains unknown.')
    elif stable and vs_old and not vs_new:
        print('Current 120-generation replay reproduces the historical 120-generation result but not the newer 300-generation run.')
        print('Investigate why the 300-generation trajectory differs (budget-sensitive random handling or runtime changes).')
    elif not stable:
        print('Even two fresh 120-generation runs differ: investigate current-process nondeterminism.')
    elif stable and vs_new and vs_old:
        print('All three histories agree; recheck earlier CSV comparison and source folders.')
    else:
        print('Neither archived history matches the fresh replay; isolate current runtime/environment changes.')
    print('No files modified.')

if __name__=='__main__':
    main()
