# Running the Model 9 optimisation

## What this runner does

`run_model9_optimisation.py` joins the two verified parts of Model 9:

1. FLORIS calculates annual energy production (AEP) for each 12-turbine
   layout using 12 wind-direction sectors and seven speed bins.
2. The Model 9 trajectory model calculates circular rotor-swept-area transit
   exposure. NSGA-II minimises the lower-noise conditional estimator
   `E_RT_CE`; sampled `E_RT` and Model 8 encounter intensity `E_enc` remain in
   the audit output.

There is no wind-input CSV. `config.py` generates and normalises the inherited
direction and Weibull speed weights programmatically, producing 84 FLORIS
conditions.

The endpoint is rotor-transit exposure. It is not collision probability,
fatality, or mortality.

## First run: smoke profile

Use **Anaconda Prompt**. Replace the path below only if you extracted the zip
somewhere else.

```bat
conda activate floris-env
cd /d "C:\Users\Deepak\Desktop\bird_layout_opt\Model_9_Code_v5\model9_code"
python -m pip install -r requirements_model9.txt
python -m unittest discover -s tests -v
python run_model9_optimisation.py --profile smoke --output results/model9_nsga2_smoke
```

The test command should report `Ran 59 tests` followed by `OK`. The smoke run
uses eight layouts, two generations, `2 × 250` optimisation trajectories per
layout evaluation, and `2 × 1,000` fresh-validation trajectories per returned
candidate. It checks the complete FLORIS→NSGA-II→fresh-validation pipeline; it
is not a thesis-quality optimisation.

When it finishes, upload this automatically created file:

```text
model9_code\results\model9_nsga2_smoke.zip
```

The runner refuses to overwrite a non-empty result directory or an existing
result zip. If you repeat a run, choose a new name such as
`results/model9_nsga2_smoke_2`.

## Second run: controlled pilot

Run this only after the smoke archive has been checked:

```bat
python run_model9_optimisation.py --profile pilot --output results/model9_nsga2_pilot
```

The pilot uses 24 layouts, 15 generations, `3 × 1,000` optimisation
trajectories per evaluation, and `3 × 5,000` independent fresh-validation
trajectories per returned candidate. It is intended to reveal whether Model 9
produces a usable trade-off before committing to the full computational cost.

## Full inherited run

The seed-142 full run has now been completed. It produced 51 training Pareto
candidates, 11 ordinary fresh nondominated candidates, and one robust 95%
candidate. Do not repeat the full search merely to increase optimisation
length: run the pre-specified endpoint confirmation in
`README_CONFIRMATION.md` next.

The command below is retained only for exact workflow reproducibility.

```bat
python run_model9_optimisation.py --profile full --output results/model9_nsga2_full
```

The full profile reproduces the Model 8 optimisation scale: population 80,
100 generations, `5 × 2,000` trajectories per objective evaluation, followed
by `3 × 10,000` fresh trajectories per Pareto candidate. The supplied Model 8
run took about 6.25 hours. Model 9 runtime may differ, so keep Anaconda Prompt
open and prevent the computer from sleeping.

## Built-in safeguards

The runner stops rather than silently continuing when:

- the configured diameter or hub height differs from FLORIS' IEA 15 MW
  turbine (`D = 242.24 m`, `H = 150 m`);
- the installed turbine definition differs from the bundled Model 8 reference;
- the wind resource does not contain exactly 84 normalised conditions;
- the northern-gannet FHD checksum changes;
- any seed layout is outside the 6 km × 6 km site or violates
  `4D = 968.96 m` spacing;
- the FLORIS wake-free AEP ceiling differs materially from the supplied Model
  8 evidence run; or
- simulated trajectories fail to complete.

The four exact Model 8 anchors are always included. The smoke and pilot modes
then choose a deterministic seeded subset of the remaining supplied layouts;
the full mode uses all 80.

## Main outputs

- `pareto_fresh_validation.csv`: training and independent fresh values for
  every returned Pareto candidate, including `E_RT`, `E_RT_CE`, intervals,
  feasibility, and robust-screening flags.
- `fresh_validation_batches.csv`: one row per candidate and independent batch.
- `all_evaluated_candidates.csv`: full optimisation audit with coordinates,
  raw metrics, normalised objectives, and spacing checks.
- `anchor_layouts/`: CSV and PNG layouts for the energy endpoint, the
  rotor-transit endpoint, and a descriptive balanced reference.
- `pareto_front_fresh_validation.png`: clean training-versus-fresh trade-off
  plot without a claimed knee point.
- `run_settings.json`: scientific assumptions, seeds, software versions,
  FLORIS/turbine checksums, wind frequencies, baseline values, and runtime.
- `checkpoint_population.csv` and `checkpoint_state.json`: latest completed
  NSGA-II population and generation if a long run is interrupted.

The “balanced reference” is only the point closest to the lower-left corner
after min–max scaling of the two fresh metrics. It is deliberately not called
a knee and does not imply a scientifically preferred layout.
