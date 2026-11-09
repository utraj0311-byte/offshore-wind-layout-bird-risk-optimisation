# Replacement Model 8 code

## Purpose

Model 8 simulates explicit two-dimensional bird trajectories through a complete
wind-farm layout. When a trajectory approaches a turbine, it may perform a
hypothetical horizontal meso-response. The model then counts close turbine
encounters and continues tracking the trajectory, so an upstream response can
change encounters with downstream turbines.

The primary metric is:

\[
E_{\mathrm{enc}} = \frac{\sum_i N_{\mathrm{enc},i}}{N_{\mathrm{in}}}.
\]

`E_enc` is mean encounters per incoming simulated trajectory. It is not a
probability, bird flux, collision risk, or mortality estimate. `P_ge_1` is the
fraction of trajectories with one or more encounters.

## Files

- `model8_meso_response.py`: standalone trajectory, response, encounter, and
  accounting engine. It depends only on NumPy.
- `model8_batching.py`: independent-batch aggregation, uncertainty diagnostics,
  fresh non-dominated filtering, and validated anchor selection.
- `tests/test_model8_meso_response.py`: deterministic unit tests for direction,
  geometry, response, encounter counting, common randomness, and accounting.
- `run_model8_validation.py`: aligned, staggered, and wider-gap fixed-layout
  comparisons; response sensitivity; Monte Carlo convergence; seed robustness;
  and step-size convergence.
- `model8_aep_problem.py`: pymoo problem coupling FLORIS AEP to `E_enc` with
  site-bound and 4D spacing constraints.
- `run_model8_multiobjective.py`: complete NSGA-II runner, multi-batch objective,
  output archive, hypervolume, whole-front fresh validation, and validated
  anchor selection.
- `run_model8_confirmation.py`: independent confirmation of every saved Pareto
  layout with new trajectory batches, paired Student-t intervals, both `E_enc`
  and `P_ge_1`, and no repeated FLORIS or NSGA-II calculation.
- `tests/test_model8_confirmation.py`: deterministic tests of the confirmation
  confidence multiplier and paired contrasts.

## Direction convention

All bird headings are **TOWARD** headings in mathematical coordinates:

- 0° = east (+x)
- 90° = north (+y)
- 180° = west (-x)
- 225° = south-west

This reproduces the bird-frame coordinates in the supplied Model 7/10 files.

## Initial hypothetical scenario

The runnable defaults are deliberately explicit:

| Parameter | Initial value | Meaning |
|---|---:|---|
| `p_response` | 0, 0.3, 0.6, 0.9 | Scenario probability of responding on an eligible turbine approach |
| `R_a` | 2D | Behavioural response radius |
| `R_e` | 1D | Counted encounter radius |
| `Delta theta` | 45° | Temporary horizontal deflection |
| `L_return` | 8D | Distance over which the heading returns toward its original direction |
| Spatial step | 0.20D | Numerical trajectory increment |

These values are implementation starting points, not empirically calibrated
species parameters. Only `p_response` should vary in the primary experiment.
The geometric parameters should be checked separately in sensitivity tests.

The official FLORIS IEA 15 MW rotor diameter, 242.24 m, is used by default.
This also makes the 4D spacing constraint 968.96 m. The older prototype
`config.py` value of 240 m should not control the FLORIS-coupled spacing rule.

## Run the standalone validation

From the folder containing the Model 8 files:

```bash
python run_model8_validation.py --n_trajectories 10000
```

The validation runner writes:

- `fixed_layout_summary.csv`
- `per_turbine_encounters.csv`
- `monte_carlo_convergence.csv`
- `random_seed_robustness.csv`
- `step_size_convergence.csv`
- fixed-layout and trajectory plots
- `run_settings.json`

Do not start optimisation until the fixed-layout results and convergence checks
are interpretable.

## Run the FLORIS–NSGA-II coupling

Extract the `model8_code` folder directly inside the existing
`bird_layout_opt` project. The runner will load the project's normalised
`config.py` and `floris_aep.py` from the parent folder. Then run one response
scenario from the project root:

```bash
python model8_code/run_model8_multiobjective.py \
  --floris_config floris_config/gch_iea15mw.yaml \
  --energy_seed_layout report_assets/model_10_final_floris/source_data/energy_endpoint_layout.csv \
  --p_response 0.6 \
  --objective_batches 5 \
  --n_trajectories 2000 \
  --fresh_batches 3 \
  --fresh_trajectories 10000 \
  --validation_z 1.96 \
  --seed 42 \
  --pop 80 \
  --gen 100
```

Do not automatically repeat the six-hour optimisation for every response
probability. First confirm the saved front below. Fixed-layout response
sensitivity is inexpensive; an additional optimiser seed is required only for
an optimiser-independent claim about the existence or absence of a trade-off.

`--n_trajectories` is the number per objective batch, not the combined number.
The command above therefore uses 10,000 optimisation trajectories per candidate
(`5 x 2,000`). `--fresh_trajectories` is likewise per batch, so every training
Pareto layout is independently reevaluated on 30,000 trajectories
(`3 x 10,000`).

## Monte Carlo overfitting control

The original single-batch pilot produced a smooth training Pareto front whose
encounter differences disappeared on fresh trajectories. That is Monte Carlo
overfitting: NSGA-II learned small chance differences in one fixed trajectory
sample. The corrected runner therefore:

1. evaluates every candidate on several independent trajectory batches;
2. sums event counts before calculating the combined `E_enc` and `P_ge_1`;
3. records batch standard deviations and standard errors;
4. reevaluates every training-front layout on independent fresh batches;
5. reconstructs both the exact and uncertainty-aware non-dominated fronts using
   fresh `E_enc`;
6. uses paired batch differences (`validation_z = 1.96` by default) to reject
   layouts whose apparent encounter gain is not distinguishable from fresh
   sample variation; and
7. chooses the reported endpoints and knee only from that robust front.

The main reporting files are now:

- `pareto_points.csv`: training-sample front;
- `training_anchor_summary.csv`: training-only anchors, retained diagnostically;
- `fresh_pareto_validation.csv`: fresh results for every training-front layout;
- `fresh_pareto_batches.csv`: individual fresh-batch results;
- `validated_pareto_points.csv`: uncertainty-aware fresh non-dominated layouts
  only;
- `validated_anchor_summary.csv` and `anchor_summary.csv`: final validated
  anchors;
- `validated_pareto_front.png`: rejected training points, validated front, and
  final anchors.

If the three validated anchor labels resolve to the same `solution_index`, the
fresh front has collapsed to a single defensible layout. This is a valid null
trade-off result and must not be replaced by the visually wider training front.

## Independently confirm a completed front

The optimiser's first fresh validation is useful screening, but it used only
three independent batches. The confirmation runner evaluates **all saved
Pareto layouts** with a completely new set of batches and uses paired 95%
Student-t intervals. The same trajectories are used for every layout within a
batch (common random numbers), while the batches and their seeds are independent
of optimisation and the first fresh validation.

For the completed `seed42`, `p_response=0.6` run, execute this one-line command
from the `bird_layout_opt` project root in Windows Command Prompt:

```bat
python model8_code\run_model8_confirmation.py --result_dir results\model8_robust_main_v2\p0p6_heading225_B5xN2000_FB3xN10000_V1p96_seed42_pop80_gen100 --output_root results\model8_confirmation --confirmation_batches 20 --trajectories_per_batch 5000 --first_seed 20042 --confidence_level 0.95
```

This runs 20 independent batches of 5,000 trajectories for each of the 19
saved layouts. It reuses their previously calculated FLORIS AEP values, so it
does not run FLORIS or NSGA-II. Expected output folder:

```text
results\model8_confirmation\p0p6_heading225_CB20xN5000_seed20042_C95_source_seed42
```

The main files are:

- `confirmation_summary.json`: automatic result classification and seed/audit
  information;
- `confirmation_layout_summary.csv`: all layout-level estimates and flags;
- `confirmation_primary_contrasts.csv`: paired comparisons of the four
  first-validation front layouts against the energy endpoint;
- `confirmation_pairwise_vs_energy.csv`: the same comparison for all layouts;
- `confirmation_batches.csv`: batch values used by the confidence intervals;
- `confirmation_per_turbine_encounters.csv`: final `N_enc_i` hand-off data;
- `confirmation_pareto_fronts.png`: AEP against confirmed `E_enc` and `P_ge_1`.

The confirmation is complete only if every layout has 100,000 started and
completed trajectories, zero failures, 20 batches, and seeds 20042 through
20061. While running, files are held in a uniquely named `_INCOMPLETE_` working
folder in the system temporary area; the final results folder appears only
after all outputs are written. If the process is interrupted, the same command
can therefore be run again safely. A completed final directory is intentionally
refused so earlier evidence cannot be silently overwritten.

## Model 8 closure decision

After confirmation, interpret `E_enc` (mean encounters per trajectory) and
`P_ge_1` (probability of one or more encounters) together:

- if the energy endpoint is the only uncertainty-aware layout for both metrics,
  report a null detectable AEP-encounter trade-off for this defined scenario;
- if another layout has a positive paired interval lower bound for both
  metrics, report a confirmed joint trade-off and its AEP cost;
- if only one metric improves, report metric dependence rather than selecting a
  universal bird-risk optimum.

Minimum remaining work before closing Model 8:

1. archive and review the independent confirmation folder;
2. report the result with its scope: 12 turbines, 6 km square, heading 225°,
   hypothetical behaviour, and `p_response=0.6`;
3. retain the completed fixed-layout sensitivity for `p_response = 0, 0.3,
   0.6, 0.9` and the step-size/accounting checks;
4. either run one additional optimiser seed before making an
   optimiser-independent null claim, or explicitly limit the conclusion to the
   completed seed-42 search;
5. freeze the code, environment, settings, seeds, raw CSVs, and figures; and
6. pass the selected layout's per-turbine `N_enc_i` to Model 10.

Multiple headings/bird roses, flight-height overlap, rotor transit, collision
probability, micro-avoidance, and mortality belong to Model 10. They are not
missing Model 8 calculations.

## Common-random-number control

Every candidate layout receives the same:

- incoming entry positions;
- initial headings;
- response random-number sequence; and
- side-choice random-number sequence.

Random numbers are assigned by approach-event order rather than turbine index.
This keeps the total metric invariant when identical turbine coordinates are
merely relabelled.

## Model 10 must change

The old Model 10 coupled FLORIS to the conservative bird-rose exposure model.
That ecological branch is no longer the final chain. The revised progression is:

1. Model 8 outputs `N_enc_i` for every turbine.
2. Flight-height overlap and rotor geometry convert horizontal encounters into
   expected rotor transits, `N_transit_i`.
3. Collision probability conditional on transit is applied.
4. Micro-avoidance is applied separately, if retained.
5. Turbine collision expectations are summed to the farm value.

Conceptually:

\[
C_i = N_{\mathrm{transit},i}
P_{\mathrm{coll}\mid\mathrm{transit},i}
(1-A_{\mathrm{micro}}),
\qquad
C_{\mathrm{farm}} = \sum_i C_i.
\]

Therefore the old Model 10 fields tied to conservative redistribution—such as
`BIRD_RISK_REFERENCE`, `REDISTRIBUTION_SCALE_D`, `BETA`, lane mass-balance
diagnostics, and the old `bird_risk` objective—must not be silently reused.
The FLORIS AEP branch can remain, but the ecological evaluator, normalisation,
saved columns, plots, summaries, and scientific labels must be revised.

## Verification completed here

- All Python files compile syntactically.
- Eighteen deterministic unit tests pass, including independent batch generation,
  count-first aggregation, fresh non-dominated filtering, and validated anchor
  selection, plus Student-t and paired-confirmation logic.
- An end-to-end confirmation smoke test loaded the completed 19-layout archive,
  evaluated every layout, produced all 13 confirmation artifacts, and passed
  trajectory-accounting and output-manifest checks.
- A 2,000-trajectory fixed-layout run completed for all three layouts and all
  four response scenarios.
- Every run satisfied `N_started = N_completed + N_failed` with zero failed
  trajectories.
- The no-response scenario generated zero response events and straight routes.

The full FLORIS-pymoo optimisation was completed separately in the project's
`floris-env`. The independent confirmation runner itself does not import either
dependency.
