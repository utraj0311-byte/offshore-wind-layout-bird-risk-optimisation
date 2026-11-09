# Metric 3b — Unique projected rotor width and FLORIS AEP

## Purpose and equations
For a fixed bird heading `θ = 225°` **TOWARD**, define cross-flight position for each turbine `s_i = -x_i sin θ + y_i cos θ`. Each rotor of diameter `D` projects to an interval `[s_i-D/2,s_i+D/2]`. The union width is `L_unique = |union_i intervals|` and the thesis Metric 3b is `Q_3b=L_unique/D` (in rotor diameters); the numerical optimiser used `Q_3b/12` as the second minimised objective.

FLORIS AEP is the first minimised objective:

`f_AEP = (AEP_ceiling - AEP(layout)) / AEP_loss_reference`

with archived constants `AEP_ceiling=854.4956585617186 GWh/year` and `AEP_loss_reference=137.96914764866568 GWh/year`. The latter is verified as a used constant, but **the historical five-layout calibration source is not established**.

## Locked thesis run (authoritative)
`results/metric_3b/final_72sectors_seed48_pop160_gen250/`: **72 wind-direction sectors at 5°**, **seven wind-speed bins**, 504 wind conditions, 12 packaged `iea_15MW` turbines with FLORIS 4.6.5, rotor D=242.24m, hub=150m, centre coordinates inside the 6000×6000m site, 4D = 968.96m minimum spacing. FLORIS wind direction uses **FROM**, unlike bird flight TOWARD.

- NSGA-II, seed **48**, population **160**, generations **250**.
- 126 non-dominated layouts in the saved external archive.
- Highest-AEP endpoint: 832.3277511113436 GWh/year, Q_3b=9.530964530757.
- Lowest-Q_3b endpoint: 775.0639685653775 GWh/year, Q_3b=2.0. This two-lane solution was present among the jittered designed starting layouts, so it was **not independently discovered**.
- The 72-sector run was a **new optimisation**, not only a re-evaluation of an earlier 12-sector front.
- Historical 12-sector optimisation is under `results/metric_3b/historical_12sectors_seed48_pop160_gen250/`; do not mix its AEP endpoints with the thesis 5° values.

## Source code and configuration
All scientific Python source under `source/` is copied unchanged from the archived historical v5 code. The **72-sector** runner is provided as `source/run_metric3b_72sectors.py`: exact bytes of the final optimised run's `source_snapshots/new_runner.py`, with only the *filesystem filename* mapped back to the user's local runner filename. The original 12-sector runner is preserved separately as `source/run_shielding_multiobjective.py`.

The 72-sector runner differs scientifically from the original by `scenario.wind.n_sectors = 72`; its remaining differences save raw final population, file hashes and refuse overwrite. Full diff is preserved by these two unmodified runner files.

`configurations/metric_3b/` holds **the exact final-run snapshots** of the 3b `config.py`, `floris_aep.py`, `gch_iea15mw.yaml` and `energy_start_layout.csv`. Do not overwrite the repository's existing shared `configurations/config.py`. The historical `seed_data/prior_pareto_seed_layouts.csv` and its README are located inside `source/seed_data/`.

## Initial population
Five designed exact starting layouts (`energy`, `two_aligned_lanes`, `three_aligned_lanes`, `along_flight_stagger`, `four_separate_lanes`), 60 layouts drawn from 61 in the **v5 development validation** archive (seed 47, pop 160, 120 generations), 75 positional jitters **of the five designed layouts** (20 energy, 11 two lanes, 11 three lanes, 18 stagger, 15 four lanes), and 20 randomly generated feasible layouts. The 60 did **not** come from the completed final 12-sector run. Parent turbines are spatially matched using the Hungarian assignment before simulated binary crossover.

## Verification / figures
Original v5 deterministic unit tests under `source/tests/`; post hoc preflight and 1° validation script under `verification/`. User's local 1° check passed both endpoints by <0.2%, but the generated `validation_1deg_endpoints.csv` was **not uploaded** and must be supplied separately for full audit.

Thesis PDF plots live in `results/metric_3b/thesis_figures/`; `plotting/build_metric3b_thesis_figures.py` reads the frozen 72-sector results folder with `--archive` and saves to `--output-dir`. It performs **no FLORIS evaluation or NSGA-II**.

## Exact reproduction workspace and dependencies
The original v5 runner assumes the following directory arrangement (stage *copies* into a separate scratch directory, not over the user's repository):

```text
<scratch>/
  config.py                    (from configurations/metric_3b/config.py)
  floris_aep.py               (from configurations/metric_3b/floris_aep.py)
  floris_config/gch_iea15mw.yaml
  report_assets/model_10_final_floris/source_data/energy_endpoint_layout.csv
  shielding_model_code/
    run_metric3b_72sectors.py
    run_shielding_multiobjective.py
    shielding_aep_problem.py
    shielding_model.py
    run_shielding_validation.py
    seed_data/prior_pareto_seed_layouts.csv
```

From scratch root, with compatible `pymoo` and FLORIS, the historical command (for an **entirely new output folder**) was:

```bat
python -u shielding_model_code\run_metric3b_72sectors.py --floris_config "floris_config\gch_iea15mw.yaml" --energy_seed_layout "report_assets\model_10_final_floris\source_data\energy_endpoint_layout.csv" --prior_seed_csv "shielding_model_code\seed_data\prior_pareto_seed_layouts.csv" --output_root "results\metric_3b_72sectors_new_reproduction" --bird_heading_deg 225 --corridor_width_m 6000 --aep_loss_reference_gwh 137.96914764866568 --seed 48 --pop 160 --gen 250 --save_evaluations
```

This command reruns **40,000 candidate evaluations** and is provided as documentation only; the packaging audit did **not** invoke it. The archived local environment reported FLORIS 4.6.5 and `pymoo` 0.6.1.5 for the 72-sector run; exact historical 12-sector pymoo version was not recorded. The full raw `all_evaluated_candidates.csv` has been omitted due size. Its omission does not remove the saved 126-point Pareto front, initial population or final population, but it prevents independent reconstruction of the full external archive from every feasible candidate in this GitHub package alone.

## Interpretation and limitations
The metric measures geometrical overlap in projected rotor widths, **not individual bird trajectories or collision mortality**. The geometric ideal Q_3b=2 was supplied via the starting population. The wind rose, turbine diameter and AEP model are not shared with the single-objective Metric 3a; keep original code configurations separate.
