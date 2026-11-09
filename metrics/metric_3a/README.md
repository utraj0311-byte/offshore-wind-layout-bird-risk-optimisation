# Metric 3a — Directional absorbing exposure

## Purpose and formulation
Twelve turbines within a 6 km × 6 km square are evaluated using an upstream/downstream **directional exposure** metric, with no AEP or wake objective. Flight direction is a **mathematical TOWARD heading of 225°**. Define the along-flight and cross-flight vectors as `d=(cos θ,sin θ)` and `n=(-sin θ,cos θ)`. For an upstream turbine `i` and downstream turbine `j`, the original kernel is

`K_ij = exp(-Δs_ij/L_s) exp(-0.5(Δn_ij/σ_n)^2)`

and remaining exposure is `E_j = product_{i upstream of j}(1-α K_ij)`. The metric is `Q_3a = sum_j E_j`. The implementation file retains historical variable names (do not modify it to relabel them).

## Actual settings (historical runs)
- Rotor diameter **240 m**, hub and FLORIS **not used**; 12 turbines, 6 km square, centres inside the site.
- Minimum spacing **4D = 960 m**.
- Downstream scale `L_s = 8D = 1920 m`, cross-flight scale `σ_n = D = 240 m`; shielding `α = 0.8`.
- **Single-objective `pymoo` `GA`**, not NSGA-II; population **100**, generation budget **120**, seed **42**.
- Both original searches are stored in `results/metric_3a/original_120_gen/`: `without_prescribed_start` and `with_aligned_start`.

## Original scientific source
`source/directional_absorbing_risk.py`, `source/risk_only_problem.py`, `source/run_directional_absorbing_risk_only.py`, and `source/run_directional_absorbing_seeded.py` are bytes from the historical verification archive. Historical scenario: `../../configurations/metric_3a/config.py`. Original `run_validation.py` is **already in `../../shared/run_validation.py`** with identical bytes; it is intentionally not duplicated.

## Original results
- Original random-start best `Q_3a = 7.688414314122593` at 120 generations.
- Original aligned-start best `Q_3a = 5.460555118557852` at 120 generations.
- Aligned two-row reference `5.525407012391223`; cross-flow reference `9.560725671350504`; unshielded baseline `12.0`.
- Verification output and four reference/final layouts in `results/metric_3a/verification_20261017/`.
- The original without-starting-layout run developed partial alignment (approximately 5+4+2+1 lane pattern), but did not recover the prescribed two-row reference.

## Extra verification / sensitivity
- Four **fixed layouts** (`aligned reference`, `cross-flow reference`, `original random-start final`, `original aligned-start final`) evaluated at `α=0.01,0.1,0.3,0.8`: `results/metric_3a/fixed_layout_eta_sweep.csv` (no optimisation in that sweep). Ranking remained unchanged.
- Three 300-generation no-prescribed-start runs (seeds 42,43,44) were run locally; their endpoint `Q_3a` values were 6.673218564, 7.764620774 and 7.522406911. **Their original result directories were not uploaded** and are not bundled here. Do not treat these numbers as replacement saved trajectories.
- The new 300-generation seed-42 history differs from the historical 120-generation seed-42 history beginning at generation 2 despite matching archived source hashes and settings. Identical replays in the current environment reproduce the new history. Exact historical dependency versions are unknown; do **not** describe the new run as an extension of the original trajectory.
- Original deterministic verification and later audit scripts are under `verification/`.

## Figure regeneration and environment
Original plotting helpers under `plotting/` are preserved, but their default paths were designed for the earlier local folder layout or the original verification ZIP. Copy required saved result data to the specified inputs or use the scripts' `--archive` / `--data-dir` arguments. The published figure PDFs are stored at `results/metric_3a/thesis_figures/`.

For a clean scientific-code execution, stage the four source `.py` files together with `configurations/metric_3a/config.py` in an **isolated scratch working directory** and ensure the already-present `shared/run_validation.py` is importable (e.g. on `PYTHONPATH`). Run `python run_directional_absorbing_risk_only.py --seed 42 --pop 100 --gen 120 --shielding 0.8`; results should be directed to a new scratch location. **Never run in place in an existing results directory.** This historical experiment is not guaranteed to match bit-for-bit without the unrecorded original dependency versions. The `pymoo` package is required; no FLORIS calculation is used in the original GA.

## Scientific limitations / progression
The directional exponential exposure kernel is an assumed proxy, not blade-strike probability or mortality. Metric 3b introduces the AEP cost of projected-rotor alignment via FLORIS. A lowest-value aligned arrangement was supplied initially in the prescribed-start experiment; separate random-start searches only demonstrate partial lane development.
