# Metric 4a — meso-scale bird response and turbine encounters

**Scientific implementation preserved; no equations, parameters, trajectories, energy values or optimisation results were regenerated to make this package.**

## Authoritative experiment

Metric 4a corresponds to the **final 72-sector (5°) re-optimisation** of the completed hypothetical meso-response Model 8 implementation. The authoritative files are `metric4a_preflight/metric4a_run72.py` and `metric4a_preflight/model8_code/`, with the 72-sector results in `../../results/metric_4a/metric_4a_72sectors/`. The independent 72-sector endpoint confirmation is a **separate subsequent evaluation**, not a second optimisation. The preceding **12-sector Model 8 optimisation** is archived unchanged in `metric4a_preflight/archives/original_optimisation.zip` for historical provenance.

**Model 8, 8a and 8b:** the available source bundle calls itself *Replacement Model 8 code* and implements trajectory encounters (`E_enc`). The names **8a** and **8b** appear in earlier project discussions about alternative exposure/flux and trajectory formulations, but **no independently identifiable original source/runners/result archives for 8a or 8b were found in the supplied materials**. Their exact code lineage and scientific-to-Metric-4a mapping are unverified. Do not attribute any historical 8a/8b results to this final metric. See `../../documentation/metric_4a/EXPERIMENT_PROVENANCE.md`.

## Mathematical metric

For incoming trajectory `j` and turbine `i`, `I_ji=1` if the horizontal simulated movement intersects the turbine's circular encounter zone at any time, and zero otherwise. The code stores a Boolean `encounter_matrix` so the same bird–turbine pair is counted **at most once**, while a bird may encounter several different turbines:

$$
Q_{4a} = E_{\mathrm{enc}} = \frac{\sum_{j=1}^{N_{\mathrm{in}}}\sum_{i=1}^{12} I_{ji}}{N_{\mathrm{in}}}.
$$

- Numerator: distinct bird–turbine horizontal encounter events (`N_enc_farm`).
- Denominator: **all incoming simulated trajectories** (`N_started`), not only completed or interacting trajectories; recorded production evaluations completed all 10,000 per layout.
- Units: **encounters per entering simulated bird** (a count per bird), not encounter probability, flux, potential strike or mortality.
- Diagnostic: `P_ge_1 = (number of entering birds with >= 1 encounter)/N_started`. Also saved: turbine-level encounters, response counts, failed trajectories and route extension.
- The two minimisation objectives passed to NSGA-II are `f0=(AEP_ceiling-AEP)/AEP_loss_ref` and `f1=Q_4a/encounter_reference`, with constants **854.4956585617186 GWh/year**, **137.96914764866568 GWh/year**, and **0.6854 encounters/bird** respectively. Normalisation changes objective scale, **not** the physical units of saved `Q_4a`.

## Scientific parameters of the archived final run

| Parameter | Actual final setting | Source |
|---|---|---|
| Site/turbines | 6,000 × 6,000 m; 12 IEA 15 MW turbines | `run_settings.json` |
| Rotor diameter/hub height | 242.24 m / 150 m | Saved FLORIS settings |
| Turbine centre bounds | x,y in [0, 6000] m | `model8_aep_problem.py` |
| Minimum pair spacing | 4D = 968.96 m (66 constraints) | `model8_aep_problem.py` |
| Bird heading | 225° TOWARD (southwest), mathematical angle | `run_settings.json`; `model8_meso_response.py` |
| Entry positions | One uniform random position within each equal-width cross-heading bin, bins shuffled, per batch | `TrajectoryPopulation.generate_parallel()` |
| Original cross-flight entry width | 6000√2 = 8,485.281374 m for 225° | Original site projection in trajectory generator |
| Start and stopping lines | Along-heading projected minimum minus (2D+8D); end at projected maximum plus same 10D margin | trajectory generator and `simulate()` |
| Up-/downwind margin | 10D = 2,422.40 m from projected extreme corners | runner sets `longitudinal_margin_m` |
| Response radius | 2D = 484.48 m | run settings |
| Encounter radius | 1D = 242.24 m | run settings |
| Response probability | 0.6 for production; 0 for no-response confirmation | saved optimisation/confirmation settings |
| Temporary turn | ±45° directed away from approached turbine; random tie-break | `MesoResponseModel.simulate()` |
| Heading recovery | After passing the response turbine, returns over 8D = 1,937.92 m | `simulate()` |
| Numerical step | 0.2D = 48.448 m | run settings |
| Flight speed | **Not specified or used**; this is a distance-stepped horizontal model without elapsed-time integration | Original simulator source |
| Repeated events | Approach processed once per bird–turbine pair; each encounter at most once per pair, but further distinct turbines remain eligible | Boolean matrices and approach tracker |
| Stopping | On crossing prescribed downwind progress line; safety cap = 4× nominal straight-flight step count; unfinished trajectories marked failed | `MesoResponseModel.simulate()` |
| Site boundary | No stop solely for leaving the 6-km square; tracked until downwind completion or failure cap | `simulate()` |

**Important corridor distinction:** the independently evaluated endpoints used an **8,787.579614 m widened corridor** so all relevant encounter circles were covered, whereas the final optimisation sampled the original 8,485.281374 m site projection. Confirmation `Q_4a` values published on the original-width basis were multiplied by the ratio of corridor widths, **not simulated again** in the original corridor.

## Optimisation and wind model

- **Population:** 80; **generations:** 100; **optimisation seed:** 42; pymoo `NSGA2(pop_size=80, sampling=FeasibleSeededSampling(...), eliminate_duplicates=True)`; termination `n_gen=100`. No explicit crossover, mutation or repair operators are passed: **pymoo 0.6.1.5 defaults** apply; do not mistake unspecified defaults for experimentally configured parameters.
- Initial population of 80 original feasible layouts: **4 exact prescribed** (`energy`, `aligned`, `staggered`, `wider_gap`), **40 feasible jittered** layouts and **36 random feasible** layouts. Generated by `FeasibleSeededSampling` with seed 42, recorded in `initial_population.csv`.
- Each candidate is evaluated using five fixed, independent bird batches of **2,000 trajectories**, seeds **42–46**, with common random numbers between candidate layouts (10,000 birds per candidate evaluation).
- `fresh_batches=3` × 10,000 and seeds 10042–10044 are **settings inherited from the earlier 12-sector protocol**. The authoritative 72-sector completion/training status says **these fresh-validation batches were not run during that optimisation stage**. Do not confuse that protocol with the later 20×5,000 independent endpoint confirmation.
- **FLORIS 4.6.5**; **72** 5° wind directions, **7** speed bins at [4,6,8,10,12,14,16] m/s, **504** direction/speed conditions. The directional weights are the normalised `exp(2*cos(angle difference from 225°))`; speed-bin weights come from the existing discrete Weibull distribution (`k=2`, `A=10 m/s`). The **exact weight generation** is in the unmodified `config.py` and `floris_aep.py`, not a new hand-entered table.
- FLORIS YAML: `iea_15MW`, `gauss` wake deficit and deflection, `crespo_hernandez` turbulence, `sosfs` combination, turbulence intensity 0.06, shear 0.12, air density 1.225 kg/m³. Yaw is not actively optimised in the source runner; other solver settings are retained in `floris_config/gch_iea15mw.yaml`.
- **72-sector results are newly optimised**, not merely the 12-sector endpoints re-evaluated at higher directional resolution. Saved controlled change record: `settings_diff.json` (12→72 sectors, 84→504 derived conditions, no other recorded scientific changes). A separate post-run **1° AEP check** script exists; its original `endpoint_aep_1deg.json` report is missing from these supplied archives.

## Saved outputs and independent evaluation

- Authoritative Pareto front: **38 non-dominated layouts** (`pareto_points.csv` and `pareto_decision_vectors.npy`); endpoint highest AEP **831.4654021246256 GWh/year** and lowest-risk AEP **810.9579417748998 GWh/year**; saved optimisation-sample encounter objective 0.2369 / 0.1058 respectively.
- Independent endpoint confirmation: **20 paired batches × 5,000 birds**, seeds **20042–20061**, shared trajectories for highest-AEP and lowest-risk, and for `p=0.6` and `p=0`. They are independent of optimisation seeds 42–46. Reported on widened corridor: 0.23510 vs 0.10405 `Q_4a`; paired difference **−0.13105** and two-sided Student-t 95% CI **[−0.134103, −0.127997]**. No-response mean encounter difference CI crosses zero.
- Confirmation `Q_4a` original-width-basis results are about 0.2435 and 0.1078. `P_ge_1` values 21.90% and 7.15% refer to **widened** corridor and are not multiplied by the corridor ratio.
- Check `results/metric_4a/metric_4a_72sectors_confirmation/confirmation_report.json` for all raw counts, uncertainty calculations and controls.
- The nearest initial layouts to the selected lowest-risk layouts suggest aligned lane patterns were **present among the initial seeds**; these optimisation results do not demonstrate independent discovery of those geometric patterns.

## Verification and code inventory

The original Model 8 source includes deterministic tests for straight-flight `p=0`, segment-based intersection and per-turbine encounter counting, labelling invariance, and response diversion. The original `README_MODEL8.md` reports earlier validation outcomes; these tests were **not rerun when building this package**. A dedicated explicit left/right turn-side sign assertion has not been identified; see `MISSING_FILES.md`.

**All 14 input/source file hashes recorded in the authoritative optimisation's `file_hashes.json` were compared to included originals and match exactly.** The original 72-sector saved source and results are not edited. Static Python syntax parsing passed without running the model. Original 12-sector source packages are retained under `metric4a_preflight/archives/` in the exact expected original locations.

Use `../../documentation/metric_4a/FILE_INVENTORY.md` for every file's origin and SHA-256, `EXPERIMENT_PROVENANCE.md` for experiment distinctions, `INTEGRATION_NOTES.md` for local installation and command suggestions, and `MISSING_FILES.md` for blockers.

**Reproduction limitation:** this archive is a scientifically auditable source/data snapshot, **not a confirmed fully executable end-to-end one-command replay**. The original runner's `verify` mode expects local preflight reports, and confirmation `verify` expects original post-run geometry/1°-AEP reports. Those reports were not found, and this package has not executed FLORIS or NSGA-II. The FLORIS bundled IEA 15 MW turbine definition must be supplied by exactly the recorded compatible FLORIS installation.
