# Metric 4b — experiment provenance and wind-resolution audit

| Experiment | Status / wind resource | Main code | Initialisation / seeds | Archived results | Interpretation |
|---|---|---|---|---|---|
| Historical full Model 9 | **Completed, original 12 directions × 7 speeds = 84 conditions** | `model9_code/run_model9_optimisation.py` | NSGA-II 80 × 100, seed 142; 5 × 2,000 trajectory seeds 142–146, height seeds 10142–10146; fixed initial 80-layout set | `results/metric_4b/historical_12_sector/model9_nsga2_full_seed142/` | 51 training Pareto solutions; **not** the final thesis AEP values. Historical settings identify v4 code; available source is v7. |
| Historical Model 9 fresh validation | Completed historical procedure (not final 72-sector evaluation) | `model9_code/run_model9_optimisation.py`, `run_model9_endpoint_confirmation.py`, `run_model9_p038_confirmation.py` | Fresh objective-excluded batches and separate endpoint confirmation, details in historical `run_settings.json` and `data/*manifest.json` | `results/metric_4b/historical_12_sector/model9_nsga2_full_seed142/fresh_validation_batches.csv`; saved endpoint ZIP | Do not replace the final independent 15×100k comparison with this. |
| **Final Metric 4b 5° optimisation** | **Completed genuine re-optimisation: 72 × 7 = 504 FLORIS conditions** | `metric4b_preflight/metric4b_run72.py` + hash-matched `model9_code` | Same 80 initial layouts; 80 × 100, seed 142; 5 × 2,000, trajectory seeds 142–146 and height seeds 10142–10146 | `results/metric_4b/final_72sectors/metric_4b_72sectors/`, 36 Pareto solutions | **Authoritative final thesis result**, AEP 828.6682 / 813.8397 GWh/year endpoints. Re-optimised, **not** AEP-only re-evaluated old layouts. |
| 5° endpoint audit / 1° AEP sensitivity | Script present; original *postrun output report missing* | `metric4b_preflight/metric4b_postrun_checks.py` | Frozen endpoint layouts, no evolution | `metric_4b_72sectors_postrun/endpoint_1degree_aep.json` **not archived here** | Previous discussion reports tests passed, but original report is absent; cannot independently substantiate saved precise values from this package. |
| **Final 5° independent evaluation** | **Completed, based on 72-sector frozen endpoints** | `metric4b_preflight/metric4b_confirm72.py` | Paired **15 × 100,000** birds per layout; seeds 30142–30156 and height seeds 40142–40156; same birds for both endpoints and four sensitivity cases | `results/metric_4b/independent_evaluation_72sectors/metric_4b_72sectors_confirmation/` | Fresh metric evaluation; does **not** recompute AEP or run optimisation. Uses widened 8,577.08 m corridor. |
| Historical model9 fixed-hub-height sensitivity | Script, manifest and predeclared plan preserved; **result archive absent** | `model9_code/run_model9_hub_height_sensitivity.py`, `model9_hub_height_sensitivity.py` | Four historical layouts, all at z=150 m, planned 15 × 20,000; manifest provides seeds | `model9_code/data/model9_hub_height_sensitivity_manifest.json`; **no completed output ZIP** | Must **not** combine with the 72-sector confirmation fixed-height p=0.6/p=0 controls, or with Model 10. |
| Earlier estimator precision check | Verification methods and summary preserved, original batch result files not present | `model9_code/tests/test_model9_rotor_transit.py`, `PRECISION_IMPROVEMENT.md` | Controlled 200,000-height draws; three fixed layouts × five 10,000-bird batches | `model9_code/PRECISION_IMPROVEMENT.md`, `VERIFICATION.md` | Recorded agreement of direct sampling and conditional FHD average; not independently rerun here. |
| Final thesis plots | Archived post-processing; not an optimisation | `plotting/metric4b_plot72_final.py`, `plot_metric4_print_ready_local.py` | Uses saved 72-sector/independent-evaluation arrays | `results/metric_4b/thesis_figures/` and `diagnostic_figures/` | Preserve plotted widths, axes and original sample/independent comparison; print-sized PDF source. |

## Version and code lineage

The `metric_4b_72sectors/file_hashes.json` manifest lists **27 exact inputs/source paths**. All 27 bytes/hashes match this package. The archived `settings_diff.json` identifies **12 → 72 directions** as the sole independently changed scientific configuration for the new optimisation, with normalised direction weights regenerated and 84 → 504 FLORIS conditions. The original historical 12-sector archive explicitly states that its implementation was v4; the available code snapshot carries later v7 changes and cannot prove byte-identical v4 reproduction. Those facts are compatible: the current **5°** run is exactly source-backed, but the historical **12-direction** run is not source-complete at the original version level.

## Wind inputs

There is **no wind-rose CSV** in the original code. `floris_aep.py` computes direction weights from the synthetic scenario and discretised Weibull wind-speed weights. The complete 504-condition and historical 84-condition weight arrays are explicitly preserved in the two `run_settings.json` files. Do not transfer the 12-direction AEP values to the 5° fronts, and do not treat a 1° post-run re-evaluation as a 1° re-optimisation.

## Code and data anchors

- Final runner: `metrics/metric_4b/metric4b_preflight/metric4b_run72.py`.
- Original scalar risk objective: `model9_code/model9_aep_problem.py` (`bird_objective_metric='E_RT_CE'`).
- Original rotor geometry, sampled height, conditional estimator and repeat-crossing handling: `model9_code/model9_rotor_transit.py`.
- Shared inherited horizontal trajectories and behaviour: `model9_code/model8_meso_response.py`; this is **not** silently substituted by Metric 4a's separate copy.
- Original FHD: `model9_code/data/northern_gannet_Johnston2014_FHD.csv` and metadata JSON, with SHA-256 `adc9ea903db9c8e34bfe5b88c382381b737c2e676d79af437f66d198208a0f2c`.
- FLORIS YAML and pinned reference turbine `model9_code/floris_config/gch_iea15mw.yaml` and `iea_15MW_REFERENCE.yaml`.
- Final 5° saved full wind-condition weights `results/metric_4b/final_72sectors/metric_4b_72sectors/run_settings.json`.
- Original historical 12-sector full weights `results/metric_4b/historical_12_sector/model9_nsga2_full_seed142/run_settings.json`.
