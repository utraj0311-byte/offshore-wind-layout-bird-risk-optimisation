# Metric 4a — experiment provenance

| Experiment | Primary saved source | Wind directions | Optimiser and birds | Scientific interpretation | Available outputs |
|---|---|---|---|---|---|
| Earlier Model 8, 12-sector | `metrics/metric_4a/metric4a_preflight/model8_code/run_model8_multiobjective.py` + `archives/original_optimisation.zip` | 12 × 30°; 84 wind conditions | NSGA-II pop80/gen100/seed42; training 5×2000 seeds 42–46; fresh 3×10000 10042–10044 | **Historical**, 12-sector AEP; the 12-sector optimisation's fresh/robust validated results are not authoritative for the final thesis | Original ZIP and original independent confirmation ZIP |
| Original 12-sector independent confirmation | `model8_code/run_model8_confirmation.py` | AEP inherited from 12-sector optimization | 20×5000 seeds 20042–20061 | Confirmation of historical front; do not merge with 72-sector endpoints | `archives/original_confirmation.zip` |
| **Metric 4a final, 72-sector** | `metric4a_run72.py` calling unmodified Model 8 modules, saved hash manifest | **72 × 5°; 504 conditions** | NSGA-II pop80/gen100/seed42; training 5×2000 seeds 42–46 | **Authoritative optimization**; 38 non-dominated layouts; 831.465402 GWh/year highest-AEP | `results/metric_4a/metric_4a_72sectors/` |
| Metric 4a post-run geometry/AEP checks | `metric4a_postrun_checks.py` | Original 72-sector AEP plus controlled 1° check | Read-only saved-layout checks; no optimisation | Necessary prerequisite for confirmation script `verify`; original output reports **not present** | Source script only; 2 scratch CSVs elsewhere not treated as a full report |
| **Metric 4a independent 72-sector endpoint confirmation** | `metric4a_confirm72.py` | AEP retained from 72-sector layouts; bird corridor widened | 20×5000 seeds 20042–20061; `p=0.6` and `p=0` | **Authoritative independent evaluation** of **two fixed 72-sector endpoints**; not optimisation or full Pareto validation | `results/metric_4a/metric_4a_72sectors_confirmation/` |
| Thesis plotting (corridor adjusted) | `plotting/metric4a_plot72_corridor_adjusted.py` | No AEP recomputation | Reads saved CSVs | Figure generation only, separate from numeric modelling | Source script |
| Final Section 3.4 print-sized figure exports | `plotting/plot_metric4_print_ready_local.py` | No AEP recomputation | Reads existing 4a and 4b result archives | Reporting figure revision only; requires 4b inputs to rerun full plotter | Three original 4a PDFs under `results/metric_4a/thesis_figures/` |

### Model 8 / 8a / 8b names

The original source bundle explicitly identifies its simulator as **Replacement Model 8 code** and its optimisation objective as `E_enc`. The earlier 8a/8b labels were associated in the user's model-development work with alternative exposure, flux, response and encounter formulations, but **no independently identifiable `model8a` or `model8b` source/runner, settings or output archive appears among the inspected original input bundles or the supplied repository**. Therefore their exact scientific equations, code ancestry, and relationship to this replacement implementation are **unverified**, and no 8a/8b code has been reconstructed or inserted here. Do not present the 8a/8b variants as tested by the final Metric 4a files.

### Wind-resolution rules

- The 12-sector archive reports 849.4074 GWh/year for a historical highest-energy training endpoint; its values **must never be silently exchanged** with the final 72-sector 831.4654 GWh/year endpoint.
- The saved 72-sector `settings_diff.json` states the independent scientific change is 12→72 wind sectors, with 84→504 derived combined conditions. Its original 80-layout initial population was checked in the 72-sector runner.
- The 72-sector final run has `completion_status.json` with `fresh_validation` **not run in that stage** and later a distinct completed `confirmation_report.json`. A `fresh_batches=3` setting in the original configuration **is not evidence those batches ran** during the final optimization.
- The independent confirmation widens **bird entry geometry** while retaining the 72-sector AEP from the final optimisation; it is **not** a 1° AEP re-optimisation or another wind-rose run.

### Provenance precision limits

- Historical 12-sector pymoo version and historical YAML SHA-256 are not recorded; the 72-sector code documents this caveat. The 72-sector saved environment reports Python 3.11.15, NumPy 2.4.6, pymoo 0.6.1.5 and FLORIS 4.6.5; the confirmation additionally reports SciPy 1.17.1.
- The 72-sector file hashes prove byte-for-byte provenance of 14 scientific runner inputs; they do **not** by themselves prove that replay produces bit-identical Pareto solutions in a new OS/BLAS environment.
- Source original result archives: `metric_4a_72sectors.zip`, `metric_4a_72sectors_confirmation.zip`, and nested original Model 8 optimisation/confirmation ZIPs from the supplied preflight bundle.
