# Metric 4a — static import/dependency audit

This checks imports from the actual source text by Python AST. It does **not** import modules, run any simulator, or run FLORIS/NSGA-II.

| Original module | Status | Use |
|---|---|---|
| `metric4a_run72.py` | Original preserved | 72-sector scientific runner and configuration checks |
| `config.py`, `floris_aep.py`, `floris_config/gch_iea15mw.yaml` | Original preserved, saved hashes match | Wind resource and FLORIS AEP |
| `model8_code/model8_meso_response.py` | Original preserved, saved hash matches | Bird trajectories, meso-response, segment-circle encounter detection |
| `model8_code/model8_batching.py` | Original preserved, saved hash matches | Batch summaries and Monte Carlo estimates |
| `model8_code/model8_aep_problem.py` | Original preserved, saved hash matches | Objective scaling and layout constraints |
| `model8_code/run_model8_multiobjective.py` | Original preserved, saved hash matches | Sampling and 12-sector historical runner |
| `model8_code/run_model8_validation.py` | Original preserved, saved hash matches | Fixed-layout validation/seed layouts |
| `metric4a_postrun_checks.py`, `metric4a_confirm72.py` | Original preserved | Post-run geometry/AEP and independent endpoint validation; generated post-run reports absent |
| `plotting/plot_metric4_print_ready_local.py` | Original preserved | Read-only plotter requiring separate Metric 4b data |

**Local flat-module imports:** config, floris_aep, model8_aep_problem, model8_batching, model8_meso_response, run_model8_confirmation, run_model8_multiobjective, run_model8_validation. Every named source module is present in the packaged `metric4a_preflight/` and `metric4a_preflight/model8_code/` directories; the original runner adds both locations to its Python path. Source files have not been rewritten to use package-style imports.

**External imports observed:** cloudpickle, floris, matplotlib, numpy, pandas, pymoo, scipy, yaml. The archived `requirements_model8.txt` is not fully pinned; the final run record supplies Python, NumPy, pymoo and FLORIS versions, and the independent confirmation supplies SciPy. `cloudpickle` is optional with a pickle fallback in the original run controller. The `iea_15MW` definition is supplied by the compatible FLORIS installation rather than a standalone turbine file in this repository.

**Other operational dependencies:** the original final run's verification requires `preflight_reports/aep_report.json`, `bird_report.json` and `audit_report.json`. The confirmation verifier requires `metric_4a_72sectors_postrun/endpoint_aep_1deg.json` and `geometry_report.json`. These outputs are not available in the supplied archives; see `MISSING_FILES.md`.
