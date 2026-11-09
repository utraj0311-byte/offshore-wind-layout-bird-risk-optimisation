# Metric 1b — reproducibility audit

**Scope:** static source, metadata, and saved-layout inspection only. The optimiser and AEP evaluator were NOT executed. No result files or Python source were changed or regenerated.

- Authoritative source: `jensen_1deg_approved.zip`.
- Authoritative completed output: `metric_1b_360sectors_seed42_pop120_gen150.zip`.
- Historical archive: `model2(1).zip`. Its content duplicate `m2.zip` was omitted.
- Source checksums recorded in final `run_settings.json`: **6/6 match**.
  - `config.py`: MATCH
  - `wake.py`: MATCH
  - `hypothetical_risk.py`: MATCH
  - `validation_problem.py`: MATCH
  - `run_validation.py`: MATCH
  - `run_jensen_1deg.py`: MATCH

- Source modules: **7/7** parse as syntactically valid Python. This does not prove imports succeed in a clean environment.
- `run_status = complete`; optimizer population/generations/seed = **120/150/42**, wind sectors **360**.
- NPZ decision/objective shapes: **(7, 24) / (7, 2)**. All seven `pareto_layouts_long.csv` coordinate sets agree with the NPZ by at most **4.55e-13 m**.
- Field check: at every saved layout, archived Q equals `1000*count(x>=3000 m)` and archived `f_bird=Q/12000`.
- Energy objective check: saved f1 equals `2-2*AEP/754.4007727706155`, to floating precision.
- All seven archived layouts satisfy 0–6000 m bounds and 960 m minimum separation (smallest pair separation **1001.726206 m**); 0 stored spacing/boundary violations.
- `pareto_aep_reported_1deg.csv` records zero differences from optimisation AEP for all 7 layouts.
- Both recorded endpoint checks against 0.5° meet strict <0.2% tolerance; the current audit **inspected saved outcomes, did not rerun Jensen**.
- `hypervolume.csv` has 150 entries.
- Historical archive has 2 Pareto points, with no full NPZ or original run settings. Both imply energy reference `754.400772770616`, but available historical runner computes `aep_max=ideal_aep` and uses different result-folder naming. Exact historical reproduction not established.
- No historical full dependency lock, original unit test transcript, or clean environment rerun was found.
- One source-compatibility detail: Python uses bare module imports; stage copies in a flat working directory for reproduction without editing originals.
