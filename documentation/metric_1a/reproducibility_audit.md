# Metric 1a reproducibility audit

**Audit only:** no optimiser run; no results or Python source code regenerated.

- Authoritative final archive: `metric_1a_360sectors_seed42_pop80_gen100.zip`; completed 360-sector Jensen run, 80x100, seed 42.
- Authoritative final implementation: `jensen_1deg_approved.zip`.
- Historical result archive: `model1(1).zip`; content-duplicate `m1.zip` intentionally omitted.
- Six recorded source SHA256 matches: **6/6**.
  - `config.py`: MATCH
  - `wake.py`: MATCH
  - `hypothetical_risk.py`: MATCH
  - `validation_problem.py`: MATCH
  - `run_validation.py`: MATCH
  - `run_jensen_1deg.py`: MATCH

- Python syntax parsed successfully for all 7 final source files. This is not a run-time import or optimiser check.
- `pareto_full.npz` shapes: `X=(1, 24)`, `F=(1, 2)`. Final anchor 12-turbine positions agree with NPZ: `True`.
- Actual final run status: `complete`; 0.5-degree endpoint check status: `pass`.
- Historical stored f1 implies `AEP_ref=754.4007727706156` GWh/year, while the available old runner uses `ideal_aep` as its maximum, and old saved folder naming lacks `_mult1.0_` in the runner's output format. Historical executable unverified.
- `config(9).py` (historical) uses 12 sectors; final `config.py` in the approved package uses 360. No modification of either file.
- Source dependencies: `numpy`, `pandas`, `matplotlib`, `pymoo` and Python standard library. No downloaded/external wind or turbine source is imported.
- No full historical environment lock, no historical `run_settings.json`, and no original exact Model 1 test log were found.
- No independent reproduction run was performed; package completeness != verified exact reexecution.
