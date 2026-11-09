# Metric 1c — reproducibility and provenance audit

**Scope:** Original Jensen 1° source and archived output; post hoc file integrity and coordinate checks, **without rerunning optimiser or AEP model**.

## Authoritative experiment
- Final archive: `1c.zip`, folder `metric_1c_360sectors_seed42_pop120_gen150/`.
- Model name: `central_corridor`; 360 × 1° wind sectors, pop 120, generations 150, seed 42.
- `run_status=complete`; three retained non-dominated layouts.
- The run's exact six source-file hashes were independently matched against `jensen_1deg_approved.zip`. The original plotter was not listed in the six source hashes but is present and identical to the previous package.

## Read-only verification outcomes
| Check | Observed result |
|---|---|
| Six original SHA-256 code signatures | 6/6 match |
| `.npz` shapes | X=(3,24), F=(3,2) |
| CSV coordinates vs NPZ | Max absolute difference 4.55e-13 m |
| Recomputed raw corridor activity from y values | Q=0,1000,2000 as archived; max difference 0.00e+00 |
| Saved normalised objectives vs formulae | Match within numerical tolerance |
| All 12 turbine positions in 6 km square | Pass, 3/3 layouts |
| All pairwise centre separations >=960 m | Pass, 3/3 layouts; recorded minimum 1043.875552 m |
| Stored minimum distances vs coordinate recalculation | Max absolute difference 2.27e-13 m |
| Independently reported 1° AEP equals optimised 1° AEP (saved report) | Pass, 3/3 layouts |
| Saved 0.5° check, highest-AEP | Pass: 0.06842515% (<0.2%) |
| Saved 0.5° check, lowest-risk | Pass: 0.02100195% (<0.2%) |
| Historical ZIP duplicate data | `model3(1).zip` and `m3(1).zip` have identical member contents |

## Remaining limits
- Original 12-sector Model 3 runner is **not authenticated**; historical outputs have no exact source hashes or run-settings JSON. The recoverable earlier Python code is a **snapshot** only.
- The `AEP_REF_GWH=754.4007727706155` provenance is recorded as reconstructed; the original reference-layout derivation was not recovered.
- No complete saved optimiser evolution, exact transitive environment lock or independent clean-environment NSGA-II reproduction was available here.
- This audit does **not** infer that the retained front is globally Pareto optimal. The report values are the saved output of a particular stochastic search.
- Files were copied byte-for-byte from the accessible archives. No Python source was modified and no figures were regenerated.
