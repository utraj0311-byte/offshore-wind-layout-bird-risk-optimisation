# Historical Model 3 (12-sector) provenance

- Historical result archive selected: `m3(1).zip`, folder `validation_central_corridor_seed42_pop120_gen150/`.
- Duplicate archive `model3(1).zip` (`03_central_corridor_wide/`) has exactly the same 11 member-file byte contents under different directory names; it was excluded to avoid double-counting.
- Contains 3 saved Pareto points (Q=1000,2000,4000) and historic plots/endpoint coordinates.
- These are **NOT** the final Jensen 1-degree (360-sector) Metric 1c results.
- `documentation/metric_1c/historical_model3/source/` preserves separately uploaded legacy Python snapshots with original module names. They are not proven to be the exact executed source of this 12-sector run; no historical `run_settings.json` or run-specific source hashes survive in the accessible result archive.
- The recovered generic legacy runner's objective normalisation does not reproduce the reference implied by the historical saved objective columns. Do not claim bitwise reproducibility of this older run.
