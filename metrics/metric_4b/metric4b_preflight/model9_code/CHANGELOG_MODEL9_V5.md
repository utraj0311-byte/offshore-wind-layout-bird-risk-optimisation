# Model 9 v5 changes

- Added a pre-specified high-sample endpoint-confirmation runner.
- Frozen the full-run `pareto_032` and `pareto_037` coordinates with SHA-256
  checksums and preserved the original energy layout as a benchmark.
- Added non-overlapping confirmation seeds and explicit guards against reusing
  earlier optimisation or validation seed ranges.
- Added sequential paired batches so every candidate receives common random
  numbers while memory use remains bounded to one 100,000-bird population.
- Added a paired two-sided Student-t primary decision rule and a secondary
  energy-layout benchmark comparison.
- Added batch checkpoints, complete raw outputs, a compact decision JSON, and
  provenance-rich run settings.
- Added nine confirmation and statistical regression tests.
- The full NSGA-II runner remains included for reproducibility, but the next
  scientific action is endpoint confirmation rather than another full search.

