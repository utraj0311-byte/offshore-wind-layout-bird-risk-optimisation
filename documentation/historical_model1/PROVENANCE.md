# Historical Model 1 evidence (not final Metric 1a)

The saved `model1(1).zip` and `m1.zip` have identical file content after stripping their different top-folder names (confirmed by SHA-256 comparison). The included historical data is from `model1(1).zip` only. It contains 11 saved files, with no `run_settings.json` or full population archive.

Its Pareto endpoint is AEP `758.5335371987206` GWh/year, Q `12000`, and f1 `-0.01095641621078180`. The inferred reference value under f1=2-2*AEP/AEP_ref is `754.4007727706156` GWh/year. The available `run_validation.py` instead derives an `aep_max` from wake-free AEP, so this cannot be certified as the exact runner for the saved result. The file naming also differs (`_mult1.0_` in current runner, missing from archive).

The source snapshot `source/config.py` is recovered from `/mnt/data/config(9).py` (wind sectors = 12). Other modules are existing `/mnt/data/*(N).py` Python sources, not newly rewritten source. The exact historical executed source/hashes are not recorded.
