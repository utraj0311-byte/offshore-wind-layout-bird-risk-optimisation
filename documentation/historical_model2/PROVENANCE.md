# Model 2 historical 12-sector archive (superseded)

`model2(1).zip` contains 11 results; `m2.zip` has exactly the same file content (all member SHA-256 comparisons match). The 2 retained historical points are not the final thesis's seven 1° Pareto layouts.

The historical archive's mean inferred AEP scaling reference from the two saved `f1` objective values is `754.400772770616` GWh/year. However, the available historical `run_validation.py` defines `aep_max` by `aep_model.ideal_aep` instead. The original archived executable is therefore not authenticated. The directory naming also differs from the available runner's expected `_mult1.0_` suffix.

12-sector-era Python source snapshots are saved in `source/`. They are actual supplied files, not reconstructed scripts, but there is no original `run_settings.json` tying their hashes to the archived result.

Keep this result separate from `results/metric_1b/metric_1b_360sectors_seed42_pop120_gen150/`.
