# Shared-file conflicts and deduplication audit

The uploaded base repository contains `configurations/config.py` (SHA-256 `f4348525c07e0e4d02ac8589a97bc6d764a951ff3122fea7a46275c28c6215f6`).

- Historical Metric 3a `config.py` SHA-256 `df56f80c33f8912eb44ef6380451d762ba71aabcd0aa002186bf506618fb074e`: **DIFFERENT** from the existing base config. Kept only at `configurations/metric_3a/config.py`.
- Metric 3b `config.py` SHA-256 `df56f80c33f8912eb44ef6380451d762ba71aabcd0aa002186bf506618fb074e`: DIFFERENT to the existing base config. Stored under `configurations/metric_3b/config.py` for exact-run isolation; no overwrite.
- Historical Metric 3a `run_validation.py` SHA-256 `e86b0cea38e8900aa82db092ffd5011ab5f95c76934cb2c830786e76c51cd981`: **IDENTICAL** to the existing `shared/run_validation.py`, and therefore not copied into this package.
- FLORIS AEP (`floris_aep.py`) and YAML (`gch_iea15mw.yaml`) are Metric 3b-specific; they are not merged into shared modules.
- Runner `source/run_shielding_multiobjective.py` is historical 12-sector; `source/run_metric3b_72sectors.py` is the final 72-sector runner. Both retained without scientific edits.

