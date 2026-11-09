# Metric 4a — missing or unverified reproduction inputs

| Item | Status | Consequence |
|---|---|---|
| Independently identifiable original Model 8a source, runner, settings and outputs | **NOT FOUND** among supplied archives | Exact 8a lineage/formulation cannot be established. No reconstruction made. |
| Independently identifiable original Model 8b source, runner, settings and outputs | **NOT FOUND** among supplied archives | Exact 8b lineage/formulation cannot be established. No reconstruction made. |
| `metric4a_preflight/preflight_reports/aep_report.json`, `bird_report.json`, `audit_report.json` | **NOT FOUND** | `metric4a_run72.py verify` explicitly requires these originally generated preflight reports; cannot claim clean one-command verification. |
| `metric_4a_72sectors_postrun/endpoint_aep_1deg.json`, `geometry_report.json` | **NOT FOUND** | `metric4a_confirm72.py verify` requires these original post-run reports; confirmed results are archived, but an unmodified full confirmation replay needs those reports. |
| Original binary checkpoints `checkpoint_latest.pkl` and `checkpoint_latest_previous.pkl` | **AVAILABLE in source 72-sector ZIP but deliberately excluded** | Required only to resume a partly completed run; ~29.6 MB uncompressed, pickle-deserialisation risk and no necessity for fixed-output audit. The original ZIP must be retained separately if exact checkpoint resume is desired. |
| `confirmation_batches_partial.csv` | **AVAILABLE but excluded as redundant** | Duplicate partial/complete batch CSV; full `confirmation_batches.csv` is retained. |
| Standalone saved original 72-sector *three-batch fresh Pareto validation* output | **NOT RUN / no saved result** | Original 12-sector protocol lists seeds 10042–10044; final 72-sector completion file says fresh validation was not run at the optimisation stage. Do not claim it happened. |
| Pre-12-sector single-batch pilot source and outputs | **NOT FOUND** | The historical `README_MODEL8.md` mentions overfitting in a pilot, but its exact results are unavailable for separate reproduction. |
| Dedicated explicit turning-side sign test | **NO RECORDED SPECIFIC ASSERTION FOUND** | Original tests include diversion and response logic, but no independent sign-direction test is identified. |
| Exact historical Python/pymoo version and historical FLORIS YAML hash | **NOT RECORDED** | Strict cross-environment replay of the earlier 12-sector result cannot be claimed. |
| Complete packaged IEA 15 MW turbine definition independent of FLORIS installation | **EXTERNAL FLORIS PACKAGE DEPENDENCY** | YAML uses `iea_15MW` built into the relevant FLORIS installation; exact package version pinned in final result settings. |
| Full scientific run replay / deterministic unit-test execution in this package-build session | **NOT PERFORMED** | Static import/syntax/data/hash checks are not substitutes for running saved model/test executables. |
| `plot_metric4_print_ready_local.py` Metric 4b input archives | **NOT INCLUDED (out of scope)** | The combined print-sized plotting utility requires Metric 4b data; its original 4a PDF outputs are included, no plot rerun done. |
| Source-level historical 12-sector `README_MODEL8` user-supplied energy layout at its original CLI path | **PATH NOT INCLUDED** | The actual seed layout *is* preserved as `metric4a_preflight/model10_energy_seed_layout.csv`; historical CLI example's `report_assets/...` directory is absent. |

**No files above were regenerated, approximated, or silently substituted.** Original source code is byte-for-byte preserved under its expected relative directory, and other original outputs available in these archives are clearly labelled as such.
