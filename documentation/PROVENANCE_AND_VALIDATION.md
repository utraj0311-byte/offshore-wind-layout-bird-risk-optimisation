# Metric 2a / 2b scientific provenance audit

 ## Method

 Source hashes were compared to the original uploaded byte streams and the SHA-256 records in the final `run_settings.json`. `pareto_full.npz` arrays were compared against `pareto_layouts_long.csv`, `pareto_points.csv`, and `anchor_summary.csv`. **Using the original supplied** `config.py`, `wake.py`, `coupled_risk.py`, and `pairwise_risk.py`, the stored 1-degree AEP and both Q metrics were evaluated for each saved Pareto layout; no optimiser was called. Full set: 19 layouts/228 coordinate rows (2a) and 20 layouts/240 coordinate rows (2b).

 - Original Jensen wake-free reference: **760.663033300808 GWh/year** (verified against `AEPModel.ideal_aep(12)`; distinct from the 754.4007727706155 normalisation used for Metrics 1a–1d).
 - Metric 2a: fixed **target index 0**, `interior_risky`, sigma 4D, Q/11. Metric 2b: all 66 unordered pairs, sigma 8D, Q/66.
 - Saved layouts: no spacing violations, no boundary violations, all mutually nondominated, all normalised objectives consistent with `2-2*AEP/760.663033300808` and Q divided by its corresponding denominator.
 - **Maximum absolute difference in recomputed 1-degree AEP**: 0.00e+00 GWh (2a) and 0.00e+00 GWh (2b).
 - **Maximum absolute difference in Q**: 9.80e-17 (2a) and 3.55e-15 (2b).
 - Minimum saved pair distances: 1054.530 m (2a) and 966.818 m (2b); required min: 960 m.
 - Archived histories: 60 (2a) / 100 (2b) generations; full final Pareto arrays retained.
 - Final archive reports both 0.5-degree endpoint checks passed, after 1-degree optimisation. This is a **recorded historical check**; no 0.5-degree calculation or GA run was performed during packaging.
 - AEP-only separate 360-sector GA archived AEP at 1 degree: 734.224321019 GWh/year, matching the saved value.

 ## Exact source-to-run relationship

 The actual final 360-sector runner SHA-256 in both saved `run_settings.json` files is `4c0923ff9431670870078344d3e29d9605cad380b774d20e57df4ba245b1c0dc`, and the preserved runner bytes match. Shared original `wake.py`, `run_validation.py`, and `validation_problem.py` match the actual files in the existing Metrics 1a–1d repository and are reused without copies in this add-on. `config.py` does not match Metric 1d's recorded config hash and is intentionally isolated. The original code's `config.py` defaults to 12 sectors; final scientific resolution is **360**, applied by `run_metric2_360.py` to the in-memory scenario without rewriting the source.

 The 360-sector rerun's archived preflight status is **PASS**, including six earlier saved 12-sector Jensen AEP reproduction cases. This is documentary evidence of the Windows run's preflight, not a rerun during ZIP creation.

 ## Limits on reproducibility claims

 The final data and recorded source hashes are robustly cross-checked and the unchanged main source modules are preserved. **A fresh full three-stage optimisation rerun is not possible from these ZIP bytes alone** because the original `aep_only_problem.py` module is missing; `risk_only_problem.py` is also missing for historical single-objective risk-only scripts. `hypothetical_risk.py` is present in the existing Metrics 1a–1d repository and its SHA-256 matches the saved reference. Historic pymoo version is unknown, while the **final** reruns used 0.6.1.5. No guarantee of identical GA population histories across a different environment. See `documentation/MISSING_FILES.md` for each missing module and recorded expected hashes.

 ## Final versus historical

 `results/metric_2a/final_360sectors/` and `results/metric_2b/final_360sectors/` are authoritative for the thesis's final Jensen metric results. Historical Model 5/6 archives are retained only to support preflight and provenance. Do not mix their AEP objectives or plotted fronts with 360-sector results. Final thesis plot versions are separate post-processing products of the saved final results.
