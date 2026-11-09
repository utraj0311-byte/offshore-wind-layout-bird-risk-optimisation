# Historical Jensen configuration for Metrics 2a–2b

The **original, unmodified** `config.py` in this folder has 12 wind-direction sectors. It is preserved as a source snapshot. The final Metric 2a/2b runner selects **360 sectors in memory**; see `shared/metric2_360_tools_original/run_metric2_360.py` and the final `run_settings.json` files.

All other shared modules were deliberately omitted from this add-on because the existing Metrics 1a–1d repository already contains **byte-identical** copies:

| Existing repository path | Verified SHA-256 |
|---|---|
| `shared/wake.py` | `122f6f86863cb85bfe4da21d4e5433ea7d0b28dc892356cd12da47ed73c070d8` |
| `shared/run_validation.py` | `e86b0cea38e8900aa82db092ffd5011ab5f95c76934cb2c830786e76c51cd981` |
| `shared/validation_problem.py` | `4f80ef448b32456116c24abea9bbb8b692e69e76634660c27ab6ef51866e6817` |
| `shared/hypothetical_risk.py` | `c49b0d07720c72996dbee08e19b8cff3fac88c29a22f28df7c326dc9c3df6d90` |

The existing `configurations/config.py` specifies **360** wind sectors and is **not** overwritten; it differs from this historical `config.py` in the wind sector count and line endings. In a staged reproduction workspace, place this historical `config.py` at the workspace root with the existing shared modules and the metric-specific scripts (without changing the source imports).

Only `aep_only_problem.py` and `risk_only_problem.py` remain missing as original source files. See `documentation/MISSING_FILES.md`.
