# Repository archive review — 9 November 2026

## Scope

Examined user-provided archive `offshore-wind-layout-bird-risk-optimisation(3).zip` as a *static* research-code snapshot. This is not a full FLORIS or optimisation execution audit. No original scientific source or result data was modified.

## Verified structure

- ZIP integrity test passed (no CRC errors).
- All 11 expected metric directories exist: `metric_1a` through `metric_4b`, plus `literature_informed`.
- All 11 metric directories contain a metric-specific `README.md`.
- Supporting `configurations/`, `results/`, `documentation/`, and `shared/` directories exist.
- 221 Python source files were parsed successfully with Python's AST parser: **0 syntax errors** (this does not test imports or execution).
- The four `run_jensen_1deg.py` files in Metrics 1a–1d share the same SHA-256 hash, consistent with one reusable runner selected by metric argument.

## Publication blockers or cautions

1. The ZIP had **no root `README.md`**. Git history in the archive shows the original README as deleted. Add the accompanying root README before committing.
2. The ZIP had **no `.gitignore`**. Add the accompanying ignore rules for caches and temporary reproducibility workspaces. Do not ignore existing archived results.
3. The archive includes a hidden `.git/` directory. This is normal for a local Git working tree but is not something to upload manually as repository content. Use GitHub Desktop's **Commit to main** and **Publish/Push**, or produce a distributable snapshot without `.git/`.
4. The saved Git index in the uploaded ZIP shows the five main research directories as **untracked** and the initial root README as deleted. This means the ZIP does **not** establish that the final archive has been committed and pushed to GitHub. Actual current GitHub status must be checked locally.
5. The contents are not guaranteed to be a one-command reproducible package. Examples documented by the metric authors include missing original helper modules for the Model 5/6 lineage, some missing Metric 4a/4b preflight/post-run records, and missing separate Model 10 sensitivity/uncertainty output datasets.
6. Runner import paths and dependencies differ between metric families. Some need staged copies in a scratch directory. Installing one configuration's `requirements.txt` does not necessarily reproduce all metrics.
7. Historical 12-sector data and authoritative 360-sector (Jensen) / 72-sector (FLORIS) data must not be mixed. Metric 3a is a GA risk-only search, not a coupled NSGA-II optimisation.
8. No repository-level software licence was found. Add one only after determining intended reuse rights and checking third-party licences.

## Recommended checks before publishing

- Review the root README and scientific terminology, particularly the status of each final experiment.
- Review `git status` in GitHub Desktop. Commit the added folders and README; push/publish the repository when ready.
- In GitHub's web view, verify all 11 metric folders and that the root README is rendered.
- Verify code and results can be reached from the published commit; optionally create a release tag aligned with thesis submission.
- Do not describe source-only or archive-only checks as full computational reproduction.

## Notes

The original ZIP included a `.git/` directory with only an initial commit visible. No claims are made about any other local clone or remote repository that may exist. ZIP file counts and Python syntax checks describe this uploaded snapshot only.
