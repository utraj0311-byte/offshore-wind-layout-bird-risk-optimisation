# Metric 1d — read-only reproducibility audit

**Scope:** File integrity, fixed-metric objective replay and archived result consistency. **No AEP model or optimiser was executed, and no figures were regenerated.**

## Authoritative source/results

- Source ZIP: `jensen_1deg_approved.zip`, path `jensen_1deg_approved/`.
- Final ZIP: `1d.zip`, path `metric_1d_360sectors_seed42_pop120_gen150/`.
- Final archive declares completion, population 120, 150 generations, seed 42, 360 wind sectors.
- All six source SHA-256 hashes recorded in `run_settings.json` match the original approved source archive.
- Original runner and shared modules have the same bytes as the Metric 1c GitHub package. Python modules passed syntax parsing, but were not executed.

| Read-only check | Result |
|---|---|
| Hash checks | 6/6 matched |
| Final duplicate `1d.zip` vs `1d(1).zip` | Byte-for-byte identical ZIP files |
| Saved NPZ coordinate matrix | X = (68,24), F = (68,2) |
| Saved long CSV | 816 rows (68 layouts × 12 turbines) |
| Maximum NPZ vs CSV coordinate discrepancy | 9.095e-13 m |
| Maximum independent Gaussian Q discrepancy (sigma=2000 m) | 9.095e-13 score units |
| Maximum saved min-spacing vs recomputed value discrepancy | 2.274e-13 m |
| Minimum pairwise turbine spacing | 969.583242 m (required at least 960 m) |
| All 68 layouts within site bounds and spacing constraints | Pass |
| Saved normalised objective columns vs raw data and fixed references | Pass |
| All 68 saved 1° reporting AEP values identical to archived optimisation AEP | Pass |
| All 68 archived layouts mutually non-dominated | Pass (in stored AEP, Q) |
| Archived highest-AEP 0.5° check | 0.08847413% (<0.2%); pass |
| Archived lowest-risk 0.5° check | 0.02094428% (<0.2%); pass |
| Historical Model 4 archives | 11 identical common files; m4.zip has 3 additional figure variants |

## Reproduction limitations

- **No clean Python environment rerun:** archived outputs were checked structurally, without reproducing the Jensen evaluation or NSGA-II candidate trajectory.
- The source's metric default `hotspot_radius_m=900` is **not the final risk definition**: it is explicitly overridden to 2000 in `run_jensen_1deg.py`. The 900 m seeding exclusion radius is a different setting.
- The historical 12-sector Model 4 result archive lacks recorded code hashes and complete run settings. Independently mounted older Python files are only recoverable code snapshots.
- `AEP_ref=754.4007727706155` is reconstructed from the final archive; original derivation unknown.
- The final archive does not preserve the entire per-generation candidate population or an original unit-test run log. Saved `hypervolume.csv` is a convergence diagnostic only, not population history.
- This check cannot establish global Pareto optimality or prove exact cross-platform reproducibility.
