# Metric 1d — smooth central hotspot (formerly Model 4)

MSc thesis: **Integrating Bird Collision Risk into Multi-Objective Offshore Wind Farm Layout Optimisation**.

**Authoritative archived experiment:** **Jensen wake model, 360 wind sectors (1°)**; population **120**, generations **150**, seed **42**, `pymoo==0.6.1.5`. This archive documents the *completed Jensen experiment*, **not** a FLORIS re-optimisation. Python source files are copied verbatim.

## Metric definition and interpretation

Twelve turbine centres `(x_i, y_i)` are optimised within a 6000 × 6000 m site. A **synthetic smooth hotspot** is centred at `(3000,3000)` m with Gaussian spread **sigma=2000 m**, maximum per-turbine field value **1000** and density multiplier **1**:

```text
q_i = 1000 * exp(-((x_i-3000)^2 + (y_i-3000)^2)/(2*(2000)^2))
Q_1d(X) = sum_i q_i
f_bird(X) = Q_1d(X)/12000
```

The score is continuous, positive and lower farther from the hotspot centre. It is an **artificial bird-activity score, not calibrated bird collision probability or mortality**. The raw value is presented in results; its division by 12000 is an optimisation scale only. The values are not a direct measure of ecological consequence.

The optimiser minimises the following Jensen energy objective, which is equivalent to maximising AEP:

```text
f_AEP(X) = 1 - (AEP - 0.5*AEP_ref)/(0.5*AEP_ref)
         = 2 - 2*AEP/AEP_ref
AEP_ref = 754.4007727706155 GWh/year
```

**Caution:** `AEP_ref` was reconstructed from the archived normalised objectives, not traced to an original reference layout or physical AEP derivation. It is **not** the Jensen wake-free ceiling (approximately 760.663 GWh/year). Preserve the exact archived reference in any reproduction. Fixed positive affine objective transformations preserve Pareto dominance; they do not assign a subjective weight to either objective.

## Settings recorded in the final archive

| Item | Value |
|---|---|
| Layout | 12 turbines; 6 km × 6 km square; 24 continuous coordinates |
| Wake model | Jensen top-hat, hub-centre overlap, `k_w=0.04`; root-sum-squares deficits (capped at 0.95) |
| Turbine | Simplified 15 MW turbine, `D=240 m`, hub height `150 m`, minimum spacing `4D=960 m` |
| Cut-in/rated/cut-out speeds | 3 / 10.6 / 25 m/s |
| Thrust coefficient | 0.8 for 3 <= U < 10.6 m/s; 0.25 for 10.6 <= U <= 25 m/s; zero otherwise |
| Wind directions | 360 × 1° from-direction sectors, dominant **225° FROM**, weights `exp(2*cos(theta-225°))` and normalisation to 1 |
| Speed bins | 4, 6, 8, 10, 12, 14, 16 m/s |
| Speed weights | Discrete Weibull density weights, `k=2`, `A=10 m/s`, renormalised over bins |
| Joint weights | Product of direction and wind-speed probabilities |
| Risk parameters | hotspot centre (3000,3000) m, Gaussian `sigma=2000 m`, `risk_high=1000`, `risk_multiplier=1` |
| Risk normalisation | `Q_1d/12000` |
| AEP normalisation | `2-2*AEP/754.4007727706155` |
| Optimiser | NSGA-II, pop 120, generations 150, seed 42, duplicate elimination |
| Initial population | 40% risk-aware; 30% random feasible; 30% jittered grid-like |
| Hypervolume reference | `(1.2,1.2)` in the objective space |
| Endpoint AEP resolution check | Each endpoint independently compared at 1° and 0.5°; strict relative tolerance 0.2% |

**Crucial default-versus-run distinction:** In `hypothetical_risk.py`, `hotspot_radius_m` has a **900 m default**, despite its radius-like name. In the Gaussian formula, this value acts as the **standard deviation sigma**. The final `run_jensen_1deg.py` expressly passes `hotspot_radius_m=2000.0`; **sigma=2000 m controls the actual Metric 1d bird objective**. The **900 m** value appears separately in `GridSampling` as a *risk-aware initial population bias* and is **not the hotspot sigma**. Retaining these different settings is essential to reproduce the archived experiment. Generic defaults in `config.py` are not the final optimisation settings; the runner fixes 120/150/42.

Wind conditions are calculated from `config.py` using synthetic distributions; no external wind rose, OpenFAST workbook, FLORIS model or imported ecological field dataset is needed. These Jensen runs do not optimise yaw angles.

## Final saved results (1° Jensen)

| Saved endpoint | AEP [GWh/year] | Raw Q_1d | Layout ID |
|---|---:|---:|---:|
| Highest AEP | 734.741831 | 2590.249892 | 50 |
| Lowest risk | 728.895074 | 2135.823192 | 32 |

The saved non-dominated set contains **68 layouts**. The score decreases by **454.426700 units**, or **17.5437%** relative to the highest-AEP endpoint, while AEP decreases by **5.846757 GWh/year** (0.7958%). These are observed endpoint differences within this particular stochastic search, not proof of the global Pareto front.

The archived endpoint resolution checks are **0.088474%** (highest-AEP) and **0.020944%** (lowest-risk), below 0.2%. These are checks on numerical wind-direction discretisation for the selected layouts, not proof that NSGA-II reached the global optimum.

`anchor_summary.csv` and `layout_knee.csv` preserve the originally selected knee candidate. Its presence is an original result artefact; this repository **does not assert that an automatically chosen knee is scientifically privileged**.

## Provenance and integrity

- **Final source:** `jensen_1deg_approved.zip::jensen_1deg_approved/`.
- **Final results:** `1d.zip::metric_1d_360sectors_seed42_pop120_gen150/`.
- **Historical Model 4:** `m4.zip::validation_point_hotspot_mult1.0_seed42_pop120_gen150/`.
- All **six recorded run-source SHA-256 hashes** match the approved source archive exactly. The seventh copied module, `plot_saved_pareto.py`, is present and byte-identical to the Metric 1c package but is not one of the original six recorded hashes.
- All 68 saved `pareto_full.npz` decision vectors agree with `pareto_layouts_long.csv`; the Gaussian formula independently recalculates the archived raw Q values; all saved layouts meet the 960 m spacing and 6 km bounds, and the archived points are mutually non-dominated.
- Archived report AEP agrees with optimisation AEP, and both saved 0.5° endpoint checks passed. See `documentation/metric_1d/archive_consistency_checks.json` and `reproducibility_audit.md`.

**Limits:** This packaging audit did *not* execute the Jensen AEP evaluator or the optimiser, and a clean-environment exact reproduction has *not* been established. There is no complete saved population evolution, original standalone test execution log, or complete transitive dependency lock. Python versions below come from the final run metadata.

## Historic Model 4 (12-sector)

`m4.zip` contains **44 historical retained layouts** and 14 result/plot files. `model4(1).zip` contains 11 output files **identical to corresponding members of `m4.zip`**, while `m4.zip` has 3 extra thesis plot variants. Only the latter superset was retained here; this is not a byte-identical ZIP duplicate. Historic highest-AEP is approximately **754.551441 GWh/year**, whereas the final 1° Metric 1d highest-AEP is **734.741831 GWh/year**. Results are not interchangeable.

No source hashes or `run_settings.json` for the historical Model 4 run were located. Separately recovered historical Python snapshots are included for provenance under `documentation/metric_1d/historical_model4/source/`; these are **not authenticated** as the exact older executed files. Do not claim bitwise reproduction of the 12-sector run.

## Recommended GitHub paths

- `metrics/metric_1d/run_jensen_1deg.py` — original runner (same source bytes as 1a–1c).
- `metrics/metric_1d/README.md` — this metric-specific README, **not the repository-root README**.
- `configurations/config.py` — shared original Python configuration.
- `configurations/metric_1d_run_settings.json` — original approved settings template.
- `configurations/requirements_metric_1d.txt` — reconstructed direct dependency versions, not original full environment export.
- `shared/` — original `wake.py`, `hypothetical_risk.py`, `validation_problem.py`, `run_validation.py`, `plot_saved_pareto.py`.
- `results/metric_1d/metric_1d_360sectors_seed42_pop120_gen150/` — final archived outputs and full retained coordinate set.
- `results/metric_1d/historical_model4_12sectors/` — separately labelled historical outputs.
- `documentation/metric_1d/` — source checks, original file locations and historical provenance.

## Reproduction instructions (not executed during package audit)

The source uses original top-level imports such as `from config import Scenario`. To reproduce **without modifying Python source**, stage the exact original scripts in one new working directory:

```powershell
$repo = 'C:\path\to\offshore-wind-layout-bird-risk-optimisation'
$work = 'C:\temp\metric_1d_replay'
New-Item -ItemType Directory -Force $work | Out-Null
Copy-Item "$repo\configurations\config.py" $work
Copy-Item "$repo\metrics\metric_1d\run_jensen_1deg.py" $work
Copy-Item "$repo\shared\*.py" $work
Set-Location $work
# Use a compatible Python 3.11 environment matching the direct recorded versions:
python -m pip install -r "$repo\configurations\requirements_metric_1d.txt"
python run_jensen_1deg.py --metric 1d
```

**Warning:** The last command launches 18,000 candidate-evaluation slots (120×150) in NSGA-II and writes to `$work\results\metric_1d_360sectors_seed42_pop120_gen150/`. It was **not executed** while creating this package. The original runner refuses to overwrite an existing output directory. To demonstrate bitwise reproducibility, compare new outputs with `run_settings.json` and the original Pareto files in the archive; library versions and platform can influence stochastic outcomes.

## How to add Metric 1d to your existing repository

Two packages are produced:
1. **`metric_1d_github_package.zip`** — complete standalone archival package (including seven original Python modules and configuration).
2. **`metric_1d_github_addon.zip`** — conflict-free addition to a repository already populated with Metrics 1a–1c. Common Python modules and `configurations/config.py` are excluded **only after verifying their exact equality** with the Metric 1c package. The add-on includes this README under `metrics/metric_1d/` and audit files under `documentation/metric_1d/`, preventing Windows filename collision prompts.

Use the **add-on** for your existing repository; keep the full package as a separate archive. Do not silently replace Jensen 1° results with a later FLORIS harmonisation: publish any such run as a separate experiment/version.
