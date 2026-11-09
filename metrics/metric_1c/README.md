# Metric 1c — central-corridor artificial bird-activity field

**MSc thesis:** *Integrating Bird Collision Risk into Multi-Objective Offshore Wind Farm Layout Optimisation*.

**Earlier name:** Model 3 — Central Corridor (also called `central_corridor`).

**Authoritative archived experiment:** **Jensen 1° / 360 wind-direction sectors**, NSGA-II **120 individuals × 150 generations, seed 42**. The original Python implementations are included verbatim. This package documents the **completed Jensen experiment**, not a FLORIS re-optimisation.

## Definition

The test site contains 12 turbine centres `(x_i,y_i)` in a 6000 m × 6000 m square. Metric 1c prescribes a **horizontal, 3000 m-wide central corridor** extending across the site. Each turbine with `1500 <= y_i <= 4500` m contributes **1000 artificial units**; every turbine outside that corridor contributes zero. The exact boundaries **are inclusive** in `hypothetical_risk.py`.

```text
q_i = 1000  if 1500 <= y_i <= 4500 m
q_i = 0     otherwise
Q_1c(X) = sum_i q_i = 1000 * (# corridor turbines)
f_bird(X) = Q_1c(X)/12000
```

This is a **synthetic positional activity score**, **not** a measured collision probability, bird mortality estimate or calibrated ecological risk. The score is reported raw in the results; division by 12000 is only an optimisation objective scale.

AEP is maximised by minimising the original Jensen objective:

```text
f_AEP(X) = 1 - (AEP - 0.5*AEP_ref)/(0.5*AEP_ref)
         = 2 - 2*AEP/AEP_ref
AEP_ref  = 754.4007727706155 GWh/year
```

The reference AEP above was **reconstructed from archived normalised objectives; its original reference layout/calculation is not documented**. It is distinct from the roughly 760.663 GWh/year Jensen wake-free ceiling. It should not be silently replaced when reproducing this run. Fixed positive objective transformations preserve Pareto dominance.

## Locked final-run assumptions

| Setting | Confirmed recorded value |
|---|---|
| Site | 6000 m × 6000 m, 12 turbines, 24 continuous x/y variables |
| Turbine model | Simplified Jensen 15 MW; D=240 m; hub height=150 m |
| Minimum separation | 4D = 960 m; 66 pairwise constraints; coordinate bounds [0,6000] m |
| Cut-in / rated / cut-out wind speed | 3 / 10.6 / 25 m/s |
| Jensen wake | `k_w=0.04`, hub-centre top-hat cone; RSS wake deficits capped at 0.95 |
| Thrust coefficient | 0.8 for 3 <= wind speed < 10.6; 0.25 for 10.6 <= speed <= 25; 0 outside |
| Wind directions | 360 sectors of 1°, weights proportional to `exp(2 cos(theta - 225°))` (FROM) |
| Wind speed bins | 4, 6, 8, 10, 12, 14, 16 m/s |
| Wind speed probabilities | Weibull density weights, k=2, A=10 m/s; renormalised over the bins |
| Joint wind weights | Product of direction and speed probabilities |
| AEP normalisation | `2 - 2*AEP/754.4007727706155` |
| Bird normalisation | `Q_1c/12000` |
| NSGA-II | `pymoo==0.6.1.5`, population 120, generations 150, seed 42; duplicate elimination |
| Initial sampling | 40% risk-aware, 30% feasible random, 30% jittered-grid |
| Risk-aware sampling | Uses an **initialisation corridor of 1200 m**; this does **not** replace the 3000 m width of the objective field |
| Hypervolume reference | `(1.2,1.2)` in normalised objective space |
| Convergence check | Re-evaluate two endpoints at 0.5° / 720 sectors; strict relative AEP difference <0.2% |

**Important implementation distinction:** `hypothetical_risk.py` defaults to a 1200 m corridor width when constructed without an override, but `run_jensen_1deg.py` **explicitly passes `corridor_width_m=3000.0`** for the authoritative experiment. Do not substitute the default for the final setting. `config.py` contains prototype `OptSettings` defaults (population 30, generations 20), but the final runner explicitly sets 120/150 for Metric 1c.

The synthetic wind distribution is generated within `config.py`; there is **no external wind rose file**, no OpenFAST workbook and no FLORIS model dependency for these Jensen runs. No turbine yaw optimisation is implemented.

## Saved output — final 1° Jensen run

All figures, tables and coordinates are reproduced from the **original result archive**; they have not been regenerated in this package.

| Saved endpoint | AEP [GWh/year] | Raw Q_1c | Corridor turbines |
|---|---:|---:|---:|
| Highest AEP | 734.078559 | 2000 | 2 |
| Intermediate retained layout | 733.581751 | 1000 | 1 |
| Lowest risk | 733.232336 | 0 | 0 |

The final archive retains **3 non-dominated layouts**, stored in `pareto_full.npz` (3 × 24 coordinates) and `pareto_layouts_long.csv` (36 turbine rows). The apparent low AEP penalty (about **0.846223 GWh/year, or 0.115%**) is an observed endpoint difference, not proof that the globally optimum zero-corridor layout has an energy penalty.

The archived endpoint 1°/0.5° checks were **0.068425%** (highest AEP) and **0.021002%** (lowest risk), both below 0.2%. They test wind-direction quadrature for the retained endpoints; they do not prove that the NSGA-II run found the global Pareto front.

The archive also includes a `layout_knee.csv` selected by the old result utility. Its presence is a historical output **not a thesis claim that it is a uniquely meaningful knee**.

## Reproducibility status

- **Exact source integrity:** all **6 source-file SHA-256 hashes** in final `run_settings.json` match the original approved source archive.
- **Package compatibility:** `config.py`, the runner and all shared Python files are byte-for-byte identical to the shared/metric-1b counterparts in the previous GitHub package. The extra Pareto plotter is also identical, but its hash was **not** among the 6 originally recorded in `run_settings.json`.
- **Saved result consistency:** all three NPZ decision vectors match the long CSV, all saved corridor scores match direct coordinate classification, and all layouts satisfy the recorded bounds and 4D spacing constraint. Normalised objectives agree with stored raw values.
- **Archived convergence:** both endpoint 0.5° checks passed as recorded. A read-only consistency audit was performed while preparing this package; see `documentation/metric_1c/archive_consistency_checks.json`.
- **Limitations:** the clean-environment optimiser was **not rerun**, so exact independently reproduced Pareto outputs are not certified. There is no preserved complete evolving population or original standalone unit-test execution log. Recorded direct dependencies are not a transitive lock.

### Historical 12-sector Model 3 archive

`results/metric_1c/historical_model3_12sectors/` contains the **old** outputs, not the authoritative 1° results. Two differently named uploaded copies — `model3(1).zip` and `m3(1).zip` — had **identical contents**; only one set has been retained in this package.

The historical archive contains three points with Q levels **1000, 2000 and 4000**, and a highest AEP of approximately **758.981737 GWh/year**. It does **not** include a saved `run_settings.json`, a `pareto_full.npz` or the entire optimiser population. Recoverable code snapshots are provided in `documentation/metric_1c/historical_model3/source/`, but they cannot be authenticated as the exact files used to generate the older results. In particular the available legacy runner's AEP normalisation reference does not match the reference implied by the archived objective columns. Do **not** describe the historical archive as exactly rerunnable or mix its values with the 1° final front.

## GitHub locations

- `metrics/metric_1c/run_jensen_1deg.py`: unmodified final runner (same bytes as Metrics 1a/1b).
- `metrics/metric_1c/README.md`: this metric-specific explanation; purposely **not** at repository root.
- `configurations/config.py`: original Scenario definitions, shared across Metrics 1a–1d.
- `configurations/metric_1c_run_settings.json`: original approved settings template, not the realised run metadata.
- `configurations/requirements_metric_1c.txt`: recorded direct package versions (new documentation, not an original environment export).
- `shared/`: unmodified `wake.py`, `hypothetical_risk.py`, `validation_problem.py`, `run_validation.py`, `plot_saved_pareto.py`.
- `results/metric_1c/metric_1c_360sectors_seed42_pop120_gen150/`: original final archive.
- `results/metric_1c/historical_model3_12sectors/`: historical outputs, separate from final.
- `documentation/metric_1c/`: manifest, provenance, integrity audit and comparison metadata.

## Reproduction from a clean, compatible environment (not executed during this audit)

The original files use top-level Python imports such as `from config import Scenario`. For a reproduction, **stage the exact unmodified files together** in an external working directory (instead of refactoring the imports):

```powershell
$repo = 'C:\path\to\offshore-wind-layout-bird-risk-optimisation'
$work = 'C:\temp\metric_1c_replay'
New-Item -ItemType Directory -Force $work | Out-Null
Copy-Item "$repo\configurations\config.py" $work
Copy-Item "$repo\metrics\metric_1c\run_jensen_1deg.py" $work
Copy-Item "$repo\shared\*.py" $work
Set-Location $work
# In a Python 3.11 environment matching the archived direct dependencies:
python -m pip install -r "$repo\configurations\requirements_metric_1c.txt"
python run_jensen_1deg.py --metric 1c
```

The final command **will launch NSGA-II** and write into `$work\results\metric_1c_360sectors_seed42_pop120_gen150/`. The runner refuses to overwrite an existing output folder. These are instructions only: the command was **not executed** when building this archival package. For a publication-grade environment recreation, pin transitive package versions and compare newly obtained results and source hashes with this archive; stochastic optimisers can be sensitive to environment/library changes.

## Safe merge into an existing Metric 1a+1b repository

Two packages are provided:

1. `metric_1c_github_package.zip`: **complete stand-alone archival package**, including all required original Python modules.
2. `metric_1c_github_addon.zip`: **conflict-free add-on** for a repository already populated from the Metric 1a/1b packages. It omits shared `*.py` and `configurations/config.py` **only because their SHA-256 bytes have been verified to match existing Metric 1b files**. The add-on still includes the Metric 1c runner, configuration template, results, history, and metric-specific README/documentation. It is not independently stand-alone until combined with the existing repository.

Use the **add-on ZIP** for your current local repository to avoid Windows' duplicate-file dialog. Do not replace repository root README or Metric 1a/1b documentation; this package intentionally uses `metrics/metric_1c/README.md` and `documentation/metric_1c/` to prevent those collisions.

## Provenance

Final source: `jensen_1deg_approved.zip` (`jensen_1deg_approved/`). Final result: `1c.zip` (`metric_1c_360sectors_seed42_pop120_gen150/`). Historical archive: `m3(1).zip` (`validation_central_corridor_seed42_pop120_gen150/`). Full original-bytes inventory and SHA-256 hashes: `documentation/metric_1c/file_manifest.csv`.
