# Metric 1b — half-site artificial bird-activity field

**MSc thesis:** *Integrating Bird Collision Risk into Multi-Objective Offshore Wind Farm Layout Optimisation*

**Previously named:** Model 2 — Half-Site Risk Field / eastern half-site field (`half_site_east_risk`).

**Authoritative experiment here:** completed **Jensen 1° (360 direction-sector)** Metric 1b rerun; **seed 42, population 120, 150 generations**. This is not a completed FLORIS optimisation. The historical 12-sector Model 2 archive is retained separately and **must not be mixed** with the final results.

## Purpose and meaning

Metric 1b tests whether NSGA-II responds to a deliberately prescribed spatially varying bird-activity field. The eastern half of the square site is assigned 1000 artificial score units per turbine; the western half receives zero. This **is not a bird-collision prediction** or empirically calibrated hazard field.

For turbine coordinates $\mathbf X=\{(x_i,y_i)\}_{i=1}^{12}$ (metres), with the vertical half-site boundary at $x=3000$ m:

```text
q_i = 1000   if x_i >= 3000 m
q_i =    0   if x_i <  3000 m
Q_1b(X) = sum_i q_i = 1000 * (number of eastern turbines)
f_bird(X) = Q_1b(X)/12,000
```

The **boundary is included on the higher-value eastern side** (`>=` in `hypothetical_risk.py`). The optimiser minimises both numerical objectives:

```text
f_AEP(X) = 1 - (AEP(X) - 0.5*AEP_ref)/(0.5*AEP_ref)
         = 2 - 2*AEP(X)/AEP_ref
AEP_ref  = 754.4007727706155 GWh/year
f_bird   = Q_1b/12,000
```

The fixed AEP reference is **not the Jensen wake-free ceiling** (approximately 760.663033 GWh/year). Its historical reference-layout calculation was not recovered; the value was reconstructed from archived objective columns. Positive normalisation does not change Pareto dominance.

## Implementation settings (verified from recorded final run)

| Setting | Value |
|---|---|
| Site | 6000 m × 6000 m; coordinates bounded in [0,6000] m |
| Number of turbines | 12; 24 continuous decision variables |
| Simplified Jensen turbine | 15 MW rated, rotor D=240 m; hub height 150 m |
| Minimum spacing | 4D=960 m; 66 distinct pairwise inequality constraints |
| Turbine cut-in / rated / cut-out | 3.0 / 10.6 / 25.0 m/s |
| Jensen | Expansion constant `k_w=0.04`; hub-centre top-hat wake contact; velocity deficits combined root-sum-squares and capped at 0.95 |
| Turbine thrust coefficient | `C_T=0.8` for 3 <= U < 10.6; `C_T=0.25` for 10.6 <= U <= 25; 0 otherwise, using undisturbed U |
| Wind directions | 360 sectors (1°), normalized $\exp[2\cos(\theta-225°)]$ weights (wind FROM 225°) |
| Speed bins | 4, 6, 8, 10, 12, 14, 16 m/s |
| Speed weights | Weibull PDF (shape 2, scale 10 m/s), discretely renormalised |
| Joint weights | Product of normalized direction and speed weights |
| Objective | Maximise AEP by minimising `2-2*AEP/AEP_ref`; minimise `Q_1b/12000` |
| NSGA-II | `pymoo 0.6.1.5`, population 120, generations 150, seed 42, eliminate duplicates |
| Initial population | `GridSampling`: 40% west-biased risk-aware / 30% random feasible / 30% jittered grid |
| Genetic operators | Installed `pymoo 0.6.1.5` defaults; the resolved operator metadata is in `run_settings.json` |
| Hypervolume diagnostics | Reference point `(1.2,1.2)` in normalised objectives |
| Endpoint convergence | Independently check 1° endpoint AEP against a 0.5° (720-sector) Jensen re-evaluation; strict difference <0.2% |

No external measured wind-rose file, IEA reference workbook or FLORIS turbine file is loaded by the supplied Jensen implementation. Site, turbine and wind assumptions are encoded in `config.py` and related Python modules. Do not substitute D=242.24 m (the later FLORIS turbine) for the simplified Jensen D=240 m in this reproduction.

## Repository file layout

- `metrics/metric_1b/run_jensen_1deg.py`: **unmodified final runner**; select Metric 1b using `--metric 1b`.
- `configurations/config.py`: exact imported 360-sector `Scenario` definition.
- `configurations/metric_1b_run_settings.json`: original **approved template**; actual realised settings and package versions are in the result `run_settings.json`.
- `shared/wake.py`: Jensen AEP evaluator and wake-free reference.
- `shared/hypothetical_risk.py`: `half_site_east_risk` and other original field branches.
- `shared/validation_problem.py`: pymoo two-objective problem; turbine spacing and coordinate bounds.
- `shared/run_validation.py`: original feasible initialisation, west-biased sampling, hypervolume callback, layout utilities; also legacy generic runner.
- `shared/plot_saved_pareto.py`: original Pareto plotter imported by the final runner.
- `results/metric_1b/metric_1b_360sectors_seed42_pop120_gen150/`: **complete archived final result**.
- `results/metric_1b/historical_model2_12sectors/`: historical initial Jensen result; **not final thesis 1° output**.
- `documentation/historical_model2/source/`: recoverable 12-sector-era source snapshot; exact executed old runner not established.
- `documentation/file_manifest.csv`: each packaged original file's source path, role, destination, classification and SHA-256.
- `documentation/reproducibility_audit.md`: the audit and limitations.

**Reusing this package alongside Metric 1a:** The seven approved source modules are shared by Metrics 1a–1d. `shared/*.py`, `configurations/config.py` and `run_jensen_1deg.py` have matching code hashes to the approved Metric 1a code supplied previously. Before merging into GitHub, compare the SHA-256 checksums; avoid redundant copies of unchanged common files if your repository already tracks them. The runner file is provided within each metric folder to make the package stand alone, not because it is metric-specific code.

### Environment

Original recorded platform: **Windows-10-10.0.22621-SP0**, Python **3.11.15**, `pymoo` **0.6.1.5**, NumPy **2.4.6**, pandas **3.0.3**, matplotlib **3.11.0**. `configurations/requirements.txt` pins these **direct packages**, not every transitive dependency. No complete original `environment.yml` or pip/conda lockfile was found.

### Clean-workspace replay instructions (NOT performed in this audit)

The original scripts import modules by bare filenames (`from config import Scenario`, `from wake import AEPModel` etc.) and the runner records source SHA-256 hashes by filenames in its current directory. Preserve source bytes and **stage copies together** rather than refactor imports. In **PowerShell**, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configurations\requirements.txt
New-Item -ItemType Directory -Force -Path .\_replay_metric_1b | Out-Null
Copy-Item .\configurations\config.py .\_replay_metric_1b\
Copy-Item .\shared\*.py .\_replay_metric_1b\
Copy-Item .\metrics\metric_1b\run_jensen_1deg.py .\_replay_metric_1b\
Set-Location .\_replay_metric_1b
python run_jensen_1deg.py --metric 1b
```

The runner refuses to overwrite an existing `results/metric_1b_360sectors_seed42_pop120_gen150/` folder and stops if the independent 0.5° endpoint check fails. The staging directory must be excluded from source control (add `_replay_metric_1b/` to the repository `.gitignore`). **This package was built without executing the optimiser**. An identical result from a clean Python environment has therefore **not** been independently demonstrated. Compare fresh results with the original `pareto_full.npz`, `pareto_points.csv`, `anchor_summary.csv` and `run_settings.json`.

## Saved result summary and validation

- Completed 1° run: **7 non-dominated layouts**; number of eastern turbines = **0, 1, 2, 3, 4, 6, 7** (each higher-side turbine contributes 1000 score units).
- Highest-AEP endpoint (`layout_id=4`): **732.8084115833284 GWh/year**, $Q_{1b}=7000$ (7 turbines in east).
- Lowest-risk endpoint (`layout_id=3`): **719.4682011910809 GWh/year**, $Q_{1b}=0$ (all turbines in west).
- Endpoint AEP difference: **13.3402103922475 GWh/year**, approximately **1.82%** of the highest-AEP endpoint.
- Independently saved 1° AEP re-evaluation of **all seven** retained layouts matches recorded optimisation AEP exactly in `pareto_aep_reported_1deg.csv`.
- Independent 0.5° (720-sector) AEP checks: highest-AEP **732.2557694751399 GWh/year** (1° vs 0.5° difference **0.075471%**) and lowest-risk **719.1309241051656 GWh/year** (difference **0.046901%**), both **PASS** against a strict 0.2% tolerance.
- `pareto_full.npz`: `X` shape `(7,24)`, `F` shape `(7,2)`. The full seven layouts are also present in `pareto_layouts_long.csv`; anchor layouts match their respective archive IDs.
- **File integrity audit:** all six recorded source-module SHA-256 hashes match the approved Python source archive byte-for-byte; the seven stored layout coordinates match the NPZ vectors; recalculating $Q_{1b}$ as `1000 * count(x>=3000)` matches all seven archived scores. No optimiser or AEP computation was run in this audit.
- All archived Pareto layouts satisfy recorded 960 m spacing and site bounds. The original NSGA-II Pareto front is a finite, search-dependent approximation, not a mathematically proven global Pareto front.
- A `knee` anchor is archived as an algorithmic diagnostic. It is not presented as an objectively preferred ecological or engineering design.

## Historical Model 2 (12-sector) — separate, superseded evidence

`model2(1).zip` contains 11 saved files under `validation_half_site_east_risk_seed42_pop120_gen150/`: **2 Pareto points**. Their archived AEPs are **759.9926178 GWh/year** at $Q_{1b}=4000$ and **754.638634 GWh/year** at $Q_{1b}=0$. The source identity and settings are not fully authenticated:

1. No recorded `run_settings.json`, full optimizer decision-vector archive or complete environment lock.
2. The historical stored numerical objectives imply `AEP_ref` ~ **754.400772771 GWh/year** under `f1=2-2*AEP/AEP_ref`, whereas the available `run_validation.py` uses `ideal_aep` as its reference. Therefore, the supplied 12-sector source snapshot is **not verified** as the exact original executable.
3. The runner's output path embeds `_mult1.0_` while the archived directory name does not. This is another provenance inconsistency.
4. `m2.zip` is a file-content duplicate of `model2(1).zip`; only one copy is included.

For investigation only, the historical generic runner invokes `python run_validation.py --model half_site_east_risk --seed 42 --pop 120 --gen 150 --risk_mult 1.0` using its **own isolated** 12-sector `config.py` source snapshot. Do **not** use those files to reproduce the final 1° study.

## Limitations / missing evidence

- Precise historical 12-sector Model 2 executed source/working directory: **not authenticated**.
- Historical input AEP reference calculation/layout: **not recovered**.
- Complete original dependency lock and historical environment: **not recovered**.
- Full historical NSGA-II population/decision history: **not found**.
- Original historical standalone test logs: **not found**.
- Re-execution in a clean Python environment to confirm output equality: **not performed**.
- No completed FLORIS Metric 1b run is included; this package documents the saved Jensen stage only.

## Preservation policy

Every Python source file in the package is copied byte-for-byte from an available original source or supplied historical snapshot. The saved result files are included byte-for-byte from the archives. Documentation, `requirements.txt`, the audit and file manifest are **new supplementary files**. No Python imports, algorithms, geometry or saved outputs were modified.
