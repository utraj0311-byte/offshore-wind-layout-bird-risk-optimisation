# Metric 1a — uniform artificial bird-activity field

**Thesis:** *Integrating Bird Collision Risk into Multi-Objective Offshore Wind Farm Layout Optimisation*

**Metric name:** Metric 1a (historical Model 1, "Constant Spatial Risk Field")

**Release represented here:** Completed, approved **Jensen 1-degree / 360-sector** rerun, seed 42, population 80, 100 generations. The historical **12-sector** Model 1 archive is preserved separately as evidence; it is **not** the final thesis result. This package contains no FLORIS rerun of Metric 1a.

## Purpose and interpretation

For each feasible 12-turbine layout, every turbine receives the same prescribed artificial bird-activity score of 1000. The metric cannot distinguish layouts. It is a null/control case for the two-objective optimiser, **not** a bird collision or mortality prediction.

Raw objective: `Q_1a(X) = sum_{i=1}^{12} 1000 = 12,000` for all 12-turbine layouts.

Bird objective sent to NSGA-II: `f_bird(X) = Q_1a(X) / 12,000 = 1`.

The **final 1-degree** energy objective (minimised) is

`f_AEP(X) = 1 - (AEP(X) - AEP_ref/2) / (AEP_ref/2) = 2 - 2*AEP(X)/AEP_ref`,

with `AEP_ref = 754.4007727706155 GWh/year`. This is a **historical optimisation scale reconstructed from saved objective columns**, not the physical Jensen wake-free ceiling. Its original reference layout/calculation has **not** been recovered. This quantity is different from the wake-free AEP (`~760.663033 GWh/year`).

## Physical and computational assumptions

| Item | Recorded value / implementation |
|---|---|
| Site | 6000 m by 6000 m |
| Turbines | 12, decision vector `(x_1,y_1,...,x_12,y_12)` = 24 continuous variables |
| Simplified turbine | 15 MW rated, rotor diameter 240 m, hub height 150 m |
| Operating range | Cut-in 3.0 m/s, rated 10.6 m/s, cut-out 25.0 m/s |
| Spacing | 4 rotor diameters = 960 m; 66 pairwise inequality constraints |
| Coordinates | `0 <= x_i,y_i <= 6000 m`; coordinate bounds enforced by pymoo |
| Jensen wake model | `k_w=0.04`, hub-centre-in-top-hat-cone wake overlap, root-sum-square velocity deficits clipped to 0.95 |
| Thrust coefficient | `C_T=0.8` for 3 <= U < 10.6; `C_T=0.25` for 10.6 <= U <= 25; 0 otherwise; based on undisturbed U |
| Turbine power | Cubic ramp from cut-in to rated; constant 15 MW through cut-out; zero outside |
| Wind directions | Final: 360 sectors, each 1 degree; source `configurations/config.py` |
| Wind directions — historical Model 1 | 12 sectors, 30 degrees apart; separate archived source snapshot |
| Dominant wind-from direction | 225 degrees |
| Direction probabilities | At sector angle `theta_j`: `p_j` proportional to `exp(2*cos(theta_j - 225deg))`, then sum-normalised |
| Speed bins | 4, 6, 8, 10, 12, 14, 16 m/s |
| Speed probabilities | Weibull PDF at discrete bins with shape `k=2`, scale `A=10 m/s`, sum-normalised |
| Joint wind weights | Product of direction frequency and speed-bin frequency |
| AEP | Sum over 8760 h/year of weighted turbine power; reported as GWh/year |
| Optimiser | `pymoo` NSGA-II, `pop_size=80`, `n_gen=100`, seed 42; duplicate elimination |
| Initialisation | 50% random feasible layouts, 50% jittered-grid layouts, via `GridSampling` in `run_validation.py` |
| Operators | Installed `pymoo 0.6.1.5` defaults; final run settings archive records resolved operator snapshots |
| Hypervolume | Fixed normalised reference point `(1.2, 1.2)`; diagnostic, not an additional objective |

No external wind-rose CSV, turbine workbook or OpenFAST/FLORIS data file is imported for this **Jensen** implementation; numerical assumptions are encoded in the Python modules. The 240 m rotor is the simplified Jensen prototype value, **not** the official FLORIS IEA-15MW diameter of 242.24 m.

## Repo layout / provenance

- `metrics/metric_1a/run_jensen_1deg.py`: verbatim approved final runner.
- `configurations/config.py`: verbatim 360-sector scenario module imported as `config`.
- `shared/wake.py`: Jensen AEP and wake calculations.
- `shared/hypothetical_risk.py`: artificial `constant` bird-risk branch.
- `shared/validation_problem.py`: pymoo objective transformation, variables and constraints.
- `shared/run_validation.py`: original grid/random sampler, HV callback, saving/figure utilities, plus historical generic runner.
- `shared/plot_saved_pareto.py`: verbatim final plotting script imported by the 1-degree runner.
- `results/metric_1a/metric_1a_360sectors_seed42_pop80_gen100/`: **complete saved 1-degree run output**.
- `results/metric_1a/historical_model1_12sectors/`: superseded original Model 1 output archive.
- `documentation/historical_model1/source/`: available 12-sector-era source snapshot. **Its identity as the precise executed historical version is unconfirmed.**
- `configurations/metric_1a_run_settings.json`: original approved template, distinct from the authoritative **recorded** `results/.../run_settings.json`.
- `documentation/file_manifest.csv`: original member location, GitHub destination, purpose, optional/required, SHA-256.

The final run recorded source-file SHA-256 hashes for six core modules. All **six hashes match** the approved source archive supplied with this package. The plotting script exists but was not one of those six hash-locked modules. See `documentation/reproducibility_audit.md`.

## Environment

Recorded local run on **Windows 10**, **Python 3.11.15**, **pymoo 0.6.1.5**, **NumPy 2.4.6**, **pandas 3.0.3**, **matplotlib 3.11.0**. See `configurations/requirements.txt`. This is a direct-dependency pin, **not a complete conda/pip environment lock**. No `environment.yml` or complete original environment export was located.

### Replaying the final 1-degree optimisation (NOT performed for this packaging audit)

The original Python modules use bare imports (`from config import Scenario`, etc.) and the final runner reads source hashes relative to its working directory. Therefore, to keep every original `.py` file **unmodified**, first stage the files together in a clean work folder. The commands below are written for **PowerShell at the GitHub repository root**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configurations\requirements.txt
New-Item -ItemType Directory -Force -Path .\_replay_metric_1a | Out-Null
Copy-Item .\configurations\config.py .\_replay_metric_1a\
Copy-Item .\shared\*.py .\_replay_metric_1a\
Copy-Item .\metrics\metric_1a\run_jensen_1deg.py .\_replay_metric_1a\
Set-Location .\_replay_metric_1a
python run_jensen_1deg.py --metric 1a
```

**Do not** run the optimisation with the archive output folder already present in the same working directory: the runner deliberately refuses to overwrite its own output directory. It will create `results/metric_1a_360sectors_seed42_pop80_gen100/` within the staging directory. The staging directory is an ephemeral workspace, not source control material; add it to `.gitignore` in your repository. The commands copy only source files and do not edit them.

The original published run is already packaged; the optimiser was **not** rerun here, so exact reproduction in a clean environment is **not independently established**. Compare the new `pareto_points.csv`, `pareto_full.npz`, and anchor summary with the archived files. Tiny differences are possible across environments; strict bit-for-bit identity has not been promised.

### Historical 12-sector development experiment (not the final thesis run)

The old saved Model 1 archive has `AEP=758.5335371987206 GWh/year`, `Q=12,000`, `pop=80`, `generations=100`, seed 42 in its result folder name. Its stored `f1_normalised_minimise=-0.010956416210781805` implies `AEP_ref=754.4007727706156 GWh/year` **if** the two-point energy objective is `2-2*AEP/AEP_ref`.

The available historical runner `run_validation.py` instead sets `aep_max = aep_model.ideal_aep(12)` (wake-free AEP, approximately 760.663033 GWh/year) and `aep_min=0.5*aep_max`. Hence it would produce a **different saved normalised f1** for the historical layout. The runner's output path also includes `_mult1.0_`, but the archive directory does **not**. We cannot prove that this is the exact historical executable. Its original python/pymoo environment is not fully recorded, and only the final retained layout, not the entire population archive, was saved. The historical scripts are provided to aid investigation, **not as an authenticated exact-original replay**.

If checking its behaviour manually, use the historical `documentation/historical_model1/source/` snapshot in its **own** isolated staging folder and invoke `python run_validation.py --model constant --seed 42 --pop 80 --gen 100 --risk_mult 1.0`. Do not mix this `config.py` (12 sectors) with the final `configurations/config.py` (360 sectors). This instruction is informational; this package creation did **not** execute it.

## Saved 1-degree results and validation

- `run_status = complete`, `1` non-dominated layout.
- Raw `Q_1a=12000` and normalised `f_bird=1`, for the retained layout.
- Highest-AEP endpoint `734.0594711076992 GWh/year`, minimum pairwise spacing `1803.9585302827331 m`; zero reported spacing or boundary violations.
- Lowest-risk and energy endpoint tags both point to the **same layout** (a consequence of constant Q); a separately meaningful risk endpoint does not exist.
- Independent 1-degree AEP recalculation of every retained layout matches the saved optimisation AEP (as recorded).
- The independently computed 0.5-degree value is `732.8864265869281 GWh/year`. The relative absolute difference is `0.1600581588383346%`, strictly smaller than the predefined 0.2% check tolerance; the saved validation reports `PASS` for both coincident anchor tags.
- The complete nondominated coordinate vector and objective pair are archived in `pareto_full.npz` (`X` shape `[1,24]`, `F` shape `[1,2]`) and in CSV.
- There is no claim that the resulting AEP is a proven global maximum.

The **saved** convergence check and source-hash checks were inspected here; no optimisation, AEP recomputation, or plot regeneration was performed. Dedicated independently executed unit-test logs for the original historical run were not found.

## Explicit limitations / missing evidence

1. **Exact executable for historical 12-sector Model 1:** not authenticated; archived f1 and directory naming disagree with available `run_validation.py`.
2. **Provenance of historical fixed `AEP_ref=754.4007727706155`:** recovered from archived objective columns, physical original reference layout/calculation unknown.
3. **Original historical `pymoo` version and full Python environment export:** not located.
4. **Clean-room rerun with result equality:** not performed, and not asserted.
5. **Full original 12-sector optimiser population/history:** no complete coordinate/decision archive located; only its final retained layout and endpoint copies were supplied.
6. **FLORIS version of Metric 1a:** no completed corresponding FLORIS run/source is included here; this package reproduces the approved Jensen study only.

## Source preservation

All included `.py` files were copied **byte-for-byte** from the supplied archives/available historical source snapshot. No Python code was edited, refactored or recreated, and no source files were invented. `README.md`, `requirements.txt`, audit text and manifest are new documentation only.
