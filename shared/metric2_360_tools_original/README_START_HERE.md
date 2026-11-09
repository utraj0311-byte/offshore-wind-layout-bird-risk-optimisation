# Jensen Metric 2a / 2b — 1° controlled rerun package

**Use in an activated Anaconda Prompt on your Windows PC. No optimisation was run when this package was prepared.**

## Installation (no original files are overwritten)

Extract the `metric2_360_tools` directory directly into:

`C:\Users\Deepak\Desktop\bird_layout_opt\metric2_360_tools`

The original Python files belong **one directory above** `metric2_360_tools`, in:

`C:\Users\Deepak\Desktop\bird_layout_opt`

The tools use these *unchanged original modules* from the project root:

- `config.py`, `wake.py`, `run_validation.py`, `validation_problem.py`, `hypothetical_risk.py`
- `coupled_risk.py`, `pairwise_risk.py`
- `run_aep_only.py`, **`aep_only_problem.py`**

**`aep_only_problem.py` was not among the files supplied in chat. It MUST already exist in your project root, or be supplied from the original project. Do not replace it with a newly invented version.** The risk-only optimisation is **not** rerun; `risk_only_problem.py` is not needed for these scripts.

The tool verifies SHA-256 hashes for original Python files provided in chat; it accepts either 12-sector or 360-sector `config.py` as long as all the other scientific parameters match. It does not modify `config.py`. The active run sets `scenario.wind.n_sectors=360` in memory.

The original 12-sector Model 5/6 results and the AEP-only/risk-only layouts are included in `metric2_360_tools\inputs` to support independent reproduction of saved AEP values.

## Anaconda Prompt commands — one job at a time

Start with:

```bat
cd /d C:\Users\Deepak\Desktop\bird_layout_opt
conda activate YOUR_EXISTING_CONDA_ENV
python -c "import pymoo; print(pymoo.__version__)"
```

`pymoo` should print **0.6.1.5**, the approved **new-run** version. The historical version is not asserted as known. Do not change this version without checking first.

### Step 1. Pre-flight only (NO optimisation)

```bat
python metric2_360_tools\run_metric2_360.py preflight
```

The script refuses the run if any of the following fail:

- Six archived 12-sector AEP reproductions, tolerance ≤ 0.001 GWh/year: Model 5 highest AEP / lowest risk; Model 6 highest AEP / lowest risk; independent AEP-only / risk-only reference.
- Correct rotor, wake, speed-bin, spacing and site settings; unchanged source file hashes where available; target index 0, 4D and 8D Gaussian kernels; objective references and normalisations.
- Installed `pymoo==0.6.1.5` for all new runs.

It also reports one 360-sector evaluation timing and the approximate 2a, 2b, AEP-only runtimes on **your** machine, then saves a uniquely named pre-flight report in the approved results directory.

**Do not run Step 2 unless pre-flight prints PASS.**

### Step 2. Run Metric 2a (Model 5), 60 × 60

```bat
python metric2_360_tools\run_metric2_360.py run-2a
```

New output only:

`jensen_1deg_approved\results\metric_2a_360sectors_seed42_pop60_gen60`

Wait for **both** newly optimised endpoints to pass the strict 1° versus 0.5° check (< 0.2%). The code records the target-only feasible-grid diagnostic with the other 11 turbines frozen. If either endpoint fails, **STOP**.

### Step 3. Run Metric 2b (Model 6), 80 × 100

```bat
python metric2_360_tools\run_metric2_360.py run-2b
```

New output only:

`jensen_1deg_approved\results\metric_2b_360sectors_seed42_pop80_gen100`

The runner first verifies Step 2 has passed. It re-evaluates the **saved risk-only reference** at 1° and 0.5° and plots it with the re-evaluated historical AEP-only reference. The risk-only GA **is never rerun**. Both 2b endpoints must pass the strict < 0.2% check. If either fails, **STOP**.

### Step 4. Run the independent Jensen AEP-only reference, GA 160 × 250

```bat
python metric2_360_tools\run_metric2_360.py run-aep
```

New output only:

`jensen_1deg_approved\results\metric_aep_only_360sectors_seed42_pop160_gen250`

The runner first verifies Step 3 has passed. It uses the original `AEPOnlyLayoutProblem`, `GA` constructor, `GridSampling`, original callback, seed, population, and generations. Its 1° best layout is independently evaluated at 0.5°. It then writes:

- The new 1° AEP-only reference and the 0.5° check result.
- A table comparing the **new reference** with available Metrics 1a–2b, without altering existing objective scaling.
- A **new** `metric_2b_pareto_final_refs.png/pdf` in the new AEP-only reference directory, showing the new AEP-only reference and the old risk-only reference re-evaluated at 1°. The completed Metric 2b files are **not modified**.

## Key scientific constants and safeguards

| Parameter | 2a | 2b | AEP-only |
|---|---|---|---|
| Original model | Model 5 | Model 6 | Independent reference |
| Optimiser | NSGA-II | NSGA-II | GA |
| Population × generations | 60 × 60 | 80 × 100 | 160 × 250 |
| Seed | 42 | 42 | 42 |
| Bird/objective definition | Fixed target index 0; Q=sum 11 Gaussian weights; σ=960 m | All 66 unique pairs; σ=1920 m | Only AEP |
| AEP optimiser scaling | 2 - 2 AEP / 760.663033300808 | Same | Original AEPOnlyLayoutProblem |
| Metric objective scaling | Q/11 | Q/66 | n/a |
| Direction sectors: original → new | 12 → 360 | 12 → 360 | 12 → 360 |

The old `754.4007727706155 GWh/year` denominator for Metrics 1a–1d stays unchanged. The new AEP-only result is a **comparison reference** for all 1a–2b, never a replacement denominator.

All completed metric archives contain the full nondominated coordinates (`pareto_full.npz`, `pareto_layouts_long.csv`), raw scores and normalised objectives (`pareto_points.csv`), endpoint coordinate CSVs and summary, settings with hashes and installed versions, endpoint 0.5° convergence CSV, hypervolume history, aligned turbine-pair tables, plus PNG/PDF Pareto and layout figures. Metric 2a also saves a no-optimisation target-only search.

**No script deletes original files or overwrites a pre-existing output directory.** If a new run crashes after creating its folder, that folder is intentionally retained for diagnosis. The same command will refuse to overwrite it; do not delete previous scientific results to rerun.

## Troubleshooting

- `Missing aep_only_problem.py`: find and place your **original** module in the project root. This package cannot verify or recreate its omitted code.
- `Missing hypothetical_risk.py`: your original `run_validation.py` imports this module. Restore it from your existing working Jensen project.
- `SHA-256 mismatch`: you may have a different original Python file. Do not bypass it; inspect the difference before launching.
- `pymoo version mismatch`: activate the same environment used for approved Metrics 1a–1d; this package requires 0.6.1.5 for new runs.
- `Output already exists`: scripts deliberately refuse to overwrite any existing run folder.
- `STOP_convergence_failed`: the next stage is blocked until the failure is reviewed. Never change the wake model to make the check pass.
