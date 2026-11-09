# Integrating the original Metric 4b archive safely

## Additive-only merge

This ZIP has only these new prefixes:

```
metrics/metric_4b/
configurations/metric_4b/
results/metric_4b/
documentation/metric_4b/
```

The packaging audit detected **zero path collisions** with both the supplied Metrics 1a–3b Git repository ZIP and the already prepared Metric 4a add-on. The ZIP contains **no `.git/`, `README.md` at repository root, or `shared/` overwrites**. Extract it **into your repository root**, then commit the added files. Do not extract the user's repository's `.git` from another ZIP or overwrite existing scientific source.

## Why the scientific code remains isolated

The original Model 9 tree uses bare imports such as `from config import Scenario` and `from model9_rotor_transit import ...`. It is preserved verbatim in `metrics/metric_4b/metric4b_preflight/model9_code/`. The package does **not** move or rewrite imports, nor place the scripts in `shared/`. Byte comparisons with the Metric 4a package show that `config.py`, `floris_aep.py`, and `model8_meso_response.py` are **not** identical, although `model8_batching.py` and `floris_config/gch_iea15mw.yaml` happen to match. Shared compatibility of all imports and scientific behaviours has therefore **not** been established.

## Runtime working-directory caveat (read this before executing)

These are **the original unmodified scripts**, not a rewritten installable module. Their project roots originally looked like this:

```
bird_layout_opt/
├── metric4b_preflight/
│   ├── metric4b_run72.py
│   ├── metric4b_postrun_checks.py
│   ├── metric4b_confirm72.py
│   ├── metric4b_preflight.py
│   ├── model9_code/
│   └── original_results/      # original 12-sector preflight inputs
├── metric_4b_72sectors/       # completed full 5° optimiser outputs
├── metric_4b_72sectors_postrun/ # post-run verification files: MISSING IN ARCHIVE
└── metric_4b_72sectors_confirmation/  # completed independent evaluation
```

The GitHub layout stores the primary source under `metrics/metric_4b/metric4b_preflight/` and immutable results under `results/metric_4b/` to avoid altering other metrics. For the unmodified original commands, work on **copies in an isolated scratch folder** with the sibling paths above. Copy the two saved completed run/confirmation directories from their corresponding `results/metric_4b/` source folders into that scratch directory **without editing scientific files**. Do not run from the GitHub repository root and assume the internal parent-folder paths resolve.

Examples of safe non-simulation checks **only as commands for the user to run**:

```bat
conda activate floris-env
cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4b_preflight
python metric4b_run72.py verify
python metric4b_preflight.py audit
```

The final runner's `run` and `resume` options **perform NSGA-II optimisation**. The confirmation script's `run`, `resume` and even `smoke` **simulate birds**. None was run during this packaging. The archived independent evaluator's `verify` checks two JSON reports in sibling `metric_4b_72sectors_postrun/`, which are absent from the supplied archive. Its saved *output CSVs* are preserved; do not generate stand-in reports or bypass these gates. The pure AST/hash/CSV checks in `STATIC_AUDIT.json` do **not** establish bitwise dynamic run reproducibility.

## Requirements and software provenance

Use the original `model9_code/requirements_model9.txt` and recorded versions in `run_settings.json`: FLORIS==4.6.5, pymoo==0.6.1.5; Python 3.11.15, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, matplotlib 3.11.0, PyYAML 6.0.3. `cloudpickle` is imported opportunistically for checkpoint serialisation, falling back to standard `pickle`; its installed version was not recorded. The original requirements file may have allowed version ranges for some packages, so do not claim a fully locked reproducible environment. The `iea_15MW_REFERENCE.yaml` is a preserved SHA-256-checked reference copy; FLORIS also loads its installed turbine library from `site-packages`, whose checksum and version should be checked before computational reproduction.

## GitHub hygiene

The package excludes original large `checkpoint_latest.pkl` and `checkpoint_previous.pkl` files (about 40 MB combined), as well as duplicated intermediate CSV checkpoints and PNG plot previews. It retains `final_population.npz`, all 36 saved Pareto layouts, both selected endpoint CSVs, the complete candidate audit, and the independent final batches. The loss of serialised checkpoint objects prevents checkpoint-exact *resume*, not inspection of the completed result. The original full ZIPs should be retained offline separately for preservation.

The northern-gannet CSV is attributed to Johnston et al. (2014) via stochLAB. Before making the public GitHub repository, verify that the original dataset's redistribution terms permit the included data to be published and keep its citation/metadata. No source or dataset in this ZIP was silently relicensed.
