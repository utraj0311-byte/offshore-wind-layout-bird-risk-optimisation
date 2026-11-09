# Reproducing Model 10 — from the recovered archive

## Source provenance
The original HPC master states it was copied from `/home/uraj/model10/` without modifying original files. The three `run_settings.json` records report complete 72-sector optimisations at seeds 730101, 730102 and 730103. Each saved run contains its **own** original `optimisation/source/` snapshot, retained intact under `results/literature_informed/01_optimisation/`.

The separate `metrics/literature_informed/original_model10/` is an assembly for easier execution: it copies the seed-730101 frozen source unchanged, adds the **original** two other seed-specific runners, and adds the separately recovered six importable `.py` files under `verification/_runtime_source/`. Original numbered files in `upload/` remain unchanged. This assembled directory is **not byte-for-byte the original frozen directory as a whole**, although each original file is preserved byte-for-byte.

## Set up a clean environment

A suitable reproducibility environment needs Python 3.11.13 and the installed FLORIS 4.6.5, pymoo 0.6.1.5 and NumPy 2.0.2 versions recorded in the archives. For example, using a separate Python 3.11 environment:

```bash
python -m pip install -r metrics/literature_informed/requirements.txt
```

Other dependencies listed there are not version-pinned by the archived run settings. For bitwise reproducibility, use the original HPC conda environment if available.

**Critical turbine-library check:** The runner verifies `floris/turbine_library/iea_15MW.yaml` SHA-256 = `93d2bcfcb619b1819efe8149ce9e8231656e51a29f2ae5c1079a46e339f02feb`. The exact bytes of this YAML are supplied in `configurations/literature_informed/floris_iea_15MW_v4.6.5.yaml`, extracted unchanged from the recovered IEA source ZIP. The original script still expects it within the FLORIS installation; if the installed file is different, its integrity check will fail. Do not bypass or edit scientific hash checks to make a run pass.

## Run an original verification stage (NOT executed when packaging)

From the repository root:

```bash
cd metrics/literature_informed/original_model10
python model10_fhd_robust_seeded_50k_72sectors.py verify --output-dir path_to_NEW_empty_output_directory
```

This uses the original frozen model code. The output directory should be new and separate from existing archived results. Other documented stages in the source are `benchmark`, `smoke`, and `optimise`; **do not start a full optimisation merely to inspect the repository**. Do not use the stale filename in the original runner's module docstring; the actual file includes `_72sectors`.

For the other recorded seeds, their original entrypoints `model10_v3_72sectors_seed730102.py` and `model10_v3_72sectors_seed730103.py` are also in the assembled root. They use the same original inputs but different optimiser seeds; each run has a frozen source snapshot under its result folder.

## Independent confirmation and plotting

`metrics/literature_informed/analysis_scripts/model10_confirm_final_selected_with_guards.py` uses relative paths for the **original HPC project root** (`hpc_full_run_v3_72sectors*` and `hpc_v3_paired_confirmation_p0`). The required saved data are in the GitHub archive under `results/literature_informed/01_optimisation/seed_*` and `03_independent_evaluation/earlier_six_guarded/`, respectively. This script has **not** been modified to run from the re-organised GitHub paths; do not claim it is a one-command reproduction without adapting paths in a separate, traceable script.

The original master plotter can be given a root containing the archived `01_optimisation`, `02_combined_pareto`, etc., but some figures require `04_sensitivities`, `05_uncertainty` and `06_numerical_resolution`, none of which were included in the supplied recovery ZIP. The script `model10_plot_master.py` also differs from its original archive-manifest checksum (see `MISSING_FILES.md`). Figures were **not regenerated** for this package.

## Verification performed during packaging

- Parsed all original `.py` files for syntax using Python's built-in AST parser.
- Verified the **23 original recorded source/input SHA-256 hashes in each of the three run settings** against the frozen snapshot, including the turbine-library YAML inside the IEA source ZIP.
- Checked the source master manifest: 183 existing entries match, 52 are absent from this partial recovery, and the supplied `model10_plot_master.py` differs from its recorded checksum.
- Preserved saved objective values and seed-specific results without rerunning simulations.

**Not tested:** FLORIS/pymoo execution of the original runner or a new stochastic Monte Carlo comparison. The package cannot be declared fully runtime-validated from these static checks alone.
