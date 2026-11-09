# Model 10 — Final 72-sector HPC master archive and plots

## Goal

Create one **new, self-contained folder on HPC12** with the complete final corrected literature-informed Model 10 results. Copy the original files **without moving, deleting, renaming, or overwriting them**. Generate figures **on HPC12** from the archived data, not in ChatGPT.

HPC user `uraj`; original directory `/home/uraj/model10`; scheduler **PBS/Torque (`qsub`)**, environment `model10`.

## Source experiment groups (all required)

1. Three corrected 72-sector NSGA-II optimisations: `hpc_full_run_v3_72sectors`, `hpc_full_run_v3_72sectors_seed730102`, `hpc_full_run_v3_72sectors_seed730103` (full result folders including original source snapshots, `.npy`, `.csv`, progress and settings).
2. Guarded independent evaluation: `hpc_v3_final_knee_confirmation_guarded`; optional original six-layout confirmation: `hpc_v3_paired_confirmation_p0`.
3. Five **final** sensitivities: `hpc_model10_frozen_final_probability`, `hpc_model10_frozen_final_radius`, `hpc_model10_frozen_final_hub`, `hpc_model10_frozen_final_speed`, `hpc_model10_frozen_final_direction_fixed_width`.
4. Bird-level bootstrap: `hpc_v3_final_optimisation_sample_bird_bootstrap` (includes individual-bird batch `.npz`, actual 10,000-replicate array, statistics and plot).
5. Numerical AEP-resolution verification: `hpc_model10_final_selected_360sector_aep`.
6. The original source runner, bird modules, input data and job scripts embedded in the v3 result folders or present at the top level of `/home/uraj/model10`.

**Not pooled:** old/pre-correction 12-sector experiments, early Model 10 sensitivity folders (`hpc_sensitivity_*`), discarded 3b wind-rose results. Keep these at their current original locations for development history.

## The archive layout

```
/home/uraj/model10/MODEL10_FINAL_72SECTOR_MASTER/
├── README_START_HERE.txt
├── archive_audit.json
├── archive_manifest.csv
├── 01_optimisation/
│   ├── seed_730101/                    (complete original run)
│   ├── seed_730102/
│   └── seed_730103/
├── 02_combined_pareto/
│   ├── combined_non_dominated_24.csv
│   ├── all_feasible_candidate_scores_513.csv
│   ├── final_selected_three.csv
│   ├── highest_aep_coordinates_xy_m.{csv,npy}
│   ├── knee_coordinates_xy_m.{csv,npy}
│   └── lowest_risk_coordinates_xy_m.{csv,npy}
├── 03_independent_evaluation/
│   ├── final_four_guarded/
│   └── earlier_six_guarded/            (if present)
├── 04_sensitivities/
│   ├── probability/
│   ├── radius/
│   ├── hub_height/
│   ├── bird_speed/
│   └── direction_fixed_width/
├── 05_uncertainty/bird_level_bootstrap/
├── 06_numerical_resolution/aep_360sector/
├── 07_scientific_code_and_job_scripts/
│   ├── frozen_source_snapshot_seed730101/
│   ├── model10_build_master.py
│   ├── model10_plot_master.py
│   ├── hpc_model10_master_plots.pbs
│   └── original .py/.pbs scripts where available
├── 08_figures/                         (generated later on HPC12)
└── 09_derived_tables/                  (generated later on HPC12)
```

**Scientific selection rule:** From all three completed optimisations, take feasible solutions from all 1,440 candidate evaluations (513 feasible), recalculate global non-dominance using the **72-sector AEP** (maximise) and **optimisation-sample** \(Q_\mathrm{lit}\) (minimise), then select the maximum positive \(d=(a-r)/\sqrt 2\) knee after end-point normalization. Validation requires exactly **24 non-dominated layouts** and highest AEP = 730101/455, knee = 730103/418, lowest risk = 730103/220. The reference is context only.

## Step 1: upload three package files

Download `Model10_HPC_Master_Archive_and_Plot_Scripts.zip` from ChatGPT, extract it on your computer, and upload these three files to `/home/uraj/model10/` in WinSCP:

- `model10_build_master.py`
- `model10_plot_master.py`
- `hpc_model10_master_plots.pbs`

`README_HPC_MASTER.md` is optional to upload; the builder copies it automatically if beside the script.

## Step 2: check before copying anything

From PuTTY:

```bash
cd ~/model10
source ~/miniconda3/etc/profile.d/conda.sh
conda activate model10
python -m py_compile model10_build_master.py model10_plot_master.py
python model10_build_master.py --check
```

Do not build if any required source folder is missing or any verification fails. The script checks completion reports, saved optimisation seeds and fixed bird samples, source hashes, 72 sectors and 480 evaluations.

## Step 3: create the master copy (only after `SOURCE AUDIT PASS`)

```bash
python model10_build_master.py --build
```

The builder creates `MODEL10_FINAL_72SECTOR_MASTER.BUILDING_INCOMPLETE` first, then renames it to `MODEL10_FINAL_72SECTOR_MASTER` **only after all checks, Pareto calculations, file copies and manifest creation succeed**. If the `.BUILDING_INCOMPLETE` folder exists, the builder refuses to overwrite it; inspect before addressing the problem. If the final folder already exists, it also refuses to overwrite it. Never delete the originals to make room.

When successful, inspect:

```bash
cat MODEL10_FINAL_72SECTOR_MASTER/archive_audit.json
ls -lh MODEL10_FINAL_72SECTOR_MASTER/02_combined_pareto/
```

## Step 4: generate thesis plots **on HPC12**

Option A (recommended, small dedicated PBS plotting job):

```bash
cd ~/model10
qsub MODEL10_FINAL_72SECTOR_MASTER/07_scientific_code_and_job_scripts/hpc_model10_master_plots.pbs
qstat -u uraj
```

Option B (interactive, if allowed by HPC policy):

```bash
cd ~/model10
source ~/miniconda3/etc/profile.d/conda.sh
conda activate model10
export MPLBACKEND=Agg
python MODEL10_FINAL_72SECTOR_MASTER/07_scientific_code_and_job_scripts/model10_plot_master.py \
  --master ~/model10/MODEL10_FINAL_72SECTOR_MASTER
```

The plotter requires `numpy`, `pandas`, `matplotlib` and `scipy`, available in the existing Model 10 environment used for the bird-level bootstrap. It runs **no bird simulation, FLORIS evaluation, optimisation or new Monte Carlo draws**. It uses saved Monte Carlo batches and saved bootstrap replicates. If data are missing, it fails rather than fabricating figures.

## Generated HPC figures (14 figure sets, each PDF/SVG/PNG)

- `01_final_combined_24_layout_pareto`: complete 24-layout optimisation-sample Pareto front across 3 seeds, selected points and reference contextual marker.
- `02_four_layout_maps_rotor_discs`: coordinates for 455, 418, 220 and reference, with schematic rotor-disc footprints.
- `03_independent_evaluation_selected_AEP_vs_Q`: mean independent-evaluation risk versus AEP, illustrative unadjusted 95% t intervals for each mean (paired adjusted **differences** stay in the saved comparison tables).
- `04_optimisation_vs_independent_evaluation`: fixed optimisation sample versus independently evaluated risk for three layouts.
- `05_sensitivity_probability`, `05_sensitivity_radius`, `05_sensitivity_hub_height`, `05_sensitivity_bird_speed`: sensitivity results using independent evaluation bird samples.
- `06_direction_fixed_width_Q`: eight headings as **markers only**, 225° baseline highlighted; W fixed for all headings.
- `06_direction_fixed_width_Q_times_W`: same headings scaled by W, for comparing with the original-width baseline.
- `07_bird_level_bootstrap_N50000_10000resamples`: KDE distributions from real paired, stratified bird-level bootstrap replicates; N=50,000 optimisation sample.
- `08_monte_carlo_convergence_empirical_and_sqrtN`: empirical SD from disjoint combinations of 10 independent batches where estimable, plus labelled 1/sqrt(N) projections; three internal paired mean differences as horizontal levels. No bird-by-bird subsamples are invented.
- `09_three_seed_feasible_candidate_exploration`: all feasible candidate scores from each seed and each seed's highest-AEP candidate.
- `10_1degree_5degree_AEP_difference`: saved AEP(360 sectors) minus AEP(72 sectors) for the final three layouts.

The script also saves processed CSVs to `09_derived_tables`, and a `08_figures/FIGURES_README.txt` with interpretation notes. All files are generated by **your HPC**.

## Quality and thesis reporting safeguards

- Pareto dominance and knee selection use **optimisation-sample** Q only. Independent evaluation values are shown separately and never used to select the knee.
- Do not use “training birds”, “fresh birds” or “balanced” in thesis descriptions.
- Directional test: fixed corridor 9696.481 m, 57,137 birds ×10 batches, heading/rotor/Band-wind rotated together, AEP held fixed.
- Bootstrap resampling is bird-level and stratified, never ten-batch bootstrap.
- The Monte Carlo convergence plot distinguishes empirical SDs from the theoretical 1/sqrt(N) continuation; it is not an ecological uncertainty analysis.
- Source CSV names `pareto_training.csv` and older source configuration labels are retained **unchanged** for reproducibility, even though thesis plots use the correct terminology.
- Original pre-correction experiments are not deleted. They are excluded from this archive to prevent accidental mixing with the final v3 conclusions.
- The SHA-256 manifest covers the copied scientific files **before the plotting step**. Plots and derived tables are generated afterward; the original manifest intentionally does not attest to those later files.

## Download figures after generation

In WinSCP open `/home/uraj/model10/MODEL10_FINAL_72SECTOR_MASTER/08_figures/`. Download PDF (vector, good for LaTeX/Word), SVG (editable vector) or PNG (convenient preview). You can also export all figures with:

```bash
cd ~/model10
# Generates an archive only; does not modify the original data.
tar -czf model10_final_hpc_figures.tar.gz \
  MODEL10_FINAL_72SECTOR_MASTER/08_figures \
  MODEL10_FINAL_72SECTOR_MASTER/09_derived_tables
```

