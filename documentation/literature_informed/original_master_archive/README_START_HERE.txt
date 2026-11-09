MODEL 10 FINAL 72-SECTOR MASTER ARCHIVE

This folder is a COPY of completed HPC source results. Original files were NOT moved, renamed, overwritten or removed.
01_optimisation/       Original complete v3 runs for 730101, 730102, 730103, with source snapshots.
02_combined_pareto/     Reconstructed 24-layout final front, 513 feasible candidate scores, selected turbine XY.
03_independent_evaluation/  Final four guarded, optionally original six-layout confirmation.
04_sensitivities/      Corrected probability, radius, hub, speed and fixed-width 8-direction results.
05_uncertainty/        Optimisation-sample individual-bird bootstrap, 10000 paired resamples.
06_numerical_resolution/  Saved 360-sector AEP verification.
07_scientific_code_and_job_scripts/  Frozen source snapshot, runner/PBS copies, this plotter.
08_figures/           Generated PDF, SVG and PNG plots (create on HPC via plot script).
09_derived_tables/    Tables generated from archived data by plotting script.
archive_manifest.csv  Per-file SHA-256 and provenance.
archive_audit.json    Verified completion, sample comparability and layout selection.

IMPORTANT: pre-correction 12-sector experiments are NOT part of this final corrected-results archive. Keep separate.
Never use the internal labels from old per-seed runner (e.g. training knee) for final selection.
Final selections: highest-AEP 730101/455; knee 730103/418; lowest-risk 730103/220.
Optimisation sample: 5x10000 birds. Independent evaluation sample: 10x50000 birds.
Directional sensitivity: fixed W=9696.481 m, 10x57137 birds; bird heading, rotor and Band wind rotate together, AEP fixed.
Plot: python 07_scientific_code_and_job_scripts/model10_plot_master.py --master .
