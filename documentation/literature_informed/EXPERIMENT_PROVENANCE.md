# Experiment provenance — literature-informed metric

| Frozen run | Optimiser seed | Original entrypoint | Saved source and settings |
|---|---:|---|---|
| Final 5°/72-sector run A | 730101 | `model10_fhd_robust_seeded_50k_72sectors.py` | `results/literature_informed/01_optimisation/seed_730101/optimisation/` |
| Final 5°/72-sector run B | 730102 | `model10_v3_72sectors_seed730102.py` | `results/literature_informed/01_optimisation/seed_730102/optimisation/` |
| Final 5°/72-sector run C | 730103 | `model10_v3_72sectors_seed730103.py` | `results/literature_informed/01_optimisation/seed_730103/optimisation/` |

All runs record `population=24`, `generations=20`, `evaluations=480`, `number_of_direction_sectors=72`, shared training trajectory seeds 731101–731105 and height seeds 732101–732105, and initial population seed 730301.

Archived master-audit selection (using training `Q_lit` and 72-sector AEP):

| Selection | Seed | Evaluation | AEP (GWh/year) | Q_lit per entering bird |
|---|---:|---:|---:|---:|
| Highest AEP | 730101 | 455 | 823.0822640294288 | 8.5322223718665e-5 |
| Knee (archived selection) | 730103 | 418 | 821.3442419232357 | 6.124694896392856e-5 |
| Lowest risk | 730103 | 220 | 807.5716007926734 | 2.9456101769728496e-5 |

The knee is an archived descriptive choice, not proof of a scientifically privileged trade-off. The final full front is saved as `results/literature_informed/02_combined_pareto/combined_non_dominated_24.csv`, while `final_selected_three.csv` holds the representative IDs.

Training and independent estimates must not be conflated. The independently paired, guarded `10 × 50,000` confirmation files are in `results/literature_informed/03_independent_evaluation/final_four_guarded/`.

The original 12-sector runs, fixed-height controls and additional sensitivity experiments are **not** the final 72-sector optimisation represented by these archived results. This upload contains no evidence of a separate completed 5° optimisation for any other metric.
