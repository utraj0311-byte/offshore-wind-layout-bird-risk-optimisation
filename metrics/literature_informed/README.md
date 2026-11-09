# Literature-informed bird-risk metric — Model 10

**MSc thesis:** *Integrating Bird Collision Risk into Multi-Objective Offshore Wind Farm Layout Optimisation*.

This folder contains the **original archived 72-sector (5°) Model 10** source files plus the separately recovered, importable Python modules placed in the path the original runner expects. No scientific Python source or original result data was rewritten.

## What is being measured?

The final bird objective is **expected Band potential strikes per entering bird**, denoted `Q_lit`. A bird that does not cross a rotor contributes zero. This is a layout-comparison and exposure metric, **not** a validated estimate of bird mortality.

The model combines the Model 8 bird-response trajectories, Model 9 northern-gannet flight-height / rotor-disc crossing calculations, and an added Band blade-strike contribution for accepted crossings. A separate FLORIS AEP objective is optimised simultaneously using NSGA-II.

## Final frozen experiment

| Item | Archived value |
|---|---|
| Optimisation wind resource | 72 wind-direction sectors at 5° spacing; speed bins 4, 6, 8, 10, 12, 14, 16 m/s |
| Farm | 12 IEA 15 MW turbines; 6 km × 6 km; minimum spacing 4D = 968.96 m |
| Turbine for FLORIS | D = 242.24 m, hub height = 150 m |
| NSGA-II | Population 24 × 20 generations; separate seeds 730101, 730102, 730103 |
| Training birds | 5 × 10,000 per objective evaluation, using frozen trajectory/height seeds |
| Independent confirmation | 10 × 50,000 per layout in the guarded final confirmation |
| Bird objective reference | `R_ref = 3.20618950134584e-5` potential strikes per entering bird |
| AEP objective ceiling | `854.4956585617186 GWh/year` |
| AEP-loss scaling reference | `137.96914764866568 GWh/year` |
| Final combined selection | 24 non-dominated layouts from 513 feasible candidates among 1,440 evaluations |
| Software recorded | Python 3.11.13; FLORIS 4.6.5; pymoo 0.6.1.5; NumPy 2.0.2 |

The optimisation minimises `(AEP_ceiling - AEP) / AEP_loss_reference` and `Q_lit / R_ref`. These numerical scales do not convert potential strikes into mortality.

## Structure

- `original_model10/` — frozen seed-730101 source tree, added original importable runtime modules, and original seed-730102/730103 runners.
- `analysis_scripts/` — archived confirmation, plotting and PBS scripts; some retain original HPC-directory assumptions.
- `../../configurations/literature_informed/` — archived run settings, wind/flight inputs and turbine data.
- `../../results/literature_informed/` — complete supplied three-seed result folders, global Pareto set and independent confirmation (including each run's untouched source snapshot).
- `../../documentation/literature_informed/` — provenance, integrity checks, limitations and integration instructions.

**Start with** `../../documentation/literature_informed/REPRODUCIBILITY.md` and `INTEGRATION_NOTES.md`.

This is the *final corrected 72-sector optimisation*, not the earlier 12-sector Model 10 experiment. The uploaded recovery archive did not include the separate sensitivity, bird-level bootstrap or 360-sector datasets. See `MISSING_FILES.md` before claiming all results can be reproduced.
