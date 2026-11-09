# Integrating Bird Collision Risk into Multi-Objective Offshore Wind Farm Layout Optimisation

**MSc Sustainable Energy Technology thesis — Utkarsh Raj (TU Delft, 2026)**

This repository archives the Python implementations, experiment configurations, representative results and verification records used to investigate how bird-related objectives can be incorporated into offshore wind-farm layout optimisation alongside annual energy production (AEP).

The development sequence moves from **synthetic spatial fields** to **turbine interactions**, **directional exposure**, **simulated trajectories**, **rotor-disc transits** and a **literature-informed Band potential-strike metric**. Early metrics are controlled methodological experiments, not calibrated predictions of bird collisions or mortality. The final literature-informed metric estimates *Band potential strikes per entering bird*, not observed collision mortality.

## Metric index

| Metric | Scientific quantity or idea | Historical label | Final archived experiment | Code and explanation |
|---|---|---|---|---|
| **1a** | Constant artificial spatial field (null control) | Model 1 | Jensen, 360 × 1° sectors | [Metric 1a](metrics/metric_1a/README.md) |
| **1b** | Eastern half-site artificial field | Model 2 | Jensen, 360 × 1° sectors | [Metric 1b](metrics/metric_1b/README.md) |
| **1c** | Central corridor artificial field | Model 3 | Jensen, 360 × 1° sectors | [Metric 1c](metrics/metric_1c/README.md) |
| **1d** | Gaussian artificial hotspot | Model 4 | Jensen, 360 × 1° sectors | [Metric 1d](metrics/metric_1d/README.md) |
| **2a** | Designated-turbine Gaussian surroundedness | Model 5 | Jensen, 360 × 1° sectors | [Metric 2a](metrics/metric_2a/README.md) |
| **2b** | Farm-wide pairwise Gaussian interaction | Model 6 | Jensen, 360 × 1° sectors | [Metric 2b](metrics/metric_2b/README.md) |
| **3a** | Directional upstream absorbing exposure | Model 7 | Risk-only genetic algorithm; no AEP wind rose | [Metric 3a](metrics/metric_3a/README.md) |
| **3b** | Unique projected rotor width (directional shielding) | Shielding model | FLORIS, 72 × 5° sectors | [Metric 3b](metrics/metric_3b/README.md) |
| **4a** | Distinct horizontal turbine encounters per entering bird | Model 8 | FLORIS, 72 × 5° sectors | [Metric 4a](metrics/metric_4a/README.md) |
| **4b** | Expected distinct rotor-disc transits per entering bird | Model 9 | FLORIS, 72 × 5° sectors | [Metric 4b](metrics/metric_4b/README.md) |
| **Literature-informed** | Expected Band potential strikes per entering bird | Model 10 | FLORIS, 72 × 5° sectors; three optimisation seeds | [Final metric](metrics/literature_informed/README.md) |

The table identifies the *authoritative experiments described by each metric's included README*; it does not imply that one software environment or wind rose was used for every development stage. Historical 12-sector archives are separately retained where available. A 1° AEP re-evaluation must not be described as an optimisation under 1° sectors.

## Repository layout

```text
metrics/           Original metric implementations, runner scripts and per-metric READMEs
configurations/    Saved model configurations, turbine/wind inputs and dependency files
results/           Final and historical optimisation results, selected layouts and figures
documentation/     Reproducibility audits, provenance, validation notes and missing-file records
shared/            Earlier common Jensen, risk, plotting and optimisation utilities
```

The `metrics/literature_informed/original_model10/` folder preserves the original final Model 10 Python source, and its dependencies and results are described by [Model 10 reproducibility notes](documentation/literature_informed/REPRODUCIBILITY.md).

## Interpretation and reproducibility

- **Different bird metrics are not interchangeable.** A synthetic field score, a projected rotor-width equivalent, an encounter count, a rotor transit and a Band potential-strike expectation have different meanings and may use different normalisation constants.
- **Wind-resource resolution matters.** Jensen Metrics 1a–2b use the later 1° archived runs; FLORIS Metrics 3b, 4a, 4b and the literature-informed metric use final 5° archived optimisation runs. Metric 3a is a bird-risk-only experiment and uses no wind rose. Historical 12-sector results must be kept separate.
- **Reproducibility is metric-specific.** Start with the corresponding metric's README and `documentation/` records. Several original runners expect historical working-directory layouts, and some archives require dependencies or inputs not fully included here. Do not assume a single `pip install` and one command will replay the entire thesis.
- **Do not rerun an expensive optimisation inadvertently.** Some scripts run thousands of layout evaluations if executed without their documented verification-only arguments. Use a separate scratch/work directory, preserve the archives, and check each metric's reproduction instructions first.
- **Archive verification is not full execution validation.** Recorded source hashes and saved results support traceability; complete end-to-end optimiser reruns have not been established for every metric from this repository alone.

For the final literature-informed metric, the saved optimisation used the scaling reference `R_ref = 3.20618950134584e-5` expected Band potential strikes per entering bird. This is a numerical objective scaling constant, not a mortality rate. The underlying Model 10 run settings and source files should be consulted for precise implementation and configuration.

## Getting started

1. Read this overview and choose a metric from the index.
2. Open that metric's `README.md` to identify its **authoritative final** script, configuration and results directory.
3. Check the metric's `documentation/` inventory and missing-file notes before attempting an exact replay.
4. Install the dependencies appropriate to that metric's recorded environment (see its README and configuration files).
5. Run only its documented verification/check procedures in a temporary workspace before attempting a fresh optimisation.

This is a research-code archive, **not** a packaged executable or an ecological forecasting service. No licence is granted implicitly: add an explicit `LICENSE` file after deciding how you wish to permit reuse of the code. Any academic citation should refer to the accompanying thesis and, if created, a versioned GitHub release/DOI.
