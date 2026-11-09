# Metric 2b — farm-wide pairwise Gaussian distance metric (formerly Model 6)

## Purpose and connection to other metrics

Metric 2b removes the designated-index dependence of Metric 2a by summing contributions over **all 66 unique pairs** of 12 turbines. It complements prior position-based metrics 1a–1d and precedes the directional metrics 3a/3b. This is a deliberately geometric verification metric, not a calibrated bird-collision probability.

## Mathematical implementation

The original `PairwiseGaussianRiskModel` in `source/pairwise_risk.py` defines

```
Q_2b = sum(i<j) exp(-||X[i]-X[j]||^2/(2*sigma_b^2))
sigma_b = 8*D = 1920 m; D = 240 m
f2 = Q_2b / 66
f1 = 2 - 2*AEP_GWh / 760.663033300808
```

`risk_multiplier=1` for the final run. Every unordered pair is included exactly once, and the total is invariant to turbine relabelling. The source function is named `collisions()` for historical compatibility and returns Q, not observed collision count.

## Common physical and wind settings

Identical to Metric 2a: 12 turbines in 6 x 6 km; min centre spacing 960 m (4D) for every pair; site coordinate bounds [0,6000] m; 240 m rotor, 150 m hub, 15 MW turbine; the original Jensen `wake.py` with `kw=.04`, hub-centre wake test, RSS clipped to .95 and undisturbed-speed CT rule. The final synthetic wind resource uses 360 one-degree sectors centred on wind **FROM 225°** with weights proportional to `exp(2*cos(delta_from_225deg))`, seven wind-speed bins (4,6,8,10,12,14,16) m/s and discretised, renormalised Weibull(k=2,A=10 m/s) probabilities. The untouched original `config.py` defaults to 12 sectors; the final runner selects 360 in memory.

## Search settings

- Multiobjective **NSGA-II population 80, generations 100, seed 42**, duplicate elimination, `GridSampling(model_name='constant')`: 50% randomly feasible and 50% jittered grid initial layouts; default pymoo 0.6.1.5 operators.
- Historical **risk-only GA**: population **80**, generations **100**, seed **42**, with `--scale_D 8.0` (IMPORTANT: the script's unmodified default is **4.0**, so the supplied default must not be used for this metric).
- Historical separate **AEP-only GA**: population **160**, generations **250**, seed **42**, implemented in `reference_scripts/run_aep_only.py`; original `aep_only_problem.py` is not available in this chat.
- The final separate 360-sector AEP-only GA has the same 160/250/42 settings and its full result archive is preserved under `results/metric_2b/references/`. The final multiobjective Pareto front was compared to it and the previously optimised risk-only layout. Risk-only was **not** rerun at 360 sectors; its original saved layout was evaluated for AEP only.

## Verification scripts

`test_pairwise_risk.py` checks pair-distance ordering, permutation invariance, expected 66 pairs and per-turbine accounting. Note: its unchanged default calls `interaction_scale_D=4.0`; this is a generic kernel test, **not** proof that an 8D GA was executed. `validate_pairwise_small_systems.py` checks two/three-turbine configurations (default 8D) with a small single-objective GA; it is verification code, not the final 12-turbine NSGA-II. These scripts and original single-objective runners are included unchanged but may require missing `risk_only_problem.py` for the original risk-only search; `hypothetical_risk.py` is already available in the existing repository.

## Final archived results

`results/metric_2b/final_360sectors/metric_2b_360sectors_seed42_pop80_gen100/` has **20** complete nondominated layouts, `pareto_full.npz` (`X`,`F`), full layout coordinates, Pareto CSV, endpoints, hypervolume 100 generations, settings, endpoint convergence, and plots. Highest-AEP endpoint: **734.176774** GWh/year at **Q≈10.449036**. Lowest-Q endpoint: **732.054312** GWh/year at **Q≈10.064129**. Original risk-only reference: **Q=9.841423**, AEP at 1° **731.907391** GWh/year. Separate new AEP-only GA reference: **734.224321** GWh/year at 1°. The latter is a separate optimiser result, not a global energy bound.

Final thesis-ready PDFs are in `results/metric_2b/thesis_figures_final/`. The source and full archive for the final GA are retained; no optimisation was run during packaging.

## Verification and reproduction

The actual final 360-sector runner `shared/metric2_360_tools_original/run_metric2_360.py` is byte-identical to the script hash recorded in the saved results. The run's preflight recorded six successful historical Jensen AEP reference checks. This package independently tested the original 360-sector AEP and Q functions against every saved Pareto coordinate set and checked bounds and normalisations. The result checks and remaining limitations are recorded in `documentation/PROVENANCE_AND_VALIDATION.md`.

Reproducing the full optimisation requires staging the original shared source files in an isolated project-root workspace, obtaining the two missing original helper modules for the relevant stages, installing the archived environment dependencies and invoking the final runner as described in `documentation/INTEGRATION_NOTES.md`. Archive output directories are protected against overwrites. The historical pymoo version is unknown; pymoo 0.6.1.5 is recorded for the new final runs.

## Limitations

- The Gaussian distance function is geometric, not an ecological collision model.
- The 8D scale is a modelling assumption; a smaller minimum pair spacing can coexist with larger mean spacing and reduced Q.
- Population-based searches do not prove global optima; the risk-only and AEP-only searches have distinct search objectives and must not be conflated.
- The package preserves the exact available implementation but does not contain all historical helper modules; see MISSING_FILES.md.

## GitHub add-on note (reuse of existing shared source)

This conflict-free add-on reuses the existing repository `shared/wake.py`, `shared/run_validation.py`, `shared/validation_problem.py`, and `shared/hypothetical_risk.py`; all four archived hashes were verified. Historical `config.py` remains as `shared/jensen_metric2_original/config.py` without replacing the existing 360-sector config. See `documentation/INTEGRATION_NOTES.md`.
