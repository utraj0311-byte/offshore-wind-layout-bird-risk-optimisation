# Shielding Model v5 refinement seeds

`prior_pareto_seed_layouts.csv` contains the 61 coordinate-exact layouts in
the external nondominated archive from the completed v5 development validation
(`seed=47`, `population=160`, `generations=120`).

The development validation combined:

- the earlier v2, v3 and v4 nondominated layouts;
- matched interpolation between spatially corresponding turbine layouts;
- permutation-aware simulated binary crossover;
- minimum-spacing repair;
- 12,109 feasible FLORIS evaluations.

Every stored seed:

- lies within the 6000 m square;
- satisfies `4D = 968.96 m` minimum spacing;
- has a non-negative FLORIS rotor-grid velocity;
- is nondominated by the other stored seeds;
- has an independently recomputable analytical `N_eff` value.

The CSV preserves AEP, `N_eff`, minimum spacing, minimum rotor-grid velocity,
and all 24 decision variables. The runner re-evaluates every seed under the
active FLORIS configuration before optimisation; stored objective columns are
provenance metadata rather than blindly trusted inputs.
