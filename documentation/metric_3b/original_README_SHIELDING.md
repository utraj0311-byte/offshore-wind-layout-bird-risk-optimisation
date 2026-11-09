# Directional Shielding Model v5

## Purpose

This model tests whether turbine layout changes bird exposure purely through
directional geometric overlap.  Birds are assumed to fly on straight,
non-responsive paths.  Turbines that appear behind one another from the chosen
flight direction contribute overlapping projected rotor widths, and overlap is
counted only once.

The model is a **directional geometric exposure surrogate**.  It is not a
collision model and does not estimate mortality.

Version 5 retains the same shielding equations, FLORIS configuration,
objectives, 12-turbine site and 4D spacing requirement. Its changes address the
incomplete and disconnected front produced by ordinary coordinate crossover:

- spacing-infeasible candidates are screened before FLORIS is called;
- candidate offspring receive deterministic minimum-spacing repair;
- a spacing-feasible candidate is excluded if any FLORIS rotor-grid velocity
  is negative;
- parent turbines are spatially matched with the Hungarian assignment before
  crossover, because turbine IDs are arbitrary;
- every feasible layout found throughout the run is retained in a persistent
  external nondominated archive, rather than reporting only the last NSGA-II
  population;
- 61 best layouts from the completed local development validation are packaged
  as refinement seeds and are re-evaluated under the active FLORIS setup;
- only the energy and shielding endpoints are reported; there is no automatic
  knee selection.

## Exact calculation

For bird heading \(\theta\), using the mathematical **TOWARD** convention
(0° east, 90° north):

\[
\mathbf d=(\cos\theta,\sin\theta),
\qquad
\mathbf n=(-\sin\theta,\cos\theta).
\]

For turbine \(i\) at \((x_i,y_i)\), its cross-flight position is:

\[
s_i=-x_i\sin\theta+y_i\cos\theta.
\]

With rotor diameter \(D\), its projected interval is:

\[
I_i=[s_i-D/2,\ s_i+D/2].
\]

The unique exposed width and primary metric are:

\[
L_{\mathrm{exp}}=\left|\bigcup_{i=1}^{N} I_i\right|,
\qquad
N_{\mathrm{eff}}=\frac{L_{\mathrm{exp}}}{D}.
\]

Interpretation:

- two completely aligned rotors: \(N_{\mathrm{eff}}=1\);
- half-diameter lateral offset: \(N_{\mathrm{eff}}=1.5\);
- one-diameter or larger lateral offset: \(N_{\mathrm{eff}}=2\).

For a stated uniform incoming corridor width \(W\):

\[
P_{\mathrm{enc}}=\frac{L_{\mathrm{exp}}}{W},
\qquad
P_{\mathrm{free}}=1-P_{\mathrm{enc}}.
\]

`P_enc` is the analytical fraction of straight incoming paths intersecting at
least one rotor-width corridor.  It is not collision probability.

The implementation rounds dimensionless `N_eff` to 12 decimal places.  This is
only a machine-precision safeguard: it prevents geometrically identical values
such as `3.0000000000000058` and `3.0000000000000067` from being treated as a
real optimisation trade-off.  It does not alter exposure at a scientifically
meaningful spatial scale.

## Scientific status

Direction-dependent turbine presentation and wind-farm configuration are
supported modelling concepts in the collision-risk literature.  The interval
union and `N_eff` equations are definitions developed for this thesis.  They
must therefore be described as a **literature-informed analytical metric
developed in this study**, not attributed as an existing equation from a
published collision model.

The closest conceptual source identified in the model brief is Smales, Muir,
Meredith and Baird (2013), *A description of the Biosis model to assess risk of
bird collisions with wind turbines*, DOI: 10.1002/wsb.257.  That paper does not
provide this exact interval-union metric.

## Locked initial scenario

| Setting | Value |
|---|---:|
| Turbines | 12 |
| Site | 6000 m × 6000 m |
| Turbine | FLORIS IEA 15 MW |
| Rotor diameter | 242.24 m, read from FLORIS |
| Minimum spacing | 4D = 968.96 m |
| Bird heading | 225° TOWARD |
| Incoming corridor width | 6000 m |

The older `config.py` prototype diameter of 240 m does not control the coupled
model.  The official 242.24 m diameter read by `FLORISAEPModel` controls both
shielding and the 4D spacing constraint.

## Locked-case geometric minimum

For this specific 12-turbine, 6 km square, 225 degree, 4D-spacing case,
`N_eff = 2` is both feasible and the geometric lower bound.  A validated
two-lane layout places six turbines on each of two projected flight-aligned
lanes and satisfies the boundary and spacing constraints.

The lower bound follows directly from the locked geometry.  Any layout with
`N_eff < 2` would have to place all projected rotor intervals in one connected
band whose turbine-centre width is less than one rotor diameter.  The maximum
along-flight span of the square at 225 degrees is
`6000*sqrt(2) = 8485.3 m`.  Among 12 turbines, one consecutive along-flight gap
would then be at most `8485.3/11 = 771.4 m`.  Even with a full-diameter
cross-flight gap, that pair could be separated by at most `808.5 m`, below the
required `4D = 968.96 m`.  Therefore one band is infeasible, while the supplied
two-lane construction demonstrates that two bands are achievable.

## Files

- `shielding_model.py` — standalone projection, interval union, `N_eff`,
  `P_enc`, and free-path calculation.  Requires only NumPy.
- `tests/` — 23 deterministic geometry, seed-integrity, candidate-screening,
  turbine-matching, spacing-repair and archive tests.
- `run_shielding_validation.py` — the three exact analytical checks, fixed
  12-turbine direction checks, and optional energy-endpoint evaluation.
- `shielding_aep_problem.py` — pymoo problem coupling FLORIS AEP to `N_eff`.
- `run_shielding_multiobjective.py` — permutation-aware NSGA-II runner with a
  persistent external Pareto archive.
- `seed_data/prior_pareto_seed_layouts.csv` — 61 validated, objective-stable
  layouts from the completed v5 development validation.
- `requirements_shielding.txt` — Python dependencies.

## Folder placement

Place this folder directly inside the existing project:

```text
bird_layout_opt/
├── config.py
├── floris_aep.py
├── floris_config/
│   └── gch_iea15mw.yaml
├── report_assets/
│   └── model_10_final_floris/source_data/energy_endpoint_layout.csv
└── shielding_model_code/
    ├── shielding_model.py
    ├── shielding_aep_problem.py
    ├── run_shielding_validation.py
    ├── run_shielding_multiobjective.py
    ├── seed_data/prior_pareto_seed_layouts.csv
    └── tests/
        ├── test_shielding_model.py
        ├── test_prior_seed_data.py
        ├── test_shielding_aep_problem.py
        └── test_optimisation_operators.py
```

If downloaded files have browser suffixes such as `config(5).py` or
`floris_aep(4).py`, restore the normal project names `config.py` and
`floris_aep.py` before running the coupled optimisation.

## 1. Run the unit tests

From the `bird_layout_opt` project root:

```bash
python -m unittest discover -s shielding_model_code/tests -v
```

Expected final lines:

```text
Ran 23 tests
OK
```

The tests verify:

- exact values 1, 1.5, and 2;
- a feasible 12-turbine two-lane layout with `N_eff = 2`;
- direction dependence;
- equivalence of headings separated by 180°;
- invariance to turbine numbering and whole-layout translation;
- interval-union correctness;
- prevalence and free-path accounting;
- rejection of invalid inputs;
- spacing-infeasible candidates are skipped before FLORIS;
- positive rotor velocities remain feasible;
- negative rotor velocities are formally excluded;
- arbitrary turbine permutations are spatially matched before crossover;
- repaired candidates satisfy the 4D spacing constraint;
- dominated and duplicate objective rows are excluded from the archive.

## 2. Run the fixed-layout validation

```bat
python shielding_model_code\run_shielding_validation.py --energy_seed_layout report_assets\model_10_final_floris\source_data\energy_endpoint_layout.csv --bird_heading_deg 225 --rotor_diameter_m 242.24 --corridor_width_m 6000
```

Main outputs:

- `shielding_validation_summary.csv`;
- `fixed_layout_coordinates.csv`;
- `energy_endpoint_projection.csv`;
- `run_settings.json`.

No Monte Carlo convergence or fresh-sample validation is required because the
same coordinates and heading always produce exactly the same shielding value.

The optimiser is seeded with the validated two-, three-, and four-lane
layouts.  Jittered variants of these fixed layouts move turbines only along
the flight axis (plus whole-layout translation), so their projected exposure
is preserved while FLORIS can distinguish their energy performance.  This is
necessary because exact projected alignment is a narrow geometric family that
generic independent x-y mutation may otherwise miss.

## 3. Run the v5 smoke test

From the `bird_layout_opt` project root on Windows:

```bat
python shielding_model_code\run_shielding_multiobjective.py --floris_config floris_config\gch_iea15mw.yaml --energy_seed_layout report_assets\model_10_final_floris\source_data\energy_endpoint_layout.csv --bird_heading_deg 225 --corridor_width_m 6000 --seed 48 --pop 20 --gen 2 --output_root results\shielding_v5_smoke
```

The 61 packaged seed layouts are re-evaluated automatically, even though only
a subset is used for breeding in the small smoke-test population. The external
archive therefore preserves the established endpoints from the start.

## 4. Run the final refinement

After the smoke test succeeds:

```bat
python shielding_model_code\run_shielding_multiobjective.py --floris_config floris_config\gch_iea15mw.yaml --energy_seed_layout report_assets\model_10_final_floris\source_data\energy_endpoint_layout.csv --bird_heading_deg 225 --corridor_width_m 6000 --seed 48 --pop 160 --gen 250 --save_evaluations --output_root results\shielding_v5_final
```

This is a refinement of the existing evidence rather than an independent run
from scratch. Seed 48 gives new variation. All 61 packaged layouts are checked
and inserted into the external archive; the evolving population combines a
spread of these seeds with projection-preserving variants and random feasible
layouts. On the machine used for development, 160 x 120 plus seed validation
took about 29 minutes. The 160 x 250 command will commonly take roughly 45 to
75 minutes, depending on the computer.

Do not judge convergence from the appearance of a smooth line. This metric can
produce separate two-band and three-band layout regimes. Treat the run as
complete only after checking that the final 25 hypervolume generations have
flattened and that the endpoints have stopped improving. A visible gap is
acceptable when layouts evaluated within it are dominated; the code must not
draw an invented connecting curve.

The optimisation minimises two numerical objectives:

\[
f_0=\frac{AEP_{\mathrm{ceiling}}-AEP}{AEP_{\mathrm{loss,ref}}},
\qquad
f_1=\frac{N_{\mathrm{eff}}}{N}.
\]

Dividing `N_eff` by `N` is only numerical scaling.  It cannot change Pareto
dominance, and all result tables retain the interpretable unnormalised `N_eff`.

Main outputs:

- `pareto_points.csv`;
- `anchor_summary.csv`;
- `layouts/energy_endpoint_layout.csv`;
- `layouts/shielding_endpoint_layout.csv`;
- `pareto_front.png`;
- `hypervolume.csv`;
- `all_evaluated_candidates.csv` when `--save_evaluations` is used;
- `pareto_constraints.npy`;
- `run_settings.json` and `summary.json`.

Only one essential result graph is produced: AEP against `N_eff`.

`summary.json` also reports:

- how many spacing-infeasible candidates were skipped before FLORIS;
- how many spacing-feasible candidates were rejected for negative rotor
  velocity;
- the minimum rotor-grid velocity observed among FLORIS-evaluated candidates;
- zero spacing and rotor-velocity violations in the saved Pareto set.

The reported Pareto files come from the external nondominated archive over all
feasible evaluations, not only the final population. `hypervolume.csv` tracks
that same persistent archive and therefore cannot lose an earlier good layout.

The knee is deliberately absent.  A knee chosen from normalised Euclidean
distance is a post-processing preference, not an optimisation objective, and
no decision-maker preference has been supplied to justify it.

## Development validation evidence

The completed local v5 validation (`seed=47`, `population=160`,
`generations=120`) was used to test the operator and create the packaged seed
archive. It is not a substitute for the final run in the project Conda
environment. It produced:

- 19,711 total recorded candidates, including 511 seed revalidations;
- 12,109 feasible FLORIS evaluations;
- zero negative rotor-velocity candidates;
- 61 externally archived Pareto layouts;
- energy endpoint: 849.3203 GWh/year at `N_eff = 3.1120`;
- shielding endpoint: 796.6728 GWh/year at `N_eff = 2.0000`;
- external-archive hypervolume: 1.00965.

The external hypervolume was still increasing at generation 120, which is why
the final Windows command uses 250 generations. The validation also evaluated
many feasible layouts between `N_eff = 2.6` and `3.0`; their AEP was below the
best layout at `N_eff = 2.5383`, so they were dominated and correctly absent
from the reported front. A gap alone is therefore not evidence of a broken
Pareto calculation.

## Correct interpretation

The optimisation answers:

> Can 12 turbines be arranged to retain high AEP while presenting less unique
> rotor width to straight-flying birds approaching on the specified axis?

Lower `N_eff` means stronger shielding.  It does not automatically mean fewer
collisions because flight height, rotor transit, blade strike, avoidance and
mortality are outside this model.

## Deliberate exclusions

- no macro, meso or micro avoidance;
- no response probability or deflection angle;
- no individual trajectories;
- no random bird sampling or batches;
- no downstream-distance effect;
- no flight-height distribution;
- no rotor-transit or blade-strike calculation;
- no mortality estimate.
