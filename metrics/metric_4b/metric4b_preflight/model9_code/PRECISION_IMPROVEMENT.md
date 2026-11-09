# Model 9 precision improvement

## Outcome

Model 9 now reports two estimators of the same expected rotor-transit
intensity:

- `E_RT`: the original estimator based on each bird's sampled constant flight
  height. It remains the audit metric.
- `E_RT_CE`: a conditional-expectation estimator that integrates the supplied
  discrete flight-height distribution exactly for each realised horizontal
  path. It is the default optimisation metric.

This changes estimator precision, not Model 9's biology or endpoint. Both
quantities describe rotor-swept-area transit exposure, not blade collision,
collision probability, fatality, or mortality.

## Derivation

For bird `j` and turbine `i`, let `L_ji` be the lateral offset where the
horizontal segment crosses the fixed rotor plane. With rotor radius `R` and
hub height `H`, a transit is geometrically possible for heights in

\[
I_{ji}=\left[H-\sqrt{R^2-L_{ji}^2},
H+\sqrt{R^2-L_{ji}^2}\right]
\]

when `|L_ji| <= R`. If the horizontal path does not cross the plane, or its
lateral offset exceeds `R`, `I_ji` is empty.

The sampled indicator is

\[
Y_{ji}=\mathbf{1}\{Z_j\in I_{ji}\},
\]

where `Z_j` is the bird's one constant height drawn from the frozen FHD. The
new contribution is

\[
w_{ji}=P_F\{Z\in I_{ji}\}.
\]

The implementation calculates `w_ji` by inclusive discrete-CDF lookup; it
does not smooth or refit the FHD. Because the frozen Model 8 horizontal path
does not depend on flight height,

\[
w_{ji}=E(Y_{ji}\mid\text{horizontal path}).
\]

Linearity of expectation then gives

\[
E_{RT,CE}=\frac{1}{N}\sum_j\sum_i w_{ji}
\]

with the same expectation as sampled `E_RT`. Independence among turbine
transits is not required for this intensity result. By the law of total
variance, conditioning cannot add variance and removes the component caused
only by sampled height.

If a bird crosses one turbine's plane more than once, the widest eligible
height interval is used. All turbines in a run share `H` and `R`, so these
same-centre intervals are nested and their union probability is their maximum.
For the same reason, `P_RT_CE` is the mean per-bird maximum probability across
turbines and estimates the expected prevalence of one or more transits.

## What remains stochastic

`E_RT_CE` still varies across trajectory batches because entry locations and
Model 8 response decisions vary. It does not account for uncertainty in the
published FHD, turbine geometry, or inherited meso-response parameters. The
flight-height seeds remain in the run for sampled `E_RT` auditing but do not
drive `E_RT_CE`.

## Verification

The test suite verifies:

- inclusive discrete-CDF boundaries and array broadcasting;
- centre, off-centre, tangent, lateral-miss, and no-plane-crossing cases;
- exact circular height-window probabilities;
- agreement with a fixed 200,000-draw Monte Carlo sample;
- integrated separation of sampled and conditional results;
- Model 8 parity and encounter-subset accounting;
- combined batch totals and per-turbine conditional totals; and
- default/explicit optimiser objective selection.

Run:

```bash
python -m unittest discover -s tests -v
```

All 50 tests pass in the packaged verification environment, including seven
new Model 8-to-Model 9 FLORIS and seed-population integration checks.

## Northern-gannet fixed-layout evidence

Five independent common-random-number batches of 10,000 trajectories were
run for each controlled layout. The exact supplied Johnston/stochLAB northern-
gannet table and the same physical/behavioural settings as v2 were used. All
150,000 layout-specific simulations completed with zero failures.

| Layout | Mean sampled E_RT | Mean E_RT_CE | Sampled batch SD | Conditional batch SD | Variance reduction |
|---|---:|---:|---:|---:|---:|
| Aligned | 0.003840 | 0.003764 | 0.001090 | 0.000103 | 111.50x |
| Staggered | 0.003260 | 0.002792 | 0.000835 | 0.000106 | 61.90x |
| Wider-gap | 0.002920 | 0.002907 | 0.000726 | 0.000042 | 305.49x |

For every layout, the paired 95% Student-t interval for sampled minus
conditional intensity included zero. This supports estimator agreement; it is
not a proof that the FHD or biological assumptions are correct.

Conditional paired layout contrasts were:

| Contrast | Mean difference | Paired 95% interval |
|---|---:|---:|
| Aligned - staggered | 0.000972 | [0.000869, 0.001074] |
| Aligned - wider-gap | 0.000856 | [0.000698, 0.001015] |
| Staggered - wider-gap | -0.000115 | [-0.000292, 0.000061] |

The conditional estimator resolves both contrasts involving the aligned
layout, whereas sampled `E_RT` resolved none. Staggered versus wider-gap
remains unresolved and must not be presented as a robust ordering.

The complete reproducible outputs are in
`results/model9_precision_validation_northern_gannet/`.

## Optimisation use

`Model9AEPRotorTransitProblem` defaults to:

\[
f_1=(AEP_{ceiling}-AEP)/s_{AEP},\qquad
f_2=E_{RT,CE}/s_{RT}.
\]

Both scaling constants are positive and therefore do not change Pareto
dominance. Evaluation logs retain `E_enc`, sampled `E_RT`, conditional
`E_RT_CE`, their batch diagnostics, and the raw objective value. An intentional
sampled-objective comparison remains available with
`bird_objective_metric="E_RT"`.

The next appropriate step is a controlled NSGA-II pilot using frozen common-
random-number trajectory batches, followed by fresh independent-batch
re-evaluation of selected Pareto candidates. A final large run should not be
treated as self-validating.
