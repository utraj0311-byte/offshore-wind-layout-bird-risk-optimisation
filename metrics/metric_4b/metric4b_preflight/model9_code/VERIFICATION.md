# Verification record

Date: 8 September 2026  
Runtime used here: Python 3.12.13

- All Python sources compile successfully.
- 59 of 59 deterministic unit/regression tests pass.
- Nine endpoint-confirmation tests verify paired Student-t calculations,
  common-random-number matching, failure rejection, count aggregation, and the
  frozen primary decision rule.
- All inherited Model 8 regression tests pass after adding the passive segment
  observer.
- The direct Model 8 versus Model 9 parity test gives identical encounters,
  responses, route lengths, completion, and failure states.
- The required below-rotor, above-rotor, rotor-centre, and lateral-miss tests
  pass.
- Circular-corner rejection, exact tangency, coarse-step intersection,
  multiple-transit accounting, turbine-order invariance, and flight-height
  reproducibility tests pass.
- The exact northern-gannet input schema, checksum, mass total, and mean height
  are protected by a regression test.
- Conditional discrete-CDF boundaries, circular height windows, a 200,000-draw
  Monte Carlo expectation check, count-first aggregation, and optimiser
  objective selection are protected by tests.
- A complete synthetic two-batch validation ran for all three controlled
  layouts and produced zero failed trajectories plus all expected outputs.
- The Model 8 integration tests confirm 12 direction sectors, seven speed
  bins, 84 normalised FLORIS conditions, a 242.24 m rotor, deterministic reuse
  of all four exact seed anchors, feasibility of the 80 supplied layouts, and
  rejection of the obsolete 240 m configuration.
- A mocked one-generation integrated runner check completed from population
  loading through optimisation, checkpointing, fresh validation, robust
  filtering, anchor layout export, plotting, and final settings output. It
  produced all 14 expected top-level output files/directories and finished
  with `run_status = completed`.

## Northern-gannet FHD validation

- Input rows: 300, representing height labels 0–299 m.
- Native columns: `height,prop`.
- Missing, non-finite, negative, or duplicated values: none.
- Raw probability sum: 1.0000000558430553; normalised internally to one.
- Mean / median height: 9.1497 m / 6 m.
- Mass in the 28.88–271.12 m vertical rotor band: 0.0492461.
- SHA-256: `adc9ea903db9c8e34bfe5b88c382381b737c2e676d79af437f66d198208a0f2c`.
- Source record: northern gannet maximum-likelihood estimate (`Gannet.est`)
  from stochLAB's `Johnston_Flight_heights_SOSS`, based on Johnston et al.
  (2014), DOI 10.1111/1365-2664.12191. The exact upstream package release or
  commit was not embedded in the supplied CSV.

## Fixed-layout precision evidence run

Five independent common-random-number batches of 10,000 trajectories were run
per layout (50,000 per layout). All 150,000 layout-specific simulations
completed, with zero failed trajectories.

| Layout | E_enc | Sampled E_RT | Conditional E_RT_CE | Conditional expected N_RT | Variance reduction |
|---|---:|---:|---:|---:|---:|
| Aligned | 0.32334 | 0.003840 | 0.003764 | 188.18 | 111.50x |
| Staggered | 0.42824 | 0.003260 | 0.002792 | 139.60 | 61.90x |
| Wider-gap | 0.27566 | 0.002920 | 0.002907 | 145.36 | 305.49x |

Model 8's point ranking from low to high exposure was wider-gap, aligned,
staggered. Model 9's point ranking was wider-gap, staggered, aligned. However,
the rotor-transit ordering changed among the five batches, and every paired
95% Student-t interval included zero:

| Paired E_RT difference | Mean | Paired 95% interval |
|---|---:|---:|
| Aligned minus staggered | 0.00058 | [-0.000304, 0.001464] |
| Aligned minus wider-gap | 0.00092 | [-0.001204, 0.003044] |
| Staggered minus wider-gap | 0.00034 | [-0.001443, 0.002123] |

Every 95% interval for sampled minus conditional `E_RT` included zero, which
is consistent with both estimators targeting the same expectation. The
conditional estimator reduced standard deviation by 7.87x–17.48x (variance by
61.90x–305.49x).

| Paired E_RT_CE difference | Mean | Paired 95% interval | Resolved? |
|---|---:|---:|:---:|
| Aligned minus staggered | 0.000972 | [0.000869, 0.001074] | Yes |
| Aligned minus wider-gap | 0.000856 | [0.000698, 0.001015] | Yes |
| Staggered minus wider-gap | -0.000115 | [-0.000292, 0.000061] | No |

The estimator therefore resolves two contrasts that sampled `E_RT` did not,
while leaving the genuinely close staggered/wider-gap contrast unresolved.
The prior outputs are retained transparently under
`results/model9_validation_northern_gannet` and
`results/pilot_6000_per_layout`.

## Optimisation status

The prior smoke-validation FHD was synthetic and produced no biological or
thesis result. A real FLORIS/pymoo optimisation was not run in this packaging
workspace because those optional runtime dependencies are not installed here.
The complete runner was instead checked end to end with deterministic test
adapters, while its scientific, geometry, aggregation, and objective cores are
covered by the 50-test suite.

The next authorised step is the real `--profile smoke` run in `floris-env`,
followed by inspection of its automatically zipped outputs. The controlled
pilot should follow only if that run passes. Every returned training Pareto
candidate is then re-evaluated automatically using independent trajectory and
height seeds. This does not justify treating every close layout contrast as
resolved.
