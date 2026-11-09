# Model 9 — Flight-Height and Rotor-Transit Exposure

For the completed three-layout confirmation, see `README_CONFIRMATION.md`.
For the completed targeted `pareto_038` extension, see
`README_P038_CONFIRMATION.md`. For the separate controlled follow-up added in
v7, start with `README_HUB_HEIGHT_SENSITIVITY.md`.

## Purpose

Model 9 keeps the complete Model 8 horizontal trajectory and meso-response
logic, assigns one constant flight height to each simulated trajectory, and
tests whether each resulting path crosses a turbine's circular rotor-swept
disk.

The underlying exposure quantity is

\[
E_{RT}=\frac{\text{total rotor-transit events}}
{\text{total entering trajectories}}.
\]

`E_RT` is the original sampled estimate of mean rotor transits per simulated
entering trajectory. `E_RT_CE` estimates the same expected intensity by
integrating the discrete flight-height distribution conditional on each
horizontal trajectory. It is the default optimisation estimator because it
removes flight-height sampling noise; sampled `E_RT` remains in every run as
an audit value. Neither is a collision probability.

`P_RT` is the sampled fraction of trajectories with one or more rotor transits.
`P_RT_CE` is its conditional expected analogue. Model 8's `E_enc` and `P_ge_1`
are retained as diagnostics.

## Frozen scientific boundary

Included:

- Model 8 individual trajectories and horizontal meso response, unchanged;
- one tabulated flight-height distribution;
- one sampled height `z_j` per trajectory, held constant through the farm;
- circular rotor geometry defined by diameter `D`, radius `R=D/2`, and hub
  height `H`;
- a fixed rotor plane perpendicular to a stated reference heading;
- individual, turbine, and farm rotor-transit counts;
- exact conditional integration of the same discrete FHD for variance
  reduction, without adding a biological process;
- independent seeded batches and uncertainty diagnostics.

Excluded:

- macro, micro, or vertical avoidance;
- changing, weather-dependent, or turbulence-dependent flight height;
- multiple species in one initial run;
- blade rotation, RPM, chord, pitch, twist, or detailed blade geometry;
- bird wingspan, body length, or conditional blade-strike probability;
- collision, fatality, or mortality estimates.

## Exact rotor-transit rule

Let `p0` and `p1` be the horizontal endpoints of one trajectory step, `t_i`
the turbine centre, and `u` the reference bird-heading unit vector. The fixed
rotor plane is perpendicular to `u`. Its intersection with the segment is

\[
q(\lambda)=p_0+\lambda(p_1-p_0),
\qquad
\lambda=-\frac{(p_0-t_i)\cdot u}{(p_1-p_0)\cdot u}.
\]

The segment crosses the plane when `0 <= lambda <= 1`. Let `v` be the
horizontal cross-heading unit vector. The crossing is inside the circular disk
when

\[
\left[(q-t_i)\cdot v\right]^2+(z_j-H)^2\le R^2.
\]

One bird-turbine pair is counted at most once. This intersection algorithm and
`E_RT` are thesis implementation definitions; they are not presented as
equations copied from the reviewed papers.

### Conditional-expectation estimator

For a horizontal plane crossing with lateral offset `l`, the circular rotor
permits heights in

\[
I(l)=\left[H-\sqrt{R^2-l^2},\;H+\sqrt{R^2-l^2}\right]
\]

when `|l| <= R`; otherwise the contribution is zero. Model 9 obtains

\[
w_{ji}=P\{Z\in I(l_{ji})\}
\]

by an exact lookup in the supplied discrete FHD and reports

\[
E_{RT,CE}=\frac{1}{N}\sum_j\sum_i w_{ji}.
\]

Because Model 8 horizontal paths do not depend on sampled height,
`w_ji` is the conditional expectation of the original 0/1 transit indicator.
The law of total expectation therefore gives `E[E_RT_CE] = E[E_RT]`, while
conditioning removes the height-draw component of variance. No smoothing,
new FHD, avoidance term, or collision probability is introduced. See
`PRECISION_IMPROVEMENT.md` for the derivation and validation.

## Model 8 preservation

`model8_meso_response.py` is the frozen Model 8 engine with one non-behavioural
addition: an optional passive segment-observer argument. With its default
`None`, Model 8 behaves exactly as before. Model 9 uses the hook only to inspect
each already-decided movement segment. It cannot alter response decisions,
headings, positions, encounters, or recovery.

The tests compare a direct Model 8 run with a Model 9 run and require identical
encounter matrices, response matrices, event counts, route lengths, completion,
and failure states.

## Files

- `config.py` — shared 242.24 m turbine, 6 km × 6 km site, inherited bird
  scenario, programmatic 12 × 7 wind resource, and full-run defaults.
- `floris_aep.py` — FLORIS v4.6.5 AEP wrapper with turbine/reference geometry
  checks and complete wind-condition provenance.
- `model8_meso_response.py` — inherited Model 8 trajectory engine plus the
  passive observation hook.
- `model8_batching.py` — inherited Model 8 Pareto/batching helpers.
- `model9_rotor_transit.py` — FHD loading/sampling, Model 9 populations,
  circular-disc intersection, Model 9 simulator, and result accounting.
- `model9_batching.py` — count-first aggregation of `E_enc`, sampled metrics,
  conditional expected metrics, per-turbine values, and batch variation.
- `run_model9_validation.py` — aligned, staggered, and wider-gap comparison
  before optimisation.
- `model9_aep_problem.py` — pymoo problem for maximising FLORIS AEP and
  minimising `E_RT_CE` by default, with sampled `E_RT` retained for audit.
- `model9_optimisation_utils.py` — seed-population, layout, spacing, JSON, and
  anchor helpers.
- `run_model9_optimisation.py` — integrated smoke, pilot, and full NSGA-II
  runner with checkpoints and independent fresh validation.
- `model9_hub_height_sensitivity.py` — fixed-height validation, paired
  comparison, and decision helpers.
- `run_model9_hub_height_sensitivity.py` — frozen four-layout, 150 m
  hub-height sensitivity runner with checkpoint/resume support.
- `README_HUB_HEIGHT_SENSITIVITY.md` — exact Anaconda Prompt commands,
  scientific boundary, and result interpretation.
- `README_OPTIMISATION.md` — exact Anaconda Prompt commands and output guide.
- `MODEL8_INTEGRATION_AUDIT.md` — supplied-input audit, hashes, and the
  240-to-242.24 m correction rationale.
- `PRECISION_IMPROVEMENT.md` — estimator proof, interpretation, tests, and
  fixed-layout precision evidence.
- `CHANGELOG_MODEL9_V3.md` — concise list of v3 additions and preserved model
  contracts.
- `CHANGELOG_MODEL9_V4.md` — Model 8/FLORIS/NSGA-II integration record.
- `tests/` — frozen Model 8 regression tests plus Model 9 geometry,
  reproducibility, parity, and accounting tests.
- `data/flight_height_distribution_TEMPLATE.csv` — required input schema.
- `data/northern_gannet_Johnston2014_FHD.csv` — exact user-supplied northern
  gannet table used by the validation, retained byte-for-byte.
- `data/northern_gannet_Johnston2014_FHD_metadata.json` — checksum,
  distribution summary, interpretation, and provenance record.
- `data/SYNTHETIC_TEST_ONLY_fhd.csv` — software smoke-test data only.
- `data/energy_endpoint_layout.csv` — supplied Model 8 energy baseline and
  exact initial-population anchor.
- `data/model8_initial_population.csv` — validated 80-layout Model 8 seed set.
- `data/model8_reference_run_settings.json` — supplied full-run scales and
  provenance used for consistency checks.
- `floris_config/gch_iea15mw.yaml` — supplied FLORIS GCH/Gauss farm config.
- `floris_config/iea_15MW_REFERENCE.yaml` — frozen supplied turbine reference.
- `results/pilot_6000_per_layout/` — the initial sparse-event pilot.
- `results/model9_validation_northern_gannet/` — the higher-evidence
  sampled-only 50,000-trajectory-per-layout validation retained from v2.
- `results/model9_precision_validation_northern_gannet/` — the matching v3
  validation with sampled and conditional estimators.

## Flight-height input used

The loader accepts the canonical schema:

```text
height_m,probability_mass
```

It also accepts the native stochLAB/Johnston schema used by the supplied file:

```text
height,prop
```

Rows must contain non-negative heights and probability masses. Heights must be
unique; the loader sorts them and normalises the masses to one. The original
file is not rewritten. Model 9 treats each integer height label as a discrete
support point and draws one height per trajectory by inverse-CDF sampling.

The selected table has 300 one-metre height bins from 0 to 299 m. Its raw mass
sums to 1.0000000558430553, a rounding difference that is normalised by the
loader. The mean discrete height is 9.1497 m; the median is 6 m; and 4.9246% of
the mass lies between the IEA 15 MW lower and upper rotor tips (28.88–271.12 m).

The input checksum is:

```text
SHA-256 adc9ea903db9c8e34bfe5b88c382381b737c2e676d79af437f66d198208a0f2c
```

The distribution is documented as the northern gannet maximum-likelihood
estimate (`Gannet.est`) from stochLAB's `Johnston_Flight_heights_SOSS` object,
based on Johnston et al. (2014), *Journal of Applied Ecology* 51, 31–41,
<https://doi.org/10.1111/1365-2664.12191>. No curve was refitted or digitised.
The upstream package release or repository commit was not embedded in the
uploaded CSV, so that one metadata item should be added if the original
download record is available. The exact bytes used here remain reproducible
through the checksum.

The Model 8 meso parameters remain hypothetical scenario values and do not
become gannet-calibrated merely because the FHD is gannet-specific.

## Run the deterministic test suite

From the `model9_code` directory:

```bash
python -m unittest discover -s tests -v
```

The required geometry checks are included: below rotor, above rotor, through
rotor centre, and correct height with a lateral miss. The suite also checks the
circular conditional-height window, discrete-CDF boundaries, Monte Carlo
equivalence, optimiser objective selection, circular boundary, coarse
numerical steps, multiple transits, Model 8 parity, common random numbers, and
count-first batch aggregation. Version 5 contains 59 tests, including seven
Model 8-to-Model 9 FLORIS and seed-population coupling checks and nine
pre-specified endpoint-confirmation checks.

## Run the integrated optimisation

Start with the small end-to-end profile:

```bash
python run_model9_optimisation.py \
  --profile smoke \
  --output results/model9_nsga2_smoke
```

This uses the frozen northern-gannet FHD automatically. The runner also
creates `results/model9_nsga2_smoke.zip`; upload that archive for review before
starting the pilot. See `README_OPTIMISATION.md` for exact Windows/Anaconda
steps, all three profiles, safeguards, runtimes, and output definitions.

## Run a software-only smoke validation

This confirms that the runner and outputs work. It is not biological evidence:

```bash
python run_model9_validation.py \
  --flight_height_csv data/SYNTHETIC_TEST_ONLY_fhd.csv \
  --fhd_name synthetic-software-test \
  --fhd_source "synthetic values; never use for thesis results" \
  --trajectories_per_batch 250 \
  --batch_count 2 \
  --output results/model9_smoke
```

## Reproduce the real pre-optimisation validation

```bash
python run_model9_validation.py \
  --flight_height_csv data/northern_gannet_Johnston2014_FHD.csv \
  --fhd_name northern-gannet-johnston-2014-mle \
  --fhd_source "Johnston et al. (2014), DOI 10.1111/1365-2664.12191; stochLAB Johnston_Flight_heights_SOSS Gannet.est export supplied by user" \
  --trajectories_per_batch 10000 \
  --batch_count 5 \
  --trajectory_seed 42 \
  --flight_height_seed 10042 \
  --output results/model9_precision_validation_northern_gannet
```

Default physical and inherited scenario values are:

| Input | Default |
|---|---:|
| Turbines / site | 12 / 6 km x 6 km |
| Bird heading | 225 degrees toward south-west |
| Rotor diameter | 242.24 m |
| Hub height | 150 m |
| Rotor height range | 28.88–271.12 m |
| Minimum spacing | 4D = 968.96 m |
| Meso-response probability | 0.6 |
| Response / encounter radius | 2D / 1D |
| Deflection / return | 45 degrees / 8D |
| Numerical step | 0.20D |

The FLORIS-coupled runner verifies diameter and hub height against the
installed official turbine and the bundled Model 8 reference. It rejects the
obsolete 240 m prototype value; every active calculation uses 242.24 m.

## Validation outputs

- `layout_encounter_transit_comparison.csv` — combined layout values, ratios,
  and ranks for `E_enc`, sampled `E_RT`, and conditional `E_RT_CE`.
- `independent_batch_results.csv` — each trajectory/height seed pair.
- `per_turbine_encounters_and_transits.csv` — `N_enc_i`, sampled `N_RT_i`,
  and conditional expected `N_RT_CE_i`.
- `paired_layout_differences.csv` — common-random-number paired differences,
  standard errors, and 95% Student-t intervals across independent batches.
- `estimator_precision_comparison.csv` — sampled-versus-conditional agreement
  intervals and empirical variance-reduction factors.
- `validation_diagnostic.json` — whether ranking changed and the range of
  transit-to-encounter ratios, plus the uncertainty qualification.
- `run_settings.json` — FHD source, seeds, geometry, behaviour, and scope.
- `encounter_vs_rotor_transit.png` — the one primary diagnostic plot.

Do not force a rank change. In the included 50,000-trajectory run, `E_RT_CE`
was 0.003764 (aligned), 0.002792 (staggered), and 0.002907 (wider-gap).
Conditional batch variance was 62–305 times lower than sampled variance, and
all three sampled-minus-conditional 95% intervals included zero. Paired
`E_RT_CE` intervals separated aligned from both alternatives, while staggered
minus wider-gap still included zero. The defensible result is therefore that
the precision improvement is real, but one tested layout contrast remains
unresolved.

## Current analysis status

The smoke run, stability pilot, full 80 × 100 NSGA-II run, independent fresh
validation, 15 × 100,000 three-layout confirmation, and matching `pareto_038`
extension have all been completed. The apparent gannet-layout differences did
not reproduce as statistically separable differences in the high-sample
confirmation. The current defensible species-specific conclusion is therefore
that no rotor-transit trade-off was confirmed among the tested candidates
under the supplied northern-gannet FHD and single 225-degree approach heading.

Version 7 does not reopen or tune that analysis. It adds one explicitly
non-species fixed-hub-height diagnostic to determine whether horizontal layout
geometry can matter when vertical exposure is guaranteed. See
`README_HUB_HEIGHT_SENSITIVITY.md`. Review that result before considering any
new hypothetical optimisation; another gannet optimisation is not currently
justified by the confirmation evidence.
