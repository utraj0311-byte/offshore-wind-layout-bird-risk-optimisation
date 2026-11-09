# Metric 4b: scientific implementation and archived result set

**Thesis:** *Integrating Bird Collision Risk into Multi-Objective Offshore Wind Farm Layout Optimisation*  
**Metric:** 4b — flight-height-integrated distinct rotor-disc transits per entering bird; historical name **Model 9**.  
**Authoritative data:** completed **72 wind sectors (5°)**, **not** the historical 12-sector front.  

## Scientific purpose and interpretation

Metric 4b extends Metric 4a's horizontal meso-response trajectories with a fixed circular rotor disc and a northern-gannet flight-height distribution. It estimates **potential transits through the rotor-swept disc**, not blades strikes, collision probability, avoidance/mortality, nor the later Band-weighted literature-informed metric.

For bird j and turbine i, the archived geometry checks a segment intersection with the fixed rotor plane perpendicular to the heading (225° **TOWARD**), then a circular cross-section of radius R. A transit can occur only for lateral offset |l_ji| ≤ R and a height Z inside:

    [H - sqrt(R² - l_ji²), H + sqrt(R² - l_ji²)].

The sampled indicator Y_ji records whether that *distinct bird–turbine pair* transits; repeated segment intersections with the same rotor cannot multiply its count. A bird may transit several different turbines. The sampled E_RT = sum_j sum_i Y_ji / N_in is retained as an **audit metric**.

The **optimised and thesis-reported** quantity is the conditional-height estimator, renamed `Q_4b`:

    Q_4b = E_RT_CE = (1 / N_in) sum_j sum_i P_F(Z in I_ji | realised horizontal path_j).

The archived implementation integrates **all entering birds**, including birds with zero rotor transits, rather than dividing only by birds entering a rotor region. The distribution is a fixed empirical input; uncertainty in the FHD itself is not modelled. For repeat crossings of the same rotor, the implementation retains the maximum attainable height-window probability, rather than adding per-segment probabilities. See `model9_rotor_transit.py` and `PRECISION_IMPROVEMENT.md`.

## Authoritative model parameters (archived run_settings.json)

| Item | Original recorded value |
|---|---|
| Number / site | 12 turbines, 6,000 × 6,000 m |
| Turbine | FLORIS packaged IEA 15 MW, D = 242.24 m, H = 150 m |
| Minimum centre spacing | 4D = 968.96 m |
| Rotor radius; sweep | R = 121.12 m; z = 28.88–271.12 m |
| Bird heading | 225° **toward** southwest (fixed) |
| Meso response | p = 0.6, initial heading / response 2D (484.48 m), encounter 1D (242.24 m) |
| Deflection / recovery | 45° deflection, 8D (1,937.92 m) return distance |
| Step size | 0.2D = 48.448 m; maximum step safety factor = 4 |
| Flight height | northern-gannet Johnston (2014), 300 original rows, 0–299 m at 1 m resolution |
| FHD weight treatment | original mass sum 1.0000000558430553, internally normalised to 1 |
| Flight-height state | one constant sampled height per bird trajectory; conditional estimator integrates same FHD |
| Rotor plane | fixed plane normal to reference bird heading; no wind-yaw-driven rotation |
| Objective simulation | 5 × 2,000 = **10,000 entering birds per evaluated layout** |
| Optimisation bird trajectory seeds | 142, 143, 144, 145, 146 |
| Height sampling audit seeds | 10142–10146; conditional Q does not depend on sampled heights |
| NSGA-II | population 80, 100 generations, seed 142, 8,000 objective evaluations |
| NSGA-II operators | `NSGA2(pop_size=80, sampling=selected_initial_population, eliminate_duplicates=True)`; **pymoo defaults** for crossover, mutation and repair (no custom repair passed) |
| FLORIS wind rose | 72 × 7 = 504 weighted conditions; direction steps 5°, speeds 4, 6, 8, 10, 12, 14, 16 m/s |
| FLORIS model / versions | Gauss deficit and deflection, Crespo–Hernández TI, SOSFS combination; FLORIS 4.6.5, pymoo 0.6.1.5 |
| Full recorded platform | Python 3.11.15, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, matplotlib 3.11.0, PyYAML 6.0.3 |

Other trajectory geometry and repeated-response stopping rules are **implemented in the original** `model8_meso_response.py`. They were inherited by the archived Metric 4b code, not reimplemented in this package. Entry points are stratified across the diagonal site's 8,485.281374 m original corridor, with one random point in each equal-width bin per batch; start and stop are 10D beyond the projected upstream/downstream site corners. See the actual source for the maximum response-event and step-stop implementation, which must not be simplified for reproduction.

### Objective normalisation supplied to NSGA-II

    f0 = (854.4956585617186 - AEP_GWh_per_year) / 137.96914764866568
    f1 = Q_4b / 0.0008704996418223796

These are **positive scaling transformations**. AEP physically has units GWh/year; the raw bird objective is expected rotor-disc transits per entering bird (dimensionless mean count). `Q_4b` itself is not additionally divided by a corridor width in the optimisation. The source files call it `E_RT_CE`; the Results chapter calls it `Q_4b`.

## Archived results: final 5° search

- Completed 72-sector run: **36 non-dominated layouts**, 8,000 evaluated candidates.
- Highest AEP: **828.668218644 GWh/year**; optimisation-sample Q_4b **0.001923926383**.
- Lowest bird metric: **813.839651506 GWh/year**; optimisation-sample Q_4b **0.000736569508**.
- Saved decision vectors, normalised objectives, Pareto CSV, initial population, final population, candidate audit and hypervolume are in `results/metric_4b/final_72sectors/metric_4b_72sectors/`.

The 12-sector run in `historical_12_sector/` is a **different actual optimisation**, not the same layouts re-evaluated with a 5° rose. Its 51-candidate training front and subsequent fresh-validated anchors cannot be mixed with the 36 solutions of the 72-sector run. A separate **1° endpoint AEP-only post-run check** was designed; its original saved report is not present in the supplied archives (see missing files). The 72-sector final confirmation itself did not re-optimise layouts.

## Independent evaluation, not the optimisation sample

The final frozen highest-AEP and lowest-risk layouts were compared with **15 paired independent batches × 100,000 birds/layout = 1.5 million entering birds per layout**, with trajectory seeds **30142–30156** and flight-height seeds **40142–40156**. The same fresh bird population was reused across both layouts and across the baseline p=0.6, no-response p=0, fixed z=150 m at p=0.6 and fixed z=150 m at p=0 conditions. The shared evaluation corridor was widened from **8,485.281374 m to 8,577.079596 m** to cover the protruding rotor, with reported original-width-basis Q values obtained by multiplying the widened-corridor results by the ratio of corridor widths. That width adjustment was *not another bird simulation*. The frozen primary values were **0.0019898263** (highest AEP) and **0.0008583859** (lowest risk) on the original-width basis, representing **56.86%** fewer transits for **1.79%** less AEP. Student-t intervals are from 15 paired batch differences; the original tables and per-batch records are shipped under `independent_evaluation_72sectors`.

**Other historical validation** used a smaller five-batch fresh 10,000-bird profile and historical seed pairs; it must be documented as distinct from the final 15 × 100,000 endpoint evaluation. The separate 150 m hub-height historical sensitivity code and predeclared design are preserved, but its completed original result archive is **not** available and must not be inferred from final 72-sector fixed-height controls.

## Original code layout and reproduction

The `metric4b_preflight/` folder is an **unaltered original working tree**, with the archived `model9_code/` modules, `original_results/` preflight references and named runners. Imports are preserved as written (`from model9_rotor_transit import ...`, `from config import ...`). It is deliberately **not** a newly rewritten pip-installable package. Nothing is replaced in the repository `shared/` directory.

Original commands, **for reference only** (NOT executed during this audit):

```bat
conda activate floris-env
cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4b_preflight
python metric4b_run72.py verify
python metric4b_preflight.py audit
python -m unittest discover -s model9_code/tests -v
```

Do not run `metric4b_run72.py run`, `resume`, or the historical `run_model9_optimisation.py` without intentionally starting an optimisation. The original project-specific relative layout and external 72-sector run directory are documented in `documentation/metric_4b/INTEGRATION_NOTES.md`. The independent evaluator's `verify` additionally requires missing original `metric_4b_72sectors_postrun/geometry_report.json` and `endpoint_1degree_aep.json`.

## Verification scope / limitations

- All **27/27 tracked source and input hashes** recorded by the original 72-sector run match this package.
- All included `.py` files parsed successfully without execution.
- Cached Pareto arrays and normalised scientific objective values agree with the archived CSV and scaling constants. 80 initial layouts, 72 wind sectors, 504 weights and 15 completed independent evaluation batches are recorded.
- **No** FLORIS model, bird trajectory, unit test or optimiser was executed to create this package. Exact runtime/re-simulation parity is **not** newly established here.
- The old historical Model 9 archive identifies **v4 code**, while available source is **v7**. The 72-sector archive hashes confirm the available v7 code was the code used for the 72-sector re-optimisation, but cannot restore absent v4 bytes for the original 12-sector run.
- The bundled historic hub-height sensitivity is **not** the literature-informed Model 10 fixed-height experiment.

See `FILE_INVENTORY.md`, `EXPERIMENT_PROVENANCE.md`, `INTEGRATION_NOTES.md`, `MISSING_FILES.md` and `STATIC_AUDIT.json` for precise provenance and limitations.
