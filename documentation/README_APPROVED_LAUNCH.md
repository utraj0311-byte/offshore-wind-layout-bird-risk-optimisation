# APPROVED: Jensen 1-degree (360-sector) NSGA-II reruns 1a–1d

Revision approved 16 October 2026. Supersedes the previous 2-degree package. **Keep your completed 2-degree 1a result as a diagnostic only.** Never overwrite or promote it to a final result. The 2-degree 1a run is not an initial seed for any new run.

## Installation and command sequence

Extract `jensen_1deg_approved.zip` within `C:\Users\Deepak\Desktop\bird_layout_opt`, keeping the top folder. Do NOT replace or edit `jensen_2deg_approved` or its results. Activate `floris-env`, then:

```bat
cd /d "C:\Users\Deepak\Desktop\bird_layout_opt\jensen_1deg_approved"
python run_jensen_1deg.py --metric 1a
```

After completion, inspect `results\metric_1a_360sectors_seed42_pop80_gen100\endpoint_0p5deg_convergence.csv` and the printed PASS. **Only if both rows pass**, launch separately:

```bat
python run_jensen_1deg.py --metric 1b
python run_jensen_1deg.py --metric 1c
python run_jensen_1deg.py --metric 1d
```

**STOP immediately if any metric fails at 0.5 degree**. The script reports FAIL, marks the run in `run_settings.json`, and exits with status 2. Do not start the next metric until the failure is reviewed. All saved data remain in that new metric's folder.

## Locked scientific settings

- Wind directions: **360 sectors**, **1 degree** each (FLORIS stays at 72 sectors).
- Jensen rotor diameter **240 m**, minimum 4D spacing **960 m**, 12 turbines in 6x6 km.
- Jensen kw=0.04, same top-hat hub-centre wake overlap and RSS combined wake deficit.
- Ct=0.8 at 3<=U<10.6 m/s, 0.25 at 10.6<=U<=25, otherwise zero, evaluated from *undisturbed* speed.
- Seven synthetic speed bins 4,6,8,10,12,14,16 m/s, Weibull k=2, A=10 m/s, FROM directional weighting centred at 225 degrees.
- Historical scaling AEP_ref **754.4007727706155 GWh/year**; f1=2-2*AEP/AEP_ref; bird f2=Q/12000.
- Same original grid, random and risk-aware initialization probabilities, seed **42**, NSGA-II with unchanged default operators from **pymoo 0.6.1.5**, duplicate elimination. Population/generations 80x100 for 1a; 120x150 for 1b-1d. Historical pymoo version was not archived.
- Only scientific input change from original runs: 12 sectors -> 360 sectors. Post-processing and archiving are expanded.

## Files saved per metric

Each unique directory under `jensen_1deg_approved\results` contains:

- `run_settings.json` (created **before** optimisation and updated after; complete settings, source hashes, version, operator snapshots, final convergence status).
- `pareto_full.npz`: ALL non-dominated coordinate vectors `X` and objectives `F`.
- `pareto_layouts_long.csv`: turbine coordinates for ALL final nondominated layouts.
- `pareto_points.csv`: original 1-degree optimisation AEP, Q, geometry diagnostics, normalized f1/f2.
- `pareto_aep_reported_1deg.csv`: **fresh, independent 1-degree AEP for every final layout**, original 1-degree AEP and difference.
- `endpoint_0p5deg_convergence.csv`: both newly optimized endpoints, AEP at 1 and 0.5 degree, absolute and relative difference; pass iff difference is <0.2% of AEP at 0.5 degree.
- `metric_1{a,b,c,d}_pareto.png` and `.pdf`, including the single point of 1a; endpoint maps over field in PNG/PDF; `anchor_summary.csv`, layout CSVs, `hypervolume.csv`.

## Plotting script location

The already-included script `plot_saved_pareto.py` lives in the SAME directory as `run_jensen_1deg.py`:

`C:\Users\Deepak\Desktop\bird_layout_opt\jensen_1deg_approved\plot_saved_pareto.py`

It is called automatically after each run. To rerender later from Anaconda Prompt in that directory:

```bat
python plot_saved_pareto.py "results\metric_1a_360sectors_seed42_pop80_gen100" --overwrite
```

This affects plots only; it never reruns NSGA-II. Figures use AEP from 1-degree optimisation (and the independently re-evaluated report file records agreement).

## FLORIS reporting (separate workflow)

FLORIS uses the official **242.24 m rotor** and **968.96 m 4D spacing**. Do not modify its 5-degree optimization output. For those runs, independently recalculate the final layout AEP at **1 degree**, report it as the final AEP, and include the difference from the originally optimized 5-degree AEP. FLORIS's prior 1-degree-vs-5-degree convergence check is separate from the Jensen 1-degree-vs-0.5-degree test.

No optimisation is performed by packaging or plotting. User must run commands in their local Windows environment.
