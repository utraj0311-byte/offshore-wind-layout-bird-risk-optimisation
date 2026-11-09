METRIC 4b POST-OPTIMISATION VERIFICATION — ANACONDA ON WINDOWS

READ ONLY: No optimisation is launched, no bird simulation is run and no original code/results are modified.

INSTALL

The ZIP contains the script and README directly (no nested folder).

1. Open the ZIP file.
2. Copy metric4b_postrun_checks.py into your existing:
   C:\Users\Deepak\Desktop\bird_layout_opt\metric4b_preflight
   It must sit alongside metric4b_preflight.py and metric4b_run72.py.
3. Leave metric_4b_72sectors, model9_code, and original_results unchanged.

COMMANDS (run one at a time in Anaconda Prompt):

conda activate floris-env
cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4b_preflight
python metric4b_postrun_checks.py audit
python metric4b_postrun_checks.py endpoints

Only if `endpoints` reports either relative AEP difference >= 0.2%:
python metric4b_postrun_checks.py fullfront1deg

RECORDS

All reports are written to a separate output folder:
C:\Users\Deepak\Desktop\bird_layout_opt\metric_4b_72sectors_postrun

The files include:
- archive_audit.json
- geometry_report.json
- corridor_rotor_margins.csv
- aligned_pairs.csv
- endpoint_1degree_aep.json
- all_36_pareto_1deg_aep.csv (ONLY if the endpoint tolerance fails)

GEOMETRY NOTE

The original cross-flight corridor is the projection of the 6 x 6 km site along the
axis perpendicular to the prescribed 225-degree TOWARD bird heading. Rotor margins are
measured using radius D/2 = 121.12 m, not the larger behavioural encounter radius D.
If either endpoint rotor leaves the original corridor, the script computes one common
expanded corridor, with 1 metre safety added on each outer edge. Independent evaluation
must use both endpoints with identical sample corridor bounds and paired seeds. The
common-width result is not substituted into the original optimisation sample.

INDEPENDENT EVALUATION (NOT PERFORMED BY THIS SCRIPT)

Same original seeds and batch schedule: 15 x 100,000 birds per endpoint.
Use pairs across layouts, and baseline p=0.6, no-response p=0 and z=150 m fixed height.
Log 95% paired intervals, Q_4b times W, and the SD for an independent 10,000-bird sample.
Those simulations will be prepared separately after both 1-degree endpoint checks.
