METRIC 4b — 72-SECTOR RESUMABLE NSGA-II RUNNER
===============================================

PURPOSE
-------
The current production driver reads the original Metric 4b v7 scientific
modules and the archived original settings. It changes ONLY the independent
scientific input n_sectors from 12 to 72 (84 -> 504 derived wind conditions).
Everything else in the optimisation settings is guarded against the source.

IMPORTANT LIMITATION
--------------------
The completed 12-sector run reported original code v4. The uploaded source is
v7. The two original 12-sector endpoint Q_4b values and AEP values have
reproduced on the laptop, but v4/v7 byte identity cannot be proven.

This driver is NEW operational code for checkpoint/resume and archiving;
therefore check the local VERIFY and SMOKE outputs before using RUN.

INSTALLATION (DO NOT REPLACE EXISTING MODEL FILES)
---------------------------------------------------
Extract the ZIP directly under:
C:\Users\Deepak\Desktop\bird_layout_opt

The resulting script must be here:
C:\Users\Deepak\Desktop\bird_layout_opt\metric4b_preflight\metric4b_run72.py

This folder already contains `model9_code`, `original_results`, and
`preflight_reports` from the successful earlier Metric 4b preflight.

ANACONDA COMMANDS — ONE STEP AT A TIME
--------------------------------------
conda activate floris-env
cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4b_preflight
python metric4b_run72.py verify

Send the complete verification output for review. Do not run smoke/run yet.
Once VERIFY passes, next is:
python metric4b_run72.py smoke

SMOKE uses 8 individuals, 3 generations, 1 x 120 birds/evaluation to compare
uninterrupted NSGA-II with checkpoint/resume for exact population, objectives,
and diagnostic history. It does NOT write the production folder.

Only after explicit approval, production starts with:
python metric4b_run72.py run

If interrupted (after a checkpoint) use:
python metric4b_run72.py resume

PRODUCTION RUN
--------------
New path, never overwritten:
C:\Users\Deepak\Desktop\bird_layout_opt\metric_4b_72sectors

80 individuals x 100 generations; seed 142; 5 x 2000 optimisation-sample
birds/layout, trajectory seeds 142..146, height seeds 10142..10146.
Rotor-transit reference exactly 0.0008704996418223796, unchanged.
AEP loss reference exactly 137.96914764866568, unchanged.

Full live checkpoints are saved at generation 0, then every five generations
and at completion. Back-up checkpoint available in case latest file is damaged.
Source/checkpoint hashes are checked when resuming. Completion saves:
- pareto_decision_vectors.npy, pareto_normalised_objectives.npy
- pareto_points.csv with ALL 24 turbine decision variables for each layout
- highest-AEP_layout.csv and lowest-risk_layout.csv
- final_population.npz (entire 80-member population, including dominated)
- all_evaluated_candidates.csv
- hypervolume.csv (descriptive; reference [2,2])
- settings_diff.json, file_hashes.json, run_settings.json,
  completion_status.json, training_summary.json

The 5 x 10000 original post-optimisation exploratory validation is recorded
in settings but intentionally deferred. The proper selected-endpoint independent
evaluation is a separate script (15 x 100000 birds with original seeds, p=0.6,
p=0 and the z=150 m control), after the 1-degree and corridor checks. No
new optimiser is run during those checks.

LAPTOP
------
Plug in the charger. Disable Windows plugged-in sleep and hibernation while
running. Leave the lid open, keep the Anaconda Prompt alive, and avoid another
Metric 4a/4b job competing for CPU/RAM. Expected optimisation wall time is
~13 hours naive, allow 14–17+ hours. Run no other heavy computations.
