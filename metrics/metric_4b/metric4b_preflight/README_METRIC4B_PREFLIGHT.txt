METRIC 4b — ANACONDA PREFLIGHT BUNDLE (read-only)
=================================================

This ZIP contains a NEW folder called metric4b_preflight. Extract the ZIP
straight into C:\Users\Deepak\Desktop\bird_layout_opt (not into a second folder).
After extracting, these files should exist:

bird_layout_opt\metric4b_preflight\metric4b_preflight.py
bird_layout_opt\metric4b_preflight\model9_code\run_model9_optimisation.py
bird_layout_opt\metric4b_preflight\original_results\run_settings.json

This is a standalone snapshot of the TWO uploaded Model 9 archives.
It will not replace or edit Metric 4a or your other Model 9 folders.

Anaconda Prompt:
   conda activate floris-env
   cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4b_preflight

Commands, ONE AT A TIME, with results reviewed between commands:
   python metric4b_preflight.py audit
   python metric4b_preflight.py bird
   python metric4b_preflight.py aep
   python metric4b_preflight.py benchmark

STOP if any check says FAIL or prints a traceback. Send the full result.
Reports are saved in metric4b_preflight\preflight_reports\.

DO NOT RUN run_model9_optimisation.py: it has an old 84-condition guard,
recalculates the original reference, and only saves a non-resumable CSV
checkpoint. A separate 72-sector resumable runner is required after all
pre-flight checks pass and the user approves it.

Scientific settings unchanged in this pre-flight bundle. The new runtime
must use a fixed value Q_ref=0.0008704996418223797 as recorded in the
archived original, and use 72 wind sectors. The preflight tests create
72-sector FLORIS objects but do not run NSGA-II.

Nuance: the code ZIP is v7 and the original NSGA-II archive declares v4.
Therefore the preflight explicitly re-evaluates the saved optimisation-
sample Q_4b values before treating the later source as reproducible.

Original independent evaluation: 15 x 100000 = 1,500,000 birds per layout;
trajectory seeds 30142..30156 and flight-height seeds 40142..40156.
This is DIFFERENT from the original automated front-validation sample
(5 x 10000); it must not be substituted during final confirmation.
