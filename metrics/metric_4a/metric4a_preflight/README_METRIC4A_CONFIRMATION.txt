Metric 4a — independent 72-sector run confirmation

Unzip this archive directly into:
C:\Users\Deepak\Desktop\bird_layout_opt\

It contains the single new script:
metric4a_preflight\metric4a_confirm72.py

Do not create an extra wrapping directory with the ZIP title.
Do not overwrite metric_4a_72sectors or metric_4a_72sectors_postrun.

Step A. Verify source files and confirmation settings (NO simulations):
    conda activate floris-env
    cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4a_preflight
    python metric4a_confirm72.py verify

Step B. Only after reviewing the verification output, run the confirmation:
    python metric4a_confirm72.py run

This runs 20 original fresh seeds (20042..20061), each with 5000 birds.
It evaluates both saved endpoints at p=0.6 and p=0.0, pairing the exact
same random birds within each batch, across endpoints and response scenarios.
It keeps the old simulator, site, longitudinal domain and parameters intact;
only the confirmation entry corridor is enlarged to the audited common width.

A completely NEW results directory is created:
metric_4a_72sectors_confirmation\

Outputs:
- confirmation_settings.json: immutable source settings and study protocol
- confirmation_batches.csv: all 80 batch-level results
- confirmation_endpoint_summary.csv: endpoints, Q_4a, P(>=1), Q_4a*W,
  SD estimated from 10 independent groups of 10000 birds
- confirmation_paired_contrasts.csv: batch-paired mean differences + 95% t CIs
- confirmation_per_turbine.csv: encounter totals per turbine by batch
- confirmation_report.json: interpretation checks/overview
- confirmation_batches_partial.csv: live progress as batches finish

Strictly read-only for the completed run and post-run verification folders.
This is NOT an NSGA-II optimisation or 1-degree wind evaluation.
