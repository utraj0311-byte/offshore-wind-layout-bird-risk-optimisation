METRIC 4b — INDEPENDENT EVALUATION ADD-ON
========================================

IMPORTANT: This ZIP contains two files at its top level, not a nested project
folder. Copy metric4b_confirm72.py into the EXISTING folder:
  C:\Users\Deepak\Desktop\bird_layout_opt\metric4b_preflight

Its original sibling folder model9_code/ must remain unchanged, and the two
results folders must already exist in bird_layout_opt/:
  metric_4b_72sectors/
  metric_4b_72sectors_postrun/

1. Open Anaconda Prompt:
   conda activate floris-env
   cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4b_preflight
   python metric4b_confirm72.py verify

   This is read-only and starts NO simulation. Send its output for review.

2. Only after successful verify, optional 100-bird functional smoke test:
   python metric4b_confirm72.py smoke
   No production output is written.

3. Full 15 x 100,000 paired independent evaluation, AFTER USER APPROVAL:
   python metric4b_confirm72.py run

   This can require multiple hours. It creates a new, separate folder:
   C:\Users\Deepak\Desktop\bird_layout_opt\metric_4b_72sectors_confirmation
   It writes a complete checkpoint AFTER EACH 100,000-bird batch.

4. If a full run stops unexpectedly, restart from saved completed batches:
   python metric4b_confirm72.py resume
   Never run 'run' again if production output exists.

What the experiment does
------------------------
- Highest-AEP and lowest-risk ONLY: freezes existing 72-sector coordinates.
- Original independent trajectory seeds 30142 ... 30156.
- Original independent flight-height seeds 40142 ... 40156.
- Original northern-gannet flight-height probability masses, 300 bins.
- Shared original meso-response trajectory code and conditional rotor-transit
  estimator, Q_4b, throughout (each bird-turbine pair at most once).
- 15 batches of 100,000 birds each, using the same entry positions and
  response random numbers for both layouts and all controls.
- Baseline p=0.6; no-response p=0; fixed z=150 m at both p settings.
- Common widened corridor W=8,577.0795956 m from saved geometry report,
  preserving the original 6,000 m x 6,000 m site.
- Saves 10,000-bird subgroup estimates for empirical sampling SD.
- Paired 95% Student-t intervals across 15 independent 100,000-bird batches.
- Reports Q_4b, Q_4b x W, and baseline Q_4b width-adjusted to the original
  W=8,485.2813742 m. These are different normalisations of the same widened
  corridor simulation, not separate original-corridor evaluations.

The original NSGA-II optimisation, bird equations, FHD, original result
archives, and FLORIS configuration are never modified. No optimisation is
performed by this add-on.

Outputs
-------
  evaluation_settings.json
  independent_evaluation_batches.csv
  independent_evaluation_10000_subsamples.csv
  independent_evaluation_endpoint_summary.csv
  independent_evaluation_paired_differences.csv

Provisional per-batch checkpoints:
  independent_batches_checkpoint.csv
  sample_10000_checkpoint.csv

WARNING: Original 12-sector study used code v4, while this calculation uses
numerically audited v7; historical byte-for-byte equivalence is not available.

Do not conclude that ALL front solutions are independently better/worse from
these two endpoints alone. These scores are conditional rotor transits, not
actual blade-strike probabilities or mortality estimates.
