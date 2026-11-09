Metric 4a strict reproduction checks — LOCAL LAPTOP ONLY

This bundle is deliberately separate from your existing project code. It does
NOT run an optimiser. Inputs are copies of the files supplied to this chat.
Results only go into metric4a_preflight/preflight_reports/.

1. Put metric4a_preflight folder INSIDE your bird_layout_opt folder:
      C:\Users\Deepak\Desktop\bird_layout_opt\metric4a_preflight\
2. Open Anaconda Prompt:
      conda activate floris-env
      cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4a_preflight
3. One command at a time, STOP and share output before next:
      python metric4a_preflight.py audit
      python metric4a_preflight.py bird
      python metric4a_preflight.py aep
      python metric4a_preflight.py benchmark

The audit checks:
- Historical settings, seeds, and sample sizes
- Recreates and compares every original initial population coordinate
- Exact original objective scaling and bird model parameters
- Corridor logic and known entry-range limitation
- Local pymoo/FLORIS versions and original project FLORIS YAML hash
- Hashes of shipped source files and historical ZIP results

The bird check uses original seeds and layouts and requires exact training
encounter counts, Q_4a and P(>=1), plus 0.6854 no-response scaling.

The AEP check uses FLORIS in your laptop's floris-env. It requires BOTH original
endpoint AEPs to reproduce within 0.001 GWh/year. It then checks 72-sector
wake-free AEP, new AEPs, and unchanged Q_4a. AEP results must be determined
locally: this chat cannot execute your Anaconda environment.

The benchmark times a real 72-sector evaluation including 10,000 birds, and
measures a single 5000-bird confirmation batch for each endpoint/p control.

Original run pymoo version and YAML hash were NOT captured in the original
saved run_settings.json: these cannot be independently proven by the archive.
If the original on-disk project YAML exists, the audit compares hashes.

Do NOT launch run_model8_multiobjective.py: it still defaults to 12 sectors,
recomputes the objective scaling, and has no restart-safe checkpoints.
The 72-sector RUNNER is a separate future step, only after all checks pass
and you explicitly approve.

Optional if you still have a separate historical configuration:
    python metric4a_preflight.py audit --reference-yaml "C:\full\path\gch_iea15mw(2).yaml"
The configured FLORIS model in THIS BUNDLE is an unchanged copy of the uploaded
'gch_iea15mw(6).yaml'; do not relabel it as historically hash-proven.
