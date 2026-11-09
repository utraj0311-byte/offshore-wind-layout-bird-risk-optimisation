# Model 9 endpoint confirmation

## Purpose

The full seed-142 optimisation completed successfully, but its 51 training
Pareto layouts compressed to one robust 95% endpoint after fresh validation.
The apparent high-energy trade-off between `pareto_037` and `pareto_032`
therefore requires a larger independent test before it can support a thesis
claim.

This runner is **not another optimisation**. The following layouts were frozen
before the new confirmation seeds are generated:

1. `original_energy_layout`: the pre-optimisation energy-layout benchmark;
2. `pareto_032`: the highest-AEP layout from the full run; and
3. `pareto_037`: the full-run candidate with an apparent 1.40% lower fresh
   `E_RT_CE` for only 0.019% lower AEP than `pareto_032`.

The layout coordinates, source-run values, file checksums, source archive
checksum, hypotheses, sample size, and decision rule are recorded in
`data/model9_confirmation_manifest.json`.

## Primary decision rule

For every batch, all three layouts receive exactly the same new simulated
birds. The primary difference is:

```text
E_RT_CE(pareto_037) - E_RT_CE(pareto_032)
```

The runner calculates a paired two-sided 95% Student-t interval across the 15
independent batches.

- Entire interval below zero: a repeatable AEP–rotor-transit trade-off is
  confirmed.
- Interval contains zero: the risk difference remains inconclusive.
- Entire interval above zero: `pareto_037` has higher rotor-transit exposure
  and the proposed trade-off fails.

A secondary comparison tests `pareto_032` against the original energy layout.
The primary trade-off gate is not changed after seeing the results.

## Run in Anaconda Prompt

Extract `Model_9_Code_v5.zip`, then run:

```bat
conda activate floris-env
cd /d "C:\Users\Deepak\Desktop\bird_layout_opt\Model_9_Code_v5\model9_code"
python -m pip install -r requirements_model9.txt
python -m unittest discover -s tests -v
python run_model9_endpoint_confirmation.py
```

The confirmation uses:

- 15 independent batches;
- 100,000 trajectories per batch;
- 1,500,000 trajectories per layout;
- 4,500,000 simulated layout-trajectories overall; and
- new trajectory seeds `30142`–`30156`, which do not overlap the optimisation,
  pilot, stability-pilot, or full-run validation seeds.

On the machine that completed the full run in 13.07 hours, this focused test is
expected to take roughly 45–60 minutes. Runtime may vary.

When it finishes, upload:

```text
model9_code\results\model9_endpoint_confirmation_seed30142.zip
```

## Main outputs

- `confirmation_decision.json`: primary gate, paired interval, p-value, AEP
  difference, and secondary benchmark result;
- `confirmation_candidate_summary.csv`: high-sample values and marginal
  batch uncertainty for each frozen layout;
- `paired_batch_differences.csv`: auditable batch-by-batch paired differences;
- `confirmation_batches.csv`: complete raw batch summaries; and
- `run_settings.json`: seeds, provenance, checksums, software versions, model
  assumptions, and completion status.

If the run is interrupted, `confirmation_batches_checkpoint.csv` retains every
fully completed batch available at that point. Start a replacement run with a
new output path rather than combining partial and complete runs manually.

## Scientific boundary

`E_RT_CE` is expected rotor-swept-area transit exposure per entering bird. It
is not blade-collision probability, fatality, or mortality.

