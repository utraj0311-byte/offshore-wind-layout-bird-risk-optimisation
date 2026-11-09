# Model 9 fixed-hub-height sensitivity

## What this run does

This is a small, controlled follow-up to the completed northern-gannet Model 9
analysis. It does **not** change, repair, or replace that result.

The confirmed gannet result remains: the tested layouts did not have a
statistically distinguishable rotor-transit difference under the supplied
gannet flight-height distribution.

The sensitivity run asks a different question:

> If every simulated trajectory is placed at the 150 m turbine hub height,
> where the rotor is widest, can the horizontal turbine layout create a
> detectable rotor-transit difference?

This removes the low-flight-height compression from the diagnostic. It is not
a prediction for gannets or any other bird species.

## Frozen design

- Four existing layouts: original, `pareto_032`, `pareto_037`, and
  `pareto_038`.
- No new optimisation.
- Every trajectory has a fixed height of exactly 150 m.
- 15 independent batches of 20,000 trajectories.
- 300,000 trajectories per layout and 1.2 million layout-trajectory
  simulations in total.
- The same trajectory population is reused for all four layouts in a batch.
- New final seeds are separate from the development pilot and all earlier
  Model 9 runs.
- The one primary comparison is `original_energy_layout - pareto_032`.
- All other pairwise comparisons are descriptive.

A five-batch development pilot was used to select that primary comparison and
the final sample size. The pilot is recorded in the manifest and is not pooled
with the final results.

## Run it in Anaconda Prompt

Extract the ZIP first. Then activate the same environment that ran the earlier
Model 9 checks:

```bat
conda activate floris-env
cd C:\Users\Deepak\Desktop\bird_layout_opt\Model_9_Code_v7\model9_code
python -m unittest discover -s tests -v
python run_model9_hub_height_sensitivity.py
```

The tests should report `Ran 72 tests` followed by `OK`.

The sensitivity runner prints progress as `Hub-height batch 1/15`, and so on.
It writes a checkpoint after each complete four-layout batch. If the run is
interrupted, continue it from the same directory with:

```bat
python run_model9_hub_height_sensitivity.py --resume
```

Do not run the normal command and `--resume` command at the same time.

The occasional FLORIS/NumPy warning about `where` and `out` is external to this
runner. If batch progress continues and no traceback appears, it does not mean
the run failed.

## What to upload afterward

The runner creates:

```text
results\model9_hub_height_sensitivity_seed90142.zip
```

Upload that result ZIP for review. The most important files inside it are:

- `hub_height_decision.json` — the primary statistical result;
- `hub_height_candidate_summary.csv` — AEP and exposure summary by layout;
- `hub_height_paired_differences.csv` — all 15 paired primary differences;
- `hub_height_pairwise_comparisons.csv` — the primary comparison plus clearly
  labelled descriptive comparisons;
- `hub_height_batches.csv` — raw audit rows;
- `run_settings.json` — complete locked settings and provenance.

## How the result will be interpreted

The primary gate passes only if both of these are true:

1. the entire paired 95% interval shows lower hub-height exposure for the
   original layout than for `pareto_032`; and
2. the original layout has lower AEP than `pareto_032`.

If it passes, the defensible conclusion is that horizontal layout geometry can
produce an AEP-versus-rotor-transit trade-off when vertical exposure is
guaranteed. Combined with the gannet result, this would support the explanation
that the empirical low gannet heights suppressed the usable signal in the
species-specific run.

If it is inconclusive, the correct conclusion is only that these four frozen
layouts did not resolve a hub-height difference at this sample size. It is not
proof that all possible layouts are identical.

No outcome from this run is a collision, fatality, mortality, or real-species
estimate. We should review this result before deciding whether a separate
hypothetical hub-height optimisation is scientifically worthwhile.

