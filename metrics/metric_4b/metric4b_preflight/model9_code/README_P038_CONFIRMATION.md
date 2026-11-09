# Model 9 targeted p038 confirmation extension

## Purpose

The completed Model 9 confirmation found no statistically confirmed
rotor-transit difference between the high-energy layouts `pareto_037` and
`pareto_032`. It also found no detected rotor-transit difference between
`pareto_032` and the original energy layout.

The earlier full-run validation identified `pareto_038` as the lowest-risk
fresh-validated candidate, but that layout was not included in the completed
confirmation. This extension answers the remaining question:

> Does `pareto_038` have reliably lower mean `E_RT_CE` than `pareto_032`, while
> accepting its approximately 2.4% reduction in AEP?

This is not another optimisation. It is one additional frozen-layout test.

## Why only p038 is simulated

The package includes the exact completed confirmation archive. That archive
already contains the `pareto_032` results for 15 batches of 100,000
trajectories. The extension regenerates the exact same trajectory populations
and evaluates only `pareto_038`.

This requires 1.5 million new layout-trajectory simulations instead of
rerunning all 4.5 million simulations from the first confirmation. Checksums,
seeds, batch size, NumPy version, flight-height input and FLORIS configuration
are locked before the run.

## Run in Anaconda Prompt

Open Anaconda Prompt, activate the same environment used for Model 9, and move
into the extracted `model9_code` folder:

```powershell
conda activate floris-env
cd C:\Users\Deepak\Desktop\bird_layout_opt\Model_9_Code_v6\model9_code
python -m unittest discover -s tests -v
python run_model9_p038_confirmation.py
```

On the same computer, the extension should take roughly one third of the time
required by the three-layout confirmation, although the exact runtime can
vary.

If the run is interrupted, continue it with:

```powershell
python run_model9_p038_confirmation.py --resume
```

Do not edit or replace the ZIP stored at
`data/model9_endpoint_confirmation_seed30142.zip`; its checksum is part of the
confirmation design.

## Result to upload

After completion, upload:

`results/model9_p038_confirmation_seed30142.zip`

## Decision rule

The program calculates the paired batch difference:

`E_RT_CE(pareto_038) - E_RT_CE(pareto_032)`

The wide energy-risk trade-off is confirmed only when the complete paired 95%
Student-t confidence interval is below zero. This would mean that `pareto_038`
has reliably lower rotor-transit exposure, but at lower AEP.

If the interval crosses zero, no reliable rotor-transit trade-off has been
demonstrated for the tested northern-gannet layouts.

## Scientific boundary

`E_RT_CE` represents expected rotor-swept-area transit exposure. It is not a
blade-collision probability, fatality estimate or mortality prediction.

`pareto_038` was selected after inspecting the first five-batch fresh
validation. Therefore, this package describes the analysis transparently as an
independent confirmation extension, not as the original v5 pre-specified
primary test.
