# Model 9 v7 changes

- Preserved the completed northern-gannet Model 9 result unchanged.
- Added a separate fixed-hub-height sensitivity that places every trajectory
  at the 150 m rotor centre to isolate horizontal layout geometry under
  guaranteed vertical exposure.
- Frozen the four evaluated layouts, their checksums, FLORIS inputs, AEP
  expectations, 15 × 20,000 sampling design, and independent final seed lists
  in `data/model9_hub_height_sensitivity_manifest.json`.
- Recorded that a five-batch development pilot informed the primary comparison
  and sample size, and excluded those pilot observations from final inference.
- Defined exactly one primary comparison: original energy layout minus
  `pareto_032`. All remaining pairwise comparisons are labelled descriptive.
- Added checkpoint/resume support and strict guards for fixed height, paired
  seeds, completed trajectories, layout hashes, FLORIS hashes, rotor geometry,
  AEP values, and equality of sampled and conditional metrics.
- Added raw-batch, candidate-summary, paired-difference, all-pairs, decision,
  and run-provenance outputs.
- Added nine automated tests for the new distribution, invariants, comparison
  direction, decision gate, and statistical labelling. The complete suite now
  contains 72 passing tests.
- Did not add a new species, fit a flight-height curve, change Model 8
  behaviour, change the rotor-transit equation, or run another optimisation.

