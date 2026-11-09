# Model 9 v6 changes

- Added the frozen `pareto_038` layout recovered from row 38 of the full-run
  `pareto_decision_vectors.npy` file.
- Added a checksum-locked copy of the completed seed-30142 confirmation.
- Added `run_model9_p038_confirmation.py`, which evaluates only `pareto_038`
  on the same 15 × 100,000 trajectory populations and pairs it with the saved
  `pareto_032` batches.
- Added explicit provenance stating that this is an independent extension,
  not the original pre-specified p037 test.
- Added strict guards for archive identity, seeds, batch size, NumPy version,
  flight-height input and FLORIS configuration.
- Added checkpoint/resume support and targeted automated tests.
- Corrected future full-run endpoint reporting so descriptive anchors span the
  fresh nondominated set rather than collapsing to one uncertainty-screened
  layout. This does not change optimisation objectives or previous results.
- Separated future optimisation and fresh-validation seed namespaces
  (`30,000 + seed` and `40,000 + seed` for fresh data). The numerical overlap
  in the historical v4 full run did not reuse data, but the new naming removes
  that avoidable reporting ambiguity.
- No scientific equation, behavioural parameter, rotor geometry, flight-height
  distribution or AEP model was changed.
