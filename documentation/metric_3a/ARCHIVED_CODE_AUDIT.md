# Metric 3a verification: code excerpts and decision

CHECK 1a: PASS (single-objective GA, no energy or wake evaluation).
- risk_only_problem.py:57-62: `n_obj=1`.
- risk_only_problem.py:82-84: `F[population_index, 0] = float(self.risk_model.collisions(xy))`.
- run_directional_absorbing_risk_only.py:644-658: risk-only problem and GA.
- run_directional_absorbing_seeded.py:247-268: the same risk-only problem and GA.
- run_validation.py:14-16 imports `AEPModel`, but is imported by these runners ONLY for GridSampling and geometry utilities; its AEP paths are not executed by these runners.

CHECK 1b: PASS (fixed bird travel direction, not derived from wind).
- directional_absorbing_risk.py:43-49: `bird_travel_deg: float = 225.0`, downstream 8D, cross 1D, alpha 0.8.
- directional_absorbing_risk.py:143-149: heading is a TOWARD angle, 0 toward +x, 90 toward +y; radians of `self.bird_travel_deg`.
- directional_absorbing_risk.py:152-168: along=(cos(theta),sin(theta)), cross=(-sin(theta),cos(theta)).
- The 225 deg numerical value is independent of the wind FROM convention and wind rose resolution.

CHECK 2: PASS. See metric3a_verification.csv for full saved and recalculated precision.

DECISION: Keep existing Metric 3a results. Do not rerun the optimisation.

Geometry counts apply hard cutoffs 1920 m along and 240 m across only for diagnostic interpretation; the model's exponential kernel itself has no hard downstream/cross-flight cutoff.

Original ZIPs and input code were copied without alteration. Output figures have no titles or turbine labels, use bird-flight coordinates with flight arrow, and put Q_3a in legend entries.
