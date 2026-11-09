from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model9_hub_height_sensitivity import (
    all_pairwise_comparisons,
    fixed_hub_height_distribution,
    hub_height_decision,
    paired_difference_frame,
    validate_hub_height_batches,
)


CANDIDATE_IDS = [
    "original_energy_layout",
    "pareto_032",
    "pareto_037",
    "pareto_038",
]


def hub_height_frame(values: dict[str, list[float]]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    batch_count = len(next(iter(values.values())))
    for candidate_id in CANDIDATE_IDS:
        metrics = values[candidate_id]
        if len(metrics) != batch_count:
            raise ValueError("Synthetic candidates must have equal batch counts.")
        for batch_index, metric in enumerate(metrics):
            started = 1000
            transit_count = int(round(metric * started))
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "batch_index": batch_index,
                    "trajectory_seed": 900 + batch_index,
                    "flight_height_seed": 1900 + batch_index,
                    "N_started": started,
                    "N_completed": started,
                    "N_failed": 0,
                    "N_enc_farm": 100,
                    "E_enc": 0.1,
                    "P_ge_1": 0.08,
                    "N_RT_farm": transit_count,
                    "E_RT": metric,
                    "P_RT": metric,
                    "N_RT_CE_farm": float(transit_count),
                    "E_RT_CE": metric,
                    "P_RT_CE": metric,
                    "P_height_band": 1.0,
                    "mean_flight_height_m": 150.0,
                    "mean_route_extension_m": 75.0,
                }
            )
    return pd.DataFrame(rows)


class FixedHeightDistributionTests(unittest.TestCase):
    def test_distribution_places_all_mass_at_hub_height(self) -> None:
        distribution = fixed_hub_height_distribution(150.0)
        heights, _ = distribution.sample(100, seed=123)
        self.assertTrue(np.all(heights == 150.0))
        self.assertEqual(distribution.mean_height_m, 150.0)
        self.assertAlmostEqual(
            float(distribution.probability_between(149.0, 151.0)), 1.0
        )
        self.assertAlmostEqual(
            float(distribution.probability_between(151.0, 200.0)), 0.0
        )

    def test_invalid_hub_height_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            fixed_hub_height_distribution(0.0)


class HubHeightValidationTests(unittest.TestCase):
    @property
    def values(self) -> dict[str, list[float]]:
        return {
            "original_energy_layout": [0.040, 0.041, 0.039],
            "pareto_032": [0.050, 0.051, 0.049],
            "pareto_037": [0.051, 0.052, 0.050],
            "pareto_038": [0.049, 0.050, 0.048],
        }

    def test_accepts_complete_deterministic_height_batches(self) -> None:
        batches = validate_hub_height_batches(
            hub_height_frame(self.values),
            candidate_ids=CANDIDATE_IDS,
            hub_height_m=150.0,
        )
        self.assertEqual(batches, [0, 1, 2])

    def test_rejects_non_hub_height_rows(self) -> None:
        frame = hub_height_frame(self.values)
        frame.loc[0, "mean_flight_height_m"] = 149.0
        with self.assertRaisesRegex(ValueError, "locked hub height"):
            validate_hub_height_batches(
                frame,
                candidate_ids=CANDIDATE_IDS,
                hub_height_m=150.0,
            )

    def test_rejects_sampled_conditional_mismatch(self) -> None:
        frame = hub_height_frame(self.values)
        frame.loc[0, "E_RT_CE"] += 0.001
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_hub_height_batches(
                frame,
                candidate_ids=CANDIDATE_IDS,
                hub_height_m=150.0,
            )


class HubHeightDecisionTests(unittest.TestCase):
    @property
    def aep(self) -> dict[str, float]:
        return {
            "original_energy_layout": 847.5,
            "pareto_032": 849.4,
            "pareto_037": 849.2,
            "pareto_038": 828.8,
        }

    @property
    def clear_values(self) -> dict[str, list[float]]:
        noise = np.asarray([-0.001, 0.000, 0.001, -0.0005, 0.0005])
        return {
            "original_energy_layout": (0.040 + noise).tolist(),
            "pareto_032": (0.050 + noise).tolist(),
            "pareto_037": (0.051 + noise).tolist(),
            "pareto_038": (0.049 + noise).tolist(),
        }

    def test_clear_original_reduction_confirms_sensitivity_tradeoff(self) -> None:
        decision = hub_height_decision(
            hub_height_frame(self.clear_values),
            aep_gwh=self.aep,
        )
        self.assertTrue(decision["primary_tradeoff_gate_passed"])
        self.assertEqual(
            decision["primary_status"],
            "confirmed_aep_hub_height_exposure_tradeoff",
        )
        self.assertFalse(decision["analysis_is_species_specific"])

    def test_sign_changing_primary_difference_is_inconclusive(self) -> None:
        values = self.clear_values
        values["original_energy_layout"] = [0.049, 0.051, 0.049, 0.051, 0.050]
        values["pareto_032"] = [0.050] * 5
        decision = hub_height_decision(
            hub_height_frame(values),
            aep_gwh=self.aep,
        )
        self.assertFalse(decision["primary_tradeoff_gate_passed"])
        self.assertEqual(
            decision["primary_status"],
            "hub_height_exposure_difference_inconclusive",
        )

    def test_only_original_vs_p032_is_labelled_primary(self) -> None:
        frame = hub_height_frame(self.clear_values)
        comparisons = all_pairwise_comparisons(
            frame,
            candidate_ids=CANDIDATE_IDS,
            aep_gwh=self.aep,
        )
        self.assertEqual(len(comparisons), 6)
        self.assertEqual(
            int((comparisons["statistical_role"] == "primary").sum()), 1
        )

    def test_paired_output_uses_locked_difference_directions(self) -> None:
        paired = paired_difference_frame(
            hub_height_frame(self.clear_values),
            candidate_ids=CANDIDATE_IDS,
            hub_height_m=150.0,
        )
        self.assertTrue(
            np.allclose(paired["primary_original_minus_pareto_032"], -0.01)
        )
        self.assertTrue(
            np.allclose(
                paired["descriptive_pareto_037_minus_pareto_032"], 0.001
            )
        )


if __name__ == "__main__":
    unittest.main()
