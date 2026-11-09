from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model9_confirmation import (
    candidate_summary_frame,
    confirmation_decision,
    paired_student_t_interval,
    validate_paired_batches,
)


def confirmation_frame(
    values: dict[str, list[float]],
    *,
    trajectory_seeds: list[int] | None = None,
) -> pd.DataFrame:
    rows = []
    count = len(next(iter(values.values())))
    seeds = list(range(300, 300 + count)) if trajectory_seeds is None else trajectory_seeds
    for candidate_id, metric_values in values.items():
        if len(metric_values) != count:
            raise ValueError("Synthetic candidates must have equal batch counts.")
        for batch_index, metric in enumerate(metric_values):
            started = 1000
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "batch_index": batch_index,
                    "trajectory_seed": seeds[batch_index],
                    "flight_height_seed": 1000 + seeds[batch_index],
                    "N_started": started,
                    "N_completed": started,
                    "N_failed": 0,
                    "N_enc_farm": 100,
                    "E_enc": 0.1,
                    "P_ge_1": 0.08,
                    "N_RT_farm": int(round(metric * started)),
                    "E_RT": metric,
                    "P_RT": metric * 0.8,
                    "N_RT_CE_farm": metric * started,
                    "E_RT_CE": metric,
                    "P_RT_CE": metric * 0.8,
                    "P_height_band": 0.05,
                    "mean_route_extension_m": 75.0,
                }
            )
    return pd.DataFrame(rows)


class PairedStudentTTests(unittest.TestCase):
    def test_interval_is_formed_from_paired_differences(self) -> None:
        result = paired_student_t_interval(
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0],
        )
        self.assertAlmostEqual(result["mean_first_minus_second"], 0.0)
        self.assertAlmostEqual(result["sample_sd"], 1.0)
        self.assertEqual(result["interval_direction"], "inconclusive")
        self.assertAlmostEqual(result["two_sided_p_value"], 1.0)

    def test_entirely_negative_interval_marks_first_as_lower(self) -> None:
        result = paired_student_t_interval(
            [0.80, 0.90, 1.00, 0.85, 0.95],
            [1.0] * 5,
        )
        self.assertLess(result["ci_upper"], 0.0)
        self.assertEqual(result["interval_direction"], "first_lower")

    def test_rejects_unpaired_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            paired_student_t_interval([1.0, 2.0], [1.0, 2.0, 3.0])


class BatchValidationTests(unittest.TestCase):
    @property
    def values(self) -> dict[str, list[float]]:
        return {
            "original_energy_layout": [0.020, 0.021, 0.019],
            "pareto_032": [0.015, 0.016, 0.014],
            "pareto_037": [0.010, 0.011, 0.009],
        }

    def test_accepts_complete_common_random_number_batches(self) -> None:
        frame = confirmation_frame(self.values)
        batches = validate_paired_batches(frame, list(self.values))
        self.assertEqual(batches, [0, 1, 2])

    def test_rejects_different_seeds_within_one_batch(self) -> None:
        frame = confirmation_frame(self.values)
        frame.loc[
            (frame["candidate_id"] == "pareto_037")
            & (frame["batch_index"] == 1),
            "trajectory_seed",
        ] = 999
        with self.assertRaisesRegex(ValueError, "same trajectory_seed"):
            validate_paired_batches(frame, list(self.values))

    def test_rejects_failed_trajectories(self) -> None:
        frame = confirmation_frame(self.values)
        frame.loc[0, "N_failed"] = 1
        frame.loc[0, "N_completed"] = 999
        with self.assertRaisesRegex(RuntimeError, "failed trajectories"):
            validate_paired_batches(frame, list(self.values))


class ConfirmationDecisionTests(unittest.TestCase):
    @property
    def aep(self) -> dict[str, float]:
        return {
            "original_energy_layout": 100.0,
            "pareto_032": 102.0,
            "pareto_037": 101.5,
        }

    def test_pre_specified_tradeoff_and_joint_improvement_can_be_confirmed(self) -> None:
        noise = np.array([-0.0002, 0.0001, 0.0000, 0.0002, -0.0001])
        frame = confirmation_frame(
            {
                "original_energy_layout": (0.020 + noise).tolist(),
                "pareto_032": (0.015 + noise).tolist(),
                "pareto_037": (0.010 + noise).tolist(),
            }
        )
        decision = confirmation_decision(frame, aep_gwh=self.aep)
        self.assertTrue(decision["primary_tradeoff_gate_passed"])
        self.assertEqual(
            decision["primary_status"],
            "confirmed_aep_rotor_transit_tradeoff",
        )
        self.assertEqual(
            decision["secondary_status"],
            "higher_aep_and_lower_rotor_transit_confirmed",
        )

    def test_sign_reversal_across_batches_remains_inconclusive(self) -> None:
        frame = confirmation_frame(
            {
                "original_energy_layout": [0.020] * 5,
                "pareto_032": [0.015] * 5,
                "pareto_037": [0.014, 0.016, 0.014, 0.016, 0.015],
            }
        )
        decision = confirmation_decision(frame, aep_gwh=self.aep)
        self.assertFalse(decision["primary_tradeoff_gate_passed"])
        self.assertEqual(
            decision["primary_status"], "risk_difference_inconclusive"
        )

    def test_summary_uses_summed_counts_and_retains_batch_uncertainty(self) -> None:
        values = {
            "original_energy_layout": [0.020, 0.022, 0.018],
            "pareto_032": [0.015, 0.017, 0.013],
            "pareto_037": [0.010, 0.012, 0.008],
        }
        frame = confirmation_frame(values)
        summary = candidate_summary_frame(
            frame,
            candidate_ids=list(values),
            aep_gwh=self.aep,
        ).set_index("candidate_id")
        self.assertEqual(summary.loc["pareto_032", "N_started"], 3000)
        self.assertAlmostEqual(summary.loc["pareto_032", "E_RT_CE"], 0.015)
        self.assertGreater(
            summary.loc["pareto_032", "E_RT_CE_batch_sd"], 0.0
        )


if __name__ == "__main__":
    unittest.main()

