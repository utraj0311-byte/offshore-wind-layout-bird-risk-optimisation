from __future__ import annotations

import unittest

import numpy as np

from run_model8_confirmation import (
    paired_reduction_statistics,
    student_t_multiplier,
)


class StudentTTests(unittest.TestCase):
    def test_95_percent_multiplier_for_twenty_batches(self) -> None:
        self.assertAlmostEqual(
            student_t_multiplier(0.95, 20), 2.093024054408263, places=12
        )

    def test_invalid_batch_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            student_t_multiplier(0.95, 1)


class PairedContrastTests(unittest.TestCase):
    def test_positive_reduction_is_supported(self) -> None:
        reference = np.array([0.12, 0.11, 0.13, 0.10])
        candidate = reference - 0.01
        result = paired_reduction_statistics(
            reference, candidate, critical_value=3.182446305284263
        )
        self.assertAlmostEqual(float(result["mean_reduction"]), 0.01)
        self.assertTrue(bool(result["reduction_supported"]))
        self.assertFalse(bool(result["worsening_supported"]))
        self.assertEqual(float(result["two_sided_p_value"]), 0.0)

    def test_identical_batches_are_not_a_supported_improvement(self) -> None:
        values = np.array([0.12, 0.11, 0.13, 0.10])
        result = paired_reduction_statistics(
            values, values, critical_value=3.182446305284263
        )
        self.assertEqual(float(result["mean_reduction"]), 0.0)
        self.assertEqual(float(result["two_sided_p_value"]), 1.0)
        self.assertFalse(bool(result["reduction_supported"]))
        self.assertFalse(bool(result["worsening_supported"]))

    def test_mismatched_batch_arrays_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_reduction_statistics(
                np.array([0.1, 0.2]),
                np.array([0.1]),
                critical_value=1.96,
            )


if __name__ == "__main__":
    unittest.main()
