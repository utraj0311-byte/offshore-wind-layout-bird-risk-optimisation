"""Deterministic tests for the analytical shielding geometry."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from shielding_model import (  # noqa: E402
    crossflight_unit_vector,
    evaluate_shielding,
    heading_unit_vector,
    merge_intervals,
)


class ShieldingGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.D = 242.24

    def test_heading_convention(self) -> None:
        np.testing.assert_allclose(heading_unit_vector(0.0), [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(heading_unit_vector(90.0), [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(crossflight_unit_vector(0.0), [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(crossflight_unit_vector(90.0), [-1.0, 0.0], atol=1e-12)

    def test_perfect_alignment_counts_as_one_rotor(self) -> None:
        xy = np.array([[0.0, 0.0], [4.0 * self.D, 0.0]])
        result = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=0.0
        )
        self.assertAlmostEqual(result.L_exp, self.D, places=10)
        self.assertAlmostEqual(result.N_eff, 1.0, places=10)

    def test_half_diameter_offset_counts_as_one_point_five(self) -> None:
        xy = np.array([[0.0, 0.0], [4.0 * self.D, 0.5 * self.D]])
        result = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=0.0
        )
        self.assertAlmostEqual(result.L_exp, 1.5 * self.D, places=10)
        self.assertAlmostEqual(result.N_eff, 1.5, places=10)

    def test_one_diameter_offset_counts_as_two(self) -> None:
        xy = np.array([[0.0, 0.0], [4.0 * self.D, self.D]])
        result = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=0.0
        )
        self.assertAlmostEqual(result.L_exp, 2.0 * self.D, places=10)
        self.assertAlmostEqual(result.N_eff, 2.0, places=10)

    def test_transitive_interval_overlap_is_merged_once(self) -> None:
        intervals = np.array([[0.0, 10.0], [8.0, 18.0], [17.0, 27.0], [40.0, 50.0]])
        merged = merge_intervals(intervals)
        np.testing.assert_allclose(merged, [[0.0, 27.0], [40.0, 50.0]])

    def test_layout_is_direction_dependent(self) -> None:
        xy = np.array(
            [[0.0, 0.0], [4.0 * self.D, 0.0], [8.0 * self.D, 0.0], [12.0 * self.D, 0.0]]
        )
        along_row = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=0.0
        )
        across_row = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=90.0
        )
        self.assertAlmostEqual(along_row.N_eff, 1.0, places=10)
        self.assertAlmostEqual(across_row.N_eff, 4.0, places=10)

    def test_heading_reversal_has_same_exposure(self) -> None:
        xy = np.array([[10.0, 30.0], [1000.0, 140.0], [2000.0, 500.0]])
        forward = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=35.0
        )
        reverse = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=215.0
        )
        self.assertAlmostEqual(forward.N_eff, reverse.N_eff, places=10)

    def test_turbine_order_does_not_change_result(self) -> None:
        xy = np.array([[0.0, 0.0], [500.0, 80.0], [1100.0, 390.0], [1800.0, 450.0]])
        original = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=20.0
        )
        reordered = evaluate_shielding(
            xy[[2, 0, 3, 1]], rotor_diameter_m=self.D, bird_heading_deg_toward=20.0
        )
        self.assertAlmostEqual(original.N_eff, reordered.N_eff, places=10)

    def test_translation_does_not_change_result(self) -> None:
        xy = np.array([[0.0, 0.0], [800.0, 100.0], [1700.0, 500.0]])
        translated = xy + np.array([1234.0, -876.0])
        first = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=225.0
        )
        second = evaluate_shielding(
            translated, rotor_diameter_m=self.D, bird_heading_deg_toward=225.0
        )
        self.assertAlmostEqual(first.N_eff, second.N_eff, places=10)

    def test_downstream_only_stagger_has_exactly_equal_objective(self) -> None:
        heading = 225.0
        along = heading_unit_vector(heading)
        cross = crossflight_unit_vector(heading)
        spacing = 4.32 * self.D
        centre = np.array([3000.0, 3000.0])

        aligned = np.asarray(
            [
                centre + a * spacing * along + c * spacing * cross
                for c in (-1.0, 0.0, 1.0)
                for a in (-1.5, -0.5, 0.5, 1.5)
            ]
        )
        staggered = np.asarray(
            [
                centre
                + (a + (-0.25 if lane % 2 == 0 else 0.25)) * spacing * along
                + c * spacing * cross
                for lane, c in enumerate((-1.0, 0.0, 1.0))
                for a in (-1.5, -0.5, 0.5, 1.5)
            ]
        )

        first = evaluate_shielding(
            aligned, rotor_diameter_m=self.D, bird_heading_deg_toward=heading
        )
        second = evaluate_shielding(
            staggered, rotor_diameter_m=self.D, bird_heading_deg_toward=heading
        )
        self.assertEqual(first.N_eff, 3.0)
        self.assertEqual(second.N_eff, 3.0)
        self.assertEqual(
            first.normalised_exposed_rotor_count,
            second.normalised_exposed_rotor_count,
        )

    def test_locked_case_has_feasible_two_lane_layout(self) -> None:
        heading = 225.0
        along = heading_unit_vector(heading)
        cross = crossflight_unit_vector(heading)
        spacing = 4.32 * self.D
        centre = np.array([3000.0, 3000.0])
        xy = np.asarray(
            [
                centre + a * spacing * along + c * spacing * cross
                for c in (-0.5, 0.5)
                for a in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
            ]
        )

        self.assertTrue(np.all(xy >= 0.0))
        self.assertTrue(np.all(xy <= 6000.0))
        differences = xy[:, None, :] - xy[None, :, :]
        distances = np.sqrt(np.sum(differences * differences, axis=2))
        pair_distances = distances[np.triu_indices(len(xy), k=1)]
        self.assertGreaterEqual(
            float(np.min(pair_distances)) + 1.0e-9,
            4.0 * self.D,
        )

        result = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=heading
        )
        self.assertEqual(result.N_eff, 2.0)

    def test_prevalence_and_free_path_fraction(self) -> None:
        xy = np.array([[0.0, 0.0], [4.0 * self.D, 0.5 * self.D]])
        result = evaluate_shielding(
            xy,
            rotor_diameter_m=self.D,
            bird_heading_deg_toward=0.0,
            corridor_width_m=6000.0,
        )
        expected = 1.5 * self.D / 6000.0
        self.assertAlmostEqual(result.P_enc, expected, places=12)
        self.assertAlmostEqual(result.free_path_fraction, 1.0 - expected, places=12)

    def test_shielding_fraction_is_relative_to_physical_rotor_count(self) -> None:
        xy = np.array([[0.0, 0.0], [4.0 * self.D, 0.0]])
        result = evaluate_shielding(
            xy, rotor_diameter_m=self.D, bird_heading_deg_toward=0.0
        )
        self.assertAlmostEqual(result.normalised_exposed_rotor_count, 0.5, places=12)
        self.assertAlmostEqual(result.shielding_fraction, 0.5, places=12)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_shielding(
                np.array([1.0, 2.0]),
                rotor_diameter_m=self.D,
                bird_heading_deg_toward=0.0,
            )
        with self.assertRaises(ValueError):
            evaluate_shielding(
                np.array([[0.0, 0.0]]),
                rotor_diameter_m=0.0,
                bird_heading_deg_toward=0.0,
            )
        with self.assertRaises(ValueError):
            evaluate_shielding(
                np.array([[0.0, 0.0], [1000.0, 1000.0]]),
                rotor_diameter_m=self.D,
                bird_heading_deg_toward=0.0,
                corridor_width_m=100.0,
            )


if __name__ == "__main__":
    unittest.main()
