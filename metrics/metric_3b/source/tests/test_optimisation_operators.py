import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_shielding_multiobjective import (
    match_layout_turbine_order,
    nondominated_indices_2d,
    repair_minimum_spacing_layout,
)


class PermutationAwareOperatorTests(unittest.TestCase):
    def test_spatial_matching_removes_arbitrary_turbine_permutation(self):
        reference = np.array(
            [[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0], [1000.0, 1000.0]]
        )
        candidate = reference[[2, 0, 3, 1]]
        matched = match_layout_turbine_order(reference, candidate).reshape(-1, 2)
        np.testing.assert_allclose(matched, reference, rtol=0.0, atol=0.0)

    def test_two_objective_archive_removes_dominated_and_duplicate_rows(self):
        objectives = np.array(
            [
                [0.10, 0.30],
                [0.20, 0.20],
                [0.30, 0.10],
                [0.25, 0.25],
                [0.20, 0.20],
            ]
        )
        indices = nondominated_indices_2d(objectives)
        self.assertEqual(indices.tolist(), [2, 1, 0])

    def test_spacing_repair_separates_colliding_turbines_inside_bounds(self):
        layout = np.array(
            [[1000.0, 1000.0], [1100.0, 1000.0], [3000.0, 3000.0]]
        )
        repaired = repair_minimum_spacing_layout(
            layout,
            minimum_spacing_m=500.0,
            lower_bounds=np.zeros(6),
            upper_bounds=np.full(6, 6000.0),
        ).reshape(-1, 2)
        distances = [
            np.linalg.norm(repaired[first] - repaired[second])
            for first in range(len(repaired))
            for second in range(first + 1, len(repaired))
        ]
        self.assertGreaterEqual(min(distances), 500.0 - 1.0e-7)
        self.assertTrue(np.all(repaired >= 0.0))
        self.assertTrue(np.all(repaired <= 6000.0))


if __name__ == "__main__":
    unittest.main()
