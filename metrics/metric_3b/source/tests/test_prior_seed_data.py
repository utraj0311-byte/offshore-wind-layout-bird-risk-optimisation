"""Integrity checks for the packaged v5 refinement seeds."""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from shielding_model import evaluate_shielding  # noqa: E402


SEED_PATH = CODE_DIR / "seed_data" / "prior_pareto_seed_layouts.csv"


def load_seed_rows() -> list[dict[str, str]]:
    with SEED_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_coordinates(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            [float(row[f"x_{turbine_id}_m"]), float(row[f"y_{turbine_id}_m"])]
            for turbine_id in range(12)
        ],
        dtype=float,
    )


class PriorSeedIntegrityTests(unittest.TestCase):
    def test_seed_count_bounds_and_spacing(self) -> None:
        rows = load_seed_rows()
        self.assertEqual(len(rows), 61)
        for row in rows:
            xy = row_coordinates(row)
            self.assertTrue(np.all(np.isfinite(xy)))
            self.assertTrue(np.all(xy >= 0.0))
            self.assertTrue(np.all(xy <= 6000.0))
            differences = xy[:, None, :] - xy[None, :, :]
            distances = np.sqrt(np.sum(differences * differences, axis=2))
            pair_distances = distances[np.triu_indices(12, k=1)]
            self.assertGreaterEqual(float(np.min(pair_distances)) + 1.0e-9, 968.96)

    def test_stored_N_eff_recomputes_from_coordinates(self) -> None:
        for row in load_seed_rows():
            result = evaluate_shielding(
                row_coordinates(row),
                rotor_diameter_m=242.24,
                bird_heading_deg_toward=225.0,
                corridor_width_m=6000.0,
            )
            self.assertAlmostEqual(result.N_eff, float(row["N_eff"]), places=10)

    def test_seed_objective_points_are_nondominated(self) -> None:
        rows = load_seed_rows()
        objectives = np.asarray(
            [[float(row["N_eff"]), float(row["aep_gwh"])] for row in rows]
        )
        for index, (n_eff, aep_gwh) in enumerate(objectives):
            others = np.delete(objectives, index, axis=0)
            dominates = (
                (others[:, 0] <= n_eff + 1.0e-12)
                & (others[:, 1] >= aep_gwh - 1.0e-9)
                & (
                    (others[:, 0] < n_eff - 1.0e-12)
                    | (others[:, 1] > aep_gwh + 1.0e-9)
                )
            )
            self.assertFalse(bool(np.any(dominates)))


if __name__ == "__main__":
    unittest.main()
