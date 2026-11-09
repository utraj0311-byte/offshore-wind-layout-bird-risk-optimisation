"""Regression tests for candidate screening and rotor-velocity validity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from shielding_aep_problem import ShieldingAEPProblem  # noqa: E402


class FakeAEPModel:
    def __init__(self, rotor_velocity_m_s: float, aep_gwh: float = 800.0) -> None:
        self.rotor_velocity_m_s = float(rotor_velocity_m_s)
        self.aep_gwh = float(aep_gwh)
        self.calls = 0
        self.floris_model = SimpleNamespace(
            core=SimpleNamespace(
                flow_field=SimpleNamespace(u=np.empty((0,), dtype=float))
            )
        )

    def evaluate(self, xy: np.ndarray) -> float:
        self.calls += 1
        number_of_turbines = len(np.asarray(xy).reshape(-1, 2))
        self.floris_model.core.flow_field.u = np.full(
            (1, number_of_turbines, 1, 1),
            self.rotor_velocity_m_s,
            dtype=float,
        )
        return self.aep_gwh


def make_problem(aep_model: FakeAEPModel) -> ShieldingAEPProblem:
    scenario = SimpleNamespace(
        site=SimpleNamespace(
            n_turbines=2,
            min_spacing_D=4.0,
            width_m=6000.0,
            height_m=6000.0,
        )
    )
    return ShieldingAEPProblem(
        scenario=scenario,
        aep_model=aep_model,
        rotor_diameter_m=100.0,
        bird_heading_deg_toward=0.0,
        corridor_width_m=6000.0,
        aep_ceiling_gwh=900.0,
        aep_loss_reference_gwh=100.0,
        record_evaluations=True,
    )


class CandidateScreeningTests(unittest.TestCase):
    def test_spacing_infeasible_layout_skips_floris(self) -> None:
        aep_model = FakeAEPModel(rotor_velocity_m_s=8.0)
        problem = make_problem(aep_model)
        out: dict = {}
        problem._evaluate(np.array([[0.0, 0.0, 100.0, 0.0]]), out)

        self.assertEqual(aep_model.calls, 0)
        self.assertEqual(problem.spacing_prescreen_skip_count, 1)
        self.assertGreater(out["G"][0, 0], 0.0)
        self.assertEqual(out["G"][0, -1], 0.0)
        self.assertTrue(np.isfinite(out["F"]).all())

    def test_valid_layout_records_positive_rotor_velocity(self) -> None:
        aep_model = FakeAEPModel(rotor_velocity_m_s=8.0)
        problem = make_problem(aep_model)
        out: dict = {}
        problem._evaluate(np.array([[0.0, 0.0, 500.0, 0.0]]), out)

        self.assertEqual(aep_model.calls, 1)
        self.assertAlmostEqual(out["F"][0, 0], 1.0)
        self.assertAlmostEqual(out["G"][0, -1], -8.0)
        self.assertEqual(problem.negative_rotor_velocity_candidate_count, 0)

    def test_negative_rotor_velocity_is_formally_infeasible(self) -> None:
        aep_model = FakeAEPModel(rotor_velocity_m_s=-0.2)
        problem = make_problem(aep_model)
        out: dict = {}
        problem._evaluate(np.array([[0.0, 0.0, 500.0, 0.0]]), out)

        self.assertEqual(aep_model.calls, 1)
        self.assertGreater(out["F"][0, 0], 2.0)
        self.assertAlmostEqual(out["G"][0, -1], 0.2)
        self.assertEqual(problem.negative_rotor_velocity_candidate_count, 1)


if __name__ == "__main__":
    unittest.main()
