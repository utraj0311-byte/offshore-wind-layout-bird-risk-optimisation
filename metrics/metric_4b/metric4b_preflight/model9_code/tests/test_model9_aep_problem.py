from __future__ import annotations

import unittest
import importlib.util
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np

from model8_meso_response import BehaviourScenario, TrajectoryPopulation

# The scientific test environment does not need pymoo's algorithms.  When the
# optional dependency is absent, this narrow adapter exercises Model 9's
# Problem subclass contract and objective plumbing; real runs install pymoo
# from requirements_model9.txt.
if importlib.util.find_spec("pymoo") is None:
    pymoo_module = ModuleType("pymoo")
    pymoo_module.__path__ = []  # type: ignore[attr-defined]
    core_module = ModuleType("pymoo.core")
    core_module.__path__ = []  # type: ignore[attr-defined]
    problem_module = ModuleType("pymoo.core.problem")

    class Problem:
        def __init__(
            self,
            *,
            n_var: int,
            n_obj: int,
            n_ieq_constr: int,
            xl: np.ndarray,
            xu: np.ndarray,
            elementwise: bool,
        ) -> None:
            self.n_var = n_var
            self.n_obj = n_obj
            self.n_ieq_constr = n_ieq_constr
            self.xl = xl
            self.xu = xu
            self.elementwise = elementwise

    problem_module.Problem = Problem  # type: ignore[attr-defined]
    sys.modules["pymoo"] = pymoo_module
    sys.modules["pymoo.core"] = core_module
    sys.modules["pymoo.core.problem"] = problem_module

from model9_aep_problem import Model9AEPRotorTransitProblem
from model9_rotor_transit import (
    DiscreteFlightHeightDistribution,
    Model9Population,
    RotorGeometry,
    RotorTransitModel,
)


def scenario() -> SimpleNamespace:
    return SimpleNamespace(
        site=SimpleNamespace(
            n_turbines=1,
            width_m=1000.0,
            height_m=1000.0,
            min_spacing_D=4.0,
        )
    )


def behaviour() -> BehaviourScenario:
    return BehaviourScenario(
        response_probability=0.0,
        response_radius_m=200.0,
        encounter_radius_m=100.0,
        deflection_angle_deg=45.0,
        return_length_m=400.0,
        step_size_m=25.0,
    )


def population() -> Model9Population:
    trajectories = TrajectoryPopulation(
        start_positions_m=np.array([[-100.0, 500.0]]),
        headings_deg_toward=np.array([0.0]),
        downstream_distance_m=np.array([1200.0]),
        response_uniforms=np.array([[0.5]]),
        side_uniforms=np.array([[0.5]]),
        seed=1,
    )
    return Model9Population(
        trajectories=trajectories,
        flight_heights_m=np.array([49.0]),
        height_uniforms=np.array([0.1]),
        flight_height_seed=2,
        flight_height_distribution_name="optimisation-test",
    )


def aep_model() -> SimpleNamespace:
    return SimpleNamespace(evaluate=lambda xy: 90.0)


class OptimisationObjectiveTests(unittest.TestCase):
    def test_conditional_estimator_is_default_bird_objective(self) -> None:
        distribution = DiscreteFlightHeightDistribution(
            name="optimisation-test",
            heights_m=np.array([49.0, 150.0, 251.0]),
            probability_mass=np.array([0.25, 0.5, 0.25]),
        )
        model = RotorTransitModel(
            behaviour(),
            RotorGeometry(rotor_diameter_m=200.0, hub_height_m=150.0),
            rotor_plane_heading_deg_toward=0.0,
            flight_height_distribution=distribution,
        )
        problem = Model9AEPRotorTransitProblem(
            scenario=scenario(),
            aep_model=aep_model(),
            rotor_transit_model=model,
            trajectory_populations=[population()],
            aep_ceiling_gwh=100.0,
            aep_loss_reference_gwh=20.0,
            rotor_transit_reference=0.25,
            record_evaluations=False,
        )
        out: dict[str, np.ndarray] = {}
        problem._evaluate(np.array([[500.0, 500.0]]), out)

        np.testing.assert_allclose(out["F"], np.array([[0.5, 2.0]]))
        self.assertEqual(problem.settings_summary()["primary_bird_objective"], "E_RT_CE")

    def test_sampled_objective_remains_available_explicitly(self) -> None:
        model = RotorTransitModel(
            behaviour(),
            RotorGeometry(rotor_diameter_m=200.0, hub_height_m=150.0),
            rotor_plane_heading_deg_toward=0.0,
        )
        problem = Model9AEPRotorTransitProblem(
            scenario=scenario(),
            aep_model=aep_model(),
            rotor_transit_model=model,
            trajectory_populations=[population()],
            aep_ceiling_gwh=100.0,
            aep_loss_reference_gwh=20.0,
            rotor_transit_reference=0.25,
            bird_objective_metric="E_RT",
            record_evaluations=False,
        )
        out: dict[str, np.ndarray] = {}
        problem._evaluate(np.array([[500.0, 500.0]]), out)
        np.testing.assert_allclose(out["F"], np.array([[0.5, 0.0]]))

    def test_conditional_objective_rejects_model_without_distribution(self) -> None:
        model = RotorTransitModel(
            behaviour(),
            RotorGeometry(rotor_diameter_m=200.0, hub_height_m=150.0),
            rotor_plane_heading_deg_toward=0.0,
        )
        with self.assertRaisesRegex(ValueError, "E_RT_CE requires"):
            Model9AEPRotorTransitProblem(
                scenario=scenario(),
                aep_model=aep_model(),
                rotor_transit_model=model,
                trajectory_populations=[population()],
                aep_ceiling_gwh=100.0,
                aep_loss_reference_gwh=20.0,
                rotor_transit_reference=0.25,
            )


if __name__ == "__main__":
    unittest.main()
