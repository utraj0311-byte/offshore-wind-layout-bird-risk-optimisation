from __future__ import annotations

import unittest

import numpy as np

from model8_meso_response import BehaviourScenario, TrajectoryPopulation
from model9_batching import summarise_model9_batches
from model9_rotor_transit import (
    DiscreteFlightHeightDistribution,
    Model9Population,
    RotorGeometry,
    RotorTransitModel,
)


def population(starts: np.ndarray, heights: np.ndarray, seed: int) -> Model9Population:
    starts = np.asarray(starts, dtype=float).reshape(-1, 2)
    heights = np.asarray(heights, dtype=float)
    n = len(starts)
    trajectories = TrajectoryPopulation(
        start_positions_m=starts,
        headings_deg_toward=np.zeros(n),
        downstream_distance_m=np.full(n, 1200.0),
        response_uniforms=np.full((n, 1), 0.5),
        side_uniforms=np.full((n, 1), 0.5),
        seed=seed,
    )
    return Model9Population(
        trajectories=trajectories,
        flight_heights_m=heights,
        height_uniforms=np.linspace(0.1, 0.9, n),
        flight_height_seed=seed + 100,
        flight_height_distribution_name="batch-test",
    )


class BatchAggregationTests(unittest.TestCase):
    def test_counts_are_summed_before_combined_metrics(self) -> None:
        scenario = BehaviourScenario(
            response_probability=0.0,
            response_radius_m=200.0,
            encounter_radius_m=100.0,
            deflection_angle_deg=45.0,
            return_length_m=400.0,
            step_size_m=25.0,
        )
        model = RotorTransitModel(
            scenario,
            RotorGeometry(rotor_diameter_m=200.0, hub_height_m=150.0),
            rotor_plane_heading_deg_toward=0.0,
            flight_height_distribution=DiscreteFlightHeightDistribution(
                name="batch-test",
                heights_m=np.array([150.0, 260.0]),
                probability_mass=np.array([0.2, 0.8]),
            ),
        )
        first = population(
            np.array([[-500.0, 0.0], [-500.0, 150.0]]),
            np.array([150.0, 150.0]),
            1,
        )
        second = population(
            np.array([[-500.0, 0.0], [-500.0, 50.0], [-500.0, 150.0]]),
            np.array([260.0, 260.0, 150.0]),
            2,
        )
        combined = summarise_model9_batches(
            model,
            np.array([[0.0, 0.0]]),
            [first, second],
        )

        self.assertEqual(combined["objective_batch_count"], 2)
        self.assertEqual(combined["N_started"], 5)
        self.assertEqual(combined["N_enc_farm"], 3)
        self.assertEqual(combined["N_RT_farm"], 1)
        self.assertAlmostEqual(combined["E_enc"], 3.0 / 5.0)
        self.assertAlmostEqual(combined["E_RT"], 1.0 / 5.0)
        self.assertAlmostEqual(combined["P_RT"], 1.0 / 5.0)
        self.assertAlmostEqual(combined["RT_to_encounter_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(combined["N_RT_CE_farm"], 0.6)
        self.assertAlmostEqual(combined["E_RT_CE"], 0.6 / 5.0)
        self.assertAlmostEqual(combined["P_RT_CE"], 0.6 / 5.0)
        self.assertAlmostEqual(combined["RT_CE_to_encounter_ratio"], 0.2)
        np.testing.assert_array_equal(combined["per_turbine_encounters"], [3])
        np.testing.assert_array_equal(combined["per_turbine_transits"], [1])
        np.testing.assert_allclose(
            combined["per_turbine_expected_transits"], [0.6]
        )
        self.assertGreater(combined["E_RT_batch_sd"], 0.0)


if __name__ == "__main__":
    unittest.main()
