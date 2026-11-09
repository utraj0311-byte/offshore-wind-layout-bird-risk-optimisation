from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from model8_meso_response import (
    BehaviourScenario,
    MesoResponseModel,
    TrajectoryPopulation,
)
from model9_rotor_transit import (
    DiscreteFlightHeightDistribution,
    Model9Population,
    RotorGeometry,
    RotorTransitModel,
    segment_rotor_disk_transit_probabilities,
    segment_rotor_disk_transits,
)


def behaviour(
    response_probability: float = 0.0,
    *,
    step_size_m: float = 10.0,
) -> BehaviourScenario:
    return BehaviourScenario(
        response_probability=response_probability,
        response_radius_m=200.0,
        encounter_radius_m=100.0,
        deflection_angle_deg=45.0,
        return_length_m=400.0,
        step_size_m=step_size_m,
    )


def manual_population(
    starts: np.ndarray,
    heights_m: np.ndarray,
    *,
    heading_deg: float = 0.0,
    downstream_distance_m: float = 1200.0,
    maximum_events: int = 4,
    response_uniform: float = 0.5,
    side_uniform: float = 0.75,
) -> Model9Population:
    starts = np.asarray(starts, dtype=float).reshape(-1, 2)
    heights = np.asarray(heights_m, dtype=float)
    n = len(starts)
    trajectories = TrajectoryPopulation(
        start_positions_m=starts,
        headings_deg_toward=np.full(n, heading_deg, dtype=float),
        downstream_distance_m=np.full(n, downstream_distance_m, dtype=float),
        response_uniforms=np.full(
            (n, maximum_events), response_uniform, dtype=float
        ),
        side_uniforms=np.full((n, maximum_events), side_uniform, dtype=float),
        seed=123,
    )
    return Model9Population(
        trajectories=trajectories,
        flight_heights_m=heights,
        height_uniforms=np.linspace(0.1, 0.9, n),
        flight_height_seed=456,
        flight_height_distribution_name="deterministic-test-heights",
    )


class DirectRotorGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rotor = RotorGeometry(rotor_diameter_m=200.0, hub_height_m=150.0)
        self.turbine = np.array([[0.0, 0.0]])

    def hit(self, *, y_m: float, z_m: float) -> bool:
        result = segment_rotor_disk_transits(
            np.array([[-150.0, y_m]]),
            np.array([[150.0, y_m]]),
            self.turbine,
            np.array([z_m]),
            self.rotor,
            0.0,
        )
        return bool(result[0, 0])

    def test_bird_below_rotor_does_not_transit(self) -> None:
        self.assertFalse(self.hit(y_m=0.0, z_m=49.0))

    def test_bird_above_rotor_does_not_transit(self) -> None:
        self.assertFalse(self.hit(y_m=0.0, z_m=251.0))

    def test_bird_through_rotor_centre_transits(self) -> None:
        self.assertTrue(self.hit(y_m=0.0, z_m=150.0))

    def test_correct_height_but_lateral_miss_does_not_transit(self) -> None:
        self.assertFalse(self.hit(y_m=101.0, z_m=150.0))

    def test_circular_disk_rejects_rectangular_corner(self) -> None:
        self.assertFalse(self.hit(y_m=90.0, z_m=200.0))

    def test_exact_disk_tangent_counts_as_transit(self) -> None:
        self.assertTrue(self.hit(y_m=100.0, z_m=150.0))

    def test_segment_that_does_not_cross_rotor_plane_is_rejected(self) -> None:
        result = segment_rotor_disk_transits(
            np.array([[-150.0, 0.0]]),
            np.array([[-50.0, 0.0]]),
            self.turbine,
            np.array([150.0]),
            self.rotor,
            0.0,
        )
        self.assertFalse(bool(result[0, 0]))

    def test_conditional_probability_uses_circular_height_window(self) -> None:
        distribution = DiscreteFlightHeightDistribution(
            name="geometry-test",
            heights_m=np.array([50.0, 100.0, 150.0, 200.0, 250.0]),
            probability_mass=np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
        )
        starts = np.array(
            [
                [-150.0, 0.0],
                [-150.0, 80.0],
                [-150.0, 100.0],
                [-150.0, 101.0],
                [-150.0, 0.0],
            ]
        )
        ends = np.array(
            [
                [150.0, 0.0],
                [150.0, 80.0],
                [150.0, 100.0],
                [150.0, 101.0],
                [-50.0, 0.0],
            ]
        )
        probabilities = segment_rotor_disk_transit_probabilities(
            starts,
            ends,
            self.turbine,
            self.rotor,
            0.0,
            distribution,
        )
        np.testing.assert_allclose(
            probabilities[:, 0],
            np.array([1.0, 0.8, 0.4, 0.0, 0.0]),
            atol=1.0e-12,
        )

    def test_conditional_probability_matches_monte_carlo_mean(self) -> None:
        distribution = DiscreteFlightHeightDistribution(
            name="monte-carlo-test",
            heights_m=np.array([50.0, 100.0, 150.0, 200.0, 250.0]),
            probability_mass=np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
        )
        sample_size = 200_000
        sampled_heights, _ = distribution.sample(sample_size, seed=8675309)
        starts = np.repeat(np.array([[-150.0, 80.0]]), sample_size, axis=0)
        ends = np.repeat(np.array([[150.0, 80.0]]), sample_size, axis=0)
        sampled = segment_rotor_disk_transits(
            starts,
            ends,
            self.turbine,
            sampled_heights,
            self.rotor,
            0.0,
        )
        conditional = segment_rotor_disk_transit_probabilities(
            starts[:1],
            ends[:1],
            self.turbine,
            self.rotor,
            0.0,
            distribution,
        )
        self.assertAlmostEqual(float(conditional[0, 0]), 0.8)
        self.assertAlmostEqual(
            float(np.mean(sampled[:, 0])),
            float(conditional[0, 0]),
            delta=0.004,
        )


class IntegratedModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rotor = RotorGeometry(rotor_diameter_m=200.0, hub_height_m=150.0)

    def test_integrated_four_case_geometry(self) -> None:
        population = manual_population(
            np.array(
                [
                    [-500.0, 0.0],
                    [-500.0, 0.0],
                    [-500.0, 0.0],
                    [-500.0, 101.0],
                ]
            ),
            np.array([49.0, 251.0, 150.0, 150.0]),
        )
        result = RotorTransitModel(
            behaviour(), self.rotor, rotor_plane_heading_deg_toward=0.0
        ).simulate(np.array([[0.0, 0.0]]), population)

        np.testing.assert_array_equal(
            result.per_trajectory_transits,
            np.array([0, 0, 1, 0]),
        )
        self.assertEqual(result.total_transits, 1)
        self.assertAlmostEqual(result.rotor_transit_intensity, 0.25)
        self.assertAlmostEqual(result.rotor_transit_prevalence, 0.25)
        result.validate_accounting()

    def test_one_bird_can_transit_multiple_rotors(self) -> None:
        turbines = np.array([[0.0, 0.0], [300.0, 0.0], [600.0, 0.0]])
        population = manual_population(
            np.array([[-500.0, 0.0]]),
            np.array([150.0]),
            downstream_distance_m=1600.0,
        )
        result = RotorTransitModel(
            behaviour(), self.rotor, rotor_plane_heading_deg_toward=0.0
        ).simulate(turbines, population)

        self.assertEqual(result.total_transits, 3)
        self.assertEqual(result.rotor_transit_intensity, 3.0)
        self.assertEqual(result.rotor_transit_prevalence, 1.0)
        np.testing.assert_array_equal(result.per_turbine_transits, [1, 1, 1])

    def test_plane_intersection_prevents_coarse_step_miss(self) -> None:
        population = manual_population(
            np.array([[-550.0, 0.0]]),
            np.array([150.0]),
            downstream_distance_m=1200.0,
        )
        result = RotorTransitModel(
            behaviour(step_size_m=300.0),
            self.rotor,
            rotor_plane_heading_deg_toward=0.0,
        ).simulate(np.array([[0.0, 0.0]]), population)
        self.assertEqual(result.total_transits, 1)

    def test_rotor_transit_is_subset_of_model8_encounter(self) -> None:
        population = manual_population(
            np.array([[-500.0, 0.0], [-500.0, 150.0]]),
            np.array([150.0, 150.0]),
        )
        result = RotorTransitModel(
            behaviour(), self.rotor, rotor_plane_heading_deg_toward=0.0
        ).simulate(np.array([[0.0, 0.0]]), population)
        self.assertTrue(
            np.all(~result.transit_matrix | result.meso_result.encounter_matrix)
        )

    def test_model8_behaviour_and_encounters_are_unchanged(self) -> None:
        turbines = np.array([[0.0, 0.0], [450.0, -80.0]])
        population = manual_population(
            np.array([[-500.0, 0.0], [-500.0, 50.0], [-500.0, -120.0]]),
            np.array([150.0, 80.0, 220.0]),
            maximum_events=2,
            response_uniform=0.4,
        )
        scenario = behaviour(response_probability=0.6)
        direct = MesoResponseModel(scenario).simulate(
            turbines,
            population.trajectories,
        )
        extended = RotorTransitModel(
            scenario, self.rotor, rotor_plane_heading_deg_toward=0.0
        ).simulate(turbines, population).meso_result

        for field in (
            "encounter_matrix",
            "response_matrix",
            "per_trajectory_encounters",
            "per_trajectory_responses",
            "approach_events_per_trajectory",
            "route_lengths_m",
            "completed_mask",
            "failed_mask",
        ):
            np.testing.assert_array_equal(getattr(direct, field), getattr(extended, field))

    def test_total_transits_are_invariant_to_turbine_index_permutation(self) -> None:
        turbines = np.array([[0.0, 40.0], [350.0, -40.0], [700.0, 80.0]])
        population = manual_population(
            np.array([[-500.0, 0.0], [-500.0, 50.0], [-500.0, -50.0]]),
            np.array([150.0, 150.0, 150.0]),
            maximum_events=3,
            downstream_distance_m=1800.0,
            response_uniform=0.4,
        )
        model = RotorTransitModel(
            behaviour(response_probability=0.6),
            self.rotor,
            rotor_plane_heading_deg_toward=0.0,
        )
        original = model.simulate(turbines, population)
        permuted = model.simulate(turbines[[2, 0, 1]], population)
        self.assertEqual(original.total_transits, permuted.total_transits)
        self.assertEqual(
            original.meso_result.total_encounters,
            permuted.meso_result.total_encounters,
        )

    def test_fixed_rotor_plane_must_match_reference_bird_heading(self) -> None:
        population = manual_population(
            np.array([[-500.0, 0.0]]),
            np.array([150.0]),
            heading_deg=0.0,
        )
        model = RotorTransitModel(
            behaviour(), self.rotor, rotor_plane_heading_deg_toward=15.0
        )
        with self.assertRaisesRegex(ValueError, "rotor-plane"):
            model.simulate(np.array([[0.0, 0.0]]), population)

    def test_conditional_estimator_integrates_height_without_changing_sample(self) -> None:
        population = manual_population(
            np.array([[-500.0, 0.0], [-500.0, 0.0]]),
            np.array([49.0, 251.0]),
        )
        distribution = DiscreteFlightHeightDistribution(
            name="deterministic-test-heights",
            heights_m=np.array([49.0, 150.0, 251.0]),
            probability_mass=np.array([0.25, 0.5, 0.25]),
        )
        result = RotorTransitModel(
            behaviour(),
            self.rotor,
            rotor_plane_heading_deg_toward=0.0,
            flight_height_distribution=distribution,
        ).simulate(np.array([[0.0, 0.0]]), population)

        self.assertEqual(result.total_transits, 0)
        self.assertEqual(result.rotor_transit_intensity, 0.0)
        self.assertTrue(result.has_conditional_estimator)
        np.testing.assert_allclose(
            result.expected_transit_probability_matrix,
            np.array([[0.5], [0.5]]),
        )
        self.assertAlmostEqual(result.expected_total_transits, 1.0)
        self.assertAlmostEqual(result.conditional_rotor_transit_intensity, 0.5)
        self.assertAlmostEqual(result.conditional_rotor_transit_prevalence, 0.5)

    def test_conditional_estimator_requires_matching_distribution_name(self) -> None:
        population = manual_population(
            np.array([[-500.0, 0.0]]),
            np.array([150.0]),
        )
        distribution = DiscreteFlightHeightDistribution(
            name="different-distribution",
            heights_m=np.array([150.0]),
            probability_mass=np.array([1.0]),
        )
        model = RotorTransitModel(
            behaviour(),
            self.rotor,
            rotor_plane_heading_deg_toward=0.0,
            flight_height_distribution=distribution,
        )
        with self.assertRaisesRegex(ValueError, "distribution names"):
            model.simulate(np.array([[0.0, 0.0]]), population)


class FlightHeightPopulationTests(unittest.TestCase):
    def test_probability_between_is_inclusive_and_broadcasts(self) -> None:
        distribution = DiscreteFlightHeightDistribution(
            name="cdf-test",
            heights_m=np.array([10.0, 20.0, 30.0, 40.0]),
            probability_mass=np.array([0.1, 0.2, 0.3, 0.4]),
        )
        probabilities = distribution.probability_between(
            np.array([10.0, 20.0, 20.1, 50.0, 30.0]),
            np.array([10.0, 30.0, 39.9, 60.0, 20.0]),
        )
        np.testing.assert_allclose(
            probabilities,
            np.array([0.1, 0.5, 0.3, 0.0, 0.0]),
        )

    def test_native_stochlab_johnston_csv_is_loaded_without_reformatting(self) -> None:
        csv_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "northern_gannet_Johnston2014_FHD.csv"
        )
        distribution = DiscreteFlightHeightDistribution.from_csv(
            csv_path,
            name="northern-gannet-johnston-2014-mle",
            source="unit-test provenance record",
        )

        self.assertEqual(len(distribution.heights_m), 300)
        self.assertEqual(distribution.input_schema, "height,prop (stochLAB/Johnston)")
        self.assertEqual(
            distribution.input_sha256,
            "adc9ea903db9c8e34bfe5b88c382381b737c2e676d79af437f66d198208a0f2c",
        )
        self.assertAlmostEqual(distribution.input_probability_mass_sum, 1.00000005584306)
        self.assertAlmostEqual(float(np.sum(distribution.probability_mass)), 1.0)
        self.assertAlmostEqual(distribution.mean_height_m, 9.149713887, places=8)

    def test_distribution_sampling_and_population_are_reproducible(self) -> None:
        distribution = DiscreteFlightHeightDistribution(
            name="test-fhd",
            heights_m=np.array([10.0, 100.0, 250.0]),
            probability_mass=np.array([0.5, 0.3, 0.2]),
            source="unit test",
        )
        kwargs = dict(
            flight_height_distribution=distribution,
            flight_height_seed=99,
            site_width_m=6000.0,
            site_height_m=6000.0,
            heading_deg_toward=225.0,
            number_of_trajectories=100,
            maximum_response_events=12,
            longitudinal_margin_m=1000.0,
            trajectory_seed=42,
        )
        first = Model9Population.generate_parallel(**kwargs)
        second = Model9Population.generate_parallel(**kwargs)
        np.testing.assert_array_equal(first.flight_heights_m, second.flight_heights_m)
        np.testing.assert_array_equal(
            first.trajectories.start_positions_m,
            second.trajectories.start_positions_m,
        )
        self.assertTrue(set(np.unique(first.flight_heights_m)) <= {10.0, 100.0, 250.0})

    def test_height_seed_is_independent_of_trajectory_seed(self) -> None:
        distribution = DiscreteFlightHeightDistribution(
            name="test-fhd",
            heights_m=np.array([50.0, 150.0]),
            probability_mass=np.array([0.5, 0.5]),
        )
        common = dict(
            flight_height_distribution=distribution,
            site_width_m=1000.0,
            site_height_m=1000.0,
            heading_deg_toward=0.0,
            number_of_trajectories=50,
            maximum_response_events=2,
            longitudinal_margin_m=300.0,
            trajectory_seed=42,
        )
        first = Model9Population.generate_parallel(
            **common, flight_height_seed=90
        )
        second = Model9Population.generate_parallel(
            **common, flight_height_seed=91
        )
        np.testing.assert_array_equal(
            first.trajectories.start_positions_m,
            second.trajectories.start_positions_m,
        )
        self.assertFalse(
            np.array_equal(first.flight_heights_m, second.flight_heights_m)
        )


if __name__ == "__main__":
    unittest.main()
