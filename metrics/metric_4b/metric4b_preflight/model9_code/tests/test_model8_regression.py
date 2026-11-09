from __future__ import annotations

import unittest

import numpy as np

from model8_batching import (
    nondominated_mask,
    robust_nondominated_mask,
    select_anchor_positions,
    summarise_meso_batches,
)
from model8_meso_response import (
    BehaviourScenario,
    MesoResponseModel,
    TrajectoryPopulation,
    bird_frame_coordinates,
    heading_unit_vectors,
)


def manual_population(
    starts: np.ndarray,
    *,
    heading_deg: float,
    downstream_distance_m: float,
    maximum_events: int,
    response_uniform: float = 0.5,
    side_uniform: float = 0.75,
) -> TrajectoryPopulation:
    starts = np.asarray(starts, dtype=float).reshape(-1, 2)
    n = len(starts)
    return TrajectoryPopulation(
        start_positions_m=starts,
        headings_deg_toward=np.full(n, heading_deg, dtype=float),
        downstream_distance_m=np.full(n, downstream_distance_m, dtype=float),
        response_uniforms=np.full(
            (n, maximum_events), response_uniform, dtype=float
        ),
        side_uniforms=np.full((n, maximum_events), side_uniform, dtype=float),
        seed=123,
    )


def behaviour(response_probability: float) -> BehaviourScenario:
    return BehaviourScenario(
        response_probability=response_probability,
        response_radius_m=200.0,
        encounter_radius_m=100.0,
        deflection_angle_deg=60.0,
        return_length_m=400.0,
        step_size_m=10.0,
    )


class DirectionConventionTests(unittest.TestCase):
    def test_toward_225_matches_existing_bird_frame(self) -> None:
        unit = heading_unit_vectors(225.0)
        np.testing.assert_allclose(
            unit,
            np.array([-np.sqrt(0.5), -np.sqrt(0.5)]),
            atol=1.0e-12,
        )

        along, cross = bird_frame_coordinates(
            np.array([[4281.656895117357, 271.9332054135207]]),
            225.0,
        )
        self.assertAlmostEqual(float(along[0]), -3219.8744388293167, places=9)
        self.assertAlmostEqual(float(cross[0]), 2835.302811673926, places=9)


class PopulationTests(unittest.TestCase):
    def test_common_random_population_is_reproducible(self) -> None:
        kwargs = dict(
            site_width_m=6000.0,
            site_height_m=6000.0,
            heading_deg_toward=225.0,
            number_of_trajectories=100,
            maximum_response_events=12,
            longitudinal_margin_m=1000.0,
            seed=42,
        )
        first = TrajectoryPopulation.generate_parallel(**kwargs)
        second = TrajectoryPopulation.generate_parallel(**kwargs)
        np.testing.assert_array_equal(first.start_positions_m, second.start_positions_m)
        np.testing.assert_array_equal(first.response_uniforms, second.response_uniforms)
        np.testing.assert_array_equal(first.side_uniforms, second.side_uniforms)

        subset = first.subset(25)
        self.assertEqual(subset.number_of_trajectories, 25)
        np.testing.assert_array_equal(
            subset.start_positions_m, first.start_positions_m[:25]
        )

    def test_independent_batch_seeds_produce_distinct_populations(self) -> None:
        kwargs = dict(
            site_width_m=6000.0,
            site_height_m=6000.0,
            heading_deg_toward=225.0,
            number_of_trajectories=100,
            maximum_response_events=12,
            longitudinal_margin_m=1000.0,
        )
        first = TrajectoryPopulation.generate_parallel(**kwargs, seed=42)
        second = TrajectoryPopulation.generate_parallel(**kwargs, seed=43)
        self.assertFalse(
            np.array_equal(first.start_positions_m, second.start_positions_m)
        )


class SimulationTests(unittest.TestCase):
    def test_no_response_produces_straight_paths_and_known_geometry(self) -> None:
        population = manual_population(
            np.array([[-500.0, 0.0], [-500.0, 99.0], [-500.0, 101.0]]),
            heading_deg=0.0,
            downstream_distance_m=2000.0,
            maximum_events=1,
        )
        result = MesoResponseModel(behaviour(0.0)).simulate(
            np.array([[0.0, 0.0]]), population
        )

        self.assertEqual(result.number_started, 3)
        self.assertEqual(result.number_completed, 3)
        self.assertEqual(result.number_failed, 0)
        self.assertEqual(result.total_response_events, 0)
        self.assertEqual(result.total_encounters, 2)
        np.testing.assert_array_equal(result.per_turbine_encounters, [2])
        np.testing.assert_allclose(result.route_extensions_m, 0.0, atol=1.0e-12)
        result.validate_accounting()

    def test_response_can_divert_a_direct_approach_outside_encounter_zone(self) -> None:
        population = manual_population(
            np.array([[-500.0, 0.0]]),
            heading_deg=0.0,
            downstream_distance_m=2000.0,
            maximum_events=1,
            response_uniform=0.0,
            side_uniform=0.75,
        )
        result = MesoResponseModel(behaviour(1.0)).simulate(
            np.array([[0.0, 0.0]]),
            population,
            store_path_indices=[0],
        )

        self.assertEqual(result.total_response_events, 1)
        self.assertEqual(result.total_encounters, 0)
        self.assertGreater(result.mean_route_extension_m, 0.0)
        self.assertGreater(len(result.stored_paths_m[0]), 2)

    def test_continuous_visit_is_counted_once(self) -> None:
        population = manual_population(
            np.array([[-500.0, 0.0]]),
            heading_deg=0.0,
            downstream_distance_m=2000.0,
            maximum_events=1,
        )
        result = MesoResponseModel(behaviour(0.0)).simulate(
            np.array([[0.0, 0.0]]), population
        )
        self.assertEqual(result.total_encounters, 1)
        self.assertEqual(result.per_trajectory_encounters[0], 1)

    def test_one_bird_can_encounter_multiple_turbines(self) -> None:
        turbines = np.array([[0.0, 0.0], [400.0, 0.0], [800.0, 0.0]])
        population = manual_population(
            np.array([[-500.0, 0.0]]),
            heading_deg=0.0,
            downstream_distance_m=2000.0,
            maximum_events=3,
        )
        result = MesoResponseModel(behaviour(0.0)).simulate(turbines, population)
        self.assertEqual(result.total_encounters, 3)
        self.assertEqual(result.per_trajectory_encounters[0], 3)
        self.assertEqual(result.encounter_intensity, 3.0)
        self.assertEqual(result.encounter_prevalence, 1.0)

    def test_total_metric_is_invariant_to_turbine_index_permutation(self) -> None:
        turbines = np.array([[0.0, 60.0], [500.0, -60.0], [1000.0, 80.0]])
        population = manual_population(
            np.array([[-500.0, 0.0], [-500.0, 50.0], [-500.0, -50.0]]),
            heading_deg=0.0,
            downstream_distance_m=2500.0,
            maximum_events=3,
            response_uniform=0.4,
        )
        model = MesoResponseModel(behaviour(0.6))
        original = model.simulate(turbines, population)
        permuted = model.simulate(turbines[[2, 0, 1]], population)
        self.assertEqual(original.total_encounters, permuted.total_encounters)
        self.assertEqual(
            original.total_response_events, permuted.total_response_events
        )

    def test_segment_intersection_prevents_step_size_misses(self) -> None:
        coarse = BehaviourScenario(
            response_probability=0.0,
            response_radius_m=200.0,
            encounter_radius_m=25.0,
            deflection_angle_deg=45.0,
            return_length_m=400.0,
            step_size_m=100.0,
        )
        population = manual_population(
            np.array([[-550.0, 0.0]]),
            heading_deg=0.0,
            downstream_distance_m=1200.0,
            maximum_events=1,
        )
        result = MesoResponseModel(coarse).simulate(
            np.array([[0.0, 0.0]]), population
        )
        self.assertEqual(result.total_encounters, 1)


class BatchAggregationTests(unittest.TestCase):
    def test_counts_are_summed_before_combined_ratios_are_formed(self) -> None:
        first = manual_population(
            np.array([[-500.0, 0.0], [-500.0, 150.0]]),
            heading_deg=0.0,
            downstream_distance_m=2000.0,
            maximum_events=1,
        )
        second = manual_population(
            np.array([[-500.0, 0.0], [-500.0, 50.0], [-500.0, 150.0]]),
            heading_deg=0.0,
            downstream_distance_m=2000.0,
            maximum_events=1,
        )
        combined = summarise_meso_batches(
            MesoResponseModel(behaviour(0.0)),
            np.array([[0.0, 0.0]]),
            [first, second],
        )

        self.assertEqual(combined["objective_batch_count"], 2)
        self.assertEqual(combined["N_started"], 5)
        self.assertEqual(combined["N_completed"], 5)
        self.assertEqual(combined["N_failed"], 0)
        self.assertEqual(combined["N_enc_farm"], 3)
        self.assertAlmostEqual(combined["E_enc"], 3.0 / 5.0)
        self.assertAlmostEqual(combined["P_ge_1"], 3.0 / 5.0)
        np.testing.assert_array_equal(combined["per_turbine_encounters"], [3])
        self.assertGreater(combined["E_enc_batch_sd"], 0.0)


class ParetoValidationTests(unittest.TestCase):
    def test_nondominated_mask_rejects_dominated_fresh_points(self) -> None:
        objectives = np.array(
            [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [1.5, 1.5], [1.0, 1.0]]
        )
        np.testing.assert_array_equal(
            nondominated_mask(objectives),
            np.array([True, True, True, False, True]),
        )

    def test_robust_mask_rejects_noise_sized_tradeoff(self) -> None:
        losses = np.array([0.0, 1.0, 2.0])
        paired_fresh_batches = np.array(
            [
                [1.00, 1.02, 0.98],
                [0.99, 1.03, 0.98],
                [0.90, 0.92, 0.88],
            ]
        )
        np.testing.assert_array_equal(
            robust_nondominated_mask(
                losses, paired_fresh_batches, z_value=1.96
            ),
            np.array([True, False, True]),
        )

    def test_anchor_selection_uses_requested_validated_values(self) -> None:
        anchors = select_anchor_positions(
            aep_gwh=np.array([10.0, 9.0, 8.0]),
            aep_loss_gwh=np.array([0.0, 1.0, 2.0]),
            encounters=np.array([3.0, 1.5, 1.0]),
        )
        self.assertEqual(anchors["energy_endpoint"], 0)
        self.assertEqual(anchors["knee"], 1)
        self.assertEqual(anchors["encounter_endpoint"], 2)


if __name__ == "__main__":
    unittest.main()
