"""Model 8: scenario-based meso-response and turbine encounters.

This module contains the ecological core of the replacement Model 8.  It is
deliberately independent of FLORIS and pymoo so that trajectory behaviour can
be validated before it is connected to the multi-objective optimiser.

Direction convention
--------------------
Headings are TOWARD directions in mathematical coordinates:

* 0 degrees: east (+x)
* 90 degrees: north (+y)
* 180 degrees: west (-x)
* 225 degrees: south-west

The convention matches the bird-frame coordinates used by the existing
directional models.

Scientific boundary
-------------------
The model counts two-dimensional turbine encounters.  It does not estimate
bird flux, rotor transits, collision probability, micro-avoidance, or
mortality.  All behavioural parameters are explicit scenario assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np


Array = np.ndarray


def _wrap_to_pi(angle_rad: Array | float) -> Array:
    """Wrap angles to the half-open interval [-pi, pi)."""
    return (np.asarray(angle_rad, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def heading_unit_vectors(headings_deg_toward: Array | float) -> Array:
    """Return unit vectors for TOWARD headings in mathematical degrees."""
    angle_rad = np.deg2rad(np.asarray(headings_deg_toward, dtype=float))
    return np.stack((np.cos(angle_rad), np.sin(angle_rad)), axis=-1)


def cross_unit_vectors(headings_deg_toward: Array | float) -> Array:
    """Return left-normal unit vectors for the supplied headings."""
    along = heading_unit_vectors(headings_deg_toward)
    return np.stack((-along[..., 1], along[..., 0]), axis=-1)


def bird_frame_coordinates(
    xy: Array,
    heading_deg_toward: float,
) -> tuple[Array, Array]:
    """Project Cartesian points into along- and cross-bird coordinates."""
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    along = heading_unit_vectors(float(heading_deg_toward))
    cross = cross_unit_vectors(float(heading_deg_toward))
    return points @ along, points @ cross


@dataclass(frozen=True)
class BehaviourScenario:
    """One hypothetical meso-response scenario, expressed in metres."""

    response_probability: float
    response_radius_m: float
    encounter_radius_m: float
    deflection_angle_deg: float
    return_length_m: float
    step_size_m: float
    max_steps_factor: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.response_probability <= 1.0:
            raise ValueError("response_probability must lie in [0, 1].")
        if self.response_radius_m <= 0.0:
            raise ValueError("response_radius_m must be positive.")
        if self.encounter_radius_m <= 0.0:
            raise ValueError("encounter_radius_m must be positive.")
        if self.encounter_radius_m >= self.response_radius_m:
            raise ValueError(
                "encounter_radius_m must be smaller than response_radius_m."
            )
        if not 0.0 < self.deflection_angle_deg < 90.0:
            raise ValueError("deflection_angle_deg must lie between 0 and 90.")
        if self.return_length_m <= 0.0:
            raise ValueError("return_length_m must be positive.")
        if self.step_size_m <= 0.0:
            raise ValueError("step_size_m must be positive.")
        if self.max_steps_factor <= 1.0:
            raise ValueError("max_steps_factor must be greater than one.")

    @classmethod
    def from_rotor_diameters(
        cls,
        *,
        rotor_diameter_m: float,
        response_probability: float,
        response_radius_D: float = 2.0,
        encounter_radius_D: float = 1.0,
        deflection_angle_deg: float = 45.0,
        return_length_D: float = 8.0,
        step_size_D: float = 0.20,
        max_steps_factor: float = 4.0,
    ) -> "BehaviourScenario":
        """Construct a scenario from transparent multiples of rotor diameter."""
        if rotor_diameter_m <= 0.0:
            raise ValueError("rotor_diameter_m must be positive.")
        return cls(
            response_probability=float(response_probability),
            response_radius_m=float(response_radius_D * rotor_diameter_m),
            encounter_radius_m=float(encounter_radius_D * rotor_diameter_m),
            deflection_angle_deg=float(deflection_angle_deg),
            return_length_m=float(return_length_D * rotor_diameter_m),
            step_size_m=float(step_size_D * rotor_diameter_m),
            max_steps_factor=float(max_steps_factor),
        )

    def settings(self, rotor_diameter_m: float | None = None) -> dict[str, Any]:
        """Return a serialisable record of all behavioural assumptions."""
        data: dict[str, Any] = {
            "response_probability": float(self.response_probability),
            "response_radius_m": float(self.response_radius_m),
            "encounter_radius_m": float(self.encounter_radius_m),
            "deflection_angle_deg": float(self.deflection_angle_deg),
            "return_length_m": float(self.return_length_m),
            "step_size_m": float(self.step_size_m),
            "max_steps_factor": float(self.max_steps_factor),
        }
        if rotor_diameter_m is not None:
            if rotor_diameter_m <= 0.0:
                raise ValueError("rotor_diameter_m must be positive.")
            data.update(
                {
                    "rotor_diameter_m": float(rotor_diameter_m),
                    "response_radius_D": float(
                        self.response_radius_m / rotor_diameter_m
                    ),
                    "encounter_radius_D": float(
                        self.encounter_radius_m / rotor_diameter_m
                    ),
                    "return_length_D": float(
                        self.return_length_m / rotor_diameter_m
                    ),
                    "step_size_D": float(self.step_size_m / rotor_diameter_m),
                }
            )
        return data


@dataclass(frozen=True)
class TrajectoryPopulation:
    """Fixed Monte Carlo population used as common random numbers."""

    start_positions_m: Array
    headings_deg_toward: Array
    downstream_distance_m: Array
    response_uniforms: Array
    side_uniforms: Array
    seed: int | None = None
    entry_cross_fractions: Array | None = None

    def __post_init__(self) -> None:
        starts = np.asarray(self.start_positions_m, dtype=float)
        headings = np.asarray(self.headings_deg_toward, dtype=float)
        downstream = np.asarray(self.downstream_distance_m, dtype=float)
        response = np.asarray(self.response_uniforms, dtype=float)
        side = np.asarray(self.side_uniforms, dtype=float)

        if starts.ndim != 2 or starts.shape[1] != 2:
            raise ValueError("start_positions_m must have shape (N, 2).")
        n = starts.shape[0]
        if headings.shape != (n,):
            raise ValueError("headings_deg_toward must have shape (N,).")
        if downstream.shape != (n,):
            raise ValueError("downstream_distance_m must have shape (N,).")
        if response.ndim != 2 or response.shape[0] != n:
            raise ValueError("response_uniforms must have shape (N, K).")
        if side.shape != response.shape:
            raise ValueError("side_uniforms must match response_uniforms.")
        if response.shape[1] <= 0:
            raise ValueError("At least one response-event random slot is required.")
        if not (
            np.all(np.isfinite(starts))
            and np.all(np.isfinite(headings))
            and np.all(np.isfinite(downstream))
            and np.all(np.isfinite(response))
            and np.all(np.isfinite(side))
        ):
            raise ValueError("Trajectory population contains non-finite values.")
        if np.any(downstream <= 0.0):
            raise ValueError("All downstream distances must be positive.")
        if np.any((response < 0.0) | (response >= 1.0)):
            raise ValueError("response_uniforms must lie in [0, 1).")
        if np.any((side < 0.0) | (side >= 1.0)):
            raise ValueError("side_uniforms must lie in [0, 1).")

        object.__setattr__(self, "start_positions_m", starts.copy())
        object.__setattr__(self, "headings_deg_toward", headings.copy())
        object.__setattr__(self, "downstream_distance_m", downstream.copy())
        object.__setattr__(self, "response_uniforms", response.copy())
        object.__setattr__(self, "side_uniforms", side.copy())

        if self.entry_cross_fractions is not None:
            fractions = np.asarray(self.entry_cross_fractions, dtype=float)
            if fractions.shape != (n,):
                raise ValueError("entry_cross_fractions must have shape (N,).")
            object.__setattr__(self, "entry_cross_fractions", fractions.copy())

    @property
    def number_of_trajectories(self) -> int:
        return int(self.start_positions_m.shape[0])

    @property
    def maximum_response_events(self) -> int:
        return int(self.response_uniforms.shape[1])

    def subset(self, count: int) -> "TrajectoryPopulation":
        """Return the first ``count`` trajectories without changing randomness."""
        if count <= 0 or count > self.number_of_trajectories:
            raise ValueError("count must lie between 1 and the population size.")
        fractions = (
            None
            if self.entry_cross_fractions is None
            else self.entry_cross_fractions[:count]
        )
        return TrajectoryPopulation(
            start_positions_m=self.start_positions_m[:count],
            headings_deg_toward=self.headings_deg_toward[:count],
            downstream_distance_m=self.downstream_distance_m[:count],
            response_uniforms=self.response_uniforms[:count],
            side_uniforms=self.side_uniforms[:count],
            seed=self.seed,
            entry_cross_fractions=fractions,
        )

    @classmethod
    def generate_parallel(
        cls,
        *,
        site_width_m: float,
        site_height_m: float,
        heading_deg_toward: float,
        number_of_trajectories: int,
        maximum_response_events: int,
        longitudinal_margin_m: float,
        seed: int,
        stratified_entries: bool = True,
    ) -> "TrajectoryPopulation":
        """Generate parallel incoming tracks that intersect a rectangular site.

        Entry locations are sampled across the site's complete cross-heading
        projection.  The trajectories begin upstream of the site and finish at
        an equivalent downstream margin.  They are normalised Monte Carlo
        samples, not a calibrated number of real birds.
        """
        if site_width_m <= 0.0 or site_height_m <= 0.0:
            raise ValueError("Site dimensions must be positive.")
        if number_of_trajectories <= 0:
            raise ValueError("number_of_trajectories must be positive.")
        if maximum_response_events <= 0:
            raise ValueError("maximum_response_events must be positive.")
        if longitudinal_margin_m <= 0.0:
            raise ValueError("longitudinal_margin_m must be positive.")

        rng = np.random.default_rng(int(seed))
        n = int(number_of_trajectories)

        if stratified_entries:
            fractions = (np.arange(n, dtype=float) + rng.random(n)) / float(n)
            rng.shuffle(fractions)
        else:
            fractions = rng.random(n)

        heading = float(heading_deg_toward) % 360.0
        along = heading_unit_vectors(heading)
        cross = cross_unit_vectors(heading)

        corners = np.array(
            [
                [0.0, 0.0],
                [site_width_m, 0.0],
                [0.0, site_height_m],
                [site_width_m, site_height_m],
            ],
            dtype=float,
        )
        along_projection = corners @ along
        cross_projection = corners @ cross

        along_min = float(along_projection.min())
        along_max = float(along_projection.max())
        cross_min = float(cross_projection.min())
        cross_max = float(cross_projection.max())

        cross_coordinates = cross_min + fractions * (cross_max - cross_min)
        upstream_coordinate = along_min - longitudinal_margin_m
        starts = (
            upstream_coordinate * along.reshape(1, 2)
            + cross_coordinates[:, None] * cross.reshape(1, 2)
        )
        downstream_distance = np.full(
            n,
            (along_max - along_min) + 2.0 * longitudinal_margin_m,
            dtype=float,
        )

        return cls(
            start_positions_m=starts,
            headings_deg_toward=np.full(n, heading, dtype=float),
            downstream_distance_m=downstream_distance,
            response_uniforms=rng.random((n, maximum_response_events)),
            side_uniforms=rng.random((n, maximum_response_events)),
            seed=int(seed),
            entry_cross_fractions=fractions,
        )


@dataclass
class SimulationResult:
    """Complete Model 8 result for one layout and behavioural scenario."""

    per_turbine_encounters: Array
    encounter_matrix: Array
    response_matrix: Array
    per_trajectory_encounters: Array
    per_trajectory_responses: Array
    approach_events_per_trajectory: Array
    route_lengths_m: Array
    straight_route_lengths_m: Array
    completed_mask: Array
    failed_mask: Array
    stored_paths_m: dict[int, Array] = field(default_factory=dict)

    @property
    def number_started(self) -> int:
        return int(len(self.per_trajectory_encounters))

    @property
    def number_completed(self) -> int:
        return int(np.count_nonzero(self.completed_mask))

    @property
    def number_failed(self) -> int:
        return int(np.count_nonzero(self.failed_mask))

    @property
    def total_encounters(self) -> int:
        return int(np.sum(self.per_turbine_encounters))

    @property
    def total_response_events(self) -> int:
        return int(np.sum(self.per_trajectory_responses))

    @property
    def total_approach_events(self) -> int:
        return int(np.sum(self.approach_events_per_trajectory))

    @property
    def encounter_intensity(self) -> float:
        """Mean encounters per incoming trajectory; this is not a probability."""
        return float(self.total_encounters / self.number_started)

    @property
    def encounter_prevalence(self) -> float:
        """Fraction of trajectories with one or more turbine encounters."""
        return float(
            np.count_nonzero(self.per_trajectory_encounters >= 1)
            / self.number_started
        )

    @property
    def route_extensions_m(self) -> Array:
        return np.maximum(self.route_lengths_m - self.straight_route_lengths_m, 0.0)

    @property
    def mean_route_extension_m(self) -> float:
        return float(np.mean(self.route_extensions_m))

    def validate_accounting(self) -> None:
        """Raise if a trajectory or event accounting identity is broken."""
        if self.number_started != self.number_completed + self.number_failed:
            raise RuntimeError(
                "Trajectory accounting failed: started != completed + failed."
            )
        if self.total_encounters != int(np.sum(self.encounter_matrix)):
            raise RuntimeError("Encounter matrix and turbine totals disagree.")
        if self.total_encounters != int(np.sum(self.per_trajectory_encounters)):
            raise RuntimeError("Encounter matrix and trajectory totals disagree.")
        if self.total_response_events != int(np.sum(self.response_matrix)):
            raise RuntimeError("Response matrix and trajectory totals disagree.")

    def summary(self) -> dict[str, Any]:
        self.validate_accounting()
        return {
            "N_started": self.number_started,
            "N_completed": self.number_completed,
            "N_failed": self.number_failed,
            "N_approach_events": self.total_approach_events,
            "N_response_events": self.total_response_events,
            "N_enc_farm": self.total_encounters,
            "E_enc": self.encounter_intensity,
            "P_ge_1": self.encounter_prevalence,
            "mean_route_extension_m": self.mean_route_extension_m,
            "maximum_encounters_per_trajectory": int(
                np.max(self.per_trajectory_encounters, initial=0)
            ),
            "per_turbine_encounters": self.per_turbine_encounters.astype(int).tolist(),
        }


class MesoResponseModel:
    """Vectorised two-dimensional trajectory and encounter simulator."""

    def __init__(self, behaviour: BehaviourScenario) -> None:
        self.behaviour = behaviour

    @staticmethod
    def _segment_circle_hits(
        start: Array,
        end: Array,
        centres: Array,
        radius_m: float,
    ) -> Array:
        """Return whether each movement segment intersects each circle."""
        segment = end - start
        length_sq = np.sum(segment * segment, axis=1)
        if np.any(length_sq <= 0.0):
            raise RuntimeError("Zero-length trajectory step encountered.")

        centre_from_start = centres[None, :, :] - start[:, None, :]
        fraction = np.sum(
            centre_from_start * segment[:, None, :],
            axis=2,
        ) / length_sq[:, None]
        fraction = np.clip(fraction, 0.0, 1.0)
        closest = start[:, None, :] + fraction[:, :, None] * segment[:, None, :]
        distance_sq = np.sum((closest - centres[None, :, :]) ** 2, axis=2)
        return distance_sq <= radius_m**2

    def simulate(
        self,
        turbine_xy_m: Array,
        population: TrajectoryPopulation,
        *,
        store_path_indices: Iterable[int] = (),
        path_stride: int = 1,
        segment_observer: Callable[[Array, Array, Array], None] | None = None,
    ) -> SimulationResult:
        """Simulate the fixed population through one complete turbine layout.

        ``segment_observer`` is an optional, passive engineering hook used by
        Model 9.  After each movement step it receives the global trajectory
        indices and the corresponding segment start/end coordinates.  The
        default remains ``None`` and therefore leaves every Model 8 behaviour,
        encounter, and accounting result unchanged.
        """
        turbines = np.asarray(turbine_xy_m, dtype=float).reshape(-1, 2)
        if len(turbines) <= 0:
            raise ValueError("At least one turbine is required.")
        if not np.all(np.isfinite(turbines)):
            raise ValueError("Turbine coordinates must be finite.")
        if population.maximum_response_events < len(turbines):
            raise ValueError(
                "The population needs at least one random response slot per turbine."
            )
        if path_stride <= 0:
            raise ValueError("path_stride must be positive.")

        n_birds = population.number_of_trajectories
        n_turbines = len(turbines)
        b = self.behaviour

        positions = population.start_positions_m.copy()
        starts = population.start_positions_m
        base_angles = np.deg2rad(population.headings_deg_toward)
        base_directions = heading_unit_vectors(population.headings_deg_toward)

        active = np.ones(n_birds, dtype=bool)
        completed = np.zeros(n_birds, dtype=bool)
        failed = np.zeros(n_birds, dtype=bool)

        approach_processed = np.zeros((n_birds, n_turbines), dtype=bool)
        encounter_matrix = np.zeros((n_birds, n_turbines), dtype=bool)
        response_matrix = np.zeros((n_birds, n_turbines), dtype=bool)
        approach_counts = np.zeros(n_birds, dtype=int)

        manoeuvre_active = np.zeros(n_birds, dtype=bool)
        holding_deflection = np.zeros(n_birds, dtype=bool)
        manoeuvre_target = np.full(n_birds, -1, dtype=int)
        manoeuvre_offset = np.zeros(n_birds, dtype=float)
        recovery_remaining_m = np.zeros(n_birds, dtype=float)

        route_lengths_m = np.zeros(n_birds, dtype=float)
        straight_route_lengths_m = (
            np.ceil(population.downstream_distance_m / b.step_size_m)
            * b.step_size_m
        )

        requested_paths = sorted({int(i) for i in store_path_indices})
        if any(i < 0 or i >= n_birds for i in requested_paths):
            raise IndexError("A stored path index is outside the population.")
        path_lists: dict[int, list[Array]] = {
            i: [positions[i].copy()] for i in requested_paths
        }

        nominal_steps = int(np.ceil(np.max(straight_route_lengths_m) / b.step_size_m))
        maximum_steps = max(1, int(np.ceil(nominal_steps * b.max_steps_factor)))
        delta_rad = np.deg2rad(b.deflection_angle_deg)
        tie_tolerance = 1.0e-12

        for step_number in range(maximum_steps):
            active_indices = np.flatnonzero(active)
            if len(active_indices) == 0:
                break

            active_positions = positions[active_indices]

            recovery_fraction = np.zeros(len(active_indices), dtype=float)
            local_manoeuvre = manoeuvre_active[active_indices]
            local_holding = holding_deflection[active_indices]
            recovery_mask = local_manoeuvre & ~local_holding
            recovery_fraction[local_holding] = 1.0
            recovery_fraction[recovery_mask] = np.clip(
                recovery_remaining_m[active_indices][recovery_mask]
                / b.return_length_m,
                0.0,
                1.0,
            )
            current_angles = (
                base_angles[active_indices]
                + manoeuvre_offset[active_indices] * recovery_fraction
            )
            current_directions = np.stack(
                (np.cos(current_angles), np.sin(current_angles)), axis=1
            )

            difference = turbines[None, :, :] - active_positions[:, None, :]
            distance_sq = np.sum(difference * difference, axis=2)
            ahead = np.sum(
                difference * current_directions[:, None, :], axis=2
            ) > 0.0
            eligible = (
                (distance_sq <= b.response_radius_m**2)
                & ahead
                & ~approach_processed[active_indices]
            )

            has_eligible = np.any(eligible, axis=1)
            if np.any(has_eligible):
                rows = np.flatnonzero(has_eligible)
                masked_distance = np.where(eligible[rows], distance_sq[rows], np.inf)
                targets = np.argmin(masked_distance, axis=1)
                birds = active_indices[rows]
                approach_processed[birds, targets] = True

                slots = approach_counts[birds]
                if np.any(slots >= population.maximum_response_events):
                    raise RuntimeError("Response random-number slots were exhausted.")
                approach_counts[birds] += 1

                responds = (
                    population.response_uniforms[birds, slots]
                    < b.response_probability
                )
                if np.any(responds):
                    responding_birds = birds[responds]
                    responding_targets = targets[responds]
                    responding_slots = slots[responds]
                    away = (
                        positions[responding_birds]
                        - turbines[responding_targets]
                    )

                    plus_angles = base_angles[responding_birds] + delta_rad
                    minus_angles = base_angles[responding_birds] - delta_rad
                    plus_directions = np.stack(
                        (np.cos(plus_angles), np.sin(plus_angles)), axis=1
                    )
                    minus_directions = np.stack(
                        (np.cos(minus_angles), np.sin(minus_angles)), axis=1
                    )
                    plus_score = np.sum(plus_directions * away, axis=1)
                    minus_score = np.sum(minus_directions * away, axis=1)

                    chosen_sign = np.where(plus_score > minus_score, 1.0, -1.0)
                    tied = np.abs(plus_score - minus_score) <= tie_tolerance
                    chosen_sign[tied] = np.where(
                        population.side_uniforms[
                            responding_birds[tied], responding_slots[tied]
                        ]
                        < 0.5,
                        -1.0,
                        1.0,
                    )

                    manoeuvre_active[responding_birds] = True
                    holding_deflection[responding_birds] = True
                    manoeuvre_target[responding_birds] = responding_targets
                    manoeuvre_offset[responding_birds] = chosen_sign * delta_rad
                    recovery_remaining_m[responding_birds] = b.return_length_m
                    response_matrix[responding_birds, responding_targets] = True

                # Recalculate the active headings because a subset may just
                # have initiated a manoeuvre at this position.
                recovery_fraction = np.zeros(len(active_indices), dtype=float)
                local_manoeuvre = manoeuvre_active[active_indices]
                local_holding = holding_deflection[active_indices]
                recovery_mask = local_manoeuvre & ~local_holding
                recovery_fraction[local_holding] = 1.0
                recovery_fraction[recovery_mask] = np.clip(
                    recovery_remaining_m[active_indices][recovery_mask]
                    / b.return_length_m,
                    0.0,
                    1.0,
                )
                current_angles = (
                    base_angles[active_indices]
                    + manoeuvre_offset[active_indices] * recovery_fraction
                )
                current_directions = np.stack(
                    (np.cos(current_angles), np.sin(current_angles)), axis=1
                )

            previous_positions = active_positions.copy()
            new_positions = previous_positions + b.step_size_m * current_directions

            if segment_observer is not None:
                segment_observer(
                    active_indices,
                    previous_positions,
                    new_positions,
                )

            segment_hits = self._segment_circle_hits(
                previous_positions,
                new_positions,
                turbines,
                b.encounter_radius_m,
            )
            new_hits = segment_hits & ~encounter_matrix[active_indices]
            encounter_matrix[active_indices] |= new_hits

            positions[active_indices] = new_positions
            route_lengths_m[active_indices] += b.step_size_m

            recovering_before_step = (
                manoeuvre_active[active_indices]
                & ~holding_deflection[active_indices]
            )
            if np.any(recovering_before_step):
                recovering_birds = active_indices[recovering_before_step]
                recovery_remaining_m[recovering_birds] = np.maximum(
                    recovery_remaining_m[recovering_birds] - b.step_size_m,
                    0.0,
                )
                finished_recovery = (
                    recovery_remaining_m[recovering_birds] <= 0.0
                )
                if np.any(finished_recovery):
                    finished_birds = recovering_birds[finished_recovery]
                    manoeuvre_active[finished_birds] = False
                    holding_deflection[finished_birds] = False
                    manoeuvre_target[finished_birds] = -1
                    manoeuvre_offset[finished_birds] = 0.0

            holding_now = (
                manoeuvre_active[active_indices]
                & holding_deflection[active_indices]
            )
            if np.any(holding_now):
                holding_birds = active_indices[holding_now]
                targets = manoeuvre_target[holding_birds]
                radial = positions[holding_birds] - turbines[targets]
                offset = manoeuvre_offset[holding_birds]
                directions = np.stack(
                    (
                        np.cos(base_angles[holding_birds] + offset),
                        np.sin(base_angles[holding_birds] + offset),
                    ),
                    axis=1,
                )
                moving_away = np.sum(radial * directions, axis=1) >= 0.0
                if np.any(moving_away):
                    released_birds = holding_birds[moving_away]
                    holding_deflection[released_birds] = False
                    recovery_remaining_m[released_birds] = b.return_length_m

            progress = np.sum(
                (positions[active_indices] - starts[active_indices])
                * base_directions[active_indices],
                axis=1,
            )
            just_completed = progress >= population.downstream_distance_m[active_indices]
            if np.any(just_completed):
                completed_birds = active_indices[just_completed]
                completed[completed_birds] = True
                active[completed_birds] = False

            if requested_paths and (
                step_number % path_stride == 0 or np.any(just_completed)
            ):
                for trajectory_index in requested_paths:
                    if np.any(active_indices == trajectory_index):
                        path_lists[trajectory_index].append(
                            positions[trajectory_index].copy()
                        )

        if np.any(active):
            failed[active] = True
            active[:] = False

        result = SimulationResult(
            per_turbine_encounters=np.sum(encounter_matrix, axis=0).astype(int),
            encounter_matrix=encounter_matrix,
            response_matrix=response_matrix,
            per_trajectory_encounters=np.sum(encounter_matrix, axis=1).astype(int),
            per_trajectory_responses=np.sum(response_matrix, axis=1).astype(int),
            approach_events_per_trajectory=approach_counts,
            route_lengths_m=route_lengths_m,
            straight_route_lengths_m=straight_route_lengths_m,
            completed_mask=completed,
            failed_mask=failed,
            stored_paths_m={
                key: np.asarray(value, dtype=float) for key, value in path_lists.items()
            },
        )
        result.validate_accounting()
        return result
