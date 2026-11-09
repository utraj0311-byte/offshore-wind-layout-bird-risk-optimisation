"""Model 9: flight-height sampling and circular rotor-transit exposure.

The module extends the frozen Model 8 horizontal meso-response trajectories.
It assigns one constant flight height to each simulated trajectory and records
a rotor transit when a movement segment crosses a turbine's fixed rotor plane
inside the circular rotor-swept disk.

The original sampled transit count is retained for audit.  When the same
flight-height distribution is supplied to ``RotorTransitModel``, the module
also integrates that distribution exactly over each horizontal rotor crossing
to provide a lower-variance conditional-expectation estimator, ``E_RT_CE``.

Scientific boundary
-------------------
This is an exposure model.  A rotor transit is not a blade collision.  The
module deliberately contains no macro avoidance, micro avoidance, vertical
avoidance, blade rotation, bird dimensions, collision probability, or
mortality calculation.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from model8_meso_response import (
    BehaviourScenario,
    MesoResponseModel,
    SimulationResult,
    TrajectoryPopulation,
    cross_unit_vectors,
    heading_unit_vectors,
)


Array = np.ndarray


@dataclass(frozen=True)
class RotorCrossingBatch:
    """Independent snapshots of rotor-plane crossings in one movement step.

    Each array has one entry per bird-turbine crossing, in movement-step order
    and then the input bird/turbine order. ``trajectory_indices`` and
    ``turbine_indices`` refer to the unchanged input population and layout.
    ``crossing_xy_m`` and ``turbine_xy_m`` use the layout's Cartesian frame.
    ``lateral_offset_m`` is signed along ``cross_unit_vectors`` of the fixed
    plane normal (``rotor_plane_heading_deg_toward``); the crossing point lies
    on that plane. ``crossing_direction_sign`` is +1 along the plane normal
    and -1 against it. It does not encode upwind or downwind flight.

    The callback receives plane crossings within the rotor's horizontal
    radius, even if the sampled bird height is outside the circular disk.
    ``height_bins_m`` and ``height_probability_mass`` are the exact discrete
    Model 9 FHD when a conditional estimator is configured; otherwise both
    are None. They allow an independent local calculation at every supported
    height. The sampled height is provided separately for diagnostics.

    Inclusive segment endpoints can report the same crossing on adjacent
    steps. Accumulate at most one contribution per bird-turbine pair, using
    the maximum across its crossings, as the existing Model 9 tracker does.
    Array fields are read-only copies; immutable tuples hold FHD values.
    """

    trajectory_indices: Array
    turbine_indices: Array
    turbine_xy_m: Array
    crossing_xy_m: Array
    lateral_offset_m: Array
    crossing_direction_sign: Array
    segment_heading_deg_toward: Array
    sampled_flight_heights_m: Array
    rotor_plane_heading_deg_toward: float
    rotor_radius_m: float
    hub_height_m: float
    height_bins_m: tuple[float, ...] | None
    height_probability_mass: tuple[float, ...] | None

    def __post_init__(self) -> None:
        for name in (
            "trajectory_indices", "turbine_indices", "turbine_xy_m",
            "crossing_xy_m", "lateral_offset_m", "crossing_direction_sign",
            "segment_heading_deg_toward", "sampled_flight_heights_m",
        ):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class RotorGeometry:
    """Uniform circular rotor geometry used by every turbine in one run."""

    rotor_diameter_m: float
    hub_height_m: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.rotor_diameter_m) or self.rotor_diameter_m <= 0.0:
            raise ValueError("rotor_diameter_m must be finite and positive.")
        if not np.isfinite(self.hub_height_m) or self.hub_height_m <= 0.0:
            raise ValueError("hub_height_m must be finite and positive.")

    @property
    def rotor_radius_m(self) -> float:
        return 0.5 * float(self.rotor_diameter_m)

    @property
    def lower_tip_height_m(self) -> float:
        return float(self.hub_height_m - self.rotor_radius_m)

    @property
    def upper_tip_height_m(self) -> float:
        return float(self.hub_height_m + self.rotor_radius_m)

    def settings(self) -> dict[str, float]:
        return {
            "rotor_diameter_m": float(self.rotor_diameter_m),
            "rotor_radius_m": self.rotor_radius_m,
            "hub_height_m": float(self.hub_height_m),
            "lower_tip_height_m": self.lower_tip_height_m,
            "upper_tip_height_m": self.upper_tip_height_m,
        }


@dataclass(frozen=True)
class DiscreteFlightHeightDistribution:
    """Traceable tabulated flight-height probability distribution.

    Each row represents probability mass at one height.  The probabilities are
    normalised internally, but they must be non-negative and have a positive
    total.  Model 9 does not fit or infer a flight-height curve.
    """

    name: str
    heights_m: Array
    probability_mass: Array
    source: str | None = None
    input_schema: str | None = None
    input_sha256: str | None = None
    input_probability_mass_sum: float = field(init=False)

    def __post_init__(self) -> None:
        heights = np.asarray(self.heights_m, dtype=float)
        mass = np.asarray(self.probability_mass, dtype=float)
        if not self.name.strip():
            raise ValueError("Flight-height distribution name cannot be empty.")
        if heights.ndim != 1 or mass.ndim != 1 or heights.shape != mass.shape:
            raise ValueError("heights_m and probability_mass must be equal 1-D arrays.")
        if len(heights) == 0:
            raise ValueError("At least one flight-height row is required.")
        if not np.all(np.isfinite(heights)) or not np.all(np.isfinite(mass)):
            raise ValueError("Flight-height inputs must be finite.")
        if np.any(heights < 0.0):
            raise ValueError("Flight heights cannot be negative.")
        if np.any(np.diff(heights) <= 0.0):
            raise ValueError("Flight heights must be strictly increasing.")
        if np.any(mass < 0.0) or float(np.sum(mass)) <= 0.0:
            raise ValueError("Probability masses must be non-negative with positive sum.")

        input_mass_sum = float(np.sum(mass))
        normalised = mass / input_mass_sum
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "heights_m", heights.copy())
        object.__setattr__(self, "probability_mass", normalised)
        object.__setattr__(self, "input_probability_mass_sum", input_mass_sum)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        source: str | None = None,
    ) -> "DiscreteFlightHeightDistribution":
        """Load a tabulated probability-mass flight-height CSV.

        The canonical schema is ``height_m,probability_mass``. The native
        stochLAB/Johnston export schema ``height,prop`` is also accepted so
        that a source table can be retained byte-for-byte rather than edited.
        """
        csv_path = Path(path)
        heights: list[float] = []
        masses: list[float] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            if {"height_m", "probability_mass"}.issubset(fieldnames):
                height_column = "height_m"
                mass_column = "probability_mass"
                input_schema = "height_m,probability_mass"
            elif {"height", "prop"}.issubset(fieldnames):
                height_column = "height"
                mass_column = "prop"
                input_schema = "height,prop (stochLAB/Johnston)"
            else:
                raise ValueError(
                    "Flight-height CSV must contain either height_m and "
                    "probability_mass columns, or the stochLAB/Johnston "
                    "height and prop columns."
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    heights.append(float(row[height_column]))
                    masses.append(float(row[mass_column]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric flight-height value on CSV row {row_number}."
                    ) from exc

        order = np.argsort(np.asarray(heights, dtype=float))
        sorted_heights = np.asarray(heights, dtype=float)[order]
        sorted_mass = np.asarray(masses, dtype=float)[order]
        return cls(
            name=csv_path.stem if name is None else name,
            heights_m=sorted_heights,
            probability_mass=sorted_mass,
            source=source,
            input_schema=input_schema,
            input_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        )

    @property
    def mean_height_m(self) -> float:
        return float(np.sum(self.heights_m * self.probability_mass))

    def sample_from_uniforms(self, uniforms: Array) -> Array:
        """Map fixed U[0,1) draws to heights by inverse discrete CDF."""
        values = np.asarray(uniforms, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(
            (values < 0.0) | (values >= 1.0)
        ):
            raise ValueError("Flight-height uniforms must lie in [0, 1).")
        cumulative = np.cumsum(self.probability_mass)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, values, side="right")
        return self.heights_m[indices]

    def sample(self, count: int, *, seed: int) -> tuple[Array, Array]:
        if count <= 0:
            raise ValueError("Flight-height sample count must be positive.")
        rng = np.random.default_rng(int(seed))
        uniforms = rng.random(int(count))
        return self.sample_from_uniforms(uniforms), uniforms

    def probability_between(self, lower_m: Array, upper_m: Array) -> Array:
        """Return probability mass inside inclusive height bounds.

        Array inputs are broadcast together. This exact discrete-CDF lookup is
        used by the conditional-expectation estimator; it does not refit or
        smooth the supplied flight-height distribution.
        """
        lower, upper = np.broadcast_arrays(
            np.asarray(lower_m, dtype=float),
            np.asarray(upper_m, dtype=float),
        )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Flight-height probability bounds must be finite.")

        prefix_mass = np.concatenate(
            (np.array([0.0]), np.cumsum(self.probability_mass))
        )
        lower_indices = np.searchsorted(self.heights_m, lower, side="left")
        upper_indices = np.searchsorted(self.heights_m, upper, side="right")
        probability = prefix_mass[upper_indices] - prefix_mass[lower_indices]
        return np.where(upper >= lower, probability, 0.0)

    def settings(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "height_row_count": int(len(self.heights_m)),
            "minimum_height_m": float(np.min(self.heights_m)),
            "maximum_height_m": float(np.max(self.heights_m)),
            "mean_height_m": self.mean_height_m,
            "input_schema": self.input_schema,
            "input_sha256": self.input_sha256,
            "input_probability_mass_sum": self.input_probability_mass_sum,
            "probability_mass_sum": float(np.sum(self.probability_mass)),
        }


@dataclass(frozen=True)
class Model9Population:
    """One common-random-number population with a constant height per bird."""

    trajectories: TrajectoryPopulation
    flight_heights_m: Array
    height_uniforms: Array
    flight_height_seed: int
    flight_height_distribution_name: str

    def __post_init__(self) -> None:
        heights = np.asarray(self.flight_heights_m, dtype=float)
        uniforms = np.asarray(self.height_uniforms, dtype=float)
        expected = (self.trajectories.number_of_trajectories,)
        if heights.shape != expected or uniforms.shape != expected:
            raise ValueError("Flight heights and uniforms must have shape (N,).")
        if not np.all(np.isfinite(heights)) or np.any(heights < 0.0):
            raise ValueError("Flight heights must be finite and non-negative.")
        if not np.all(np.isfinite(uniforms)) or np.any(
            (uniforms < 0.0) | (uniforms >= 1.0)
        ):
            raise ValueError("Flight-height uniforms must lie in [0, 1).")
        if not self.flight_height_distribution_name.strip():
            raise ValueError("flight_height_distribution_name cannot be empty.")
        object.__setattr__(self, "flight_heights_m", heights.copy())
        object.__setattr__(self, "height_uniforms", uniforms.copy())

    @property
    def number_of_trajectories(self) -> int:
        return self.trajectories.number_of_trajectories

    @property
    def trajectory_seed(self) -> int | None:
        return self.trajectories.seed

    def subset(self, count: int) -> "Model9Population":
        return Model9Population(
            trajectories=self.trajectories.subset(count),
            flight_heights_m=self.flight_heights_m[:count],
            height_uniforms=self.height_uniforms[:count],
            flight_height_seed=self.flight_height_seed,
            flight_height_distribution_name=self.flight_height_distribution_name,
        )

    @classmethod
    def generate_parallel(
        cls,
        *,
        flight_height_distribution: DiscreteFlightHeightDistribution,
        flight_height_seed: int,
        site_width_m: float,
        site_height_m: float,
        heading_deg_toward: float,
        number_of_trajectories: int,
        maximum_response_events: int,
        longitudinal_margin_m: float,
        trajectory_seed: int,
        stratified_entries: bool = True,
    ) -> "Model9Population":
        trajectories = TrajectoryPopulation.generate_parallel(
            site_width_m=site_width_m,
            site_height_m=site_height_m,
            heading_deg_toward=heading_deg_toward,
            number_of_trajectories=number_of_trajectories,
            maximum_response_events=maximum_response_events,
            longitudinal_margin_m=longitudinal_margin_m,
            seed=trajectory_seed,
            stratified_entries=stratified_entries,
        )
        heights, uniforms = flight_height_distribution.sample(
            number_of_trajectories,
            seed=flight_height_seed,
        )
        return cls(
            trajectories=trajectories,
            flight_heights_m=heights,
            height_uniforms=uniforms,
            flight_height_seed=int(flight_height_seed),
            flight_height_distribution_name=flight_height_distribution.name,
        )


def _segment_rotor_plane_crossings(
    segment_start_xy_m: Array,
    segment_end_xy_m: Array,
    turbine_xy_m: Array,
    rotor_plane_heading_deg_toward: float,
    *,
    tolerance_m: float = 1.0e-9,
) -> tuple[Array, Array]:
    """Return plane-crossing mask and lateral offset for every segment/turbine."""
    starts = np.asarray(segment_start_xy_m, dtype=float).reshape(-1, 2)
    ends = np.asarray(segment_end_xy_m, dtype=float).reshape(-1, 2)
    turbines = np.asarray(turbine_xy_m, dtype=float).reshape(-1, 2)
    if starts.shape != ends.shape:
        raise ValueError("Segment start and end arrays must have the same shape.")
    if len(turbines) == 0:
        raise ValueError("At least one turbine is required.")
    if tolerance_m < 0.0 or not np.isfinite(tolerance_m):
        raise ValueError("tolerance_m must be finite and non-negative.")
    if not (
        np.all(np.isfinite(starts))
        and np.all(np.isfinite(ends))
        and np.all(np.isfinite(turbines))
    ):
        raise ValueError("Rotor-plane geometry inputs must be finite.")

    normal = heading_unit_vectors(float(rotor_plane_heading_deg_toward))
    cross = cross_unit_vectors(float(rotor_plane_heading_deg_toward))
    relative_start = starts[:, None, :] - turbines[None, :, :]
    relative_end = ends[:, None, :] - turbines[None, :, :]
    signed_start = np.sum(relative_start * normal, axis=2)
    signed_end = np.sum(relative_end * normal, axis=2)
    denominator = signed_end - signed_start

    non_parallel = np.abs(denominator) > tolerance_m
    crossing_fraction = np.zeros_like(denominator)
    np.divide(
        -signed_start,
        denominator,
        out=crossing_fraction,
        where=non_parallel,
    )
    fraction_tolerance = 1.0e-12
    crosses_plane = (
        non_parallel
        & (crossing_fraction >= -fraction_tolerance)
        & (crossing_fraction <= 1.0 + fraction_tolerance)
    )
    crossing_fraction = np.clip(crossing_fraction, 0.0, 1.0)
    segment = ends - starts
    crossing_xy = (
        starts[:, None, :]
        + crossing_fraction[:, :, None] * segment[:, None, :]
    )
    lateral_offset_m = np.sum(
        (crossing_xy - turbines[None, :, :]) * cross,
        axis=2,
    )
    return crosses_plane, lateral_offset_m


def _inside_rotor_disk(
    crosses_plane: Array,
    lateral_offset_m: Array,
    flight_heights_m: Array,
    rotor_geometry: RotorGeometry,
    *,
    tolerance_m: float,
) -> Array:
    heights = np.asarray(flight_heights_m, dtype=float)
    if heights.shape != (crosses_plane.shape[0],):
        raise ValueError("flight_heights_m must have one value per segment.")
    if not np.all(np.isfinite(heights)):
        raise ValueError("Flight heights must be finite.")
    if np.any(heights < 0.0):
        raise ValueError("Flight heights cannot be negative.")

    vertical_offset_m = heights[:, None] - rotor_geometry.hub_height_m
    radius_with_tolerance_m = rotor_geometry.rotor_radius_m + tolerance_m
    inside_disk = (
        lateral_offset_m**2 + vertical_offset_m**2
        <= radius_with_tolerance_m**2
    )
    return crosses_plane & inside_disk


def _conditional_disk_probabilities(
    crosses_plane: Array,
    lateral_offset_m: Array,
    rotor_geometry: RotorGeometry,
    flight_height_distribution: DiscreteFlightHeightDistribution,
    *,
    tolerance_m: float,
) -> Array:
    radius_m = rotor_geometry.rotor_radius_m
    within_lateral_extent = np.abs(lateral_offset_m) <= radius_m + tolerance_m
    vertical_half_width_m = np.sqrt(
        np.maximum(0.0, radius_m**2 - lateral_offset_m**2)
    )
    lower_m = rotor_geometry.hub_height_m - vertical_half_width_m
    upper_m = rotor_geometry.hub_height_m + vertical_half_width_m
    probability = flight_height_distribution.probability_between(lower_m, upper_m)
    return np.where(crosses_plane & within_lateral_extent, probability, 0.0)


def segment_rotor_disk_transits(
    segment_start_xy_m: Array,
    segment_end_xy_m: Array,
    turbine_xy_m: Array,
    flight_heights_m: Array,
    rotor_geometry: RotorGeometry,
    rotor_plane_heading_deg_toward: float,
    *,
    tolerance_m: float = 1.0e-9,
) -> Array:
    """Return the sampled bird-by-turbine circular-disc transit mask."""
    crosses_plane, lateral_offset_m = _segment_rotor_plane_crossings(
        segment_start_xy_m,
        segment_end_xy_m,
        turbine_xy_m,
        rotor_plane_heading_deg_toward,
        tolerance_m=tolerance_m,
    )
    return _inside_rotor_disk(
        crosses_plane,
        lateral_offset_m,
        flight_heights_m,
        rotor_geometry,
        tolerance_m=tolerance_m,
    )


def segment_rotor_disk_transit_probabilities(
    segment_start_xy_m: Array,
    segment_end_xy_m: Array,
    turbine_xy_m: Array,
    rotor_geometry: RotorGeometry,
    rotor_plane_heading_deg_toward: float,
    flight_height_distribution: DiscreteFlightHeightDistribution,
    *,
    tolerance_m: float = 1.0e-9,
) -> Array:
    """Return conditional rotor-transit probabilities for horizontal segments.

    Flight height is integrated out exactly over the supplied discrete FHD.
    The returned value has the same expectation as the sampled transit
    indicator but avoids rare-height Monte Carlo noise.
    """
    crosses_plane, lateral_offset_m = _segment_rotor_plane_crossings(
        segment_start_xy_m,
        segment_end_xy_m,
        turbine_xy_m,
        rotor_plane_heading_deg_toward,
        tolerance_m=tolerance_m,
    )
    return _conditional_disk_probabilities(
        crosses_plane,
        lateral_offset_m,
        rotor_geometry,
        flight_height_distribution,
        tolerance_m=tolerance_m,
    )


class _RotorTransitTracker:
    """Accumulate at most one circular-disc transit per bird-turbine pair."""

    def __init__(
        self,
        *,
        turbine_xy_m: Array,
        flight_heights_m: Array,
        rotor_geometry: RotorGeometry,
        rotor_plane_heading_deg_toward: float,
        number_of_trajectories: int,
        flight_height_distribution: DiscreteFlightHeightDistribution | None,
    ) -> None:
        self.turbine_xy_m = np.asarray(turbine_xy_m, dtype=float).reshape(-1, 2)
        self.flight_heights_m = np.asarray(flight_heights_m, dtype=float)
        self.rotor_geometry = rotor_geometry
        self.rotor_plane_heading_deg_toward = float(
            rotor_plane_heading_deg_toward
        )
        self.transit_matrix = np.zeros(
            (number_of_trajectories, len(self.turbine_xy_m)), dtype=bool
        )
        self.flight_height_distribution = flight_height_distribution
        self.expected_transit_probability_matrix = (
            None
            if flight_height_distribution is None
            else np.zeros(
                (number_of_trajectories, len(self.turbine_xy_m)), dtype=float
            )
        )

    def observe(
        self,
        trajectory_indices: Array,
        segment_start_xy_m: Array,
        segment_end_xy_m: Array,
    ) -> None:
        bird_indices = np.asarray(trajectory_indices, dtype=int)
        crosses_plane, lateral_offset_m = _segment_rotor_plane_crossings(
            segment_start_xy_m,
            segment_end_xy_m,
            self.turbine_xy_m,
            self.rotor_plane_heading_deg_toward,
        )
        hits = _inside_rotor_disk(
            crosses_plane,
            lateral_offset_m,
            self.flight_heights_m[bird_indices],
            self.rotor_geometry,
            tolerance_m=1.0e-9,
        )
        not_previously_counted = ~self.transit_matrix[bird_indices]
        self.transit_matrix[bird_indices] |= hits & not_previously_counted

        if self.expected_transit_probability_matrix is not None:
            assert self.flight_height_distribution is not None
            probabilities = _conditional_disk_probabilities(
                crosses_plane,
                lateral_offset_m,
                self.rotor_geometry,
                self.flight_height_distribution,
                tolerance_m=1.0e-9,
            )
            current = self.expected_transit_probability_matrix[bird_indices]
            self.expected_transit_probability_matrix[bird_indices] = np.maximum(
                current,
                probabilities,
            )


@dataclass
class Model9SimulationResult:
    """Joint Model 8 encounter and Model 9 rotor-transit result."""

    meso_result: SimulationResult
    flight_heights_m: Array
    transit_matrix: Array
    expected_transit_probability_matrix: Array | None
    rotor_geometry: RotorGeometry
    rotor_plane_heading_deg_toward: float

    @property
    def number_started(self) -> int:
        return self.meso_result.number_started

    @property
    def number_completed(self) -> int:
        return self.meso_result.number_completed

    @property
    def number_failed(self) -> int:
        return self.meso_result.number_failed

    @property
    def per_turbine_transits(self) -> Array:
        return np.sum(self.transit_matrix, axis=0).astype(int)

    @property
    def per_trajectory_transits(self) -> Array:
        return np.sum(self.transit_matrix, axis=1).astype(int)

    @property
    def total_transits(self) -> int:
        return int(np.sum(self.transit_matrix))

    @property
    def rotor_transit_intensity(self) -> float:
        """Mean rotor transits per entering trajectory; not collision risk."""
        return float(self.total_transits / self.number_started)

    @property
    def rotor_transit_prevalence(self) -> float:
        return float(
            np.count_nonzero(self.per_trajectory_transits >= 1)
            / self.number_started
        )

    @property
    def has_conditional_estimator(self) -> bool:
        return self.expected_transit_probability_matrix is not None

    def _expected_matrix(self) -> Array:
        if self.expected_transit_probability_matrix is None:
            raise RuntimeError(
                "Conditional rotor-transit estimates require a flight-height "
                "distribution on RotorTransitModel."
            )
        return self.expected_transit_probability_matrix

    @property
    def per_turbine_expected_transits(self) -> Array:
        return np.sum(self._expected_matrix(), axis=0)

    @property
    def per_trajectory_expected_transits(self) -> Array:
        return np.sum(self._expected_matrix(), axis=1)

    @property
    def expected_total_transits(self) -> float:
        return float(np.sum(self._expected_matrix()))

    @property
    def conditional_rotor_transit_intensity(self) -> float:
        """Conditional-expectation estimate of mean rotor transits per bird."""
        return float(self.expected_total_transits / self.number_started)

    @property
    def conditional_rotor_transit_prevalence(self) -> float:
        """Expected fraction with one or more transits under uniform geometry."""
        return float(np.mean(np.max(self._expected_matrix(), axis=1)))

    @property
    def rotor_height_band_prevalence(self) -> float:
        within = (
            (self.flight_heights_m >= self.rotor_geometry.lower_tip_height_m)
            & (self.flight_heights_m <= self.rotor_geometry.upper_tip_height_m)
        )
        return float(np.count_nonzero(within) / self.number_started)

    def validate_accounting(self) -> None:
        self.meso_result.validate_accounting()
        expected_shape = self.meso_result.encounter_matrix.shape
        if self.transit_matrix.shape != expected_shape:
            raise RuntimeError("Transit and encounter matrices have different shapes.")
        if self.flight_heights_m.shape != (self.number_started,):
            raise RuntimeError("Flight-height and trajectory counts disagree.")
        if self.total_transits != int(np.sum(self.per_turbine_transits)):
            raise RuntimeError("Transit matrix and turbine totals disagree.")
        if self.total_transits != int(np.sum(self.per_trajectory_transits)):
            raise RuntimeError("Transit matrix and trajectory totals disagree.")
        if self.expected_transit_probability_matrix is not None:
            expected = self.expected_transit_probability_matrix
            if expected.shape != expected_shape:
                raise RuntimeError(
                    "Conditional and sampled transit matrices have different shapes."
                )
            if not np.all(np.isfinite(expected)) or np.any(
                (expected < 0.0) | (expected > 1.0 + 1.0e-12)
            ):
                raise RuntimeError(
                    "Conditional transit probabilities must lie in [0, 1]."
                )

    def summary(self) -> dict[str, Any]:
        self.validate_accounting()
        meso = self.meso_result.summary()
        result = {
            **meso,
            "N_RT_farm": self.total_transits,
            "E_RT": self.rotor_transit_intensity,
            "P_RT": self.rotor_transit_prevalence,
            "P_height_band": self.rotor_height_band_prevalence,
            "mean_flight_height_m": float(np.mean(self.flight_heights_m)),
            "minimum_flight_height_m": float(np.min(self.flight_heights_m)),
            "maximum_flight_height_m": float(np.max(self.flight_heights_m)),
            "maximum_transits_per_trajectory": int(
                np.max(self.per_trajectory_transits, initial=0)
            ),
            "per_turbine_transits": self.per_turbine_transits.tolist(),
        }
        if self.has_conditional_estimator:
            result.update(
                {
                    "N_RT_CE_farm": self.expected_total_transits,
                    "E_RT_CE": self.conditional_rotor_transit_intensity,
                    "P_RT_CE": self.conditional_rotor_transit_prevalence,
                    "per_turbine_expected_transits": (
                        self.per_turbine_expected_transits.tolist()
                    ),
                }
            )
        return result


class RotorTransitModel:
    """Run frozen Model 8 behaviour while passively recording rotor transits."""

    def __init__(
        self,
        behaviour: BehaviourScenario,
        rotor_geometry: RotorGeometry,
        *,
        rotor_plane_heading_deg_toward: float,
        flight_height_distribution: DiscreteFlightHeightDistribution | None = None,
    ) -> None:
        self.behaviour = behaviour
        self.rotor_geometry = rotor_geometry
        self.rotor_plane_heading_deg_toward = float(
            rotor_plane_heading_deg_toward
        ) % 360.0
        self.flight_height_distribution = flight_height_distribution
        self.meso_model = MesoResponseModel(behaviour)

    def simulate(
        self,
        turbine_xy_m: Array,
        population: Model9Population,
        *,
        store_path_indices: Iterable[int] = (),
        path_stride: int = 1,
        crossing_callback: Callable[[RotorCrossingBatch], None] | None = None,
    ) -> Model9SimulationResult:
        """Run Model 8/9, optionally reporting copied rotor-plane crossings.

        The optional callback observes crossings only. It cannot supply
        response decisions or replace Model 9's original exposure tracker.
        """
        turbines = np.asarray(turbine_xy_m, dtype=float).reshape(-1, 2)
        population_headings = population.trajectories.headings_deg_toward
        heading_difference = (
            population_headings
            - self.rotor_plane_heading_deg_toward
            + 180.0
        ) % 360.0 - 180.0
        if np.any(np.abs(heading_difference) > 1.0e-9):
            raise ValueError(
                "The stripped-down Model 9 requires the fixed rotor-plane "
                "normal to match every trajectory's reference heading."
            )
        if (
            self.flight_height_distribution is not None
            and population.flight_height_distribution_name
            != self.flight_height_distribution.name
        ):
            raise ValueError(
                "Population and conditional-estimator flight-height "
                "distribution names do not match."
            )
        tracker = _RotorTransitTracker(
            turbine_xy_m=turbines,
            flight_heights_m=population.flight_heights_m,
            rotor_geometry=self.rotor_geometry,
            rotor_plane_heading_deg_toward=self.rotor_plane_heading_deg_toward,
            number_of_trajectories=population.number_of_trajectories,
            flight_height_distribution=self.flight_height_distribution,
        )
        observer = tracker.observe
        if crossing_callback is not None:
            distribution = self.flight_height_distribution
            height_bins = (
                None if distribution is None
                else tuple(float(x) for x in distribution.heights_m)
            )
            height_mass = (
                None if distribution is None
                else tuple(float(x) for x in distribution.probability_mass)
            )
            heading = self.rotor_plane_heading_deg_toward
            normal = heading_unit_vectors(heading)
            lateral = cross_unit_vectors(heading)
            radius = self.rotor_geometry.rotor_radius_m

            def observe_with_crossings(
                trajectory_indices: Array,
                segment_start_xy_m: Array,
                segment_end_xy_m: Array,
            ) -> None:
                # Keep original exposure accounting first, with identical inputs.
                tracker.observe(
                    trajectory_indices, segment_start_xy_m, segment_end_xy_m
                )
                plane_mask, offsets = _segment_rotor_plane_crossings(
                    segment_start_xy_m,
                    segment_end_xy_m,
                    turbines,
                    heading,
                )
                rows, columns = np.nonzero(
                    plane_mask & (np.abs(offsets) <= radius + 1.0e-9)
                )
                if len(rows) == 0:
                    return
                segment = segment_end_xy_m[rows] - segment_start_xy_m[rows]
                offset = offsets[rows, columns]
                crossing_callback(RotorCrossingBatch(
                    trajectory_indices=trajectory_indices[rows],
                    turbine_indices=columns,
                    turbine_xy_m=turbines[columns],
                    crossing_xy_m=turbines[columns] + offset[:, None] * lateral,
                    lateral_offset_m=offset,
                    crossing_direction_sign=np.where(segment @ normal > 0.0, 1, -1),
                    segment_heading_deg_toward=(
                        np.degrees(np.arctan2(segment[:, 1], segment[:, 0])) % 360.0
                    ),
                    sampled_flight_heights_m=(
                        population.flight_heights_m[trajectory_indices[rows]]
                    ),
                    rotor_plane_heading_deg_toward=heading,
                    rotor_radius_m=radius,
                    hub_height_m=self.rotor_geometry.hub_height_m,
                    height_bins_m=height_bins,
                    height_probability_mass=height_mass,
                ))

            observer = observe_with_crossings
        meso_result = self.meso_model.simulate(
            turbines,
            population.trajectories,
            store_path_indices=store_path_indices,
            path_stride=path_stride,
            segment_observer=observer,
        )
        result = Model9SimulationResult(
            meso_result=meso_result,
            flight_heights_m=population.flight_heights_m.copy(),
            transit_matrix=tracker.transit_matrix,
            expected_transit_probability_matrix=(
                tracker.expected_transit_probability_matrix
            ),
            rotor_geometry=self.rotor_geometry,
            rotor_plane_heading_deg_toward=self.rotor_plane_heading_deg_toward,
        )
        result.validate_accounting()

        if self.behaviour.encounter_radius_m + 1.0e-9 >= self.rotor_geometry.rotor_radius_m:
            impossible = result.transit_matrix & ~meso_result.encounter_matrix
            if np.any(impossible):
                raise RuntimeError(
                    "A rotor transit was not also counted as a Model 8 encounter."
                )
            if result.has_conditional_estimator:
                impossible_expected = (
                    result._expected_matrix() > 0.0
                ) & ~meso_result.encounter_matrix
                if np.any(impossible_expected):
                    raise RuntimeError(
                        "A positive conditional rotor-transit probability was "
                        "not also counted as a Model 8 encounter."
                    )
        return result
