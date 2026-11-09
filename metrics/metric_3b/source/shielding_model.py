"""Deterministic directional shielding metric for wind-farm layouts.

The model projects every turbine centre onto the axis perpendicular to a
specified bird-flight heading.  Each projected rotor occupies an interval of
one rotor diameter.  Overlapping intervals are merged and counted only once.

This is a geometric exposure surrogate.  It does not calculate collision
probability, mortality, bird avoidance, or encounter intensity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray

# The interval calculation uses trigonometric projections, so geometrically
# identical layouts can differ at approximately 1e-15 in dimensionless N_eff.
# Rounding far below any meaningful spatial resolution prevents an optimiser
# from interpreting that machine noise as a genuine exposure improvement.
N_EFF_DECIMALS = 12


@dataclass(frozen=True)
class ShieldingResult:
    """Complete result for one layout and one bird-flight heading."""

    bird_heading_deg_toward: float
    rotor_diameter_m: float
    number_of_turbines: int
    direction_unit_vector: Array
    crossflight_unit_vector: Array
    projected_centres_m: Array
    rotor_intervals_m: Array
    merged_intervals_m: Array
    unique_exposed_width_m: float
    effective_exposed_rotor_count: float
    normalised_exposed_rotor_count: float
    shielding_fraction: float
    corridor_width_m: float | None
    encounter_prevalence: float | None
    free_path_fraction: float | None

    @property
    def N_eff(self) -> float:
        """Short mathematical name for effective exposed rotor count."""

        return self.effective_exposed_rotor_count

    @property
    def L_exp(self) -> float:
        """Short mathematical name for unique exposed width."""

        return self.unique_exposed_width_m

    @property
    def P_enc(self) -> float | None:
        """Short mathematical name for analytical encounter prevalence."""

        return self.encounter_prevalence

    def summary(self) -> dict[str, Any]:
        """Return the scalar result fields in a JSON-safe dictionary."""

        return {
            "bird_heading_deg_toward": self.bird_heading_deg_toward,
            "rotor_diameter_m": self.rotor_diameter_m,
            "number_of_turbines": self.number_of_turbines,
            "unique_exposed_width_m": self.unique_exposed_width_m,
            "N_eff": self.effective_exposed_rotor_count,
            "N_eff_over_N": self.normalised_exposed_rotor_count,
            "shielding_fraction": self.shielding_fraction,
            "corridor_width_m": self.corridor_width_m,
            "P_enc": self.encounter_prevalence,
            "P_free": self.free_path_fraction,
            "number_of_merged_intervals": int(len(self.merged_intervals_m)),
        }


def _validated_layout(xy: Any) -> Array:
    coordinates = np.asarray(xy, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "xy must be a two-dimensional array with shape "
            "(number_of_turbines, 2)."
        )
    if coordinates.shape[0] == 0:
        raise ValueError("At least one turbine is required.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Turbine coordinates must be finite.")
    return coordinates


def heading_unit_vector(heading_deg_toward: float) -> Array:
    """Return the unit vector in the mathematical TOWARD heading convention.

    ``0 degrees`` is east (+x) and ``90 degrees`` is north (+y).
    """

    heading = float(heading_deg_toward)
    if not np.isfinite(heading):
        raise ValueError("heading_deg_toward must be finite.")
    theta = np.deg2rad(heading)
    return np.array([np.cos(theta), np.sin(theta)], dtype=float)


def crossflight_unit_vector(heading_deg_toward: float) -> Array:
    """Return the unit vector perpendicular to the bird-flight heading."""

    direction = heading_unit_vector(heading_deg_toward)
    return np.array([-direction[1], direction[0]], dtype=float)


def project_crossflight_positions(
    xy: Any,
    bird_heading_deg_toward: float,
) -> Array:
    """Project turbine centres onto the axis perpendicular to bird flight."""

    coordinates = _validated_layout(xy)
    crossflight = crossflight_unit_vector(bird_heading_deg_toward)
    return np.asarray(coordinates @ crossflight, dtype=float)


def projected_rotor_intervals(
    projected_centres_m: Any,
    rotor_diameter_m: float,
) -> Array:
    """Create one width-``D`` interval around every projected centre."""

    centres = np.asarray(projected_centres_m, dtype=float)
    if centres.ndim != 1 or centres.size == 0:
        raise ValueError("projected_centres_m must be a non-empty 1-D array.")
    if not np.all(np.isfinite(centres)):
        raise ValueError("Projected turbine centres must be finite.")

    diameter = float(rotor_diameter_m)
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("rotor_diameter_m must be finite and positive.")

    half_diameter = 0.5 * diameter
    return np.column_stack((centres - half_diameter, centres + half_diameter))


def merge_intervals(intervals_m: Any, tolerance_m: float = 1.0e-9) -> Array:
    """Merge overlapping or touching intervals and return them in order."""

    intervals = np.asarray(intervals_m, dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals_m must have shape (number_of_intervals, 2).")
    if intervals.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    if not np.all(np.isfinite(intervals)):
        raise ValueError("Intervals must be finite.")
    if np.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError("Every interval end must be at or after its start.")

    tolerance = float(tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_m must be finite and non-negative.")

    ordered = intervals[np.argsort(intervals[:, 0], kind="mergesort")]
    merged: list[list[float]] = [[float(ordered[0, 0]), float(ordered[0, 1])]]
    for start, end in ordered[1:]:
        if float(start) <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], float(end))
        else:
            merged.append([float(start), float(end)])
    return np.asarray(merged, dtype=float)


def evaluate_shielding(
    xy: Any,
    *,
    rotor_diameter_m: float,
    bird_heading_deg_toward: float,
    corridor_width_m: float | None = None,
    interval_tolerance_m: float = 1.0e-9,
) -> ShieldingResult:
    """Evaluate directional geometric shielding for one turbine layout.

    Parameters
    ----------
    xy:
        Turbine centre coordinates in metres, with shape ``(N, 2)``.
    rotor_diameter_m:
        Width assigned to every projected rotor interval.
    bird_heading_deg_toward:
        Flight direction using 0=east and 90=north.  Reversing a heading by
        180 degrees gives the same shielding because this model depends on the
        directional axis, not which end of the axis birds enter from.
    corridor_width_m:
        Optional width ``W`` across which straight paths are assumed uniformly
        distributed.  When supplied, ``P_enc = L_exp / W`` and
        ``P_free = 1 - P_enc`` are returned.  The corridor is assumed to cover
        every projected rotor interval.
    interval_tolerance_m:
        Numerical tolerance used when deciding whether intervals touch.
    """

    coordinates = _validated_layout(xy)
    diameter = float(rotor_diameter_m)
    direction = heading_unit_vector(bird_heading_deg_toward)
    crossflight = crossflight_unit_vector(bird_heading_deg_toward)
    projected_centres = np.asarray(coordinates @ crossflight, dtype=float)
    intervals = projected_rotor_intervals(projected_centres, diameter)
    merged = merge_intervals(intervals, tolerance_m=interval_tolerance_m)
    raw_exposed_width = float(np.sum(merged[:, 1] - merged[:, 0]))

    number_of_turbines = int(len(coordinates))
    effective_count = float(
        np.round(raw_exposed_width / diameter, decimals=N_EFF_DECIMALS)
    )
    exposed_width = float(effective_count * diameter)
    normalised_count = float(effective_count / number_of_turbines)
    shielding_fraction = float(1.0 - normalised_count)

    corridor: float | None
    prevalence: float | None
    free_fraction: float | None
    if corridor_width_m is None:
        corridor = None
        prevalence = None
        free_fraction = None
    else:
        corridor = float(corridor_width_m)
        if not np.isfinite(corridor) or corridor <= 0.0:
            raise ValueError("corridor_width_m must be finite and positive.")
        if exposed_width > corridor + interval_tolerance_m:
            raise ValueError(
                "corridor_width_m is smaller than the unique exposed width; "
                "P_enc would exceed one."
            )
        prevalence = float(np.clip(exposed_width / corridor, 0.0, 1.0))
        free_fraction = float(1.0 - prevalence)

    return ShieldingResult(
        bird_heading_deg_toward=float(bird_heading_deg_toward) % 360.0,
        rotor_diameter_m=diameter,
        number_of_turbines=number_of_turbines,
        direction_unit_vector=direction,
        crossflight_unit_vector=crossflight,
        projected_centres_m=projected_centres,
        rotor_intervals_m=intervals,
        merged_intervals_m=merged,
        unique_exposed_width_m=exposed_width,
        effective_exposed_rotor_count=effective_count,
        normalised_exposed_rotor_count=normalised_count,
        shielding_fraction=shielding_fraction,
        corridor_width_m=corridor,
        encounter_prevalence=prevalence,
        free_path_fraction=free_fraction,
    )
