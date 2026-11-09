"""Shared physical, wind-resource, behavioural, and optimiser configuration.

The wind rose is intentionally generated in code. It contains 12 direction
sectors and seven discrete Weibull-weighted speed bins, matching the supplied
Model 8 FLORIS workflow; no wind-input CSV is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Turbine:
    """Official FLORIS IEA 15 MW geometry used throughout Model 9."""

    rotor_diameter_m: float = 242.24
    hub_height_m: float = 150.0
    rated_power_mw: float = 15.0
    cut_in_ms: float = 3.0
    rated_ms: float = 10.6
    cut_out_ms: float = 25.0


@dataclass
class Site:
    """Rectangular offshore wind-farm design domain."""

    width_m: float = 6000.0
    height_m: float = 6000.0
    n_turbines: int = 12
    min_spacing_D: float = 4.0
    wind_from_deg: float = 225.0


@dataclass
class WindResource:
    """Programmatic directional and discrete Weibull wind weights."""

    n_sectors: int = 12
    weibull_k: float = 2.0
    weibull_A: float = 10.0
    speed_bins: np.ndarray = field(
        default_factory=lambda: np.array(
            [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            dtype=float,
        )
    )

    def sector_frequencies(self, dominant_from_deg: float) -> np.ndarray:
        """Return normalised von-Mises-like weights for equal sectors."""
        if self.n_sectors <= 0:
            raise ValueError("n_sectors must be positive.")
        directions = np.arange(self.n_sectors, dtype=float) * (
            360.0 / self.n_sectors
        )
        angle_diff = np.radians(
            (directions - float(dominant_from_deg) + 180.0) % 360.0 - 180.0
        )
        weights = np.exp(2.0 * np.cos(angle_diff))
        return weights / weights.sum()


@dataclass
class BirdParams:
    """Frozen Model 9 meso-response and flight-direction assumptions."""

    heading_deg_toward: float = 225.0
    response_probability: float = 0.6
    response_radius_D: float = 2.0
    encounter_radius_D: float = 1.0
    deflection_angle_deg: float = 45.0
    return_length_D: float = 8.0
    step_size_D: float = 0.20
    maximum_response_events: int = 12


@dataclass
class OptSettings:
    """Full-run settings inherited from the supplied Model 8 evidence run."""

    pop_size: int = 80
    n_gen: int = 100
    seed: int = 42
    objective_batch_count: int = 5
    objective_trajectories_per_batch: int = 2000
    fresh_batch_count: int = 3
    fresh_trajectories_per_batch: int = 10000


@dataclass
class Scenario:
    """One complete Model 9 optimisation scenario."""

    turbine: Turbine = field(default_factory=Turbine)
    site: Site = field(default_factory=Site)
    wind: WindResource = field(default_factory=WindResource)
    birds: BirdParams = field(default_factory=BirdParams)
    opt: OptSettings = field(default_factory=OptSettings)
