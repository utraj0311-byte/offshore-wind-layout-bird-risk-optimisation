from dataclasses import dataclass, field
import numpy as np


@dataclass
class Turbine:
    """Simplified IEA 15 MW reference turbine assumptions."""
    rotor_diameter_m: float = 240.0
    hub_height_m: float = 150.0
    rated_power_mw: float = 15.0
    cut_in_ms: float = 3.0
    rated_ms: float = 10.6
    cut_out_ms: float = 25.0


@dataclass
class Site:
    """Simple rectangular offshore wind farm site."""
    width_m: float = 6000.0
    height_m: float = 6000.0
    n_turbines: int = 12
    min_spacing_D: float = 4.0

    # Main wind direction and bird migration axis.
    # 225° means southwest-to-northeast style alignment.
    wind_from_deg: float = 225.0
    migration_axis_deg: float = 225.0


@dataclass
class WindResource:
    """Synthetic wind rose for first prototype."""
    n_sectors: int = 12
    weibull_k: float = 2.0
    weibull_A: float = 10.0
    speed_bins: np.ndarray = field(
        default_factory=lambda: np.array([4, 6, 8, 10, 12, 14, 16], dtype=float)
    )

    def sector_frequencies(self, dominant_from_deg: float) -> np.ndarray:
        """
        Creates a simple directional wind distribution.
        More frequent winds come from the dominant direction.
        """
        directions = np.arange(self.n_sectors) * 360.0 / self.n_sectors
        angle_diff = np.radians((directions - dominant_from_deg + 180) % 360 - 180)

        # von-Mises-like directional weighting
        weights = np.exp(2.0 * np.cos(angle_diff))
        return weights / weights.sum()


@dataclass
class BirdParams:
    """Placeholder bird parameters for prototype testing."""
    bird_density_per_km: float = 80.0
    flight_height_overlap: float = 0.25
    avoidance_rate: float = 0.98
    migration_width_m: float = 6000.0
    escape_distance_m: float = 600.0

    # This can be changed later:
    # "uniform", "edge_flux", or "trajectory"
    bird_model: str = "trajectory"


@dataclass
class OptSettings:
    """NSGA-II optimisation settings."""
    pop_size: int = 30
    n_gen: int = 20
    seed: int = 42


@dataclass
class Scenario:
    """One complete model scenario."""
    turbine: Turbine = field(default_factory=Turbine)
    site: Site = field(default_factory=Site)
    wind: WindResource = field(default_factory=WindResource)
    birds: BirdParams = field(default_factory=BirdParams)
    opt: OptSettings = field(default_factory=OptSettings)