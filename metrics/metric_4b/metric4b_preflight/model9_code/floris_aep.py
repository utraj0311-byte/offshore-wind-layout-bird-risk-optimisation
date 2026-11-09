"""FLORIS-backed annual-energy calculation for Model 9.

The wind resource is the inherited Model 8 discretisation: 12 direction
sectors crossed with seven discrete Weibull-weighted speed bins.  The bundled
IEA 15 MW YAML is a provenance reference; FLORIS itself loads its packaged
``iea_15MW`` definition from the configuration file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import floris
from floris import FlorisModel

from config import Scenario


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "floris_config" / "gch_iea15mw.yaml"
DEFAULT_TURBINE_REFERENCE_PATH = (
    MODULE_DIR / "floris_config" / "iea_15MW_REFERENCE.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FLORISAEPModel:
    """Calculate total and per-turbine AEP with FLORIS v4.

    ``evaluate`` returns GWh/year.  Construction verifies that the Scenario,
    FLORIS' installed turbine, and the supplied reference turbine all describe
    the same hub height and rotor diameter.  This explicitly guards against
    the obsolete 240 m prototype value found in the earlier Python config.
    """

    def __init__(
        self,
        scenario: Scenario,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        turbine_reference_path: str | Path = DEFAULT_TURBINE_REFERENCE_PATH,
        installed_turbine_path: str | Path | None = None,
        turbulence_intensity: float = 0.06,
        hours_per_year: float = 8760.0,
        strict_turbine_reference: bool = True,
    ) -> None:
        self.s = scenario
        self.turbine = scenario.turbine
        self.site = scenario.site
        self.wind = scenario.wind
        self.config_path = Path(config_path).resolve()
        self.turbine_reference_path = Path(turbine_reference_path).resolve()

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"FLORIS configuration not found: {self.config_path}"
            )
        if not self.turbine_reference_path.is_file():
            raise FileNotFoundError(
                "Bundled turbine reference not found: "
                f"{self.turbine_reference_path}"
            )
        if not np.isfinite(turbulence_intensity) or turbulence_intensity < 0.0:
            raise ValueError("turbulence_intensity must be finite and non-negative.")
        if not np.isfinite(hours_per_year) or hours_per_year <= 0.0:
            raise ValueError("hours_per_year must be finite and positive.")

        self.turbulence_intensity = float(turbulence_intensity)
        self.hours_per_year = float(hours_per_year)
        self.strict_turbine_reference = bool(strict_turbine_reference)

        config_data = self._load_yaml(self.config_path)
        turbine_types = config_data.get("farm", {}).get("turbine_type", [])
        if not turbine_types:
            raise ValueError("No turbine_type found in the FLORIS configuration.")
        self.turbine_type = str(turbine_types[0])

        if installed_turbine_path is None:
            package_root = Path(floris.__file__).resolve().parent
            installed_turbine_path = (
                package_root / "turbine_library" / f"{self.turbine_type}.yaml"
            )
        self.installed_turbine_path = Path(installed_turbine_path).resolve()
        if not self.installed_turbine_path.is_file():
            raise FileNotFoundError(
                "Packaged FLORIS turbine file not found: "
                f"{self.installed_turbine_path}"
            )

        installed = self._load_yaml(self.installed_turbine_path)
        reference = self._load_yaml(self.turbine_reference_path)
        self.floris_hub_height_m = float(installed["hub_height"])
        self.floris_rotor_diameter_m = float(installed["rotor_diameter"])
        self._validate_turbine_identity(installed, reference)
        self._validate_scenario_geometry()

        self.floris_model = FlorisModel(self.config_path)
        (
            self.wind_directions_deg,
            self.wind_speeds_ms,
            self.condition_frequencies,
        ) = self._build_wind_conditions()
        self.turbulence_intensities = np.full(
            self.number_of_conditions,
            self.turbulence_intensity,
            dtype=float,
        )

        self.last_farm_aep_gwh_from_api: float | None = None
        self.last_total_aep_gwh_from_powers: float | None = None
        self.last_turbine_powers_shape: tuple[int, ...] | None = None
        self._single_turbine_aep_gwh: float | None = None

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a YAML mapping in {path}.")
        return value

    def _validate_turbine_identity(
        self,
        installed: dict[str, Any],
        reference: dict[str, Any],
    ) -> None:
        if str(installed.get("turbine_type")) != self.turbine_type:
            raise ValueError(
                "The installed FLORIS turbine name does not match the farm "
                f"configuration: {installed.get('turbine_type')!r} versus "
                f"{self.turbine_type!r}."
            )
        identity_keys = (
            "turbine_type",
            "hub_height",
            "rotor_diameter",
            "power_thrust_table",
        )
        differences = [
            key for key in identity_keys if installed.get(key) != reference.get(key)
        ]
        if self.strict_turbine_reference and differences:
            raise ValueError(
                "The installed FLORIS turbine differs from the bundled Model 8 "
                f"reference in: {', '.join(differences)}. Use FLORIS 4.6.5 or "
                "review the turbine change before running Model 9."
            )

    def _validate_scenario_geometry(self) -> None:
        checks = (
            (
                "rotor diameter",
                float(self.turbine.rotor_diameter_m),
                self.floris_rotor_diameter_m,
            ),
            (
                "hub height",
                float(self.turbine.hub_height_m),
                self.floris_hub_height_m,
            ),
        )
        for label, configured, installed in checks:
            if not np.isclose(configured, installed, rtol=0.0, atol=1.0e-9):
                raise ValueError(
                    f"Scenario {label} ({configured:g} m) does not match the "
                    f"installed FLORIS {self.turbine_type} value "
                    f"({installed:g} m)."
                )

    def _build_wind_conditions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        directions = np.arange(self.wind.n_sectors, dtype=float) * (
            360.0 / float(self.wind.n_sectors)
        )
        direction_frequencies = np.asarray(
            self.wind.sector_frequencies(self.site.wind_from_deg),
            dtype=float,
        )
        speeds = np.asarray(self.wind.speed_bins, dtype=float)
        if speeds.ndim != 1 or len(speeds) == 0:
            raise ValueError("speed_bins must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(speeds)) or np.any(speeds <= 0.0):
            raise ValueError("speed_bins must be finite and positive.")

        shape = float(self.wind.weibull_k)
        scale = float(self.wind.weibull_A)
        if not np.isfinite(shape) or not np.isfinite(scale) or shape <= 0 or scale <= 0:
            raise ValueError("Weibull shape and scale must be finite and positive.")
        speed_weights = (
            (shape / scale)
            * (speeds / scale) ** (shape - 1.0)
            * np.exp(-((speeds / scale) ** shape))
        )
        if not np.all(np.isfinite(speed_weights)) or speed_weights.sum() <= 0.0:
            raise ValueError("Discrete Weibull weights must be finite and positive.")
        speed_frequencies = speed_weights / speed_weights.sum()

        condition_directions = np.repeat(directions, len(speeds))
        condition_speeds = np.tile(speeds, len(directions))
        condition_frequencies = np.repeat(
            direction_frequencies, len(speeds)
        ) * np.tile(speed_frequencies, len(directions))
        if not np.isclose(condition_frequencies.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError("Combined wind-condition frequencies do not sum to one.")
        return (
            condition_directions.astype(float),
            condition_speeds.astype(float),
            condition_frequencies.astype(float),
        )

    @property
    def number_of_conditions(self) -> int:
        return int(len(self.condition_frequencies))

    @staticmethod
    def _validate_layout(xy: Any) -> np.ndarray:
        coordinates = np.asarray(xy, dtype=float).reshape(-1, 2)
        if len(coordinates) == 0:
            raise ValueError("At least one turbine is required.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Layout coordinates must be finite.")
        return coordinates

    def _run_layout(self, xy: Any) -> np.ndarray:
        coordinates = self._validate_layout(xy)
        n_turbines = len(coordinates)
        self.floris_model.set(
            layout_x=coordinates[:, 0],
            layout_y=coordinates[:, 1],
            turbine_type=[self.turbine_type],
            reference_wind_height=self.floris_hub_height_m,
            wind_directions=self.wind_directions_deg,
            wind_speeds=self.wind_speeds_ms,
            turbulence_intensities=self.turbulence_intensities,
        )
        self.floris_model.run()
        powers_raw = np.asarray(self.floris_model.get_turbine_powers(), dtype=float)
        self.last_turbine_powers_shape = tuple(powers_raw.shape)
        expected_size = self.number_of_conditions * n_turbines
        if powers_raw.size != expected_size:
            raise RuntimeError(
                "Unexpected FLORIS turbine-power array size: expected "
                f"{expected_size}, received {powers_raw.size} with shape "
                f"{powers_raw.shape}."
            )
        powers_w = powers_raw.reshape(self.number_of_conditions, n_turbines)
        if not np.all(np.isfinite(powers_w)) or np.any(powers_w < 0.0):
            raise RuntimeError("FLORIS returned invalid turbine powers.")

        farm_aep_wh = float(
            self.floris_model.get_farm_AEP(
                freq=self.condition_frequencies,
                hours_per_year=self.hours_per_year,
            )
        )
        self.last_farm_aep_gwh_from_api = farm_aep_wh / 1.0e9
        return powers_w

    def evaluate_per_turbine(self, xy: Any) -> np.ndarray:
        """Return turbine-level annual energy in GWh/year."""
        powers_w = self._run_layout(xy)
        weighted_mean_power_w = self.condition_frequencies @ powers_w
        aep_per_turbine_gwh = weighted_mean_power_w * self.hours_per_year / 1.0e9
        self.last_total_aep_gwh_from_powers = float(aep_per_turbine_gwh.sum())
        if self.last_farm_aep_gwh_from_api is not None and not np.isclose(
            self.last_total_aep_gwh_from_powers,
            self.last_farm_aep_gwh_from_api,
            rtol=1.0e-9,
            atol=1.0e-8,
        ):
            raise RuntimeError(
                "FLORIS AEP API and turbine-power integration disagree: "
                f"{self.last_farm_aep_gwh_from_api:.12g} versus "
                f"{self.last_total_aep_gwh_from_powers:.12g} GWh/year."
            )
        return np.asarray(aep_per_turbine_gwh, dtype=float)

    def evaluate(self, xy: Any) -> float:
        """Return total farm annual energy in GWh/year."""
        return float(self.evaluate_per_turbine(xy).sum())

    def ideal_aep(self, n_turbines: int) -> float:
        """Return a wake-free AEP ceiling from a one-turbine FLORIS run."""
        if n_turbines <= 0:
            raise ValueError("n_turbines must be positive.")
        if self._single_turbine_aep_gwh is None:
            self._single_turbine_aep_gwh = self.evaluate([[0.0, 0.0]])
        return float(n_turbines * self._single_turbine_aep_gwh)

    def settings_summary(self) -> dict[str, Any]:
        """Return serialisable inputs and provenance for a run archive."""
        return {
            "model": "FLORISAEPModel",
            "config_path": str(self.config_path),
            "config_sha256": _sha256(self.config_path),
            "turbine_type": self.turbine_type,
            "installed_turbine_path": str(self.installed_turbine_path),
            "installed_turbine_sha256": _sha256(self.installed_turbine_path),
            "bundled_turbine_reference_path": str(self.turbine_reference_path),
            "bundled_turbine_reference_sha256": _sha256(
                self.turbine_reference_path
            ),
            "strict_turbine_reference": self.strict_turbine_reference,
            "floris_hub_height_m": self.floris_hub_height_m,
            "floris_rotor_diameter_m": self.floris_rotor_diameter_m,
            "scenario_hub_height_m": float(self.turbine.hub_height_m),
            "scenario_rotor_diameter_m": float(self.turbine.rotor_diameter_m),
            "number_of_direction_sectors": int(self.wind.n_sectors),
            "number_of_speed_bins": int(len(self.wind.speed_bins)),
            "number_of_combined_conditions": self.number_of_conditions,
            "wind_directions_deg": self.wind_directions_deg.tolist(),
            "wind_speeds_ms": self.wind_speeds_ms.tolist(),
            "condition_frequencies": self.condition_frequencies.tolist(),
            "condition_frequency_sum": float(self.condition_frequencies.sum()),
            "dominant_wind_from_deg": float(self.site.wind_from_deg),
            "weibull_shape": float(self.wind.weibull_k),
            "weibull_scale_ms": float(self.wind.weibull_A),
            "turbulence_intensity": self.turbulence_intensity,
            "hours_per_year": self.hours_per_year,
        }
