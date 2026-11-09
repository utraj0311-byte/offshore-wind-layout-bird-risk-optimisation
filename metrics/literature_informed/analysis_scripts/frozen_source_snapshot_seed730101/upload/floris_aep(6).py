from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from floris import FlorisModel

from config import Scenario


class FLORISAEPModel:
    """
    FLORIS-backed AEP model with the same public interface as AEPModel.

    Public methods
    --------------
    evaluate(xy)
        Total farm AEP in GWh/year.

    evaluate_per_turbine(xy)
        Turbine-level AEP in GWh/year.

    ideal_aep(n_turbines)
        Wake-free AEP ceiling in GWh/year, calculated as
        n_turbines times the one-turbine AEP under the same wind resource.

    Notes
    -----
    The wind-resource discretisation and probabilities intentionally match
    the existing Jensen prototype:
    - 12 equally spaced direction sectors by default;
    - the existing directional sector-frequency function;
    - the existing discrete Weibull speed-bin weighting.

    The official packaged FLORIS IEA 15 MW turbine is retained unchanged.
    Its rotor diameter is 242.24 m, whereas the Jensen prototype uses 240 m.
    """

    def __init__(
        self,
        scenario: Scenario,
        config_path: str | Path = (
            Path("floris_config") / "gch_iea15mw.yaml"
        ),
        turbulence_intensity: float = 0.06,
        hours_per_year: float = 8760.0,
    ) -> None:
        self.s = scenario
        self.turbine = scenario.turbine
        self.site = scenario.site
        self.wind = scenario.wind

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"FLORIS configuration not found: {self.config_path}"
            )

        if turbulence_intensity < 0.0:
            raise ValueError(
                "turbulence_intensity must be non-negative."
            )

        if hours_per_year <= 0.0:
            raise ValueError(
                "hours_per_year must be positive."
            )

        self.turbulence_intensity = float(
            turbulence_intensity
        )
        self.hours_per_year = float(
            hours_per_year
        )

        self.floris_model = FlorisModel(
            self.config_path
        )

        (
            self.turbine_type,
            self.floris_hub_height_m,
            self.floris_rotor_diameter_m,
        ) = self._read_turbine_metadata()

        (
            self.wind_directions_deg,
            self.wind_speeds_ms,
            self.condition_frequencies,
        ) = self._build_wind_conditions()

        self.turbulence_intensities = np.full(
            self.wind_directions_deg.shape,
            self.turbulence_intensity,
            dtype=float,
        )

        self.last_farm_aep_gwh_from_api: (
            float | None
        ) = None

        self.last_total_aep_gwh_from_powers: (
            float | None
        ) = None

        self.last_turbine_powers_shape: (
            tuple[int, ...] | None
        ) = None

        self._single_turbine_aep_gwh: (
            float | None
        ) = None

    def _read_turbine_metadata(
        self,
    ) -> tuple[str, float, float]:
        config = yaml.safe_load(
            self.config_path.read_text(
                encoding="utf-8"
            )
        )

        farm = config.get("farm", {})
        turbine_types = farm.get(
            "turbine_type",
            [],
        )

        if not turbine_types:
            raise ValueError(
                "No turbine_type found in the FLORIS configuration."
            )

        turbine_type = str(
            turbine_types[0]
        )

        import floris

        package_root = (
            Path(floris.__file__)
            .resolve()
            .parent
        )

        turbine_path = (
            package_root
            / "turbine_library"
            / f"{turbine_type}.yaml"
        )

        if not turbine_path.exists():
            raise FileNotFoundError(
                "Packaged FLORIS turbine file not found: "
                f"{turbine_path}"
            )

        turbine_data = yaml.safe_load(
            turbine_path.read_text(
                encoding="utf-8"
            )
        )

        return (
            turbine_type,
            float(turbine_data["hub_height"]),
            float(turbine_data["rotor_diameter"]),
        )

    def _build_wind_conditions(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        directions = (
            np.arange(
                self.wind.n_sectors,
                dtype=float,
            )
            * 360.0
            / float(self.wind.n_sectors)
        )

        direction_frequencies = np.asarray(
            self.wind.sector_frequencies(
                self.site.wind_from_deg
            ),
            dtype=float,
        )

        speeds = np.asarray(
            self.wind.speed_bins,
            dtype=float,
        )

        k = float(
            self.wind.weibull_k
        )
        weibull_scale = float(
            self.wind.weibull_A
        )

        speed_pdf = (
            (k / weibull_scale)
            * (speeds / weibull_scale)
            ** (k - 1.0)
            * np.exp(
                -(
                    speeds / weibull_scale
                )
                ** k
            )
        )

        if not np.all(
            np.isfinite(speed_pdf)
        ):
            raise ValueError(
                "Non-finite discrete Weibull weights."
            )

        if speed_pdf.sum() <= 0.0:
            raise ValueError(
                "Discrete Weibull weights sum to zero."
            )

        speed_frequencies = (
            speed_pdf
            / speed_pdf.sum()
        )

        condition_directions = np.repeat(
            directions,
            len(speeds),
        )

        condition_speeds = np.tile(
            speeds,
            len(directions),
        )

        condition_frequencies = (
            np.repeat(
                direction_frequencies,
                len(speeds),
            )
            * np.tile(
                speed_frequencies,
                len(directions),
            )
        )

        if not np.isclose(
            condition_frequencies.sum(),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Combined wind-condition frequencies do not sum to one."
            )

        return (
            condition_directions.astype(
                float
            ),
            condition_speeds.astype(
                float
            ),
            condition_frequencies.astype(
                float
            ),
        )

    @property
    def number_of_conditions(
        self,
    ) -> int:
        return int(
            len(self.condition_frequencies)
        )

    def _validate_layout(
        self,
        xy: Any,
    ) -> np.ndarray:
        coordinates = np.asarray(
            xy,
            dtype=float,
        ).reshape(-1, 2)

        if coordinates.shape[0] <= 0:
            raise ValueError(
                "At least one turbine is required."
            )

        if not np.all(
            np.isfinite(coordinates)
        ):
            raise ValueError(
                "Layout coordinates must be finite."
            )

        return coordinates

    def _run_layout(
        self,
        xy: Any,
    ) -> np.ndarray:
        coordinates = self._validate_layout(
            xy
        )

        n_turbines = int(
            coordinates.shape[0]
        )

        self.floris_model.set(
            layout_x=coordinates[:, 0],
            layout_y=coordinates[:, 1],
            turbine_type=[
                self.turbine_type
            ],
            reference_wind_height=(
                self.floris_hub_height_m
            ),
            wind_directions=(
                self.wind_directions_deg
            ),
            wind_speeds=(
                self.wind_speeds_ms
            ),
            turbulence_intensities=(
                self.turbulence_intensities
            ),
        )

        self.floris_model.run()

        powers_raw = np.asarray(
            self.floris_model.get_turbine_powers(),
            dtype=float,
        )

        self.last_turbine_powers_shape = (
            tuple(powers_raw.shape)
        )

        expected_size = (
            self.number_of_conditions
            * n_turbines
        )

        if powers_raw.size != expected_size:
            raise RuntimeError(
                "Unexpected FLORIS turbine-power array size. "
                f"Expected {expected_size}, received "
                f"{powers_raw.size} with shape "
                f"{powers_raw.shape}."
            )

        powers_w = powers_raw.reshape(
            self.number_of_conditions,
            n_turbines,
        )

        if not np.all(
            np.isfinite(powers_w)
        ):
            raise RuntimeError(
                "FLORIS returned non-finite turbine powers."
            )

        farm_aep_wh = float(
            self.floris_model.get_farm_AEP(
                freq=self.condition_frequencies,
                hours_per_year=(
                    self.hours_per_year
                ),
            )
        )

        self.last_farm_aep_gwh_from_api = (
            farm_aep_wh / 1.0e9
        )

        return powers_w

    def evaluate_per_turbine(
        self,
        xy: Any,
    ) -> np.ndarray:
        """
        Calculate annual energy per turbine in GWh/year.
        """
        powers_w = self._run_layout(
            xy
        )

        weighted_mean_power_w = (
            self.condition_frequencies
            @ powers_w
        )

        aep_per_turbine_gwh = (
            weighted_mean_power_w
            * self.hours_per_year
            / 1.0e9
        )

        self.last_total_aep_gwh_from_powers = float(
            aep_per_turbine_gwh.sum()
        )

        return np.asarray(
            aep_per_turbine_gwh,
            dtype=float,
        )

    def evaluate(
        self,
        xy: Any,
    ) -> float:
        """
        Calculate total farm AEP in GWh/year.
        """
        return float(
            self.evaluate_per_turbine(
                xy
            ).sum()
        )

    def ideal_aep(
        self,
        n_turbines: int,
    ) -> float:
        """
        Calculate the FLORIS wake-free AEP ceiling.
        """
        if n_turbines <= 0:
            raise ValueError(
                "n_turbines must be positive."
            )

        if self._single_turbine_aep_gwh is None:
            self._single_turbine_aep_gwh = float(
                self.evaluate(
                    np.array(
                        [[0.0, 0.0]],
                        dtype=float,
                    )
                )
            )

        return float(
            n_turbines
            * self._single_turbine_aep_gwh
        )

    def settings_summary(
        self,
    ) -> dict[str, Any]:
        """
        Return serialisable settings for run archives.
        """
        return {
            "model": "FLORISAEPModel",
            "config_path": str(
                self.config_path
            ),
            "turbine_type": (
                self.turbine_type
            ),
            "floris_hub_height_m": (
                self.floris_hub_height_m
            ),
            "floris_rotor_diameter_m": (
                self.floris_rotor_diameter_m
            ),
            "jensen_scenario_rotor_diameter_m": float(
                self.turbine.rotor_diameter_m
            ),
            "number_of_direction_sectors": int(
                self.wind.n_sectors
            ),
            "number_of_speed_bins": int(
                len(self.wind.speed_bins)
            ),
            "number_of_combined_conditions": (
                self.number_of_conditions
            ),
            "dominant_wind_from_deg": float(
                self.site.wind_from_deg
            ),
            "turbulence_intensity": (
                self.turbulence_intensity
            ),
            "hours_per_year": (
                self.hours_per_year
            ),
            "condition_frequency_sum": float(
                self.condition_frequencies.sum()
            ),
        }
