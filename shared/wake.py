import numpy as np

from config import Scenario


class AEPModel:
    """
    Simplified Jensen wake model for the first prototype.

    Main functions:
    - evaluate(xy) -> total farm AEP in GWh/year
    - evaluate_per_turbine(xy) -> turbine-level AEP in GWh/year

    Later, this class can be replaced by a FLORIS-backed model
    with the same interface.
    """

    def __init__(self, scenario: Scenario):
        self.s = scenario
        self.turbine = scenario.turbine
        self.site = scenario.site
        self.wind = scenario.wind

        self.rotor_radius_m = (
            self.turbine.rotor_diameter_m / 2.0
        )

        # Offshore Jensen wake-expansion coefficient.
        self.wake_decay_k = 0.04

    def power_curve_w(self, wind_speed_ms):
        """
        Very simple turbine power curve.

        Below cut-in:
            zero power.

        Between cut-in and rated:
            cubic increase.

        Between rated and cut-out:
            rated power.

        Above cut-out:
            zero power.
        """
        u = np.asarray(
            wind_speed_ms,
            dtype=float,
        )

        power_w = np.zeros_like(
            u,
            dtype=float,
        )

        rated_power_w = (
            self.turbine.rated_power_mw
            * 1e6
        )

        cut_in = self.turbine.cut_in_ms
        rated = self.turbine.rated_ms
        cut_out = self.turbine.cut_out_ms

        # Region 1: between cut-in and rated.
        mask_partial = (
            (u >= cut_in)
            & (u < rated)
        )

        power_w[mask_partial] = (
            rated_power_w
            * (
                (
                    u[mask_partial]
                    - cut_in
                )
                / (
                    rated
                    - cut_in
                )
            )
            ** 3
        )

        # Region 2: rated power.
        mask_rated = (
            (u >= rated)
            & (u <= cut_out)
        )

        power_w[mask_rated] = rated_power_w

        return power_w

    def ct_curve(self, wind_speed_ms):
        """
        Simplified thrust-coefficient curve.

        Ct is high below rated wind speed and lower above
        rated wind speed.

        This remains a placeholder for the prototype.
        """
        u = np.asarray(
            wind_speed_ms,
            dtype=float,
        )

        ct = np.zeros_like(
            u,
            dtype=float,
        )

        cut_in = self.turbine.cut_in_ms
        rated = self.turbine.rated_ms
        cut_out = self.turbine.cut_out_ms

        mask_operating = (
            (u >= cut_in)
            & (u <= cut_out)
        )

        ct[mask_operating] = 0.8

        mask_above_rated = (
            (u >= rated)
            & (u <= cut_out)
        )

        ct[mask_above_rated] = 0.25

        return ct

    def _wind_frame_coordinates(
        self,
        xy,
        wind_from_deg,
    ):
        """
        Rotate coordinates into the wind-flow frame.

        wind_from_deg specifies the direction from which
        the wind originates.

        The flow direction is therefore:

            wind_from_deg + 180 degrees
        """
        xy = np.asarray(
            xy,
            dtype=float,
        ).reshape(-1, 2)

        flow_deg = (
            wind_from_deg
            + 180.0
        ) % 360.0

        theta = np.radians(flow_deg)

        # Unit vector along the flow direction.
        along = np.array(
            [
                np.cos(theta),
                np.sin(theta),
            ]
        )

        # Unit vector perpendicular to the flow direction.
        cross = np.array(
            [
                -np.sin(theta),
                np.cos(theta),
            ]
        )

        x_along = xy @ along
        y_cross = xy @ cross

        return x_along, y_cross

    def effective_wind_speeds(
        self,
        xy,
        wind_speed_ms,
        wind_from_deg,
    ):
        """
        Calculate the effective wind speed at each turbine
        after Jensen wake losses.
        """
        xy = np.asarray(
            xy,
            dtype=float,
        ).reshape(-1, 2)

        n_turbines = xy.shape[0]

        x_along, y_cross = (
            self._wind_frame_coordinates(
                xy,
                wind_from_deg,
            )
        )

        deficits_squared = np.zeros(
            n_turbines,
            dtype=float,
        )

        ct = float(
            self.ct_curve(
                np.array(
                    [wind_speed_ms],
                    dtype=float,
                )
            )[0]
        )

        if ct <= 0.0:
            return np.zeros(
                n_turbines,
                dtype=float,
            )

        # Jensen maximum velocity deficit at wake centre.
        initial_deficit = (
            1.0
            - np.sqrt(
                1.0
                - ct
            )
        )

        for i in range(n_turbines):
            for j in range(n_turbines):
                if i == j:
                    continue

                dx = (
                    x_along[i]
                    - x_along[j]
                )

                dy = abs(
                    y_cross[i]
                    - y_cross[j]
                )

                # Turbine j wakes turbine i only when j
                # is upstream of i.
                if dx <= 0.0:
                    continue

                wake_radius = (
                    self.rotor_radius_m
                    + self.wake_decay_k
                    * dx
                )

                # Check whether turbine i lies inside
                # turbine j's wake cone.
                if dy <= wake_radius:
                    deficit = (
                        initial_deficit
                        / (
                            1.0
                            + self.wake_decay_k
                            * dx
                            / self.rotor_radius_m
                        )
                        ** 2
                    )

                    deficits_squared[i] += (
                        deficit**2
                    )

        total_deficit = np.sqrt(
            deficits_squared
        )

        total_deficit = np.clip(
            total_deficit,
            0.0,
            0.95,
        )

        effective_speeds = (
            wind_speed_ms
            * (
                1.0
                - total_deficit
            )
        )

        return effective_speeds

    def evaluate_per_turbine(self, xy):
        """
        Calculate annual energy production per turbine
        in GWh/year.
        """
        xy = np.asarray(
            xy,
            dtype=float,
        ).reshape(-1, 2)

        n_turbines = xy.shape[0]

        aep_per_turbine_gwh = np.zeros(
            n_turbines,
            dtype=float,
        )

        wind_directions = (
            np.arange(
                self.wind.n_sectors
            )
            * 360.0
            / self.wind.n_sectors
        )

        direction_freqs = (
            self.wind.sector_frequencies(
                self.site.wind_from_deg
            )
        )

        speeds = np.asarray(
            self.wind.speed_bins,
            dtype=float,
        )

        k = self.wind.weibull_k
        weibull_scale = self.wind.weibull_A

        # Discrete Weibull probability for the selected
        # wind-speed bins.
        speed_pdf = (
            (k / weibull_scale)
            * (
                speeds
                / weibull_scale
            )
            ** (k - 1.0)
            * np.exp(
                -(
                    speeds
                    / weibull_scale
                )
                ** k
            )
        )

        speed_freqs = (
            speed_pdf
            / speed_pdf.sum()
        )

        hours_per_year = 8760.0

        for direction_index, direction in enumerate(
            wind_directions
        ):
            for speed_index, speed in enumerate(
                speeds
            ):
                probability = (
                    direction_freqs[
                        direction_index
                    ]
                    * speed_freqs[
                        speed_index
                    ]
                )

                effective_speeds = (
                    self.effective_wind_speeds(
                        xy=xy,
                        wind_speed_ms=speed,
                        wind_from_deg=direction,
                    )
                )

                power_w = self.power_curve_w(
                    effective_speeds
                )

                # Convert W to GWh/year.
                energy_gwh = (
                    power_w
                    * hours_per_year
                    * probability
                    / 1e9
                )

                aep_per_turbine_gwh += (
                    energy_gwh
                )

        return aep_per_turbine_gwh

    def evaluate(self, xy):
        """
        Calculate total farm AEP in GWh/year.
        """
        return float(
            np.sum(
                self.evaluate_per_turbine(
                    xy
                )
            )
        )

    def ideal_aep(self, n_turbines):
        """
        Calculate the true wake-free AEP upper bound.

        A single turbine cannot experience a wake from
        another turbine. Its wake-free AEP is therefore
        multiplied by the requested number of turbines.
        """
        if n_turbines <= 0:
            raise ValueError(
                "n_turbines must be positive."
            )

        single_turbine_xy = np.array(
            [
                [0.0, 0.0],
            ],
            dtype=float,
        )

        single_turbine_aep = self.evaluate(
            single_turbine_xy
        )

        return float(
            n_turbines
            * single_turbine_aep
        )


if __name__ == "__main__":
    scenario = Scenario()
    model = AEPModel(scenario)

    # Feasible 12-turbine test layout:
    # 4 columns × 3 rows with spacing above 4D.
    spacing_m = 1200.0

    xy = []

    for row in range(3):
        for column in range(4):
            xy.append(
                [
                    column * spacing_m,
                    row * spacing_m,
                ]
            )

    xy = np.asarray(
        xy,
        dtype=float,
    )

    farm_aep = model.evaluate(xy)

    per_turbine_aep = (
        model.evaluate_per_turbine(
            xy
        )
    )

    wake_free_aep = model.ideal_aep(
        scenario.site.n_turbines
    )

    wake_loss_gwh = (
        wake_free_aep
        - farm_aep
    )

    wake_loss_fraction = (
        wake_loss_gwh
        / wake_free_aep
    )

    print("Wake model test successful.")
    print(
        f"Farm AEP: "
        f"{farm_aep:.6f} GWh/year"
    )
    print(
        f"Wake-free AEP: "
        f"{wake_free_aep:.6f} GWh/year"
    )
    print(
        f"Wake loss: "
        f"{wake_loss_gwh:.6f} GWh/year"
    )
    print(
        f"Wake-loss fraction: "
        f"{wake_loss_fraction:.6f}"
    )
    print("Per-turbine AEP:")
    print(
        np.round(
            per_turbine_aep,
            3,
        )
    )