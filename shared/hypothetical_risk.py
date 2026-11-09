import numpy as np


class HypotheticalCollisionModel:
    """
    Artificial collision-risk models for optimiser validation.

    These are not ecological bird models.
    They are deliberately simple risk fields with known expected behaviour.
    """

    def __init__(
    self,
    scenario,
    model_name="constant",
    risk_high=1000.0,
    risk_low=0.0,
    corridor_width_m=1200.0,
    hotspot_radius_m=900.0,
    risk_multiplier=1.0,
):
        self.scenario = scenario
        self.site = scenario.site
        self.turbine = scenario.turbine

        self.model_name = model_name
        self.risk_high = float(risk_high)
        self.risk_low = float(risk_low)
        self.corridor_width_m = float(corridor_width_m)
        self.hotspot_radius_m = float(hotspot_radius_m)
        self.risk_multiplier = float(risk_multiplier)
        self.width = self.site.width_m
        self.height = self.site.height_m
        self.n_turbines = self.site.n_turbines

    def collisions_per_turbine(self, xy):
        """
        Returns artificial collision-risk value per turbine.
        """
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)

        x = xy[:, 0]
        y = xy[:, 1]

        if self.model_name == "constant":
            # Every turbine has identical risk.
            # Expected result: collision risk should not affect layout.
            return np.full(len(xy), self.risk_high)

        elif self.model_name == "half_site_east_risk":
            # East half is risky, west half is safe.
            # Expected result: collision-optimal layout moves west.
            return np.where(
                x >= self.width / 2.0,
                self.risk_high,
                self.risk_low,
            )

        elif self.model_name == "central_corridor":
            # High-risk horizontal corridor through the centre.
            # Expected result: collision-optimal layout avoids centre band.
            corridor_center = self.height / 2.0
            in_corridor = np.abs(y - corridor_center) <= self.corridor_width_m / 2.0

            return np.where(
                in_corridor,
                self.risk_high,
                self.risk_low,
            )

        elif self.model_name == "point_hotspot":
            # Smooth central hotspot.
            # Risk is highest at the centre and gradually decreases with distance.
            # Expected result:
            # collision-optimal layout should push turbines away from the centre.
            cx = self.width / 2.0
            cy = self.height / 2.0

            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            sigma = self.hotspot_radius_m

            risk = self.risk_high * np.exp(-(dist ** 2) / (2.0 * sigma ** 2))

            return risk

        else:
            raise ValueError(
                f"Unknown model_name: {self.model_name}. "
                "Use constant, half_site_east_risk, central_corridor, or point_hotspot."
            )

    def collisions(self, xy):
        """
        Total artificial collision risk for the whole layout.
        The risk multiplier represents density scaling.
        """
        return float(self.risk_multiplier * np.sum(self.collisions_per_turbine(xy)))

    def risk_scale(self):
        """
        Maximum expected artificial risk.
        Used for objective normalisation.
        """
        return self.risk_multiplier * self.n_turbines * self.risk_high

    def diagnostics(self, xy):
        """
        Diagnostics for checking whether turbines are inside risky regions.
        """
        risk_per_turbine = self.risk_multiplier * self.collisions_per_turbine(xy)

        threshold = 0.5 * self.risk_high * self.risk_multiplier

        high_risk_count = int(np.sum(risk_per_turbine >= threshold))
        high_risk_fraction = high_risk_count / len(risk_per_turbine)

        return {
            "high_risk_count": high_risk_count,
            "high_risk_fraction": high_risk_fraction,
            "total_risk": float(np.sum(risk_per_turbine)),
        }

    def draw_background(self, ax):
        """
        Draw artificial risk field behind layout plots.
        """
        if self.model_name == "half_site_east_risk":
            ax.axvspan(
                self.width / 2.0,
                self.width,
                alpha=0.15,
                label="High-risk east half",
            )

        elif self.model_name == "central_corridor":
            y0 = self.height / 2.0 - self.corridor_width_m / 2.0
            y1 = self.height / 2.0 + self.corridor_width_m / 2.0

            ax.axhspan(
                y0,
                y1,
                alpha=0.15,
                label="High-risk corridor",
            )

        elif self.model_name == "point_hotspot":
            import matplotlib.pyplot as plt

            circle = plt.Circle(
                (self.width / 2.0, self.height / 2.0),
                self.hotspot_radius_m,
                alpha=0.15,
                label="High-risk hotspot",
            )
            ax.add_patch(circle)