import numpy as np


class SurroundednessRiskModel:
    """
    Artificial coupled-risk model for optimiser validation.

    The risk assigned to one designated turbine depends on
    the positions of all other turbines.

    Modes
    -----
    interior_risky:
        The designated turbine has higher risk when it is
        surrounded by nearby turbines.

    interior_safe:
        The designated turbine has lower risk when it is
        surrounded by nearby turbines.

    This is a geometric validation model, not an ecological
    bird-collision prediction.
    """

    VALID_MODES = {
        "interior_risky",
        "interior_safe",
    }

    def __init__(
        self,
        scenario,
        target_index=0,
        interaction_scale_D=4.0,
        mode="interior_risky",
        risk_multiplier=1.0,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(self.VALID_MODES)}"
            )

        if interaction_scale_D <= 0.0:
            raise ValueError(
                "interaction_scale_D must be positive."
            )

        if risk_multiplier <= 0.0:
            raise ValueError(
                "risk_multiplier must be positive."
            )

        self.scenario = scenario
        self.site = scenario.site
        self.turbine = scenario.turbine

        self.n_turbines = self.site.n_turbines
        self.target_index = int(target_index)
        self.mode = mode
        self.risk_multiplier = float(risk_multiplier)

        if not 0 <= self.target_index < self.n_turbines:
            raise IndexError(
                "target_index is outside the turbine layout."
            )

        self.interaction_scale_D = float(
            interaction_scale_D
        )

        self.sigma_m = (
            self.interaction_scale_D
            * self.turbine.rotor_diameter_m
        )

        # Theoretical maximum: all other turbines contribute 1.
        self.maximum_surroundedness = float(
            self.n_turbines - 1
        )

    def _validate_layout(self, xy):
        xy = np.asarray(
            xy,
            dtype=float,
        )

        expected_shape = (
            self.n_turbines,
            2,
        )

        if xy.shape != expected_shape:
            raise ValueError(
                "xy must have shape "
                f"{expected_shape}, received {xy.shape}."
            )

        if not np.all(np.isfinite(xy)):
            raise ValueError(
                "Layout coordinates must be finite."
            )

        return xy

    def surroundedness(self, xy):
        """
        Calculate the surroundedness of the designated turbine.

        Each neighbouring turbine contributes:

            exp(-d^2 / (2 sigma^2))

        Nearby turbines therefore contribute more strongly
        than distant turbines.
        """
        xy = self._validate_layout(xy)

        target_position = xy[self.target_index]

        displacement = (
            xy
            - target_position
        )

        all_distances = np.linalg.norm(
            displacement,
            axis=1,
        )

        neighbour_mask = (
            np.arange(self.n_turbines)
            != self.target_index
        )

        neighbour_distances = all_distances[
            neighbour_mask
        ]

        interaction_weights = np.exp(
            -(
                neighbour_distances**2
            )
            / (
                2.0
                * self.sigma_m**2
            )
        )

        score = float(
            np.sum(interaction_weights)
        )

        return (
            score,
            neighbour_distances,
            interaction_weights,
        )

    def collisions_per_turbine(self, xy):
        """
        Return artificial risk assigned to each turbine.

        Only the designated turbine directly carries risk.
        The other turbines influence that risk through their
        positions.
        """
        score, _, _ = self.surroundedness(xy)

        if self.mode == "interior_risky":
            target_risk = score

        else:
            target_risk = (
                self.maximum_surroundedness
                - score
            )

        target_risk = max(
            0.0,
            float(target_risk),
        )

        per_turbine_risk = np.zeros(
            self.n_turbines,
            dtype=float,
        )

        per_turbine_risk[
            self.target_index
        ] = (
            self.risk_multiplier
            * target_risk
        )

        return per_turbine_risk

    def collisions(self, xy):
        """
        Return total artificial risk for the complete layout.

        This method matches the interface currently used by
        ValidationLayoutProblem.
        """
        return float(
            np.sum(
                self.collisions_per_turbine(xy)
            )
        )

    def risk_scale(self):
        """
        Conservative upper scale for objective normalisation.
        """
        return float(
            self.risk_multiplier
            * self.maximum_surroundedness
        )

    def diagnostics(self, xy):
        """
        Return values useful for validation and plotting.
        """
        xy = self._validate_layout(xy)

        score, distances, weights = (
            self.surroundedness(xy)
        )

        centroid = np.mean(
            xy,
            axis=0,
        )

        target_position = xy[
            self.target_index
        ]

        target_radius = float(
            np.linalg.norm(
                target_position
                - centroid
            )
        )

        all_radii = np.linalg.norm(
            xy - centroid,
            axis=1,
        )

        radial_rank = int(
            np.argsort(
                np.argsort(all_radii)
            )[self.target_index]
        )

        neighbour_radius_m = (
            5.0
            * self.turbine.rotor_diameter_m
        )

        neighbour_count = int(
            np.sum(
                distances
                <= neighbour_radius_m
            )
        )

        return {
            "mode": self.mode,
            "target_index": self.target_index,
            "target_x_m": float(
                target_position[0]
            ),
            "target_y_m": float(
                target_position[1]
            ),
            "surroundedness": score,
            "total_risk": self.collisions(xy),
            "distance_to_centroid_m": target_radius,
            "radial_rank": radial_rank,
            "neighbour_count_within_5D": neighbour_count,
            "neighbour_distances_m": distances,
            "interaction_weights": weights,
        }