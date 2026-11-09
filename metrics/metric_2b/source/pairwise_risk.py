import numpy as np


class PairwiseGaussianRiskModel:
    """
    Artificial permutation-invariant pairwise risk model.

    Every unique turbine pair contributes:

        exp(-d_ij^2 / (2 sigma^2))

    where:
        d_ij is the distance between turbines i and j;
        sigma controls the interaction length scale.

    Minimising the total objective encourages turbines to
    separate from one another.

    This is an optimiser-validation function. It is not a
    biologically calibrated bird-collision model.
    """

    def __init__(
        self,
        scenario,
        interaction_scale_D=4.0,
        risk_multiplier=1.0,
    ):
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

        self.interaction_scale_D = float(
            interaction_scale_D
        )

        self.risk_multiplier = float(
            risk_multiplier
        )

        self.sigma_m = (
            self.interaction_scale_D
            * self.turbine.rotor_diameter_m
        )

    @staticmethod
    def _validate_layout(xy):
        """
        Validate and return a turbine-coordinate array.

        The model accepts any layout containing at least
        two turbines. This allows exact 2-turbine checks
        before applying it to the full 12-turbine scenario.
        """
        xy = np.asarray(
            xy,
            dtype=float,
        )

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(
                "xy must have shape (n_turbines, 2)."
            )

        if len(xy) < 2:
            raise ValueError(
                "At least two turbines are required."
            )

        if not np.all(np.isfinite(xy)):
            raise ValueError(
                "All coordinates must be finite."
            )

        return xy

    def pair_data(self, xy):
        """
        Calculate distance and Gaussian weight for every
        unique turbine pair.

        Returns
        -------
        pair_indices:
            Integer array with shape (n_pairs, 2).

        pair_distances:
            Distance for every unique pair.

        pair_weights:
            Gaussian contribution from every unique pair.
        """
        xy = self._validate_layout(xy)

        pair_indices = []
        pair_distances = []
        pair_weights = []

        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                displacement = (
                    xy[i]
                    - xy[j]
                )

                distance = float(
                    np.linalg.norm(
                        displacement
                    )
                )

                weight = float(
                    np.exp(
                        -(
                            distance**2
                        )
                        / (
                            2.0
                            * self.sigma_m**2
                        )
                    )
                )

                pair_indices.append(
                    [i, j]
                )

                pair_distances.append(
                    distance
                )

                pair_weights.append(
                    weight
                )

        return (
            np.asarray(
                pair_indices,
                dtype=int,
            ),
            np.asarray(
                pair_distances,
                dtype=float,
            ),
            np.asarray(
                pair_weights,
                dtype=float,
            ),
        )

    def collisions(self, xy):
        """
        Return the total artificial pairwise risk.
        """
        _, _, pair_weights = (
            self.pair_data(xy)
        )

        return float(
            self.risk_multiplier
            * np.sum(pair_weights)
        )

    def collisions_per_turbine(self, xy):
        """
        Allocate half of each pair contribution to each
        turbine participating in that pair.

        The per-turbine values sum exactly to total risk.
        """
        xy = self._validate_layout(xy)

        pair_indices, _, pair_weights = (
            self.pair_data(xy)
        )

        per_turbine_risk = np.zeros(
            len(xy),
            dtype=float,
        )

        for pair_index, (i, j) in enumerate(
            pair_indices
        ):
            pair_contribution = (
                self.risk_multiplier
                * pair_weights[pair_index]
            )

            per_turbine_risk[i] += (
                0.5
                * pair_contribution
            )

            per_turbine_risk[j] += (
                0.5
                * pair_contribution
            )

        return per_turbine_risk

    def risk_scale(self, n_turbines=None):
        """
        Return the theoretical maximum pairwise risk.

        Each Gaussian term is at most one. Therefore:

            maximum = number of unique pairs

        Co-located turbines are infeasible in the actual
        optimisation, so this is a conservative scaling
        bound rather than a feasible maximum.
        """
        if n_turbines is None:
            n_turbines = (
                self.site.n_turbines
            )

        n_turbines = int(
            n_turbines
        )

        if n_turbines < 2:
            raise ValueError(
                "n_turbines must be at least two."
            )

        number_of_pairs = (
            n_turbines
            * (n_turbines - 1)
            // 2
        )

        return float(
            self.risk_multiplier
            * number_of_pairs
        )

    def diagnostics(self, xy):
        """
        Return pairwise and layout-level diagnostics.
        """
        xy = self._validate_layout(xy)

        (
            pair_indices,
            pair_distances,
            pair_weights,
        ) = self.pair_data(xy)

        per_turbine_risk = (
            self.collisions_per_turbine(
                xy
            )
        )

        total_risk = float(
            self.risk_multiplier
            * np.sum(pair_weights)
        )

        return {
            "number_of_turbines": int(
                len(xy)
            ),
            "number_of_pairs": int(
                len(pair_weights)
            ),
            "interaction_scale_D": (
                self.interaction_scale_D
            ),
            "sigma_m": self.sigma_m,
            "total_risk": total_risk,
            "minimum_pair_distance_m": float(
                np.min(pair_distances)
            ),
            "mean_pair_distance_m": float(
                np.mean(pair_distances)
            ),
            "maximum_pair_distance_m": float(
                np.max(pair_distances)
            ),
            "maximum_pair_weight": float(
                np.max(pair_weights)
            ),
            "mean_pair_weight": float(
                np.mean(pair_weights)
            ),
            "pair_indices": pair_indices,
            "pair_distances_m": (
                pair_distances
            ),
            "pair_weights": pair_weights,
            "per_turbine_risk": (
                per_turbine_risk
            ),
        }