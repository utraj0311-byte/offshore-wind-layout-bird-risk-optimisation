from __future__ import annotations

import numpy as np


class DirectionalAbsorbingRiskModel:
    """
    Newly proposed diagnostic directional-interaction model.

    Status
    ------
    DIAGNOSTIC ONLY / NEWLY PROPOSED

    Purpose
    -------
    Test whether a non-conservative directional exposure
    formulation encourages turbines to collapse into files
    parallel to an assumed bird-travel direction.

    This is not an empirical bird-behaviour model and does
    not predict expected bird casualties.

    Model
    -----
    For each physically upstream-downstream turbine pair:

        K_ij =
            exp(-delta_s / lambda_s)
            * exp(-0.5 * (delta_n / sigma_n)^2)

    The remaining exposure at receiver turbine j is:

        E_j = product_i(1 - alpha * K_ij)

    Total diagnostic risk is:

        R = risk_multiplier * sum_j(E_j)

    Because removed exposure is not redistributed, this is
    deliberately a non-conservative benchmark.
    """

    def __init__(
        self,
        scenario,
        bird_travel_deg: float = 225.0,
        downstream_scale_D: float = 8.0,
        cross_scale_D: float = 1.0,
        maximum_shielding: float = 0.8,
        risk_multiplier: float = 1.0,
    ) -> None:
        if downstream_scale_D <= 0.0:
            raise ValueError(
                "downstream_scale_D must be positive."
            )

        if cross_scale_D <= 0.0:
            raise ValueError(
                "cross_scale_D must be positive."
            )

        if not 0.0 < maximum_shielding < 1.0:
            raise ValueError(
                "maximum_shielding must lie between 0 and 1."
            )

        if risk_multiplier <= 0.0:
            raise ValueError(
                "risk_multiplier must be positive."
            )

        self.scenario = scenario
        self.site = scenario.site
        self.turbine = scenario.turbine

        self.bird_travel_deg = (
            float(bird_travel_deg) % 360.0
        )

        rotor_diameter_m = float(
            self.turbine.rotor_diameter_m
        )

        self.downstream_scale_D = float(
            downstream_scale_D
        )

        self.cross_scale_D = float(
            cross_scale_D
        )

        self.downstream_scale_m = (
            self.downstream_scale_D
            * rotor_diameter_m
        )

        self.cross_scale_m = (
            self.cross_scale_D
            * rotor_diameter_m
        )

        self.maximum_shielding = float(
            maximum_shielding
        )

        self.risk_multiplier = float(
            risk_multiplier
        )

    @staticmethod
    def _validate_layout(
        xy: np.ndarray,
    ) -> np.ndarray:
        xy = np.asarray(
            xy,
            dtype=float,
        )

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(
                "xy must have shape (n_turbines, 2)."
            )

        if len(xy) < 1:
            raise ValueError(
                "At least one turbine is required."
            )

        if not np.all(np.isfinite(xy)):
            raise ValueError(
                "All turbine coordinates must be finite."
            )

        return xy

    def direction_vectors(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return unit vectors parallel and perpendicular to
        bird travel.

        Convention:
            0 degrees   -> travel toward +x
            90 degrees  -> travel toward +y
            180 degrees -> travel toward -x
        """
        angle_rad = np.radians(
            self.bird_travel_deg
        )

        along_unit = np.array(
            [
                np.cos(angle_rad),
                np.sin(angle_rad),
            ],
            dtype=float,
        )

        cross_unit = np.array(
            [
                -np.sin(angle_rad),
                np.cos(angle_rad),
            ],
            dtype=float,
        )

        return along_unit, cross_unit

    def bird_frame_coordinates(
        self,
        xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project turbine coordinates into the bird-travel
        coordinate system.
        """
        xy = self._validate_layout(xy)

        along_unit, cross_unit = (
            self.direction_vectors()
        )

        along = xy @ along_unit
        cross = xy @ cross_unit

        return along, cross

    def directional_kernel(
        self,
        downstream_distance_m: float,
        cross_distance_m: float,
    ) -> float:
        """
        Smooth directional-overlap kernel.

        Returns zero when the proposed receiver is not
        downstream.
        """
        downstream_distance_m = float(
            downstream_distance_m
        )

        cross_distance_m = abs(
            float(cross_distance_m)
        )

        if downstream_distance_m <= 0.0:
            return 0.0

        downstream_factor = np.exp(
            -downstream_distance_m
            / self.downstream_scale_m
        )

        cross_factor = np.exp(
            -0.5
            * (
                cross_distance_m
                / self.cross_scale_m
            ) ** 2
        )

        return float(
            downstream_factor
            * cross_factor
        )

    def pair_data(
        self,
        xy: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Return one physically ordered interaction for every
        turbine pair with non-zero along-flow separation.

        Each unordered turbine pair appears at most once.
        """
        xy = self._validate_layout(xy)

        along, cross = (
            self.bird_frame_coordinates(xy)
        )

        source_indices: list[int] = []
        receiver_indices: list[int] = []
        downstream_distances: list[float] = []
        cross_distances: list[float] = []
        kernels: list[float] = []
        shielding_fractions: list[float] = []

        tolerance = 1e-12

        for first in range(len(xy) - 1):
            for second in range(
                first + 1,
                len(xy),
            ):
                along_difference = (
                    along[second]
                    - along[first]
                )

                if abs(along_difference) <= tolerance:
                    continue

                if along_difference > 0.0:
                    source = first
                    receiver = second
                else:
                    source = second
                    receiver = first

                downstream_distance = (
                    along[receiver]
                    - along[source]
                )

                cross_distance = abs(
                    cross[receiver]
                    - cross[source]
                )

                kernel = self.directional_kernel(
                    downstream_distance_m=(
                        downstream_distance
                    ),
                    cross_distance_m=(
                        cross_distance
                    ),
                )

                shielding_fraction = (
                    self.maximum_shielding
                    * kernel
                )

                source_indices.append(source)
                receiver_indices.append(receiver)

                downstream_distances.append(
                    downstream_distance
                )

                cross_distances.append(
                    cross_distance
                )

                kernels.append(kernel)

                shielding_fractions.append(
                    shielding_fraction
                )

        return {
            "source_indices": np.asarray(
                source_indices,
                dtype=int,
            ),
            "receiver_indices": np.asarray(
                receiver_indices,
                dtype=int,
            ),
            "downstream_distances_m": np.asarray(
                downstream_distances,
                dtype=float,
            ),
            "cross_distances_m": np.asarray(
                cross_distances,
                dtype=float,
            ),
            "kernels": np.asarray(
                kernels,
                dtype=float,
            ),
            "shielding_fractions": np.asarray(
                shielding_fractions,
                dtype=float,
            ),
        }

    def exposure_per_turbine(
        self,
        xy: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate remaining dimensionless exposure at each
        turbine.

        Every turbine starts with exposure 1. Upstream
        directional interactions remove a fraction of the
        remaining exposure.
        """
        xy = self._validate_layout(xy)

        exposure = np.ones(
            len(xy),
            dtype=float,
        )

        pair_data = self.pair_data(xy)

        for receiver, shielding in zip(
            pair_data["receiver_indices"],
            pair_data["shielding_fractions"],
        ):
            exposure[receiver] *= (
                1.0 - shielding
            )

        return exposure

    def collisions_per_turbine(
        self,
        xy: np.ndarray,
    ) -> np.ndarray:
        """
        Return per-turbine diagnostic risk values.

        Method name retained for compatibility with the
        existing optimisation problem interface.
        """
        return (
            self.risk_multiplier
            * self.exposure_per_turbine(xy)
        )

    def collisions(
        self,
        xy: np.ndarray,
    ) -> float:
        """
        Return total diagnostic risk.
        """
        return float(
            np.sum(
                self.collisions_per_turbine(xy)
            )
        )

    def risk_scale(
        self,
        n_turbines: int | None = None,
    ) -> float:
        """
        Upper bound when every turbine receives baseline
        exposure and no directional shielding occurs.
        """
        if n_turbines is None:
            n_turbines = (
                self.site.n_turbines
            )

        n_turbines = int(n_turbines)

        if n_turbines < 1:
            raise ValueError(
                "n_turbines must be positive."
            )

        return float(
            self.risk_multiplier
            * n_turbines
        )

    def diagnostics(
        self,
        xy: np.ndarray,
    ) -> dict:
        """
        Return pair- and turbine-level model diagnostics.
        """
        xy = self._validate_layout(xy)

        along, cross = (
            self.bird_frame_coordinates(xy)
        )

        pair_data = self.pair_data(xy)

        exposure = self.exposure_per_turbine(
            xy
        )

        per_turbine_risk = (
            self.risk_multiplier
            * exposure
        )

        shielding = pair_data[
            "shielding_fractions"
        ]

        receiver_counts = np.zeros(
            len(xy),
            dtype=int,
        )

        for receiver in pair_data[
            "receiver_indices"
        ]:
            receiver_counts[receiver] += 1

        effective_mask = shielding > 1e-6

        if len(shielding) == 0:
            maximum_shielding = 0.0
            mean_shielding = 0.0
        else:
            maximum_shielding = float(
                np.max(shielding)
            )

            mean_shielding = float(
                np.mean(shielding)
            )

        baseline_risk = self.risk_scale(
            len(xy)
        )

        total_risk = float(
            np.sum(per_turbine_risk)
        )

        return {
            "model_status": (
                "NEWLY PROPOSED / DIAGNOSTIC ONLY"
            ),
            "number_of_turbines": int(
                len(xy)
            ),
            "bird_travel_deg": (
                self.bird_travel_deg
            ),
            "downstream_scale_D": (
                self.downstream_scale_D
            ),
            "cross_scale_D": (
                self.cross_scale_D
            ),
            "downstream_scale_m": (
                self.downstream_scale_m
            ),
            "cross_scale_m": (
                self.cross_scale_m
            ),
            "maximum_shielding_parameter": (
                self.maximum_shielding
            ),
            "baseline_total_risk": (
                baseline_risk
            ),
            "total_risk": total_risk,
            "total_risk_reduction": (
                baseline_risk
                - total_risk
            ),
            "mean_exposure": float(
                np.mean(exposure)
            ),
            "minimum_exposure": float(
                np.min(exposure)
            ),
            "maximum_exposure": float(
                np.max(exposure)
            ),
            "number_of_directional_pairs": int(
                len(shielding)
            ),
            "number_of_effective_pairs": int(
                np.sum(effective_mask)
            ),
            "maximum_pair_shielding": (
                maximum_shielding
            ),
            "mean_pair_shielding": (
                mean_shielding
            ),
            "along_coordinates_m": along,
            "cross_coordinates_m": cross,
            "exposure_per_turbine": exposure,
            "per_turbine_risk": (
                per_turbine_risk
            ),
            "upstream_interaction_counts": (
                receiver_counts
            ),
            **pair_data,
        }