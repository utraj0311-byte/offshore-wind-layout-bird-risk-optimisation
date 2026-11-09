"""pymoo coupling for FLORIS AEP and analytical directional shielding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

from shielding_model import N_EFF_DECIMALS, evaluate_shielding


class ShieldingAEPProblem(Problem):
    """Two-objective continuous wind-farm layout problem.

    NSGA-II minimises:

    ``f0 = (AEP_ceiling - AEP) / AEP_loss_reference``

    ``f1 = N_eff / N``

    Dividing ``N_eff`` by the physical turbine count gives a stable objective
    between ``1/N`` and ``1`` without changing Pareto dominance.  The reported
    scientific metric remains the unnormalised ``N_eff``.

    Candidate layouts that violate the minimum-spacing constraint are screened
    before FLORIS is called.  A final inequality constraint also requires every
    FLORIS rotor-grid velocity to be non-negative.  These two safeguards keep
    layouts outside the valid wake-model domain off the reported Pareto front.
    """

    INVALID_AEP_OBJECTIVE = 2.0

    def __init__(
        self,
        *,
        scenario: Any,
        aep_model: Any,
        rotor_diameter_m: float,
        bird_heading_deg_toward: float,
        corridor_width_m: float,
        aep_ceiling_gwh: float,
        aep_loss_reference_gwh: float,
        record_evaluations: bool = False,
        spacing_constraint_tolerance_m: float = 1.0e-6,
    ) -> None:
        self.scenario = scenario
        self.aep_model = aep_model
        self.rotor_diameter_m = float(rotor_diameter_m)
        self.bird_heading_deg_toward = float(bird_heading_deg_toward)
        self.corridor_width_m = float(corridor_width_m)
        self.aep_ceiling_gwh = float(aep_ceiling_gwh)
        self.aep_loss_reference_gwh = float(aep_loss_reference_gwh)
        self.record_evaluations = bool(record_evaluations)
        self.spacing_constraint_tolerance_m = float(spacing_constraint_tolerance_m)

        if not np.isfinite(self.rotor_diameter_m) or self.rotor_diameter_m <= 0.0:
            raise ValueError("rotor_diameter_m must be finite and positive.")
        if not np.isfinite(self.bird_heading_deg_toward):
            raise ValueError("bird_heading_deg_toward must be finite.")
        if not np.isfinite(self.corridor_width_m) or self.corridor_width_m <= 0.0:
            raise ValueError("corridor_width_m must be finite and positive.")
        if not np.isfinite(self.aep_ceiling_gwh) or self.aep_ceiling_gwh <= 0.0:
            raise ValueError("aep_ceiling_gwh must be finite and positive.")
        if (
            not np.isfinite(self.aep_loss_reference_gwh)
            or self.aep_loss_reference_gwh <= 0.0
        ):
            raise ValueError("aep_loss_reference_gwh must be finite and positive.")
        if self.spacing_constraint_tolerance_m < 0.0:
            raise ValueError("spacing_constraint_tolerance_m cannot be negative.")

        self.number_of_turbines = int(scenario.site.n_turbines)
        self.minimum_spacing_m = float(
            scenario.site.min_spacing_D * self.rotor_diameter_m
        )
        pair_count = self.number_of_turbines * (self.number_of_turbines - 1) // 2
        self.pair_spacing_constraint_count = int(pair_count)

        lower = np.tile(np.array([0.0, 0.0]), self.number_of_turbines)
        upper = np.tile(
            np.array([scenario.site.width_m, scenario.site.height_m], dtype=float),
            self.number_of_turbines,
        )

        super().__init__(
            n_var=2 * self.number_of_turbines,
            n_obj=2,
            n_ieq_constr=pair_count + 1,
            xl=lower,
            xu=upper,
            elementwise=False,
        )
        self._evaluation_rows: list[dict[str, Any]] = []
        self.total_candidate_evaluations = 0
        self.floris_evaluation_count = 0
        self.spacing_prescreen_skip_count = 0
        self.negative_rotor_velocity_candidate_count = 0
        self.minimum_observed_rotor_grid_velocity_m_s: float | None = None

    def spacing_constraints(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=float).reshape(self.number_of_turbines, 2)
        constraints = []
        for first in range(self.number_of_turbines):
            for second in range(first + 1, self.number_of_turbines):
                distance_m = float(np.linalg.norm(points[first] - points[second]))
                constraints.append(
                    self.minimum_spacing_m
                    - distance_m
                    - self.spacing_constraint_tolerance_m
                )
        return np.asarray(constraints, dtype=float)

    def evaluate_raw_layout(self, xy: np.ndarray) -> dict[str, Any]:
        points = np.asarray(xy, dtype=float).reshape(self.number_of_turbines, 2)
        aep_gwh = float(self.aep_model.evaluate(points))
        rotor_velocity = self.rotor_velocity_diagnostics()
        shielding = evaluate_shielding(
            points,
            rotor_diameter_m=self.rotor_diameter_m,
            bird_heading_deg_toward=self.bird_heading_deg_toward,
            corridor_width_m=self.corridor_width_m,
        )
        return {
            "aep_gwh": aep_gwh,
            "aep_loss_gwh": float(self.aep_ceiling_gwh - aep_gwh),
            **rotor_velocity,
            **shielding.summary(),
        }

    def rotor_velocity_diagnostics(self) -> dict[str, Any]:
        """Inspect the FLORIS rotor grid produced by the most recent run."""
        try:
            velocities = np.asarray(
                self.aep_model.floris_model.core.flow_field.u,
                dtype=float,
            )
        except AttributeError as exc:
            raise RuntimeError(
                "The AEP model does not expose FLORIS rotor-grid velocities at "
                "floris_model.core.flow_field.u."
            ) from exc

        if velocities.size == 0:
            raise RuntimeError("FLORIS returned an empty rotor-velocity array.")
        if not np.all(np.isfinite(velocities)):
            raise RuntimeError("FLORIS returned non-finite rotor-grid velocities.")

        minimum_velocity = float(np.min(velocities))
        negative_count = int(np.count_nonzero(velocities < 0.0))
        return {
            "minimum_rotor_grid_velocity_m_s": minimum_velocity,
            "negative_rotor_grid_point_count": negative_count,
            "rotor_velocity_feasible": bool(negative_count == 0),
        }

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        decision_vectors = np.atleast_2d(np.asarray(X, dtype=float))
        objective_rows = []
        constraint_rows = []

        for vector in decision_vectors:
            xy = vector.reshape(self.number_of_turbines, 2)
            self.total_candidate_evaluations += 1
            spacing_constraints = self.spacing_constraints(xy)
            maximum_spacing_constraint = float(np.max(spacing_constraints))
            spacing_feasible = bool(maximum_spacing_constraint <= 0.0)

            shielding = evaluate_shielding(
                xy,
                rotor_diameter_m=self.rotor_diameter_m,
                bird_heading_deg_toward=self.bird_heading_deg_toward,
                corridor_width_m=self.corridor_width_m,
            )
            raw: dict[str, Any] = shielding.summary()
            f1 = float(raw["N_eff_over_N"])

            if not spacing_feasible:
                self.spacing_prescreen_skip_count += 1
                raw.update(
                    {
                        "aep_gwh": np.nan,
                        "aep_loss_gwh": np.nan,
                        "minimum_rotor_grid_velocity_m_s": np.nan,
                        "negative_rotor_grid_point_count": 0,
                        "rotor_velocity_feasible": True,
                    }
                )
                normalised_spacing_violation = max(
                    0.0,
                    maximum_spacing_constraint / self.minimum_spacing_m,
                )
                f0 = self.INVALID_AEP_OBJECTIVE + normalised_spacing_violation
                rotor_velocity_constraint = 0.0
                floris_evaluated = False
            else:
                raw = self.evaluate_raw_layout(xy)
                self.floris_evaluation_count += 1
                floris_evaluated = True
                minimum_velocity = float(
                    raw["minimum_rotor_grid_velocity_m_s"]
                )
                if self.minimum_observed_rotor_grid_velocity_m_s is None:
                    self.minimum_observed_rotor_grid_velocity_m_s = minimum_velocity
                else:
                    self.minimum_observed_rotor_grid_velocity_m_s = min(
                        self.minimum_observed_rotor_grid_velocity_m_s,
                        minimum_velocity,
                    )

                rotor_velocity_constraint = -minimum_velocity
                if bool(raw["rotor_velocity_feasible"]):
                    f0 = raw["aep_loss_gwh"] / self.aep_loss_reference_gwh
                else:
                    self.negative_rotor_velocity_candidate_count += 1
                    f0 = self.INVALID_AEP_OBJECTIVE + (
                        max(0.0, -minimum_velocity) / self.rotor_diameter_m
                    )

            constraints = np.concatenate(
                [spacing_constraints, np.asarray([rotor_velocity_constraint])]
            )
            objective_rows.append([f0, f1])
            constraint_rows.append(constraints)

            if self.record_evaluations:
                self._evaluation_rows.append(
                    {
                        "evaluation_index": len(self._evaluation_rows),
                        **{
                            coordinate_name: float(coordinate_value)
                            for turbine_id, point in enumerate(xy)
                            for coordinate_name, coordinate_value in (
                                (f"x_{turbine_id}_m", point[0]),
                                (f"y_{turbine_id}_m", point[1]),
                            )
                        },
                        "aep_gwh": raw["aep_gwh"],
                        "aep_loss_gwh": raw["aep_loss_gwh"],
                        "floris_evaluated": floris_evaluated,
                        "spacing_feasible": spacing_feasible,
                        "minimum_rotor_grid_velocity_m_s": raw[
                            "minimum_rotor_grid_velocity_m_s"
                        ],
                        "negative_rotor_grid_point_count": raw[
                            "negative_rotor_grid_point_count"
                        ],
                        "rotor_velocity_feasible": raw[
                            "rotor_velocity_feasible"
                        ],
                        "unique_exposed_width_m": raw["unique_exposed_width_m"],
                        "N_eff": raw["N_eff"],
                        "N_eff_over_N": raw["N_eff_over_N"],
                        "shielding_fraction": raw["shielding_fraction"],
                        "P_enc": raw["P_enc"],
                        "P_free": raw["P_free"],
                        "number_of_merged_intervals": raw[
                            "number_of_merged_intervals"
                        ],
                        "f0_normalised_aep_loss": f0,
                        "f1_normalised_N_eff": f1,
                        "maximum_spacing_constraint": maximum_spacing_constraint,
                        "rotor_velocity_constraint": rotor_velocity_constraint,
                    }
                )

        out["F"] = np.asarray(objective_rows, dtype=float)
        out["G"] = np.asarray(constraint_rows, dtype=float)

    def evaluation_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._evaluation_rows)

    def candidate_screening_summary(self) -> dict[str, Any]:
        return {
            "total_candidate_evaluations": int(self.total_candidate_evaluations),
            "floris_evaluation_count": int(self.floris_evaluation_count),
            "spacing_prescreen_skip_count": int(self.spacing_prescreen_skip_count),
            "negative_rotor_velocity_candidate_count": int(
                self.negative_rotor_velocity_candidate_count
            ),
            "minimum_observed_rotor_grid_velocity_m_s": (
                None
                if self.minimum_observed_rotor_grid_velocity_m_s is None
                else float(self.minimum_observed_rotor_grid_velocity_m_s)
            ),
        }

    def settings_summary(self) -> dict[str, Any]:
        return {
            "problem": "ShieldingAEPProblem",
            "number_of_turbines": self.number_of_turbines,
            "rotor_diameter_m": self.rotor_diameter_m,
            "minimum_spacing_D": float(self.scenario.site.min_spacing_D),
            "minimum_spacing_m": self.minimum_spacing_m,
            "spacing_constraint_count": self.pair_spacing_constraint_count,
            "rotor_velocity_constraint_count": 1,
            "total_inequality_constraint_count": int(self.n_ieq_constr),
            "spacing_constraint_tolerance_m": self.spacing_constraint_tolerance_m,
            "bird_heading_deg_toward": self.bird_heading_deg_toward % 360.0,
            "corridor_width_m": self.corridor_width_m,
            "aep_ceiling_gwh": self.aep_ceiling_gwh,
            "aep_loss_reference_gwh": self.aep_loss_reference_gwh,
            "aep_objective": (
                "(AEP_ceiling - AEP) / aep_loss_reference_gwh"
            ),
            "shielding_objective": "N_eff / number_of_turbines",
            "reported_primary_metric": "N_eff = L_exp / D",
            "N_eff_numerical_decimals": N_EFF_DECIMALS,
            "candidate_prescreen": (
                "minimum spacing is checked before FLORIS evaluation"
            ),
            "rotor_velocity_constraint": (
                "minimum FLORIS rotor-grid velocity must be >= 0 m/s"
            ),
        }
