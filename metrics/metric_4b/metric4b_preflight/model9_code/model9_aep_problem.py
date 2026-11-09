"""pymoo coupling for FLORIS AEP and Model 9 rotor-transit intensity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

from model9_batching import summarise_model9_batches
from model9_rotor_transit import Model9Population, RotorTransitModel


class Model9AEPRotorTransitProblem(Problem):
    """Two-objective wind-farm layout problem for AEP and rotor exposure.

    The minimisation objectives returned to NSGA-II are normalised versions of

    ``f0 = AEP_ceiling - AEP``

    and

    ``f1 = E_RT_CE`` by default.

    Positive normalisation constants do not change Pareto dominance.  The
    original sampled ``E_RT``, conditional ``E_RT_CE``, Model 8 encounter,
    and unnormalised AEP metrics are all retained in the evaluation log.
    """

    def __init__(
        self,
        *,
        scenario: Any,
        aep_model: Any,
        rotor_transit_model: RotorTransitModel,
        trajectory_populations: Sequence[Model9Population],
        aep_ceiling_gwh: float,
        aep_loss_reference_gwh: float,
        rotor_transit_reference: float,
        bird_objective_metric: str = "E_RT_CE",
        record_evaluations: bool = True,
        spacing_constraint_tolerance_m: float = 1.0e-6,
    ) -> None:
        populations = tuple(trajectory_populations)
        if not populations:
            raise ValueError("At least one Model 9 population is required.")
        self.scenario = scenario
        self.aep_model = aep_model
        self.rotor_transit_model = rotor_transit_model
        self.trajectory_populations = populations
        self.aep_ceiling_gwh = float(aep_ceiling_gwh)
        self.aep_loss_reference_gwh = float(aep_loss_reference_gwh)
        self.rotor_transit_reference = float(rotor_transit_reference)
        self.bird_objective_metric = str(bird_objective_metric)
        self.record_evaluations = bool(record_evaluations)
        self.spacing_constraint_tolerance_m = float(
            spacing_constraint_tolerance_m
        )

        if self.aep_loss_reference_gwh <= 0.0:
            raise ValueError("aep_loss_reference_gwh must be positive.")
        if self.rotor_transit_reference <= 0.0:
            raise ValueError("rotor_transit_reference must be positive.")
        if self.bird_objective_metric not in {"E_RT", "E_RT_CE"}:
            raise ValueError(
                "bird_objective_metric must be either 'E_RT' or 'E_RT_CE'."
            )
        if (
            self.bird_objective_metric == "E_RT_CE"
            and rotor_transit_model.flight_height_distribution is None
        ):
            raise ValueError(
                "E_RT_CE requires RotorTransitModel to have a flight-height "
                "distribution. Use bird_objective_metric='E_RT' only when the "
                "sampled audit estimator is intentionally required."
            )
        if self.spacing_constraint_tolerance_m < 0.0:
            raise ValueError("spacing_constraint_tolerance_m cannot be negative.")

        self.number_of_turbines = int(scenario.site.n_turbines)
        self.rotor_diameter_m = float(
            rotor_transit_model.rotor_geometry.rotor_diameter_m
        )
        self.minimum_spacing_m = float(
            scenario.site.min_spacing_D * self.rotor_diameter_m
        )
        pair_count = self.number_of_turbines * (self.number_of_turbines - 1) // 2
        lower = np.tile(np.array([0.0, 0.0]), self.number_of_turbines)
        upper = np.tile(
            np.array([scenario.site.width_m, scenario.site.height_m], dtype=float),
            self.number_of_turbines,
        )
        super().__init__(
            n_var=2 * self.number_of_turbines,
            n_obj=2,
            n_ieq_constr=pair_count,
            xl=lower,
            xu=upper,
            elementwise=False,
        )
        self._evaluation_rows: list[dict[str, Any]] = []

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
        exposure = summarise_model9_batches(
            self.rotor_transit_model,
            points,
            self.trajectory_populations,
        )
        return {
            "aep_gwh": aep_gwh,
            "aep_loss_gwh": float(self.aep_ceiling_gwh - aep_gwh),
            **exposure,
        }

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        decision_vectors = np.atleast_2d(np.asarray(X, dtype=float))
        objective_rows = []
        constraint_rows = []
        for vector in decision_vectors:
            xy = vector.reshape(self.number_of_turbines, 2)
            raw = self.evaluate_raw_layout(xy)
            f0 = raw["aep_loss_gwh"] / self.aep_loss_reference_gwh
            f1 = raw[self.bird_objective_metric] / self.rotor_transit_reference
            objectives = [f0, f1]
            constraints = self.spacing_constraints(xy)
            objective_rows.append(objectives)
            constraint_rows.append(constraints)

            if self.record_evaluations:
                coordinate_values = {
                    name: float(xy[turbine_id, coordinate_id])
                    for turbine_id in range(self.number_of_turbines)
                    for coordinate_id, name in enumerate(
                        (f"x_{turbine_id}_m", f"y_{turbine_id}_m")
                    )
                }
                pairwise_distances = self.minimum_spacing_m - (
                    constraints + self.spacing_constraint_tolerance_m
                )
                self._evaluation_rows.append(
                    {
                        "evaluation_index": len(self._evaluation_rows),
                        **coordinate_values,
                        "objective_batch_count": raw["objective_batch_count"],
                        "aep_gwh": raw["aep_gwh"],
                        "aep_loss_gwh": raw["aep_loss_gwh"],
                        "N_started": raw["N_started"],
                        "N_completed": raw["N_completed"],
                        "N_failed": raw["N_failed"],
                        "N_approach_events": raw["N_approach_events"],
                        "N_response_events": raw["N_response_events"],
                        "N_enc_farm": raw["N_enc_farm"],
                        "E_enc": raw["E_enc"],
                        "E_enc_batch_sd": raw["E_enc_batch_sd"],
                        "E_enc_batch_se": raw["E_enc_batch_se"],
                        "P_ge_1": raw["P_ge_1"],
                        "N_RT_farm": raw["N_RT_farm"],
                        "E_RT": raw["E_RT"],
                        "E_RT_batch_sd": raw["E_RT_batch_sd"],
                        "E_RT_batch_se": raw["E_RT_batch_se"],
                        "P_RT": raw["P_RT"],
                        "P_RT_batch_sd": raw["P_RT_batch_sd"],
                        "P_RT_batch_se": raw["P_RT_batch_se"],
                        "N_RT_CE_farm": raw.get("N_RT_CE_farm"),
                        "E_RT_CE": raw.get("E_RT_CE"),
                        "E_RT_CE_batch_sd": raw.get("E_RT_CE_batch_sd"),
                        "E_RT_CE_batch_se": raw.get("E_RT_CE_batch_se"),
                        "P_RT_CE": raw.get("P_RT_CE"),
                        "P_RT_CE_batch_sd": raw.get("P_RT_CE_batch_sd"),
                        "P_RT_CE_batch_se": raw.get("P_RT_CE_batch_se"),
                        "P_height_band": raw["P_height_band"],
                        "RT_to_encounter_ratio": raw[
                            "RT_to_encounter_ratio"
                        ],
                        "RT_CE_to_encounter_ratio": raw.get(
                            "RT_CE_to_encounter_ratio"
                        ),
                        "mean_route_extension_m": raw[
                            "mean_route_extension_m"
                        ],
                        "f0_normalised_aep_loss": f0,
                        "f1_normalised_rotor_transits": f1,
                        "bird_objective_metric": self.bird_objective_metric,
                        "bird_objective_raw": raw[self.bird_objective_metric],
                        "minimum_pairwise_spacing_m": float(
                            np.min(pairwise_distances)
                        ),
                        "spacing_violation_count": int(
                            np.count_nonzero(constraints > 0.0)
                        ),
                        "maximum_spacing_constraint": float(
                            np.max(constraints)
                        ),
                    }
                )

        out["F"] = np.asarray(objective_rows, dtype=float)
        out["G"] = np.asarray(constraint_rows, dtype=float)

    def evaluation_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._evaluation_rows)

    def settings_summary(self) -> dict[str, Any]:
        model = self.rotor_transit_model
        return {
            "problem": "Model9AEPRotorTransitProblem",
            "primary_bird_objective": self.bird_objective_metric,
            "sampled_audit_metric": "E_RT",
            "conditional_estimator": (
                "E_RT_CE integrates the discrete flight-height distribution "
                "conditional on each horizontal path"
                if model.flight_height_distribution is not None
                else None
            ),
            "endpoint": "rotor transit, not blade collision",
            "number_of_turbines": self.number_of_turbines,
            "minimum_spacing_D": float(self.scenario.site.min_spacing_D),
            "minimum_spacing_m": self.minimum_spacing_m,
            "spacing_constraint_count": int(self.n_ieq_constr),
            "spacing_constraint_tolerance_m": self.spacing_constraint_tolerance_m,
            "aep_ceiling_gwh": self.aep_ceiling_gwh,
            "aep_loss_reference_gwh": self.aep_loss_reference_gwh,
            "rotor_transit_reference": self.rotor_transit_reference,
            "objective_batch_count": len(self.trajectory_populations),
            "N_in": int(
                sum(
                    population.number_of_trajectories
                    for population in self.trajectory_populations
                )
            ),
            "N_in_per_batch": [
                population.number_of_trajectories
                for population in self.trajectory_populations
            ],
            "trajectory_seeds": [
                population.trajectory_seed
                for population in self.trajectory_populations
            ],
            "flight_height_seeds": [
                population.flight_height_seed
                for population in self.trajectory_populations
            ],
            "flight_height_distribution_names": sorted(
                {
                    population.flight_height_distribution_name
                    for population in self.trajectory_populations
                }
            ),
            "rotor_plane_heading_deg_toward": (
                model.rotor_plane_heading_deg_toward
            ),
            "rotor_geometry": model.rotor_geometry.settings(),
            "behaviour": model.behaviour.settings(self.rotor_diameter_m),
        }
