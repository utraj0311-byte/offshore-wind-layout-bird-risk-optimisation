"""pymoo coupling for FLORIS AEP and Model 8 turbine encounters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

from model8_batching import summarise_meso_batches
from model8_meso_response import MesoResponseModel, TrajectoryPopulation


class Model8AEPEncounterProblem(Problem):
    """Two-objective continuous wind-farm layout problem.

    The minimisation objectives returned to NSGA-II are

    ``f0 = (AEP_ceiling - AEP) / AEP_loss_reference``

    and

    ``f1 = E_enc / encounter_reference``.

    The first reference can retain the locked Model 10 energy normalisation.
    The second must be fixed before optimisation, preferably from the
    no-response energy-endpoint layout using the same trajectory population.
    """

    def __init__(
        self,
        *,
        scenario: Any,
        aep_model: Any,
        meso_model: MesoResponseModel,
        trajectory_population: TrajectoryPopulation | None = None,
        trajectory_populations: Sequence[TrajectoryPopulation] | None = None,
        rotor_diameter_m: float,
        aep_ceiling_gwh: float,
        aep_loss_reference_gwh: float,
        encounter_reference: float,
        record_evaluations: bool = True,
        spacing_constraint_tolerance_m: float = 1.0e-6,
    ) -> None:
        self.scenario = scenario
        self.aep_model = aep_model
        self.meso_model = meso_model
        if trajectory_population is not None and trajectory_populations is not None:
            raise ValueError(
                "Pass trajectory_population or trajectory_populations, not both."
            )
        if trajectory_populations is None:
            if trajectory_population is None:
                raise ValueError("At least one trajectory population is required.")
            populations = (trajectory_population,)
        else:
            populations = tuple(trajectory_populations)
            if not populations:
                raise ValueError("At least one trajectory population is required.")
        self.trajectory_populations = populations
        # Retain the old attribute for callers that inspect single-batch runs.
        self.trajectory_population = populations[0]
        self.rotor_diameter_m = float(rotor_diameter_m)
        self.aep_ceiling_gwh = float(aep_ceiling_gwh)
        self.aep_loss_reference_gwh = float(aep_loss_reference_gwh)
        self.encounter_reference = float(encounter_reference)
        self.record_evaluations = bool(record_evaluations)
        self.spacing_constraint_tolerance_m = float(
            spacing_constraint_tolerance_m
        )

        if self.rotor_diameter_m <= 0.0:
            raise ValueError("rotor_diameter_m must be positive.")
        if self.aep_loss_reference_gwh <= 0.0:
            raise ValueError("aep_loss_reference_gwh must be positive.")
        if self.encounter_reference <= 0.0:
            raise ValueError("encounter_reference must be positive.")
        if self.spacing_constraint_tolerance_m < 0.0:
            raise ValueError("spacing_constraint_tolerance_m cannot be negative.")

        self.number_of_turbines = int(scenario.site.n_turbines)
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
        for i in range(self.number_of_turbines):
            for j in range(i + 1, self.number_of_turbines):
                distance_m = float(np.linalg.norm(points[i] - points[j]))
                constraints.append(
                    self.minimum_spacing_m
                    - distance_m
                    - self.spacing_constraint_tolerance_m
                )
        return np.asarray(constraints, dtype=float)

    def evaluate_raw_layout(self, xy: np.ndarray) -> dict[str, Any]:
        points = np.asarray(xy, dtype=float).reshape(self.number_of_turbines, 2)
        aep_gwh = float(self.aep_model.evaluate(points))
        meso = summarise_meso_batches(
            self.meso_model, points, self.trajectory_populations
        )
        return {
            "aep_gwh": aep_gwh,
            "aep_loss_gwh": float(self.aep_ceiling_gwh - aep_gwh),
            **meso,
        }

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        decision_vectors = np.atleast_2d(np.asarray(X, dtype=float))
        objective_rows = []
        constraint_rows = []

        for vector in decision_vectors:
            xy = vector.reshape(self.number_of_turbines, 2)
            raw = self.evaluate_raw_layout(xy)
            f0 = raw["aep_loss_gwh"] / self.aep_loss_reference_gwh
            f1 = raw["E_enc"] / self.encounter_reference
            objective_rows.append([f0, f1])
            constraints = self.spacing_constraints(xy)
            constraint_rows.append(constraints)

            if self.record_evaluations:
                self._evaluation_rows.append(
                    {
                        "evaluation_index": len(self._evaluation_rows),
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
                        "E_enc_batch_min": raw["E_enc_batch_min"],
                        "E_enc_batch_max": raw["E_enc_batch_max"],
                        "P_ge_1": raw["P_ge_1"],
                        "P_ge_1_batch_sd": raw["P_ge_1_batch_sd"],
                        "P_ge_1_batch_se": raw["P_ge_1_batch_se"],
                        "mean_route_extension_m": raw["mean_route_extension_m"],
                        "f0_normalised_aep_loss": f0,
                        "f1_normalised_encounters": f1,
                        "maximum_spacing_constraint": float(np.max(constraints)),
                    }
                )

        out["F"] = np.asarray(objective_rows, dtype=float)
        out["G"] = np.asarray(constraint_rows, dtype=float)

    def evaluation_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._evaluation_rows)

    def settings_summary(self) -> dict[str, Any]:
        return {
            "problem": "Model8AEPEncounterProblem",
            "number_of_turbines": self.number_of_turbines,
            "rotor_diameter_m": self.rotor_diameter_m,
            "minimum_spacing_D": float(self.scenario.site.min_spacing_D),
            "minimum_spacing_m": self.minimum_spacing_m,
            "spacing_constraint_count": int(self.n_ieq_constr),
            "spacing_constraint_tolerance_m": self.spacing_constraint_tolerance_m,
            "aep_ceiling_gwh": self.aep_ceiling_gwh,
            "aep_loss_reference_gwh": self.aep_loss_reference_gwh,
            "encounter_reference": self.encounter_reference,
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
            "common_random_number_seeds": [
                population.seed for population in self.trajectory_populations
            ],
            "behaviour": self.meso_model.behaviour.settings(self.rotor_diameter_m),
        }
