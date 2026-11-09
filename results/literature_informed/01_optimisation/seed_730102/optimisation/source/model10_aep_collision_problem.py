"""Two-objective Model 10 layout evaluation using the frozen Model 8/9 path.

The bird objective is the sum of height-weighted local Band contributions
across accepted bird-turbine crossings divided by the number of entrants.
This is expected *potential strikes per entering bird*: there is no extra
avoidance factor, annual bird flux, or mortality scaling. Model 9's sampled
transits and conditional nominal-disc exposure remain separate diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pymoo.core.problem import Problem

from model10_band_strike import BandCrossingAccumulator, BandScenario, ProjectedBladeStations
from model9_rotor_transit import Model9Population, RotorTransitModel


class Model10AEPCollisionProblem(Problem):
    """Maximise unchanged FLORIS AEP and minimise Model 10 potential strikes.

    ``F[:,0] = (wake_free_AEP - layout_AEP) / aep_loss_reference``;
    ``F[:,1] = expected_potential_strikes_per_entrant / strike_reference``.
    Exactly the existing 4D pairwise spacing constraints apply.
    """

    def __init__(
        self, *, scenario: Any, aep_model: Any, rotor_model: RotorTransitModel,
        populations: Sequence[Model9Population], blade: ProjectedBladeStations,
        band_scenario: BandScenario, aep_ceiling_gwh: float,
        aep_loss_reference_gwh: float, strike_reference: float,
        spacing_constraint_tolerance_m: float = 1e-6,
        record_evaluations: bool = True,
    ) -> None:
        self.scenario = scenario
        self.aep_model = aep_model
        self.rotor_model = rotor_model
        self.populations = tuple(populations)
        self.blade = blade
        self.band_scenario = band_scenario
        self.aep_ceiling_gwh = float(aep_ceiling_gwh)
        self.aep_loss_reference_gwh = float(aep_loss_reference_gwh)
        self.strike_reference = float(strike_reference)
        self.spacing_constraint_tolerance_m = float(spacing_constraint_tolerance_m)
        self.number_of_turbines = int(scenario.site.n_turbines)
        self.minimum_spacing_m = float(scenario.site.min_spacing_D * rotor_model.rotor_geometry.rotor_diameter_m)
        self.record_evaluations = bool(record_evaluations)
        self.evaluation_rows: list[dict[str, float | int]] = []
        if not self.populations or any(p.number_of_trajectories <= 0 for p in self.populations):
            raise ValueError("At least one nonempty population is required")
        if not all(np.isfinite(v) and v > 0 for v in (
            self.aep_ceiling_gwh, self.aep_loss_reference_gwh, self.strike_reference,
            self.minimum_spacing_m,
        )):
            raise ValueError("AEP and strike references and spacing must be finite and positive")
        if self.spacing_constraint_tolerance_m < 0 or not np.isfinite(self.spacing_constraint_tolerance_m):
            raise ValueError("Spacing tolerance must be finite and nonnegative")
        if rotor_model.flight_height_distribution is None:
            raise ValueError("The Band objective requires Model 9's discrete FHD")
        lower = np.tile([0.0, 0.0], self.number_of_turbines)
        upper = np.tile([scenario.site.width_m, scenario.site.height_m], self.number_of_turbines)
        super().__init__(
            n_var=2 * self.number_of_turbines, n_obj=2,
            n_ieq_constr=self.number_of_turbines * (self.number_of_turbines - 1) // 2,
            xl=lower, xu=upper, elementwise=False,
        )

    def spacing_constraints(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=float).reshape(self.number_of_turbines, 2)
        return np.asarray([
            self.minimum_spacing_m - float(np.linalg.norm(points[i] - points[j]))
            - self.spacing_constraint_tolerance_m
            for i in range(self.number_of_turbines)
            for j in range(i + 1, self.number_of_turbines)
        ])

    def evaluate_raw_layout(self, xy: np.ndarray) -> dict[str, Any]:
        points = np.asarray(xy, dtype=float).reshape(self.number_of_turbines, 2)
        aep_gwh = float(self.aep_model.evaluate(points))
        if not np.isfinite(aep_gwh):
            raise ValueError("FLORIS AEP must be finite")
        trajectory_count = 0
        sampled_transits = 0
        conditional_transits = 0.0
        potential_strikes = 0.0
        per_turbine_strikes = np.zeros(self.number_of_turbines)
        accepted_pairs = 0
        batch_rows: list[dict[str, float | int]] = []
        for population in self.populations:
            accumulator = BandCrossingAccumulator(
                blade=self.blade, scenario=self.band_scenario,
                number_of_trajectories=population.number_of_trajectories,
                number_of_turbines=self.number_of_turbines,
            )
            result = self.rotor_model.simulate(points, population, crossing_callback=accumulator)
            result.validate_accounting()
            n = result.number_started
            strike_sum = float(np.sum(accumulator.contribution_matrix))
            if not np.isfinite(strike_sum) or strike_sum < 0:
                raise ValueError("Band contribution must be finite and nonnegative")
            potential_strikes += strike_sum
            per_turbine_strikes += np.sum(accumulator.contribution_matrix, axis=0)
            accepted_pairs += int(np.count_nonzero(accumulator.contribution_matrix))
            trajectory_count += n
            sampled_transits += result.total_transits
            conditional_transits += result.expected_total_transits
            batch_rows.append({
                "trajectory_seed": int(population.trajectory_seed),
                "flight_height_seed": int(population.flight_height_seed),
                "N_started": n,
                "E_RT_CE": result.conditional_rotor_transit_intensity,
                "E_potential_strike": strike_sum / n,
            })
        return {
            "aep_gwh": aep_gwh,
            "aep_loss_gwh": self.aep_ceiling_gwh - aep_gwh,
            "N_started": trajectory_count,
            "N_RT_farm": sampled_transits,
            "E_RT": sampled_transits / trajectory_count,
            "N_RT_CE_farm": conditional_transits,
            "E_RT_CE": conditional_transits / trajectory_count,
            "N_potential_strikes": potential_strikes,
            "E_potential_strike": potential_strikes / trajectory_count,
            "accepted_bird_turbine_pairs": accepted_pairs,
            "per_turbine_potential_strikes": per_turbine_strikes,
            "batch_rows": batch_rows,
        }

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        objectives = []
        constraints = []
        for vector in np.atleast_2d(np.asarray(X, dtype=float)):
            xy = vector.reshape(self.number_of_turbines, 2)
            raw = self.evaluate_raw_layout(xy)
            f0 = raw["aep_loss_gwh"] / self.aep_loss_reference_gwh
            f1 = raw["E_potential_strike"] / self.strike_reference
            g = self.spacing_constraints(xy)
            objectives.append((f0, f1))
            constraints.append(g)
            if self.record_evaluations:
                self.evaluation_rows.append({
                    "evaluation_index": len(self.evaluation_rows),
                    "aep_gwh": raw["aep_gwh"],
                    "aep_loss_gwh": raw["aep_loss_gwh"],
                    "N_started": raw["N_started"],
                    "N_RT_farm": raw["N_RT_farm"],
                    "E_RT_CE": raw["E_RT_CE"],
                    "N_potential_strikes": raw["N_potential_strikes"],
                    "E_potential_strike": raw["E_potential_strike"],
                    "accepted_bird_turbine_pairs": raw["accepted_bird_turbine_pairs"],
                    "f0_normalised_aep_loss": f0,
                    "f1_normalised_potential_strikes": f1,
                    "maximum_spacing_constraint": float(np.max(g)),
                })
        out["F"] = np.asarray(objectives, dtype=float)
        out["G"] = np.asarray(constraints, dtype=float)
