"""Run FLORIS AEP versus Model 8 encounter optimisation with NSGA-II.

Place this file beside the project's normalised ``config.py`` and
``floris_aep.py`` modules.  The supplied FLORIS YAML and a feasible reference
layout are passed as command-line paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

# Permit the packaged ``model8_code`` folder to sit directly inside the
# existing bird_layout_opt project while importing its config.py and
# floris_aep.py modules from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Scenario
from floris_aep import FLORISAEPModel
from model8_aep_problem import Model8AEPEncounterProblem
from model8_batching import (
    nondominated_mask,
    robust_nondominated_mask,
    select_anchor_positions,
    summarise_meso_batches,
)
from model8_meso_response import (
    BehaviourScenario,
    MesoResponseModel,
    TrajectoryPopulation,
)
from run_model8_validation import make_fixed_layouts, pairwise_distances


def safe_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def load_layout_csv(path: Path, number_of_turbines: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Layout file not found: {path}")
    dataframe = pd.read_csv(path)
    if "turbine_id" in dataframe.columns:
        dataframe = dataframe.sort_values("turbine_id")
    if not {"x_m", "y_m"}.issubset(dataframe.columns):
        raise ValueError(f"{path} must contain x_m and y_m columns.")
    xy = dataframe[["x_m", "y_m"]].to_numpy(dtype=float)
    if xy.shape != (number_of_turbines, 2):
        raise ValueError(
            f"Expected {(number_of_turbines, 2)} coordinates, received {xy.shape}."
        )
    if not np.all(np.isfinite(xy)):
        raise ValueError("Layout coordinates must be finite.")
    return xy


def layout_is_feasible(
    xy: np.ndarray,
    *,
    width_m: float,
    height_m: float,
    minimum_spacing_m: float,
) -> bool:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] > width_m)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] > height_m)
    ):
        return False
    return bool(np.min(pairwise_distances(points)) + 1.0e-9 >= minimum_spacing_m)


def random_feasible_layout(
    *,
    rng: np.random.Generator,
    number_of_turbines: int,
    width_m: float,
    height_m: float,
    minimum_spacing_m: float,
    maximum_restarts: int = 300,
    attempts_per_turbine: int = 3000,
) -> np.ndarray:
    for _ in range(maximum_restarts):
        points: list[np.ndarray] = []
        for _turbine in range(number_of_turbines):
            accepted = False
            for _ in range(attempts_per_turbine):
                candidate = rng.uniform([0.0, 0.0], [width_m, height_m])
                if not points or all(
                    np.linalg.norm(candidate - existing) >= minimum_spacing_m
                    for existing in points
                ):
                    points.append(candidate)
                    accepted = True
                    break
            if not accepted:
                break
        if len(points) == number_of_turbines:
            return np.asarray(points, dtype=float)
    raise RuntimeError("Could not construct a random feasible layout.")


class FeasibleSeededSampling(Sampling):
    """Deterministic feasible initial population with controlled seed layouts."""

    def __init__(
        self,
        *,
        scenario: Scenario,
        rotor_diameter_m: float,
        exact_seed_layouts: dict[str, np.ndarray],
        seed: int,
        jitter_scale_D: float = 0.20,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.rotor_diameter_m = float(rotor_diameter_m)
        self.exact_seed_layouts = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in exact_seed_layouts.items()
        }
        self.seed = int(seed)
        self.jitter_scale_m = float(jitter_scale_D * rotor_diameter_m)
        self.last_labels: list[str] = []

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        width = float(self.scenario.site.width_m)
        height = float(self.scenario.site.height_m)
        n_turbines = int(self.scenario.site.n_turbines)
        minimum_spacing = float(
            self.scenario.site.min_spacing_D * self.rotor_diameter_m
        )
        accepted: list[np.ndarray] = []
        labels: list[str] = []

        def add(candidate_xy: np.ndarray, label: str) -> bool:
            candidate_xy = np.asarray(candidate_xy, dtype=float).reshape(n_turbines, 2)
            if not layout_is_feasible(
                candidate_xy,
                width_m=width,
                height_m=height,
                minimum_spacing_m=minimum_spacing,
            ):
                return False
            vector = candidate_xy.reshape(-1)
            if any(np.max(np.abs(vector - item)) <= 1.0e-7 for item in accepted):
                return False
            accepted.append(vector)
            labels.append(label)
            return True

        for name, layout in self.exact_seed_layouts.items():
            if len(accepted) >= n_samples:
                break
            if not add(layout, f"exact_{name}"):
                raise ValueError(f"Exact seed layout is infeasible or duplicated: {name}")

        seed_items = list(self.exact_seed_layouts.items())
        jitter_target = max(0, min(n_samples // 2, n_samples - len(accepted)))
        attempts = 0
        while len(accepted) < len(self.exact_seed_layouts) + jitter_target:
            attempts += 1
            if attempts > 10000:
                break
            name, base = seed_items[attempts % len(seed_items)]
            translation = rng.normal(0.0, 0.35 * self.jitter_scale_m, size=2)
            local = rng.normal(0.0, self.jitter_scale_m, size=base.shape)
            add(base + translation + local, f"jitter_{name}")

        while len(accepted) < n_samples:
            candidate = random_feasible_layout(
                rng=rng,
                number_of_turbines=n_turbines,
                width_m=width,
                height_m=height,
                minimum_spacing_m=minimum_spacing,
            )
            add(candidate, "random_feasible")

        self.last_labels = labels[:n_samples]
        return np.asarray(accepted[:n_samples], dtype=float)


class HypervolumeCallback(Callback):
    def __init__(self, reference_point: tuple[float, float] = (1.1, 1.1)) -> None:
        super().__init__()
        self.reference_point = np.asarray(reference_point, dtype=float)
        self.indicator = HV(ref_point=self.reference_point)
        self.data["generation"] = []
        self.data["normalised_hypervolume"] = []
        self.data["minimum_f0"] = []
        self.data["minimum_f1"] = []

    def notify(self, algorithm) -> None:
        objectives = np.asarray(algorithm.pop.get("F"), dtype=float)
        constraints = algorithm.pop.get("G")
        if constraints is not None:
            constraints = np.asarray(constraints, dtype=float)
            if constraints.ndim == 1:
                constraints = constraints[:, None]
            objectives = objectives[np.all(constraints <= 1.0e-9, axis=1)]
        if len(objectives) == 0:
            hypervolume = np.nan
            minimum_f0 = np.nan
            minimum_f1 = np.nan
        else:
            hypervolume = float(self.indicator(objectives))
            minimum_f0 = float(np.min(objectives[:, 0]))
            minimum_f1 = float(np.min(objectives[:, 1]))
        self.data["generation"].append(int(algorithm.n_gen))
        self.data["normalised_hypervolume"].append(hypervolume)
        self.data["minimum_f0"].append(minimum_f0)
        self.data["minimum_f1"].append(minimum_f1)


def build_pareto_dataframe(
    X: np.ndarray,
    F: np.ndarray,
    problem: Model8AEPEncounterProblem,
) -> pd.DataFrame:
    rows = []
    for solution_index, (vector, objectives) in enumerate(zip(X, F)):
        xy = vector.reshape(problem.number_of_turbines, 2)
        raw = problem.evaluate_raw_layout(xy)
        constraints = problem.spacing_constraints(xy)
        rows.append(
            {
                "solution_index": solution_index,
                "objective_batch_count": raw["objective_batch_count"],
                "aep_gwh": raw["aep_gwh"],
                "aep_loss_gwh": raw["aep_loss_gwh"],
                "N_started": raw["N_started"],
                "N_completed": raw["N_completed"],
                "N_failed": raw["N_failed"],
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
                "f0_normalised_aep_loss": objectives[0],
                "f1_normalised_encounters": objectives[1],
                "minimum_spacing_m": float(np.min(pairwise_distances(xy))),
                "maximum_spacing_constraint": float(np.max(constraints)),
                "spacing_violations": int(np.count_nonzero(constraints > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def find_anchor_indices(
    dataframe: pd.DataFrame,
    *,
    encounter_column: str = "E_enc",
) -> dict[str, int]:
    """Return dataframe indices for the two endpoints and ideal-point knee."""

    positions = select_anchor_positions(
        dataframe["aep_gwh"].to_numpy(dtype=float),
        dataframe["aep_loss_gwh"].to_numpy(dtype=float),
        dataframe[encounter_column].to_numpy(dtype=float),
    )
    return {
        label: int(dataframe.index[position])
        for label, position in positions.items()
    }


def save_anchor_layouts(
    *,
    X: np.ndarray,
    dataframe: pd.DataFrame,
    anchors: dict[str, int],
    problem: Model8AEPEncounterProblem,
    output_dir: Path,
    raw_by_solution: dict[int, dict] | None = None,
    filename_prefix: str = "",
) -> pd.DataFrame:
    layout_dir = output_dir / "layouts"
    layout_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, dataframe_index in anchors.items():
        row = dataframe.loc[dataframe_index]
        solution_index = int(row["solution_index"])
        xy = X[solution_index].reshape(problem.number_of_turbines, 2)
        raw = (
            problem.evaluate_raw_layout(xy)
            if raw_by_solution is None
            else raw_by_solution[solution_index]
        )
        layout_frame = pd.DataFrame(
            {
                "turbine_id": np.arange(problem.number_of_turbines),
                "x_m": xy[:, 0],
                "y_m": xy[:, 1],
                "N_enc_i": raw["per_turbine_encounters"],
            }
        )
        layout_frame.to_csv(
            layout_dir / f"{filename_prefix}{label}_layout.csv", index=False
        )
        rows.append({"layout": label, **row.to_dict()})
    return pd.DataFrame(rows)


def plot_pareto(
    dataframe: pd.DataFrame,
    anchors: dict[str, int],
    output_path: Path,
    *,
    encounter_column: str = "E_enc",
    nondominated_column: str | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    if nondominated_column is None:
        axis.scatter(
            dataframe[encounter_column], dataframe["aep_gwh"], s=28, alpha=0.75
        )
    else:
        mask = dataframe[nondominated_column].astype(bool).to_numpy()
        axis.scatter(
            dataframe.loc[~mask, encounter_column],
            dataframe.loc[~mask, "aep_gwh"],
            s=24,
            alpha=0.35,
            color="0.55",
            label="training-front point rejected by fresh validation",
        )
        axis.scatter(
            dataframe.loc[mask, encounter_column],
            dataframe.loc[mask, "aep_gwh"],
            s=32,
            alpha=0.8,
            label="fresh-validated Pareto point",
        )
    for label, index in anchors.items():
        row = dataframe.loc[index]
        axis.scatter(
            row[encounter_column],
            row["aep_gwh"],
            s=80,
            marker="*",
            label=label,
        )
    axis.set_xlabel(
        "Fresh encounter intensity, E_enc"
        if encounter_column.startswith("fresh_")
        else "Optimisation encounter intensity, E_enc"
    )
    axis.set_ylabel("AEP (GWh/year)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_population_batches(
    *,
    scenario: Scenario,
    heading_deg_toward: float,
    trajectories_per_batch: int,
    batch_count: int,
    longitudinal_margin_m: float,
    first_seed: int,
) -> tuple[TrajectoryPopulation, ...]:
    """Create independent batches while retaining CRN within each batch."""

    return tuple(
        TrajectoryPopulation.generate_parallel(
            site_width_m=scenario.site.width_m,
            site_height_m=scenario.site.height_m,
            heading_deg_toward=heading_deg_toward,
            number_of_trajectories=trajectories_per_batch,
            maximum_response_events=scenario.site.n_turbines,
            longitudinal_margin_m=longitudinal_margin_m,
            seed=first_seed + batch_index,
        )
        for batch_index in range(batch_count)
    )


def fresh_validate_pareto(
    *,
    X: np.ndarray,
    pareto: pd.DataFrame,
    model: MesoResponseModel,
    populations: tuple[TrajectoryPopulation, ...],
    number_of_turbines: int,
    validation_z: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict]]:
    """Evaluate every training-front layout on independent trajectory batches."""

    validation_rows: list[dict] = []
    batch_rows: list[dict] = []
    raw_by_solution: dict[int, dict] = {}
    batch_encounters: list[list[float]] = []
    total = len(pareto)
    for position, (_, row) in enumerate(pareto.iterrows(), start=1):
        solution_index = int(row["solution_index"])
        xy = X[solution_index].reshape(number_of_turbines, 2)
        print(
            f"Fresh-validating Pareto layout {position}/{total} "
            f"(solution {solution_index})",
            flush=True,
        )
        fresh = summarise_meso_batches(model, xy, populations)
        raw_by_solution[solution_index] = fresh
        batch_encounters.append(
            [float(batch["E_enc"]) for batch in fresh["batch_summaries"]]
        )
        validation_rows.append(
            {
                **row.to_dict(),
                "fresh_N_started": fresh["N_started"],
                "fresh_N_completed": fresh["N_completed"],
                "fresh_N_failed": fresh["N_failed"],
                "fresh_N_response_events": fresh["N_response_events"],
                "fresh_N_enc_farm": fresh["N_enc_farm"],
                "fresh_E_enc": fresh["E_enc"],
                "fresh_E_enc_batch_sd": fresh["E_enc_batch_sd"],
                "fresh_E_enc_batch_se": fresh["E_enc_batch_se"],
                "fresh_E_enc_batch_min": fresh["E_enc_batch_min"],
                "fresh_E_enc_batch_max": fresh["E_enc_batch_max"],
                "fresh_P_ge_1": fresh["P_ge_1"],
                "fresh_P_ge_1_batch_sd": fresh["P_ge_1_batch_sd"],
                "fresh_P_ge_1_batch_se": fresh["P_ge_1_batch_se"],
                "fresh_mean_route_extension_m": fresh[
                    "mean_route_extension_m"
                ],
                "E_enc_generalisation_gap": fresh["E_enc"] - row["E_enc"],
                "E_enc_generalisation_gap_pct": (
                    100.0 * (fresh["E_enc"] - row["E_enc"]) / row["E_enc"]
                    if row["E_enc"] > 0.0
                    else np.nan
                ),
            }
        )
        for batch in fresh["batch_summaries"]:
            batch_rows.append({"solution_index": solution_index, **batch})

    validation = pd.DataFrame(validation_rows)
    validation["fresh_is_nondominated"] = nondominated_mask(
        validation[["aep_loss_gwh", "fresh_E_enc"]].to_numpy(dtype=float)
    )
    validation["fresh_is_robustly_nondominated"] = robust_nondominated_mask(
        validation["aep_loss_gwh"].to_numpy(dtype=float),
        np.asarray(batch_encounters, dtype=float),
        z_value=validation_z,
    )
    return validation, pd.DataFrame(batch_rows), raw_by_solution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimise FLORIS AEP against Model 8 encounter intensity."
    )
    parser.add_argument("--floris_config", type=Path, required=True)
    parser.add_argument("--energy_seed_layout", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("results/model8_aep"))
    parser.add_argument("--p_response", type=float, default=0.6)
    parser.add_argument("--response_radius_D", type=float, default=2.0)
    parser.add_argument("--encounter_radius_D", type=float, default=1.0)
    parser.add_argument("--deflection_angle_deg", type=float, default=45.0)
    parser.add_argument("--return_length_D", type=float, default=8.0)
    parser.add_argument("--step_size_D", type=float, default=0.20)
    parser.add_argument("--bird_heading_deg", type=float, default=225.0)
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=2000,
        help="Trajectories per objective batch.",
    )
    parser.add_argument(
        "--objective_batches",
        type=int,
        default=5,
        help="Independent common-random-number batches per candidate.",
    )
    parser.add_argument(
        "--fresh_trajectories",
        type=int,
        default=10000,
        help="Trajectories per independent fresh-validation batch.",
    )
    parser.add_argument(
        "--fresh_batches",
        type=int,
        default=3,
        help="Independent batches used to validate every Pareto layout.",
    )
    parser.add_argument(
        "--validation_z",
        type=float,
        default=1.96,
        help=(
            "Paired fresh-batch uncertainty multiplier used for robust "
            "non-dominated filtering."
        ),
    )
    parser.add_argument("--aep_loss_reference_gwh", type=float, default=137.96914764866568)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--gen", type=int, default=100)
    parser.add_argument("--no_evaluation_archive", action="store_true")
    args = parser.parse_args()

    if args.pop <= 1 or args.gen <= 0:
        raise ValueError("pop must exceed one and gen must be positive.")
    if args.n_trajectories <= 0 or args.fresh_trajectories <= 0:
        raise ValueError("Trajectory counts must be positive.")
    if args.objective_batches <= 0 or args.fresh_batches <= 0:
        raise ValueError("Batch counts must be positive.")
    if args.validation_z < 0.0 or not np.isfinite(args.validation_z):
        raise ValueError("validation_z must be finite and non-negative.")

    scenario = Scenario()
    scenario.site.n_turbines = 12
    scenario.site.width_m = 6000.0
    scenario.site.height_m = 6000.0
    scenario.site.min_spacing_D = 4.0
    scenario.site.migration_axis_deg = float(args.bird_heading_deg)

    aep_model = FLORISAEPModel(scenario, config_path=args.floris_config)
    rotor_diameter_m = float(aep_model.floris_rotor_diameter_m)
    scenario.turbine.rotor_diameter_m = rotor_diameter_m
    minimum_spacing_m = scenario.site.min_spacing_D * rotor_diameter_m

    energy_seed_xy = load_layout_csv(
        args.energy_seed_layout, scenario.site.n_turbines
    )
    if not layout_is_feasible(
        energy_seed_xy,
        width_m=scenario.site.width_m,
        height_m=scenario.site.height_m,
        minimum_spacing_m=minimum_spacing_m,
    ):
        raise ValueError("The energy seed layout is infeasible under FLORIS diameter.")

    behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=rotor_diameter_m,
        response_probability=args.p_response,
        response_radius_D=args.response_radius_D,
        encounter_radius_D=args.encounter_radius_D,
        deflection_angle_deg=args.deflection_angle_deg,
        return_length_D=args.return_length_D,
        step_size_D=args.step_size_D,
    )
    longitudinal_margin_m = behaviour.response_radius_m + behaviour.return_length_m
    objective_populations = generate_population_batches(
        scenario=scenario,
        heading_deg_toward=args.bird_heading_deg,
        trajectories_per_batch=args.n_trajectories,
        batch_count=args.objective_batches,
        longitudinal_margin_m=longitudinal_margin_m,
        first_seed=args.seed,
    )
    meso_model = MesoResponseModel(behaviour)

    no_response = BehaviourScenario(
        response_probability=0.0,
        response_radius_m=behaviour.response_radius_m,
        encounter_radius_m=behaviour.encounter_radius_m,
        deflection_angle_deg=behaviour.deflection_angle_deg,
        return_length_m=behaviour.return_length_m,
        step_size_m=behaviour.step_size_m,
        max_steps_factor=behaviour.max_steps_factor,
    )
    encounter_reference_result = summarise_meso_batches(
        MesoResponseModel(no_response), energy_seed_xy, objective_populations
    )
    encounter_reference = float(encounter_reference_result["E_enc"])
    if encounter_reference <= 0.0:
        raise RuntimeError("The no-response encounter reference is zero.")

    aep_ceiling_gwh = float(aep_model.ideal_aep(scenario.site.n_turbines))
    problem = Model8AEPEncounterProblem(
        scenario=scenario,
        aep_model=aep_model,
        meso_model=meso_model,
        trajectory_populations=objective_populations,
        rotor_diameter_m=rotor_diameter_m,
        aep_ceiling_gwh=aep_ceiling_gwh,
        aep_loss_reference_gwh=args.aep_loss_reference_gwh,
        encounter_reference=encounter_reference,
        record_evaluations=not args.no_evaluation_archive,
    )

    fixed_layouts = make_fixed_layouts(
        site_width_m=scenario.site.width_m,
        site_height_m=scenario.site.height_m,
        bird_heading_deg_toward=args.bird_heading_deg,
        minimum_spacing_m=minimum_spacing_m,
    )
    exact_seeds = {"energy": energy_seed_xy, **fixed_layouts}
    sampling = FeasibleSeededSampling(
        scenario=scenario,
        rotor_diameter_m=rotor_diameter_m,
        exact_seed_layouts=exact_seeds,
        seed=args.seed,
    )
    algorithm = NSGA2(
        pop_size=args.pop,
        sampling=sampling,
        eliminate_duplicates=True,
    )
    callback = HypervolumeCallback()

    case_label = (
        f"p{safe_label(args.p_response)}_heading{safe_label(args.bird_heading_deg)}"
        f"_B{args.objective_batches}xN{args.n_trajectories}"
        f"_FB{args.fresh_batches}xN{args.fresh_trajectories}"
        f"_V{safe_label(args.validation_z)}"
        f"_seed{args.seed}_pop{args.pop}_gen{args.gen}"
    )
    output_dir = args.output_root / case_label
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight_X = sampling._do(problem, args.pop)
    pd.DataFrame(
        preflight_X,
        columns=[
            item
            for turbine_id in range(scenario.site.n_turbines)
            for item in (f"x_{turbine_id}_m", f"y_{turbine_id}_m")
        ],
    ).assign(seed_source=sampling.last_labels).to_csv(
        output_dir / "initial_population.csv", index=False
    )

    run_settings = {
        "model": "Model 8 FLORIS AEP versus meso-response encounters",
        "scientific_status": "hypothetical behavioural sensitivity model",
        "objectives": ["maximise AEP", "minimise encounter intensity E_enc"],
        "not_modelled": [
            "calibrated bird flux",
            "flight-height overlap",
            "rotor transit",
            "collision probability",
            "micro-avoidance",
            "mortality",
        ],
        "seed": args.seed,
        "population": args.pop,
        "generations": args.gen,
        "objective_batches": args.objective_batches,
        "objective_trajectories_per_batch": args.n_trajectories,
        "objective_trajectories_total": (
            args.objective_batches * args.n_trajectories
        ),
        "fresh_batches": args.fresh_batches,
        "fresh_trajectories_per_batch": args.fresh_trajectories,
        "fresh_trajectories_total_per_layout": (
            args.fresh_batches * args.fresh_trajectories
        ),
        "validation_z": args.validation_z,
        "objective_random_number_seeds": [
            population.seed for population in objective_populations
        ],
        "fresh_random_number_seeds": [
            args.seed + 10000 + batch_index
            for batch_index in range(args.fresh_batches)
        ],
        "bird_heading_deg_toward": args.bird_heading_deg,
        "site_width_m": float(scenario.site.width_m),
        "site_height_m": float(scenario.site.height_m),
        "problem": problem.settings_summary(),
        "aep_model": aep_model.settings_summary(),
        "encounter_reference_definition": (
            "E_enc for the no-response energy-seed layout aggregated over the "
            "same independent objective batches"
        ),
        "energy_seed_layout": str(args.energy_seed_layout),
        "floris_config": str(args.floris_config),
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(run_settings, indent=2), encoding="utf-8"
    )

    start_time = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", args.gen),
        seed=args.seed,
        callback=callback,
        verbose=True,
    )
    runtime_seconds = float(time.perf_counter() - start_time)
    if result.X is None:
        raise RuntimeError("No feasible Pareto layouts were found.")

    X = np.atleast_2d(np.asarray(result.X, dtype=float))
    F = np.atleast_2d(np.asarray(result.F, dtype=float))
    np.save(output_dir / "pareto_decision_vectors.npy", X)
    np.save(output_dir / "pareto_normalised_objectives.npy", F)

    pareto = build_pareto_dataframe(X, F, problem)
    pareto.to_csv(output_dir / "pareto_points.csv", index=False)
    training_anchors = find_anchor_indices(pareto)
    training_anchor_summary = save_anchor_layouts(
        X=X,
        dataframe=pareto,
        anchors=training_anchors,
        problem=problem,
        output_dir=output_dir,
        filename_prefix="training_",
    )
    training_anchor_summary.to_csv(
        output_dir / "training_anchor_summary.csv", index=False
    )
    plot_pareto(pareto, training_anchors, output_dir / "pareto_front.png")

    hypervolume = pd.DataFrame(callback.data)
    hypervolume.to_csv(output_dir / "hypervolume.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    axis.plot(hypervolume["generation"], hypervolume["normalised_hypervolume"])
    axis.set_xlabel("Generation")
    axis.set_ylabel("Normalised hypervolume")
    axis.grid(alpha=0.25)
    fig.savefig(output_dir / "hypervolume.png", dpi=180)
    plt.close(fig)

    if not args.no_evaluation_archive:
        problem.evaluation_dataframe().to_csv(
            output_dir / "all_evaluated_candidates.csv", index=False
        )

    validation_start_time = time.perf_counter()
    fresh_populations = generate_population_batches(
        scenario=scenario,
        heading_deg_toward=args.bird_heading_deg,
        trajectories_per_batch=args.fresh_trajectories,
        batch_count=args.fresh_batches,
        longitudinal_margin_m=longitudinal_margin_m,
        first_seed=args.seed + 10000,
    )
    fresh_model = MesoResponseModel(behaviour)
    fresh_validation, fresh_batches, fresh_raw = fresh_validate_pareto(
        X=X,
        pareto=pareto,
        model=fresh_model,
        populations=fresh_populations,
        number_of_turbines=scenario.site.n_turbines,
        validation_z=args.validation_z,
    )
    fresh_validation.to_csv(output_dir / "fresh_pareto_validation.csv", index=False)
    fresh_batches.to_csv(output_dir / "fresh_pareto_batches.csv", index=False)

    validated_pareto = fresh_validation.loc[
        fresh_validation["fresh_is_robustly_nondominated"]
    ].copy()
    validated_pareto.to_csv(
        output_dir / "validated_pareto_points.csv", index=False
    )
    validated_anchors = find_anchor_indices(
        validated_pareto, encounter_column="fresh_E_enc"
    )
    anchor_summary = save_anchor_layouts(
        X=X,
        dataframe=validated_pareto,
        anchors=validated_anchors,
        problem=problem,
        output_dir=output_dir,
        raw_by_solution=fresh_raw,
    )
    anchor_summary.to_csv(output_dir / "anchor_summary.csv", index=False)
    anchor_summary.to_csv(
        output_dir / "validated_anchor_summary.csv", index=False
    )
    anchor_summary[
        [
            "layout",
            "solution_index",
            "E_enc",
            "fresh_E_enc",
            "P_ge_1",
            "fresh_P_ge_1",
            "fresh_N_started",
            "fresh_N_failed",
        ]
    ].rename(
        columns={
            "E_enc": "optimisation_E_enc",
            "P_ge_1": "optimisation_P_ge_1",
            "fresh_N_started": "fresh_N_in",
        }
    ).to_csv(output_dir / "fresh_sample_anchor_validation.csv", index=False)
    plot_pareto(
        fresh_validation,
        validated_anchors,
        output_dir / "validated_pareto_front.png",
        encounter_column="fresh_E_enc",
        nondominated_column="fresh_is_robustly_nondominated",
    )
    fresh_validation_runtime_seconds = float(
        time.perf_counter() - validation_start_time
    )

    summary = {
        **run_settings,
        "runtime_seconds": runtime_seconds,
        "optimisation_runtime_seconds": runtime_seconds,
        "fresh_validation_runtime_seconds": fresh_validation_runtime_seconds,
        "total_model_runtime_seconds": (
            runtime_seconds + fresh_validation_runtime_seconds
        ),
        "number_of_pareto_layouts": int(len(pareto)),
        "number_of_exact_fresh_pareto_layouts": int(
            fresh_validation["fresh_is_nondominated"].sum()
        ),
        "number_of_validated_pareto_layouts": int(len(validated_pareto)),
        "total_spacing_violations": int(pareto["spacing_violations"].sum()),
        "maximum_failed_trajectories": int(pareto["N_failed"].max()),
        "maximum_fresh_failed_trajectories": int(
            fresh_validation["fresh_N_failed"].max()
        ),
        "maximum_absolute_E_enc_generalisation_gap": float(
            fresh_validation["E_enc_generalisation_gap"].abs().max()
        ),
        "validated_distinct_anchor_count": int(
            anchor_summary["solution_index"].nunique()
        ),
        "final_normalised_hypervolume": (
            None
            if len(hypervolume) == 0
            else float(hypervolume["normalised_hypervolume"].iloc[-1])
        ),
        "training_anchors": json.loads(
            training_anchor_summary.to_json(orient="records")
        ),
        "anchors": json.loads(anchor_summary.to_json(orient="records")),
        "anchor_selection": (
            "Robust fresh-validated front using paired batch differences and "
            "the configured validation_z uncertainty multiplier"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(anchor_summary.to_string(index=False))
    print(f"\nSaved Model 8 optimisation outputs to {output_dir}")


if __name__ == "__main__":
    main()
