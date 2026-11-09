"""Run the integrated FLORIS + Model 9 NSGA-II layout optimisation.

The default ``smoke`` profile is deliberately small.  It checks the complete
scientific and software path before a longer pilot or the inherited full
Model 8-sized run is attempted.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from scipy.stats import t as student_t

from config import OptSettings, Scenario
from floris_aep import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_TURBINE_REFERENCE_PATH,
    FLORISAEPModel,
)
from model8_batching import nondominated_mask, robust_nondominated_mask
from model8_meso_response import BehaviourScenario
from model9_aep_problem import Model9AEPRotorTransitProblem
from model9_batching import summarise_model9_batches
from model9_optimisation_utils import (
    coordinate_columns,
    layout_frame,
    load_initial_population,
    load_layout_csv,
    select_anchor_positions,
    serialisable,
    validate_layout,
)
from model9_rotor_transit import (
    DiscreteFlightHeightDistribution,
    Model9Population,
    RotorGeometry,
    RotorTransitModel,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_FHD_PATH = MODULE_DIR / "data" / "northern_gannet_Johnston2014_FHD.csv"
DEFAULT_FHD_METADATA_PATH = (
    MODULE_DIR / "data" / "northern_gannet_Johnston2014_FHD_metadata.json"
)
DEFAULT_BASELINE_LAYOUT_PATH = MODULE_DIR / "data" / "energy_endpoint_layout.csv"
DEFAULT_INITIAL_POPULATION_PATH = (
    MODULE_DIR / "data" / "model8_initial_population.csv"
)
DEFAULT_MODEL8_SETTINGS_PATH = (
    MODULE_DIR / "data" / "model8_reference_run_settings.json"
)


@dataclass(frozen=True)
class RunProfile:
    population: int
    generations: int
    objective_batches: int
    objective_trajectories_per_batch: int
    fresh_batches: int
    fresh_trajectories_per_batch: int


PROFILES = {
    "smoke": RunProfile(8, 2, 2, 250, 2, 1000),
    "pilot": RunProfile(24, 15, 3, 1000, 3, 5000),
    "full": RunProfile(80, 100, 5, 2000, 3, 10000),
}


class CheckpointCallback(Callback):
    """Keep a recoverable snapshot of the current NSGA-II population."""

    def __init__(self, output_dir: Path, number_of_turbines: int) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.columns = coordinate_columns(number_of_turbines)

    def notify(self, algorithm: Any) -> None:
        x = np.asarray(algorithm.pop.get("X"), dtype=float)
        f = np.asarray(algorithm.pop.get("F"), dtype=float)
        g = np.asarray(algorithm.pop.get("G"), dtype=float)
        frame = pd.DataFrame(x, columns=self.columns)
        frame["f0_normalised_aep_loss"] = f[:, 0]
        frame["f1_normalised_rotor_transits"] = f[:, 1]
        frame["maximum_constraint"] = np.max(g, axis=1)
        frame.to_csv(self.output_dir / "checkpoint_population.csv", index=False)
        state = {
            "generation": int(algorithm.n_gen),
            "function_evaluations": int(algorithm.evaluator.n_eval),
        }
        (self.output_dir / "checkpoint_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )


def make_populations(
    *,
    distribution: DiscreteFlightHeightDistribution,
    scenario: Scenario,
    behaviour: BehaviourScenario,
    batch_count: int,
    trajectories_per_batch: int,
    first_trajectory_seed: int,
    first_height_seed: int,
) -> tuple[Model9Population, ...]:
    return tuple(
        Model9Population.generate_parallel(
            flight_height_distribution=distribution,
            flight_height_seed=first_height_seed + batch_index,
            site_width_m=scenario.site.width_m,
            site_height_m=scenario.site.height_m,
            heading_deg_toward=scenario.birds.heading_deg_toward,
            number_of_trajectories=trajectories_per_batch,
            maximum_response_events=scenario.birds.maximum_response_events,
            longitudinal_margin_m=(
                behaviour.response_radius_m + behaviour.return_length_m
            ),
            trajectory_seed=first_trajectory_seed + batch_index,
        )
        for batch_index in range(batch_count)
    )


def summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "batch_summaries",
        "per_turbine_encounters",
        "per_turbine_transits",
        "per_turbine_expected_transits",
    }
    return {key: value for key, value in summary.items() if key not in excluded}


def add_interval_columns(frame: pd.DataFrame, batch_count: int) -> None:
    if batch_count <= 1:
        critical = 0.0
    else:
        critical = float(student_t.ppf(0.975, df=batch_count - 1))
    for metric in ("E_RT", "E_RT_CE"):
        se_column = f"{metric}_batch_se"
        if metric in frame and se_column in frame:
            half_width = critical * frame[se_column]
            frame[f"{metric}_ci95_lower"] = frame[metric] - half_width
            frame[f"{metric}_ci95_upper"] = frame[metric] + half_width


def save_initial_population(
    vectors: np.ndarray,
    labels: list[str],
    source_rows: list[int],
    output_path: Path,
    number_of_turbines: int,
) -> None:
    frame = pd.DataFrame(vectors, columns=coordinate_columns(number_of_turbines))
    frame["seed_source"] = labels
    frame["source_row_model8"] = source_rows
    frame.to_csv(output_path, index=False)


def plot_front(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    axis.scatter(
        frame["training_aep_gwh"],
        frame["training_E_RT_CE"],
        facecolors="none",
        edgecolors="#9c5f35",
        label="Optimisation batches",
    )
    axis.scatter(
        frame["aep_gwh"],
        frame["E_RT_CE"],
        color="#315f73",
        label="Fresh validation batches",
        zorder=3,
    )
    for row in frame.loc[frame["anchor_label"].notna()].itertuples(index=False):
        axis.annotate(
            str(row.anchor_label).replace("_", " "),
            (row.aep_gwh, row.E_RT_CE),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Annual energy production (GWh/year; higher is better)")
    axis.set_ylabel("Rotor-transit intensity E_RT_CE (lower is better)")
    axis.set_title("Model 9 NSGA-II trade-off and fresh validation")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layout(
    xy: np.ndarray,
    *,
    title: str,
    site_width_m: float,
    site_height_m: float,
    rotor_diameter_m: float,
    output_path: Path,
) -> None:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    fig, axis = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    axis.scatter(points[:, 0], points[:, 1], s=38, color="#315f73", zorder=3)
    for turbine_id, (x_m, y_m) in enumerate(points):
        axis.annotate(str(turbine_id), (x_m, y_m), xytext=(4, 3),
                      textcoords="offset points", fontsize=7)
        axis.add_patch(
            plt.Circle(
                (x_m, y_m),
                rotor_diameter_m / 2.0,
                fill=False,
                linewidth=0.7,
                alpha=0.35,
                color="#315f73",
            )
        )
    axis.set(xlim=(0.0, site_width_m), ylim=(0.0, site_height_m))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def model8_reference(path: Path, scenario: Scenario) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    problem = data["problem"]
    aep_settings = data["aep_model"]
    checks = {
        "number_of_turbines": scenario.site.n_turbines,
        "rotor_diameter_m": scenario.turbine.rotor_diameter_m,
        "minimum_spacing_D": scenario.site.min_spacing_D,
    }
    for key, expected in checks.items():
        actual = problem[key]
        if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Model 8 reference {key}={actual} does not match Model 9 "
                f"configuration value {expected}."
            )
    if int(aep_settings["number_of_direction_sectors"]) != scenario.wind.n_sectors:
        raise ValueError("Model 8 and Model 9 direction-sector counts differ.")
    if int(aep_settings["number_of_speed_bins"]) != len(scenario.wind.speed_bins):
        raise ValueError("Model 8 and Model 9 speed-bin counts differ.")
    return data


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in ("floris", "pymoo", "numpy", "pandas", "scipy", "matplotlib", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not reported"
    return versions


def prepare_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {resolved}. Choose a new --output path."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise Model 9 FLORIS AEP versus rotor-transit exposure."
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--objective_batches", type=int, default=None)
    parser.add_argument(
        "--objective_trajectories_per_batch", type=int, default=None
    )
    parser.add_argument("--fresh_batches", type=int, default=None)
    parser.add_argument("--fresh_trajectories_per_batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flight_height_csv", type=Path, default=DEFAULT_FHD_PATH)
    parser.add_argument("--floris_config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--turbine_reference", type=Path, default=DEFAULT_TURBINE_REFERENCE_PATH
    )
    parser.add_argument(
        "--initial_population", type=Path, default=DEFAULT_INITIAL_POPULATION_PATH
    )
    parser.add_argument(
        "--baseline_layout", type=Path, default=DEFAULT_BASELINE_LAYOUT_PATH
    )
    parser.add_argument(
        "--model8_settings", type=Path, default=DEFAULT_MODEL8_SETTINGS_PATH
    )
    parser.add_argument("--no_zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = PROFILES[args.profile]
    profile = RunProfile(
        population=base.population if args.population is None else args.population,
        generations=base.generations if args.generations is None else args.generations,
        objective_batches=(
            base.objective_batches
            if args.objective_batches is None
            else args.objective_batches
        ),
        objective_trajectories_per_batch=(
            base.objective_trajectories_per_batch
            if args.objective_trajectories_per_batch is None
            else args.objective_trajectories_per_batch
        ),
        fresh_batches=(
            base.fresh_batches if args.fresh_batches is None else args.fresh_batches
        ),
        fresh_trajectories_per_batch=(
            base.fresh_trajectories_per_batch
            if args.fresh_trajectories_per_batch is None
            else args.fresh_trajectories_per_batch
        ),
    )
    if any(value <= 0 for value in asdict(profile).values()):
        raise ValueError("All profile counts must be positive.")
    if profile.population < 4:
        raise ValueError("population must be at least four.")

    default_output = MODULE_DIR / "results" / f"model9_nsga2_{args.profile}"
    output_dir = prepare_output_dir(
        default_output if args.output is None else args.output
    )
    planned_archive = output_dir.with_suffix(".zip")
    if not args.no_zip and planned_archive.exists():
        raise FileExistsError(
            f"Result archive already exists: {planned_archive}. Move it or "
            "choose a new --output path before starting the run."
        )
    started = time.time()
    scenario = Scenario(opt=OptSettings(
        pop_size=profile.population,
        n_gen=profile.generations,
        seed=args.seed,
        objective_batch_count=profile.objective_batches,
        objective_trajectories_per_batch=(
            profile.objective_trajectories_per_batch
        ),
        fresh_batch_count=profile.fresh_batches,
        fresh_trajectories_per_batch=profile.fresh_trajectories_per_batch,
    ))
    reference = model8_reference(args.model8_settings.resolve(), scenario)
    aep_model = FLORISAEPModel(
        scenario,
        args.floris_config,
        turbine_reference_path=args.turbine_reference,
    )
    if aep_model.number_of_conditions != 84:
        raise RuntimeError(
            f"Expected 84 inherited wind conditions, got {aep_model.number_of_conditions}."
        )

    metadata = json.loads(DEFAULT_FHD_METADATA_PATH.read_text(encoding="utf-8"))
    distribution = DiscreteFlightHeightDistribution.from_csv(
        args.flight_height_csv,
        name="northern_gannet_Johnston2014_Gannet.est",
        source=metadata["primary_citation"],
    )
    if distribution.input_sha256 != metadata["sha256"]:
        raise ValueError(
            "The flight-height CSV does not match the frozen northern-gannet "
            f"input checksum {metadata['sha256']}."
        )
    rotor = RotorGeometry(
        rotor_diameter_m=aep_model.floris_rotor_diameter_m,
        hub_height_m=aep_model.floris_hub_height_m,
    )
    birds = scenario.birds
    behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=rotor.rotor_diameter_m,
        response_probability=birds.response_probability,
        response_radius_D=birds.response_radius_D,
        encounter_radius_D=birds.encounter_radius_D,
        deflection_angle_deg=birds.deflection_angle_deg,
        return_length_D=birds.return_length_D,
        step_size_D=birds.step_size_D,
    )
    transit_model = RotorTransitModel(
        behaviour,
        rotor,
        rotor_plane_heading_deg_toward=birds.heading_deg_toward,
        flight_height_distribution=distribution,
    )

    objective_populations = make_populations(
        distribution=distribution,
        scenario=scenario,
        behaviour=behaviour,
        batch_count=profile.objective_batches,
        trajectories_per_batch=profile.objective_trajectories_per_batch,
        first_trajectory_seed=args.seed,
        first_height_seed=10_000 + args.seed,
    )
    minimum_spacing_m = scenario.site.min_spacing_D * rotor.rotor_diameter_m
    baseline_xy = load_layout_csv(
        args.baseline_layout,
        number_of_turbines=scenario.site.n_turbines,
        site_width_m=scenario.site.width_m,
        site_height_m=scenario.site.height_m,
        minimum_spacing_m=minimum_spacing_m,
    )
    baseline_exposure = summarise_model9_batches(
        transit_model, baseline_xy, objective_populations
    )
    if baseline_exposure["N_failed"] != 0:
        raise RuntimeError("Trajectory failures occurred for the baseline layout.")
    rotor_reference = float(baseline_exposure["E_RT_CE"])
    if rotor_reference <= 0.0:
        raise RuntimeError("Baseline E_RT_CE must be positive for objective scaling.")

    aep_ceiling_gwh = aep_model.ideal_aep(scenario.site.n_turbines)
    inherited_ceiling = float(reference["problem"]["aep_ceiling_gwh"])
    if not np.isclose(aep_ceiling_gwh, inherited_ceiling, rtol=0.0, atol=0.01):
        raise RuntimeError(
            "FLORIS wake-free AEP ceiling differs from the supplied Model 8 run: "
            f"{aep_ceiling_gwh:.9f} versus {inherited_ceiling:.9f} GWh/year."
        )
    aep_loss_reference_gwh = float(
        reference["problem"]["aep_loss_reference_gwh"]
    )
    baseline_aep_gwh = aep_model.evaluate(baseline_xy)

    initial_x, initial_labels, source_rows = load_initial_population(
        args.initial_population,
        population_size=profile.population,
        seed=args.seed,
        number_of_turbines=scenario.site.n_turbines,
        site_width_m=scenario.site.width_m,
        site_height_m=scenario.site.height_m,
        minimum_spacing_m=minimum_spacing_m,
    )
    save_initial_population(
        initial_x,
        initial_labels,
        source_rows,
        output_dir / "selected_initial_population.csv",
        scenario.site.n_turbines,
    )
    layout_frame(baseline_xy).to_csv(
        output_dir / "baseline_energy_layout.csv", index=False
    )

    problem = Model9AEPRotorTransitProblem(
        scenario=scenario,
        aep_model=aep_model,
        rotor_transit_model=transit_model,
        trajectory_populations=objective_populations,
        aep_ceiling_gwh=aep_ceiling_gwh,
        aep_loss_reference_gwh=aep_loss_reference_gwh,
        rotor_transit_reference=rotor_reference,
        bird_objective_metric="E_RT_CE",
    )
    preliminary_settings = {
        "model": "Model 9 FLORIS AEP versus rotor-transit exposure",
        "endpoint": "rotor-swept-area transit exposure, not collision or mortality",
        "run_status": "optimisation_started",
        "profile_name": args.profile,
        "profile": asdict(profile),
        "seed": args.seed,
        "runtime_versions": runtime_versions(),
        "objective_trajectory_seeds": [
            args.seed + index for index in range(profile.objective_batches)
        ],
        "objective_flight_height_seeds": [
            10_000 + args.seed + index for index in range(profile.objective_batches)
        ],
        "fresh_trajectory_seeds": [
            30_000 + args.seed + index for index in range(profile.fresh_batches)
        ],
        "fresh_flight_height_seeds": [
            40_000 + args.seed + index for index in range(profile.fresh_batches)
        ],
        "wind_input": "programmatic 12-sector x 7-speed-bin weights; no wind CSV",
        "aep_model": aep_model.settings_summary(),
        "problem": problem.settings_summary(),
        "flight_height_distribution": distribution.settings(),
        "baseline": {
            "layout_path": str(args.baseline_layout.resolve()),
            "aep_gwh": baseline_aep_gwh,
            **summary_row(baseline_exposure),
        },
        "model8_reference_settings_path": str(args.model8_settings.resolve()),
        "initial_population_path": str(args.initial_population.resolve()),
        "initial_population_source_rows": source_rows,
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(serialisable(preliminary_settings), indent=2), encoding="utf-8"
    )

    print(
        f"Starting {args.profile} run: population={profile.population}, "
        f"generations={profile.generations}, 84 FLORIS conditions, "
        f"{profile.objective_batches} x "
        f"{profile.objective_trajectories_per_batch} bird trajectories/evaluation."
    )
    algorithm = NSGA2(
        pop_size=profile.population,
        sampling=initial_x,
        eliminate_duplicates=True,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", profile.generations),
        seed=args.seed,
        callback=CheckpointCallback(output_dir, scenario.site.n_turbines),
        save_history=True,
        verbose=True,
    )
    if result.X is None or result.F is None:
        raise RuntimeError("NSGA-II returned no feasible Pareto solutions.")
    pareto_x = np.atleast_2d(np.asarray(result.X, dtype=float))
    pareto_f = np.atleast_2d(np.asarray(result.F, dtype=float))
    np.save(output_dir / "pareto_decision_vectors.npy", pareto_x)
    np.save(output_dir / "pareto_normalised_objectives.npy", pareto_f)
    problem.evaluation_dataframe().to_csv(
        output_dir / "all_evaluated_candidates.csv", index=False
    )

    convergence_rows = []
    for history_item in result.history:
        optimum = np.atleast_2d(np.asarray(history_item.opt.get("F"), dtype=float))
        convergence_rows.append(
            {
                "generation": int(history_item.n_gen),
                "function_evaluations": int(history_item.evaluator.n_eval),
                "pareto_size": len(optimum),
                "minimum_normalised_aep_loss": float(np.min(optimum[:, 0])),
                "minimum_normalised_E_RT_CE": float(np.min(optimum[:, 1])),
            }
        )
    pd.DataFrame(convergence_rows).to_csv(
        output_dir / "convergence.csv", index=False
    )

    training_rows: list[dict[str, Any]] = []
    for candidate_index, xy_vector in enumerate(pareto_x):
        xy = xy_vector.reshape(scenario.site.n_turbines, 2)
        raw = problem.evaluate_raw_layout(xy)
        recalculated_objectives = np.array(
            [
                raw["aep_loss_gwh"] / aep_loss_reference_gwh,
                raw["E_RT_CE"] / rotor_reference,
            ],
            dtype=float,
        )
        if not np.allclose(
            recalculated_objectives,
            pareto_f[candidate_index],
            rtol=1.0e-9,
            atol=1.0e-10,
        ):
            raise RuntimeError(
                f"Re-evaluated training objectives changed for pareto_{candidate_index:03d}."
            )
        validation = validate_layout(
            xy,
            number_of_turbines=scenario.site.n_turbines,
            site_width_m=scenario.site.width_m,
            site_height_m=scenario.site.height_m,
            minimum_spacing_m=minimum_spacing_m,
        )
        training_rows.append(
            {
                "candidate_id": f"pareto_{candidate_index:03d}",
                "training_aep_gwh": raw["aep_gwh"],
                "training_aep_loss_gwh": raw["aep_loss_gwh"],
                "training_E_RT": raw["E_RT"],
                "training_E_RT_CE": raw["E_RT_CE"],
                "training_E_enc": raw["E_enc"],
                "f0_normalised_aep_loss": pareto_f[candidate_index, 0],
                "f1_normalised_rotor_transits": pareto_f[candidate_index, 1],
                **validation,
            }
        )

    fresh_populations = make_populations(
        distribution=distribution,
        scenario=scenario,
        behaviour=behaviour,
        batch_count=profile.fresh_batches,
        trajectories_per_batch=profile.fresh_trajectories_per_batch,
        first_trajectory_seed=30_000 + args.seed,
        first_height_seed=40_000 + args.seed,
    )
    fresh_rows: list[dict[str, Any]] = []
    fresh_batch_rows: list[dict[str, Any]] = []
    for candidate_index, xy_vector in enumerate(pareto_x):
        candidate_id = f"pareto_{candidate_index:03d}"
        xy = xy_vector.reshape(scenario.site.n_turbines, 2)
        combined = summarise_model9_batches(transit_model, xy, fresh_populations)
        if combined["N_failed"] != 0:
            raise RuntimeError(f"Trajectory failures occurred for {candidate_id}.")
        fresh_rows.append(
            {
                "candidate_id": candidate_id,
                "aep_gwh": training_rows[candidate_index]["training_aep_gwh"],
                "aep_loss_gwh": training_rows[candidate_index][
                    "training_aep_loss_gwh"
                ],
                **summary_row(combined),
            }
        )
        for batch in combined["batch_summaries"]:
            fresh_batch_rows.append({"candidate_id": candidate_id, **batch})

    fresh = pd.DataFrame(fresh_rows)
    training = pd.DataFrame(training_rows)
    candidates = training.merge(fresh, on="candidate_id", validate="one_to_one")
    add_interval_columns(candidates, profile.fresh_batches)
    candidates["fresh_nondominated"] = nondominated_mask(
        candidates[["aep_loss_gwh", "E_RT_CE"]].to_numpy(dtype=float)
    )
    batch_frame = pd.DataFrame(fresh_batch_rows)
    batch_matrix = batch_frame.pivot(
        index="candidate_id", columns="batch_index", values="E_RT_CE"
    ).reindex(candidates["candidate_id"])
    candidates["robust_nondominated_95pct"] = robust_nondominated_mask(
        candidates["aep_loss_gwh"].to_numpy(dtype=float),
        batch_matrix.to_numpy(dtype=float),
        z_value=1.96,
    )

    # Descriptive endpoints must span the fresh nondominated front.  The
    # uncertainty-screened set can legitimately collapse to one layout; using
    # that singleton for all endpoint labels previously produced a misleading
    # "rotor-transit endpoint" label on the highest-risk energy candidate.
    selection_mask = candidates["fresh_nondominated"].to_numpy(dtype=bool)
    if not np.any(selection_mask):
        selection_mask = np.ones(len(candidates), dtype=bool)
    selection_positions = np.flatnonzero(selection_mask)
    selected = candidates.iloc[selection_positions]
    anchors_local = select_anchor_positions(
        selected["aep_gwh"].to_numpy(dtype=float),
        selected["aep_loss_gwh"].to_numpy(dtype=float),
        selected["E_RT_CE"].to_numpy(dtype=float),
    )
    candidates["anchor_label"] = pd.Series([None] * len(candidates), dtype="object")
    anchor_records: dict[str, Any] = {}
    layout_dir = output_dir / "anchor_layouts"
    layout_dir.mkdir()
    for label, local_position in anchors_local.items():
        global_position = int(selection_positions[local_position])
        candidate_id = str(candidates.iloc[global_position]["candidate_id"])
        existing = candidates.at[global_position, "anchor_label"]
        candidates.at[global_position, "anchor_label"] = (
            label if existing is None else f"{existing};{label}"
        )
        xy = pareto_x[global_position].reshape(scenario.site.n_turbines, 2)
        layout_path = layout_dir / f"{label}_{candidate_id}.csv"
        layout_frame(xy).to_csv(layout_path, index=False)
        plot_layout(
            xy,
            title=f"{label.replace('_', ' ').title()} ({candidate_id})",
            site_width_m=scenario.site.width_m,
            site_height_m=scenario.site.height_m,
            rotor_diameter_m=rotor.rotor_diameter_m,
            output_path=layout_dir / f"{label}_{candidate_id}.png",
        )
        anchor_records[label] = {
            "candidate_id": candidate_id,
            "layout_csv": str(layout_path),
            "aep_gwh": float(candidates.iloc[global_position]["aep_gwh"]),
            "aep_loss_gwh": float(
                candidates.iloc[global_position]["aep_loss_gwh"]
            ),
            "fresh_E_RT_CE": float(candidates.iloc[global_position]["E_RT_CE"]),
        }

    candidates.to_csv(output_dir / "pareto_fresh_validation.csv", index=False)
    batch_frame.to_csv(output_dir / "fresh_validation_batches.csv", index=False)
    plot_front(candidates, output_dir / "pareto_front_fresh_validation.png")

    elapsed = float(time.time() - started)
    final_settings = {
        **preliminary_settings,
        "run_status": "completed",
        "elapsed_seconds": elapsed,
        "function_evaluations": int(result.algorithm.evaluator.n_eval),
        "training_pareto_size": int(len(candidates)),
        "fresh_nondominated_count": int(candidates["fresh_nondominated"].sum()),
        "robust_nondominated_95pct_count": int(
            candidates["robust_nondominated_95pct"].sum()
        ),
        "robust_filter_note": (
            "Uses paired fresh-batch E_RT_CE differences and z=1.96; this is a "
            "screening rule, not proof of ecological superiority."
        ),
        "anchor_definitions": {
            "energy_endpoint": "highest fresh-validated AEP",
            "rotor_transit_endpoint": "lowest fresh-validated E_RT_CE",
            "balanced_reference": (
                "minimum Euclidean distance after min-max scaling of AEP loss "
                "and E_RT_CE; descriptive reference, not a claimed Pareto knee"
            ),
        },
        "anchor_selection_note": (
            "Descriptive anchors are selected across the fresh nondominated "
            "set, independently of the uncertainty-screened robust set."
        ),
        "anchors": anchor_records,
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(serialisable(final_settings), indent=2), encoding="utf-8"
    )
    (output_dir / "anchor_summary.json").write_text(
        json.dumps(serialisable(anchor_records), indent=2), encoding="utf-8"
    )

    archive_path: Path | None = None
    if not args.no_zip:
        archive_path = planned_archive
        if archive_path.exists():
            raise FileExistsError(
                f"Result archive already exists: {archive_path}. Move it first."
            )
        shutil.make_archive(
            str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
        )

    print(
        f"Completed in {elapsed / 60.0:.2f} minutes. "
        f"Fresh-validated {len(candidates)} training Pareto candidates."
    )
    print(f"Results: {output_dir}")
    if archive_path is not None:
        print(f"Upload this archive for review: {archive_path}")


if __name__ == "__main__":
    main()
