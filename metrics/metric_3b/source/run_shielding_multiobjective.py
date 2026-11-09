"""Refine FLORIS AEP versus deterministic directional shielding with NSGA-II.

Place the ``shielding_model_code`` folder directly inside the existing
``bird_layout_opt`` project.  The runner imports ``config.py`` and
``floris_aep.py`` from that project root.  Version 5 pre-screens and repairs
spacing, constrains rotor-grid velocity, matches unordered turbines before
crossover, preserves an external Pareto archive, and reports endpoints without
an automatic knee.
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
from pymoo.operators.crossover.sbx import SBX
from pymoo.optimize import minimize
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
DEFAULT_PRIOR_SEED_CSV = (
    CODE_DIR / "seed_data" / "prior_pareto_seed_layouts.csv"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Scenario
from floris_aep import FLORISAEPModel
from run_shielding_validation import make_fixed_layouts, pairwise_distances
from shielding_aep_problem import ShieldingAEPProblem
from shielding_model import evaluate_shielding, heading_unit_vector


def safe_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def match_layout_turbine_order(
    reference_vector: np.ndarray,
    candidate_vector: np.ndarray,
) -> np.ndarray:
    """Match an unordered candidate layout to a reference before crossover.

    Turbine IDs are arbitrary in a layout optimisation.  The Hungarian
    assignment prevents crossover from mixing unrelated physical locations
    merely because they happen to have the same array index.
    """

    reference = np.asarray(reference_vector, dtype=float).reshape(-1, 2)
    candidate = np.asarray(candidate_vector, dtype=float).reshape(-1, 2)
    if reference.shape != candidate.shape or reference.shape[0] == 0:
        raise ValueError("Layouts must have equal, non-empty (N, 2) shapes.")
    squared_distance = np.sum(
        (reference[:, None, :] - candidate[None, :, :]) ** 2,
        axis=2,
    )
    reference_indices, candidate_indices = linear_sum_assignment(
        squared_distance
    )
    if not np.array_equal(reference_indices, np.arange(len(reference))):
        raise RuntimeError("Unexpected non-sequential assignment result.")
    return candidate[candidate_indices].reshape(-1)


def repair_minimum_spacing_layout(
    vector: np.ndarray,
    *,
    minimum_spacing_m: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    maximum_iterations: int = 120,
) -> np.ndarray:
    """Iteratively separate colliding turbines while respecting site bounds."""

    points = np.asarray(vector, dtype=float).reshape(-1, 2).copy()
    lower = np.asarray(lower_bounds, dtype=float).reshape(-1, 2)
    upper = np.asarray(upper_bounds, dtype=float).reshape(-1, 2)
    if points.shape != lower.shape or points.shape != upper.shape:
        raise ValueError("Coordinate and bound shapes must match.")
    clearance = float(minimum_spacing_m)
    if clearance <= 0.0:
        raise ValueError("minimum_spacing_m must be positive.")

    points = np.clip(points, lower, upper)
    for _ in range(maximum_iterations):
        worst_deficit = 0.0
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                delta = points[second] - points[first]
                distance = float(np.linalg.norm(delta))
                deficit = clearance - distance
                if deficit <= 1.0e-7:
                    continue
                worst_deficit = max(worst_deficit, deficit)
                if distance <= 1.0e-12:
                    angle = (
                        (first + 1) * 1.61803398875
                        + (second + 1) * 0.61803398875
                    ) * np.pi
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = delta / distance
                shift = 0.5 * (deficit + 1.0e-6) * direction
                points[first] = np.clip(
                    points[first] - shift,
                    lower[first],
                    upper[first],
                )
                points[second] = np.clip(
                    points[second] + shift,
                    lower[second],
                    upper[second],
                )
        if worst_deficit <= 1.0e-7:
            break
    return points.reshape(-1)


class PermutationAwareSBX(SBX):
    """Spatially matched SBX with deterministic minimum-spacing repair."""

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        aligned = np.asarray(X, dtype=float).copy()
        for mating_index in range(aligned.shape[1]):
            aligned[1, mating_index] = match_layout_turbine_order(
                aligned[0, mating_index],
                aligned[1, mating_index],
            )
        offspring = super()._do(
            problem,
            aligned,
            *args,
            random_state=random_state,
            **kwargs,
        )
        for offspring_index in range(offspring.shape[0]):
            for mating_index in range(offspring.shape[1]):
                offspring[offspring_index, mating_index] = (
                    repair_minimum_spacing_layout(
                        offspring[offspring_index, mating_index],
                        minimum_spacing_m=problem.minimum_spacing_m,
                        lower_bounds=problem.xl,
                        upper_bounds=problem.xu,
                    )
                )
        return offspring


def nondominated_indices_2d(objectives: np.ndarray) -> np.ndarray:
    """Return a deterministic two-objective minimisation archive."""

    values = np.atleast_2d(np.asarray(objectives, dtype=float))
    if values.shape[1] != 2:
        raise ValueError("Exactly two objectives are required.")
    if len(values) == 0:
        return np.empty(0, dtype=int)
    if not np.all(np.isfinite(values)):
        raise ValueError("Archive objectives must be finite.")

    order = np.lexsort((values[:, 0], values[:, 1]))
    keep: list[int] = []
    best_f0 = np.inf
    for index in order:
        f0 = float(values[index, 0])
        if f0 < best_f0:
            keep.append(int(index))
            best_f0 = f0
    return np.asarray(keep, dtype=int)


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


def load_prior_seed_csv(
    path: Path,
    number_of_turbines: int,
) -> tuple[np.ndarray, list[str]]:
    """Load exact prior Pareto layouts used to initialise refinement."""
    if not path.exists():
        raise FileNotFoundError(f"Prior-seed file not found: {path}")
    dataframe = pd.read_csv(path)
    coordinate_columns = [
        column
        for turbine_id in range(number_of_turbines)
        for column in (f"x_{turbine_id}_m", f"y_{turbine_id}_m")
    ]
    missing = [column for column in coordinate_columns if column not in dataframe]
    if missing:
        raise ValueError(
            f"{path} is missing prior-seed coordinate columns: {missing}"
        )
    vectors = dataframe[coordinate_columns].to_numpy(dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2 * number_of_turbines:
        raise ValueError("Unexpected prior-seed decision-vector shape.")
    if len(vectors) == 0 or not np.all(np.isfinite(vectors)):
        raise ValueError("Prior-seed coordinates must be non-empty and finite.")

    if "seed_id" in dataframe:
        labels = [f"prior_{value}" for value in dataframe["seed_id"].astype(str)]
    else:
        labels = [f"prior_pareto_{index}" for index in range(len(vectors))]
    return vectors.reshape(-1, number_of_turbines, 2), labels


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
    """Feasible initial population containing exact interpretable layouts."""

    def __init__(
        self,
        *,
        scenario: Scenario,
        rotor_diameter_m: float,
        exact_seed_layouts: dict[str, np.ndarray],
        prior_seed_layouts: np.ndarray,
        prior_seed_labels: list[str],
        projection_preserving_seed_names: set[str],
        seed: int,
        jitter_scale_D: float = 0.20,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.rotor_diameter_m = float(rotor_diameter_m)
        self.exact_seed_layouts = {
            name: np.asarray(layout, dtype=float).copy()
            for name, layout in exact_seed_layouts.items()
        }
        self.prior_seed_layouts = np.asarray(
            prior_seed_layouts,
            dtype=float,
        ).reshape(-1, int(scenario.site.n_turbines), 2)
        self.prior_seed_labels = list(prior_seed_labels)
        if len(self.prior_seed_layouts) != len(self.prior_seed_labels):
            raise ValueError(
                "prior_seed_layouts and prior_seed_labels must have equal length."
            )
        self.projection_preserving_seed_names = set(
            projection_preserving_seed_names
        )
        self.seed = int(seed)
        self.jitter_scale_D = float(jitter_scale_D)
        self.jitter_scale_m = float(jitter_scale_D * rotor_diameter_m)
        self.last_labels: list[str] = []
        self.last_prior_seed_count = 0

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        if n_samples < len(self.exact_seed_layouts):
            raise ValueError(
                "Population size must be at least the number of exact seed layouts."
            )

        rng = np.random.default_rng(self.seed)
        width = float(self.scenario.site.width_m)
        height = float(self.scenario.site.height_m)
        number_of_turbines = int(self.scenario.site.n_turbines)
        minimum_spacing = float(
            self.scenario.site.min_spacing_D * self.rotor_diameter_m
        )
        accepted: list[np.ndarray] = []
        labels: list[str] = []

        def add(candidate_xy: np.ndarray, label: str) -> bool:
            candidate = np.asarray(candidate_xy, dtype=float).reshape(
                number_of_turbines, 2
            )
            if not layout_is_feasible(
                candidate,
                width_m=width,
                height_m=height,
                minimum_spacing_m=minimum_spacing,
            ):
                return False
            vector = candidate.reshape(-1)
            if any(np.max(np.abs(vector - existing)) <= 1.0e-7 for existing in accepted):
                return False
            accepted.append(vector)
            labels.append(label)
            return True

        for name, layout in self.exact_seed_layouts.items():
            if not add(layout, f"exact_{name}"):
                raise ValueError(f"Exact seed layout is infeasible or duplicated: {name}")

        prior_count_before = len(accepted)
        number_of_prior = len(self.prior_seed_layouts)
        exploration_reserve = min(
            max(20, n_samples // 4),
            n_samples - len(accepted),
        )
        prior_capacity = max(
            0,
            n_samples - len(accepted) - exploration_reserve,
        )
        if number_of_prior > 0 and prior_capacity > 0:
            if number_of_prior <= prior_capacity:
                preferred = list(range(number_of_prior))
            else:
                spread = np.linspace(0, number_of_prior - 1, prior_capacity)
                preferred = list(dict.fromkeys(np.rint(spread).astype(int).tolist()))
                preferred.extend(
                    index for index in range(number_of_prior) if index not in preferred
                )

            for index in preferred:
                if (
                    len(accepted) >= n_samples
                    or len(accepted) - prior_count_before >= prior_capacity
                ):
                    break
                layout = self.prior_seed_layouts[index]
                if not layout_is_feasible(
                    layout,
                    width_m=width,
                    height_m=height,
                    minimum_spacing_m=minimum_spacing,
                ):
                    raise ValueError(
                        "Prior Pareto seed is infeasible under the active model: "
                        f"{self.prior_seed_labels[index]}"
                    )
                add(layout, self.prior_seed_labels[index])
        self.last_prior_seed_count = len(accepted) - prior_count_before

        seed_items = list(self.exact_seed_layouts.items())
        random_reserve = min(max(10, n_samples // 8), n_samples - len(accepted))
        jitter_population_target = n_samples - random_reserve
        attempts = 0
        while len(accepted) < jitter_population_target:
            attempts += 1
            if attempts > 10000:
                break
            name, base = seed_items[attempts % len(seed_items)]
            translation = rng.normal(0.0, 0.35 * self.jitter_scale_m, size=2)
            if name in self.projection_preserving_seed_names:
                along = heading_unit_vector(self.scenario.site.migration_axis_deg)
                along_offsets = rng.normal(
                    0.0, self.jitter_scale_m, size=(len(base), 1)
                )
                local = along_offsets * along[None, :]
                label = f"jitter_projection_preserving_{name}"
            else:
                local = rng.normal(0.0, self.jitter_scale_m, size=base.shape)
                label = f"jitter_{name}"
            add(base + translation + local, label)

        while len(accepted) < n_samples:
            add(
                random_feasible_layout(
                    rng=rng,
                    number_of_turbines=number_of_turbines,
                    width_m=width,
                    height_m=height,
                    minimum_spacing_m=minimum_spacing,
                ),
                "random_feasible",
            )

        self.last_labels = labels[:n_samples]
        return np.asarray(accepted[:n_samples], dtype=float)


class HypervolumeCallback(Callback):
    def __init__(self, reference_point: tuple[float, float] = (1.2, 1.05)) -> None:
        super().__init__()
        self.reference_point = np.asarray(reference_point, dtype=float)
        self.indicator = HV(ref_point=self.reference_point)
        self.archive_objectives = np.empty((0, 2), dtype=float)
        self.last_evaluation_index = 0
        self.data["generation"] = []
        self.data["normalised_hypervolume"] = []
        self.data["minimum_f0"] = []
        self.data["minimum_f1"] = []
        self.data["archive_size"] = []

    def notify(self, algorithm) -> None:
        evaluations = algorithm.problem.evaluation_dataframe()
        new_rows = evaluations.iloc[self.last_evaluation_index :]
        self.last_evaluation_index = len(evaluations)
        feasible = new_rows.loc[
            new_rows["floris_evaluated"].astype(bool)
            & new_rows["spacing_feasible"].astype(bool)
            & new_rows["rotor_velocity_feasible"].astype(bool)
        ]
        if len(feasible):
            new_objectives = feasible[
                ["f0_normalised_aep_loss", "f1_normalised_N_eff"]
            ].to_numpy(dtype=float)
            combined = np.vstack([self.archive_objectives, new_objectives])
            self.archive_objectives = combined[
                nondominated_indices_2d(combined)
            ]

        objectives = self.archive_objectives
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
        self.data["archive_size"].append(int(len(objectives)))


def build_pareto_dataframe(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    problem: ShieldingAEPProblem,
) -> pd.DataFrame:
    rows: list[dict] = []
    for solution_index, (vector, objectives, all_constraints) in enumerate(
        zip(X, F, G)
    ):
        xy = vector.reshape(problem.number_of_turbines, 2)
        shielding = evaluate_shielding(
            xy,
            rotor_diameter_m=problem.rotor_diameter_m,
            bird_heading_deg_toward=problem.bird_heading_deg_toward,
            corridor_width_m=problem.corridor_width_m,
        )
        constraints = problem.spacing_constraints(xy)
        rotor_velocity_constraint = float(all_constraints[-1])
        minimum_rotor_velocity = -rotor_velocity_constraint
        aep_loss_gwh = float(objectives[0] * problem.aep_loss_reference_gwh)
        rows.append(
            {
                "solution_index": solution_index,
                "aep_gwh": float(problem.aep_ceiling_gwh - aep_loss_gwh),
                "aep_loss_gwh": aep_loss_gwh,
                **shielding.summary(),
                "f0_normalised_aep_loss": float(objectives[0]),
                "f1_normalised_N_eff": float(objectives[1]),
                "minimum_spacing_m": float(np.min(pairwise_distances(xy))),
                "maximum_spacing_constraint": float(np.max(constraints)),
                "spacing_violations": int(np.count_nonzero(constraints > 0.0)),
                "minimum_rotor_grid_velocity_m_s": minimum_rotor_velocity,
                "rotor_velocity_constraint": rotor_velocity_constraint,
                "rotor_velocity_violations": int(rotor_velocity_constraint > 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["N_eff", "aep_gwh"], ascending=[True, False]
    ).reset_index(drop=True)


def find_endpoint_indices(dataframe: pd.DataFrame) -> dict[str, int]:
    """Select only the two objective endpoints."""

    energy_candidates = dataframe.loc[
        dataframe["aep_gwh"] == dataframe["aep_gwh"].max()
    ]
    energy_index = int(energy_candidates["N_eff"].idxmin())

    shielding_candidates = dataframe.loc[
        dataframe["N_eff"] == dataframe["N_eff"].min()
    ]
    shielding_index = int(shielding_candidates["aep_gwh"].idxmax())

    return {
        "energy_endpoint": energy_index,
        "shielding_endpoint": shielding_index,
    }


def save_anchor_layouts(
    *,
    X: np.ndarray,
    dataframe: pd.DataFrame,
    anchors: dict[str, int],
    problem: ShieldingAEPProblem,
    output_dir: Path,
) -> pd.DataFrame:
    layout_dir = output_dir / "layouts"
    layout_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, dataframe_index in anchors.items():
        row = dataframe.loc[dataframe_index]
        solution_index = int(row["solution_index"])
        xy = X[solution_index].reshape(problem.number_of_turbines, 2)
        shielding = evaluate_shielding(
            xy,
            rotor_diameter_m=problem.rotor_diameter_m,
            bird_heading_deg_toward=problem.bird_heading_deg_toward,
            corridor_width_m=problem.corridor_width_m,
        )
        pd.DataFrame(
            {
                "turbine_id": np.arange(problem.number_of_turbines, dtype=int),
                "x_m": xy[:, 0],
                "y_m": xy[:, 1],
                "projected_crossflight_m": shielding.projected_centres_m,
                "interval_start_m": shielding.rotor_intervals_m[:, 0],
                "interval_end_m": shielding.rotor_intervals_m[:, 1],
            }
        ).sort_values("projected_crossflight_m").to_csv(
            layout_dir / f"{label}_layout.csv", index=False
        )
        rows.append({"layout": label, **row.to_dict()})
    return pd.DataFrame(rows)


def plot_pareto(
    dataframe: pd.DataFrame,
    anchors: dict[str, int],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.scatter(dataframe["N_eff"], dataframe["aep_gwh"], s=30, alpha=0.8)
    for label, index in anchors.items():
        row = dataframe.loc[index]
        axis.scatter(row["N_eff"], row["aep_gwh"], s=90, marker="*", label=label)
    axis.set_xlabel("Effective exposed rotor count, N_eff")
    axis.set_ylabel("AEP (GWh/year)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimise FLORIS AEP against directional shielding."
    )
    parser.add_argument("--floris_config", type=Path, required=True)
    parser.add_argument("--energy_seed_layout", type=Path, required=True)
    parser.add_argument(
        "--output_root", type=Path, default=Path("results/shielding_aep")
    )
    parser.add_argument("--bird_heading_deg", type=float, default=225.0)
    parser.add_argument("--corridor_width_m", type=float, default=6000.0)
    parser.add_argument(
        "--aep_loss_reference_gwh", type=float, default=137.96914764866568
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--gen", type=int, default=100)
    parser.add_argument(
        "--prior_seed_csv",
        type=Path,
        default=DEFAULT_PRIOR_SEED_CSV,
        help="CSV of prior Pareto layouts used for refinement seeding.",
    )
    parser.add_argument(
        "--disable_prior_seeds",
        action="store_true",
        help="Run without the packaged prior Pareto seed layouts.",
    )
    parser.add_argument("--save_evaluations", action="store_true")
    args = parser.parse_args()

    if args.pop < 5 or args.gen <= 0:
        raise ValueError("pop must be at least five and gen must be positive.")
    if not np.isfinite(args.bird_heading_deg):
        raise ValueError("bird_heading_deg must be finite.")
    if args.corridor_width_m <= 0.0:
        raise ValueError("corridor_width_m must be positive.")

    scenario = Scenario()
    scenario.site.n_turbines = 12
    scenario.site.width_m = 6000.0
    scenario.site.height_m = 6000.0
    scenario.site.min_spacing_D = 4.0
    scenario.site.migration_axis_deg = float(args.bird_heading_deg)
    scenario.birds.migration_width_m = float(args.corridor_width_m)

    aep_model = FLORISAEPModel(scenario, config_path=args.floris_config)
    rotor_diameter_m = float(aep_model.floris_rotor_diameter_m)
    scenario.turbine.rotor_diameter_m = rotor_diameter_m
    minimum_spacing_m = float(scenario.site.min_spacing_D * rotor_diameter_m)

    if args.corridor_width_m + 1.0e-9 < scenario.site.n_turbines * rotor_diameter_m:
        raise ValueError(
            "corridor_width_m must be at least N*D so P_enc remains valid for "
            "every possible layout considered by the optimiser."
        )

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

    aep_ceiling_gwh = float(aep_model.ideal_aep(scenario.site.n_turbines))
    problem = ShieldingAEPProblem(
        scenario=scenario,
        aep_model=aep_model,
        rotor_diameter_m=rotor_diameter_m,
        bird_heading_deg_toward=args.bird_heading_deg,
        corridor_width_m=args.corridor_width_m,
        aep_ceiling_gwh=aep_ceiling_gwh,
        aep_loss_reference_gwh=args.aep_loss_reference_gwh,
        record_evaluations=True,
    )

    fixed_layouts = make_fixed_layouts(
        site_width_m=scenario.site.width_m,
        site_height_m=scenario.site.height_m,
        bird_heading_deg_toward=args.bird_heading_deg,
        minimum_spacing_m=minimum_spacing_m,
    )
    exact_seeds = {"energy": energy_seed_xy, **fixed_layouts}
    if args.disable_prior_seeds:
        prior_seed_layouts = np.empty(
            (0, scenario.site.n_turbines, 2), dtype=float
        )
        prior_seed_labels: list[str] = []
        prior_seed_path: str | None = None
    else:
        prior_seed_layouts, prior_seed_labels = load_prior_seed_csv(
            args.prior_seed_csv,
            scenario.site.n_turbines,
        )
        prior_seed_path = str(args.prior_seed_csv)
    sampling = FeasibleSeededSampling(
        scenario=scenario,
        rotor_diameter_m=rotor_diameter_m,
        exact_seed_layouts=exact_seeds,
        prior_seed_layouts=prior_seed_layouts,
        prior_seed_labels=prior_seed_labels,
        projection_preserving_seed_names=set(fixed_layouts),
        seed=args.seed,
    )
    algorithm = NSGA2(
        pop_size=args.pop,
        sampling=sampling,
        crossover=PermutationAwareSBX(prob=0.9, eta=15),
        eliminate_duplicates=True,
    )
    callback = HypervolumeCallback()

    case_label = (
        f"heading{safe_label(args.bird_heading_deg)}"
        f"_W{safe_label(args.corridor_width_m)}"
        f"_seed{args.seed}_pop{args.pop}_gen{args.gen}"
    )
    output_dir = args.output_root / case_label
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_X = sampling._do(problem, args.pop)
    pd.DataFrame(
        initial_X,
        columns=[
            column
            for turbine_id in range(scenario.site.n_turbines)
            for column in (f"x_{turbine_id}_m", f"y_{turbine_id}_m")
        ],
    ).assign(seed_source=sampling.last_labels).to_csv(
        output_dir / "initial_population.csv", index=False
    )

    start_time = time.perf_counter()
    if len(prior_seed_layouts):
        problem.evaluate(
            prior_seed_layouts.reshape(len(prior_seed_layouts), -1),
            return_values_of=["F", "G"],
        )

    run_settings = {
        "model": (
            "Shielding Model v5: FLORIS AEP versus directional analytical "
            "shielding"
        ),
        "scientific_status": "literature-informed geometric surrogate developed in this study",
        "objectives": ["maximise AEP", "minimise N_eff"],
        "primary_bird_metric": "N_eff = L_exp / D",
        "optimiser_bird_objective": "N_eff / N",
        "secondary_metrics": [
            "P_enc = L_exp / W",
            "P_free = 1 - P_enc",
            "shielding_fraction = 1 - N_eff/N",
        ],
        "deterministic_bird_objective": True,
        "trajectory_batches": 0,
        "bird_heading_convention": "TOWARD; 0=east (+x), 90=north (+y)",
        "bird_heading_deg_toward": float(args.bird_heading_deg) % 360.0,
        "corridor_width_m": float(args.corridor_width_m),
        "locked_case_validated_minimum_N_eff": 2.0,
        "seed": int(args.seed),
        "population": int(args.pop),
        "generations": int(args.gen),
        "site_width_m": float(scenario.site.width_m),
        "site_height_m": float(scenario.site.height_m),
        "problem": problem.settings_summary(),
        "aep_model": aep_model.settings_summary(),
        "energy_seed_layout": str(args.energy_seed_layout),
        "floris_config": str(args.floris_config),
        "initialisation": {
            "exact_seed_layouts": list(exact_seeds),
            "prior_seed_csv": prior_seed_path,
            "available_prior_pareto_seed_count": int(len(prior_seed_layouts)),
            "validated_external_prior_seed_count": int(
                len(prior_seed_layouts)
            ),
            "included_prior_pareto_seed_count": int(
                sampling.last_prior_seed_count
            ),
            "projection_preserving_jitter_seed_layouts": list(fixed_layouts),
            "jitter_scale_D": sampling.jitter_scale_D,
        },
        "optimisation_safeguards": {
            "permutation_aware_crossover": (
                "Hungarian spatial matching before simulated binary crossover"
            ),
            "reported_front": (
                "external nondominated archive over every feasible evaluation"
            ),
        },
        "not_modelled": [
            "bird response or avoidance",
            "individual bird trajectories",
            "encounter intensity",
            "flight height",
            "rotor transit",
            "collision probability",
            "mortality",
        ],
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(run_settings, indent=2), encoding="utf-8"
    )

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

    evaluations = problem.evaluation_dataframe()
    valid_evaluations = evaluations.loc[
        evaluations["floris_evaluated"].astype(bool)
        & evaluations["spacing_feasible"].astype(bool)
        & evaluations["rotor_velocity_feasible"].astype(bool)
    ].copy()
    if len(valid_evaluations) == 0:
        raise RuntimeError("No valid FLORIS evaluations were recorded.")
    objective_columns = [
        "f0_normalised_aep_loss",
        "f1_normalised_N_eff",
    ]
    archive_local_indices = nondominated_indices_2d(
        valid_evaluations[objective_columns].to_numpy(dtype=float)
    )
    archive_rows = valid_evaluations.iloc[archive_local_indices].copy()
    coordinate_columns = [
        column
        for turbine_id in range(scenario.site.n_turbines)
        for column in (f"x_{turbine_id}_m", f"y_{turbine_id}_m")
    ]
    X = archive_rows[coordinate_columns].to_numpy(dtype=float)
    F = archive_rows[objective_columns].to_numpy(dtype=float)
    G = np.vstack(
        [
            np.concatenate(
                [
                    problem.spacing_constraints(
                        vector.reshape(problem.number_of_turbines, 2)
                    ),
                    np.asarray([float(rotor_constraint)]),
                ]
            )
            for vector, rotor_constraint in zip(
                X,
                archive_rows["rotor_velocity_constraint"],
            )
        ]
    )
    np.save(output_dir / "pareto_decision_vectors.npy", X)
    np.save(output_dir / "pareto_normalised_objectives.npy", F)
    np.save(output_dir / "pareto_constraints.npy", G)

    pareto = build_pareto_dataframe(X, F, G, problem)
    source_evaluation_by_solution = dict(
        enumerate(archive_rows["evaluation_index"].astype(int).to_list())
    )
    pareto["source_evaluation_index"] = pareto["solution_index"].map(
        source_evaluation_by_solution
    )
    if int(pareto["spacing_violations"].sum()) != 0:
        raise RuntimeError("A saved Pareto layout violates minimum spacing.")
    if int(pareto["rotor_velocity_violations"].sum()) != 0:
        raise RuntimeError("A saved Pareto layout has negative rotor velocity.")
    pareto.to_csv(output_dir / "pareto_points.csv", index=False)
    anchors = find_endpoint_indices(pareto)
    anchor_summary = save_anchor_layouts(
        X=X,
        dataframe=pareto,
        anchors=anchors,
        problem=problem,
        output_dir=output_dir,
    )
    anchor_summary.to_csv(output_dir / "anchor_summary.csv", index=False)
    plot_pareto(pareto, anchors, output_dir / "pareto_front.png")

    hypervolume = pd.DataFrame(callback.data)
    hypervolume.to_csv(output_dir / "hypervolume.csv", index=False)
    if args.save_evaluations:
        evaluations.to_csv(
            output_dir / "all_evaluated_candidates.csv", index=False
        )

    summary = {
        **run_settings,
        "runtime_seconds": runtime_seconds,
        "number_of_pareto_layouts": int(len(pareto)),
        "pareto_source": "external archive over all feasible evaluations",
        "total_spacing_violations": int(pareto["spacing_violations"].sum()),
        "total_rotor_velocity_violations": int(
            pareto["rotor_velocity_violations"].sum()
        ),
        "distinct_anchor_count": int(anchor_summary["solution_index"].nunique()),
        "candidate_screening": problem.candidate_screening_summary(),
        "final_normalised_hypervolume": (
            None
            if len(hypervolume) == 0
            else float(hypervolume["normalised_hypervolume"].iloc[-1])
        ),
        "anchors": json.loads(anchor_summary.to_json(orient="records")),
        "validation_note": (
            "No fresh trajectory validation is required because the shielding "
            "objective is analytical and deterministic."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(
        anchor_summary[
            [
                "layout",
                "solution_index",
                "aep_gwh",
                "N_eff",
                "P_enc",
                "P_free",
                "minimum_rotor_grid_velocity_m_s",
            ]
        ].to_string(index=False)
    )
    screening = problem.candidate_screening_summary()
    print(
        "\nCandidate screening: "
        f"{screening['spacing_prescreen_skip_count']} spacing-infeasible "
        "layouts skipped before FLORIS; "
        f"{screening['negative_rotor_velocity_candidate_count']} "
        "spacing-feasible layouts rejected for negative rotor velocity."
    )
    print(f"\nSaved Shielding Model optimisation outputs to {output_dir}")


if __name__ == "__main__":
    main()
