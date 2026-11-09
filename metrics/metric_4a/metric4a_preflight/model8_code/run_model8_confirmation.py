"""Confirm a saved Model 8 Pareto front on completely new trajectories.

This runner deliberately does not import FLORIS or pymoo.  It reuses the AEP
values and decision vectors saved by ``run_model8_multiobjective.py`` and only
repeats the stochastic meso-response calculation.  The confirmation sample is
therefore inexpensive and independent of both the optimisation trajectories
and the first fresh-validation trajectories.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

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


def safe_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def student_t_multiplier(confidence_level: float, batch_count: int) -> float:
    """Return the two-sided Student-t multiplier for independent batches."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    if batch_count < 2:
        raise ValueError("At least two independent batches are required.")
    probability = 0.5 + confidence_level / 2.0
    return float(student_t.ppf(probability, df=batch_count - 1))


def paired_reduction_statistics(
    reference_batches: np.ndarray,
    candidate_batches: np.ndarray,
    *,
    critical_value: float,
) -> dict[str, float | bool]:
    """Summarise a paired reduction, positive when the candidate is better."""

    reference = np.asarray(reference_batches, dtype=float)
    candidate = np.asarray(candidate_batches, dtype=float)
    if reference.ndim != 1 or candidate.shape != reference.shape:
        raise ValueError("Paired batch arrays must be one-dimensional and equal.")
    if len(reference) < 2:
        raise ValueError("At least two paired batches are required.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("Paired batch values must be finite.")
    if critical_value < 0.0 or not np.isfinite(critical_value):
        raise ValueError("critical_value must be finite and non-negative.")

    reduction = reference - candidate
    mean_reduction = float(np.mean(reduction))
    batch_sd = float(np.std(reduction, ddof=1))
    batch_se = float(batch_sd / np.sqrt(len(reduction)))
    margin = float(critical_value * batch_se)
    lower = float(mean_reduction - margin)
    upper = float(mean_reduction + margin)
    tolerance = 1.0e-15
    if batch_se <= tolerance:
        if abs(mean_reduction) <= tolerance:
            t_statistic = 0.0
            p_value = 1.0
        else:
            t_statistic = float(np.copysign(np.inf, mean_reduction))
            p_value = 0.0
    else:
        t_statistic = float(mean_reduction / batch_se)
        p_value = float(
            2.0 * student_t.sf(abs(t_statistic), df=len(reduction) - 1)
        )
    return {
        "mean_reduction": mean_reduction,
        "paired_batch_sd": batch_sd,
        "paired_batch_se": batch_se,
        "interval_margin": margin,
        "interval_lower": lower,
        "interval_upper": upper,
        "t_statistic": t_statistic,
        "two_sided_p_value": p_value,
        "reduction_supported": bool(lower > 0.0),
        "worsening_supported": bool(upper < 0.0),
    }


def generate_confirmation_batches(
    *,
    site_width_m: float,
    site_height_m: float,
    heading_deg_toward: float,
    number_of_trajectories: int,
    batch_count: int,
    maximum_response_events: int,
    longitudinal_margin_m: float,
    first_seed: int,
) -> tuple[TrajectoryPopulation, ...]:
    """Create independent batches shared across every saved Pareto layout."""

    return tuple(
        TrajectoryPopulation.generate_parallel(
            site_width_m=site_width_m,
            site_height_m=site_height_m,
            heading_deg_toward=heading_deg_toward,
            number_of_trajectories=number_of_trajectories,
            maximum_response_events=maximum_response_events,
            longitudinal_margin_m=longitudinal_margin_m,
            seed=first_seed + batch_index,
        )
        for batch_index in range(batch_count)
    )


def load_source_result(
    result_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], pd.DataFrame | None]:
    """Load and validate the minimum saved state needed for confirmation."""

    required = [
        result_dir / "pareto_decision_vectors.npy",
        result_dir / "pareto_points.csv",
        result_dir / "run_settings.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The source result is incomplete; missing: " + ", ".join(missing)
        )

    X = np.load(required[0], allow_pickle=False)
    pareto = pd.read_csv(required[1])
    settings = json.loads(required[2].read_text(encoding="utf-8"))
    if X.ndim != 2 or len(X) == 0 or not np.all(np.isfinite(X)):
        raise ValueError("pareto_decision_vectors.npy must be a finite 2-D array.")
    required_columns = {
        "solution_index",
        "aep_gwh",
        "aep_loss_gwh",
        "E_enc",
        "P_ge_1",
    }
    absent = sorted(required_columns.difference(pareto.columns))
    if absent:
        raise ValueError("pareto_points.csv is missing columns: " + ", ".join(absent))
    if len(pareto) == 0 or pareto["solution_index"].duplicated().any():
        raise ValueError("pareto_points.csv must contain unique non-empty solutions.")

    try:
        number_of_turbines = int(settings["problem"]["number_of_turbines"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "run_settings.json does not define problem.number_of_turbines."
        ) from error
    if X.shape[1] != 2 * number_of_turbines:
        raise ValueError(
            "Decision-vector width does not match the saved turbine count."
        )
    solution_indices = pareto["solution_index"].to_numpy(dtype=int)
    if np.any(solution_indices < 0) or np.any(solution_indices >= len(X)):
        raise ValueError("A saved solution_index lies outside the decision array.")

    previous_path = result_dir / "fresh_pareto_validation.csv"
    previous = pd.read_csv(previous_path) if previous_path.exists() else None
    if previous is not None:
        if "solution_index" not in previous.columns:
            raise ValueError("fresh_pareto_validation.csv lacks solution_index.")
        if previous["solution_index"].duplicated().any():
            raise ValueError("Fresh-validation solution indices must be unique.")
    return X, pareto, settings, previous


def behaviour_from_settings(settings: dict[str, Any]) -> BehaviourScenario:
    try:
        data = settings["problem"]["behaviour"]
        return BehaviourScenario(
            response_probability=float(data["response_probability"]),
            response_radius_m=float(data["response_radius_m"]),
            encounter_radius_m=float(data["encounter_radius_m"]),
            deflection_angle_deg=float(data["deflection_angle_deg"]),
            return_length_m=float(data["return_length_m"]),
            step_size_m=float(data["step_size_m"]),
            max_steps_factor=float(data.get("max_steps_factor", 4.0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "run_settings.json does not contain a complete behaviour scenario."
        ) from error


def previous_validation_columns(previous: pd.DataFrame | None) -> pd.DataFrame | None:
    if previous is None:
        return None
    mappings = {
        "fresh_E_enc": "initial_validation_E_enc",
        "fresh_P_ge_1": "initial_validation_P_ge_1",
        "fresh_mean_route_extension_m": (
            "initial_validation_mean_route_extension_m"
        ),
        "fresh_is_nondominated": "initial_validation_exact_nondominated",
        "fresh_is_robustly_nondominated": (
            "initial_validation_robustly_nondominated"
        ),
    }
    available = ["solution_index"] + [
        column for column in mappings if column in previous.columns
    ]
    return previous[available].rename(columns=mappings).copy()


def add_pairwise_fields(
    row: dict[str, Any],
    prefix: str,
    statistics: dict[str, float | bool],
) -> None:
    for key, value in statistics.items():
        row[f"{prefix}_{key}"] = value


def plot_confirmation(
    summary: pd.DataFrame,
    *,
    critical_value: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    panels = [
        (
            "confirmation_E_enc",
            "confirmation_E_enc_batch_se",
            "confirmation_exact_nondominated_E_enc",
            "confirmation_robustly_nondominated_E_enc",
            "Encounter intensity, E_enc",
        ),
        (
            "confirmation_P_ge_1",
            "confirmation_P_ge_1_batch_se",
            "confirmation_exact_nondominated_P_ge_1",
            "confirmation_robustly_nondominated_P_ge_1",
            "Probability of at least one encounter, P_ge_1",
        ),
    ]
    for axis, (metric, se_column, exact_column, robust_column, xlabel) in zip(
        axes, panels
    ):
        exact = summary[exact_column].astype(bool).to_numpy()
        robust = summary[robust_column].astype(bool).to_numpy()
        axis.errorbar(
            summary[metric],
            summary["aep_gwh"],
            xerr=critical_value * summary[se_column],
            fmt="o",
            markersize=4,
            color="0.65",
            ecolor="0.80",
            elinewidth=0.8,
            capsize=2,
            label="all saved layouts",
        )
        axis.scatter(
            summary.loc[exact, metric],
            summary.loc[exact, "aep_gwh"],
            s=32,
            color="tab:blue",
            label="exact confirmation front",
            zorder=3,
        )
        axis.scatter(
            summary.loc[robust, metric],
            summary.loc[robust, "aep_gwh"],
            s=95,
            marker="*",
            color="tab:red",
            label="uncertainty-aware front",
            zorder=4,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("FLORIS AEP (GWh/year)")
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Independent confirmation of the saved Model 8 Pareto layouts")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm saved Model 8 Pareto layouts on new independent trajectory "
            "batches without repeating FLORIS or NSGA-II."
        )
    )
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/model8_confirmation"),
    )
    parser.add_argument("--confirmation_batches", type=int, default=20)
    parser.add_argument("--trajectories_per_batch", type=int, default=5000)
    parser.add_argument(
        "--first_seed",
        type=int,
        default=None,
        help="Defaults to the source optimisation seed plus 20000.",
    )
    parser.add_argument("--confidence_level", type=float, default=0.95)
    parser.add_argument(
        "--site_width_m",
        type=float,
        default=None,
        help="Defaults to the source setting, or 6000 m for older results.",
    )
    parser.add_argument(
        "--site_height_m",
        type=float,
        default=None,
        help="Defaults to the source setting, or 6000 m for older results.",
    )
    args = parser.parse_args()

    if args.confirmation_batches < 2:
        raise ValueError("confirmation_batches must be at least two.")
    if args.trajectories_per_batch <= 0:
        raise ValueError("trajectories_per_batch must be positive.")
    result_dir = args.result_dir.resolve()
    X, pareto, source_settings, previous = load_source_result(result_dir)
    behaviour = behaviour_from_settings(source_settings)
    problem_settings = source_settings["problem"]
    site_width_m = float(
        args.site_width_m
        if args.site_width_m is not None
        else source_settings.get("site_width_m", 6000.0)
    )
    site_height_m = float(
        args.site_height_m
        if args.site_height_m is not None
        else source_settings.get("site_height_m", 6000.0)
    )
    if site_width_m <= 0.0 or site_height_m <= 0.0:
        raise ValueError("Site dimensions must be positive.")
    number_of_turbines = int(problem_settings["number_of_turbines"])
    heading_deg = float(source_settings["bird_heading_deg_toward"])
    source_seed = int(source_settings["seed"])
    first_seed = (
        int(args.first_seed)
        if args.first_seed is not None
        else source_seed + 20000
    )
    confirmation_seeds = [
        first_seed + index for index in range(args.confirmation_batches)
    ]
    previous_seeds = {
        int(seed)
        for key in (
            "objective_random_number_seeds",
            "fresh_random_number_seeds",
        )
        for seed in source_settings.get(key, [])
    }
    overlap = sorted(previous_seeds.intersection(confirmation_seeds))
    if overlap:
        raise ValueError(
            "Confirmation seeds overlap earlier samples: "
            + ", ".join(str(seed) for seed in overlap)
        )

    coordinates = X.reshape(len(X), number_of_turbines, 2)
    if (
        np.any(coordinates[:, :, 0] < 0.0)
        or np.any(coordinates[:, :, 0] > site_width_m)
        or np.any(coordinates[:, :, 1] < 0.0)
        or np.any(coordinates[:, :, 1] > site_height_m)
    ):
        raise ValueError("A saved layout lies outside the configured site.")

    critical_value = student_t_multiplier(
        args.confidence_level, args.confirmation_batches
    )
    label = (
        f"p{safe_label(behaviour.response_probability)}"
        f"_heading{safe_label(heading_deg)}"
        f"_CB{args.confirmation_batches}xN{args.trajectories_per_batch}"
        f"_seed{first_seed}"
        f"_C{safe_label(100.0 * args.confidence_level)}"
        f"_source_seed{source_seed}"
    )
    final_output_dir = args.output_root / label
    if final_output_dir.exists():
        raise FileExistsError(
            f"Confirmation output already exists: {final_output_dir}. "
            "Choose another output_root or first_seed."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(
        tempfile.mkdtemp(prefix=f"model8_{label}_INCOMPLETE_")
    )

    settings = {
        "model": "Model 8 independent confirmation validation",
        "scientific_status": source_settings.get("scientific_status"),
        "source_result_dir": str(result_dir),
        "final_output_dir": str(final_output_dir),
        "source_optimisation_seed": source_seed,
        "number_of_saved_layouts": int(len(pareto)),
        "site_width_m": site_width_m,
        "site_height_m": site_height_m,
        "bird_heading_deg_toward": heading_deg,
        "behaviour": behaviour.settings(
            float(problem_settings["rotor_diameter_m"])
        ),
        "confirmation_batches": int(args.confirmation_batches),
        "trajectories_per_batch": int(args.trajectories_per_batch),
        "trajectories_per_layout": int(
            args.confirmation_batches * args.trajectories_per_batch
        ),
        "first_seed": first_seed,
        "confirmation_random_number_seeds": confirmation_seeds,
        "confidence_level": float(args.confidence_level),
        "student_t_degrees_of_freedom": int(args.confirmation_batches - 1),
        "student_t_critical_value": critical_value,
        "comparison_design": (
            "paired common-random-number batches, independent of optimisation "
            "and first fresh validation"
        ),
    }
    (output_dir / "confirmation_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    populations = generate_confirmation_batches(
        site_width_m=site_width_m,
        site_height_m=site_height_m,
        heading_deg_toward=heading_deg,
        number_of_trajectories=args.trajectories_per_batch,
        batch_count=args.confirmation_batches,
        maximum_response_events=number_of_turbines,
        longitudinal_margin_m=(
            behaviour.response_radius_m + behaviour.return_length_m
        ),
        first_seed=first_seed,
    )
    model = MesoResponseModel(behaviour)
    prior = previous_validation_columns(previous)
    prior_by_solution = (
        None if prior is None else prior.set_index("solution_index")
    )

    start_time = time.perf_counter()
    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    turbine_rows: list[dict[str, Any]] = []
    total = len(pareto)
    for position, (_, source_row) in enumerate(pareto.iterrows(), start=1):
        solution_index = int(source_row["solution_index"])
        xy = X[solution_index].reshape(number_of_turbines, 2)
        print(
            f"Confirming saved Pareto layout {position}/{total} "
            f"(solution {solution_index})",
            flush=True,
        )
        result = summarise_meso_batches(model, xy, populations)
        row: dict[str, Any] = {
            "solution_index": solution_index,
            "aep_gwh": float(source_row["aep_gwh"]),
            "aep_loss_gwh": float(source_row["aep_loss_gwh"]),
            "optimisation_E_enc": float(source_row["E_enc"]),
            "optimisation_P_ge_1": float(source_row["P_ge_1"]),
            "confirmation_batch_count": int(result["objective_batch_count"]),
            "confirmation_N_started": int(result["N_started"]),
            "confirmation_N_completed": int(result["N_completed"]),
            "confirmation_N_failed": int(result["N_failed"]),
            "confirmation_N_approach_events": int(result["N_approach_events"]),
            "confirmation_N_response_events": int(result["N_response_events"]),
            "confirmation_N_enc_farm": int(result["N_enc_farm"]),
            "confirmation_E_enc": float(result["E_enc"]),
            "confirmation_E_enc_batch_sd": float(result["E_enc_batch_sd"]),
            "confirmation_E_enc_batch_se": float(result["E_enc_batch_se"]),
            "confirmation_E_enc_batch_min": float(result["E_enc_batch_min"]),
            "confirmation_E_enc_batch_max": float(result["E_enc_batch_max"]),
            "confirmation_P_ge_1": float(result["P_ge_1"]),
            "confirmation_P_ge_1_batch_sd": float(
                result["P_ge_1_batch_sd"]
            ),
            "confirmation_P_ge_1_batch_se": float(
                result["P_ge_1_batch_se"]
            ),
            "confirmation_mean_route_extension_m": float(
                result["mean_route_extension_m"]
            ),
            "confirmation_E_enc_gap_vs_optimisation": float(
                result["E_enc"] - source_row["E_enc"]
            ),
        }
        if prior_by_solution is not None and solution_index in prior_by_solution.index:
            for column, value in prior_by_solution.loc[solution_index].items():
                row[column] = value
            if "initial_validation_E_enc" in row:
                row["confirmation_E_enc_gap_vs_initial_validation"] = float(
                    result["E_enc"] - row["initial_validation_E_enc"]
                )
            if "initial_validation_P_ge_1" in row:
                row["confirmation_P_ge_1_gap_vs_initial_validation"] = float(
                    result["P_ge_1"] - row["initial_validation_P_ge_1"]
                )
        summary_rows.append(row)
        for batch in result["batch_summaries"]:
            batch_rows.append({"solution_index": solution_index, **batch})
        for turbine_id in range(number_of_turbines):
            turbine_rows.append(
                {
                    "solution_index": solution_index,
                    "turbine_id": turbine_id,
                    "x_m": float(xy[turbine_id, 0]),
                    "y_m": float(xy[turbine_id, 1]),
                    "confirmation_N_enc_i": int(
                        result["per_turbine_encounters"][turbine_id]
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values("solution_index").reset_index(
        drop=True
    )
    batches = pd.DataFrame(batch_rows)
    per_turbine = pd.DataFrame(turbine_rows)
    solution_order = summary["solution_index"].to_numpy(dtype=int)
    e_batches = batches.pivot(
        index="solution_index", columns="batch_index", values="E_enc"
    ).loc[solution_order]
    p_batches = batches.pivot(
        index="solution_index", columns="batch_index", values="P_ge_1"
    ).loc[solution_order]
    route_batches = batches.pivot(
        index="solution_index",
        columns="batch_index",
        values="mean_route_extension_m",
    ).loc[solution_order]

    summary["confirmation_exact_nondominated_E_enc"] = nondominated_mask(
        summary[["aep_loss_gwh", "confirmation_E_enc"]].to_numpy(dtype=float)
    )
    summary["confirmation_robustly_nondominated_E_enc"] = (
        robust_nondominated_mask(
            summary["aep_loss_gwh"].to_numpy(dtype=float),
            e_batches.to_numpy(dtype=float),
            z_value=critical_value,
        )
    )
    summary["confirmation_exact_nondominated_P_ge_1"] = nondominated_mask(
        summary[["aep_loss_gwh", "confirmation_P_ge_1"]].to_numpy(dtype=float)
    )
    summary["confirmation_robustly_nondominated_P_ge_1"] = (
        robust_nondominated_mask(
            summary["aep_loss_gwh"].to_numpy(dtype=float),
            p_batches.to_numpy(dtype=float),
            z_value=critical_value,
        )
    )

    energy_position = int(summary["aep_gwh"].to_numpy().argmax())
    energy_solution = int(summary.loc[energy_position, "solution_index"])
    energy_aep = float(summary.loc[energy_position, "aep_gwh"])
    e_reference = e_batches.loc[energy_solution].to_numpy(dtype=float)
    p_reference = p_batches.loc[energy_solution].to_numpy(dtype=float)
    route_reference = route_batches.loc[energy_solution].to_numpy(dtype=float)
    energy_row = summary.loc[energy_position]

    pairwise_rows: list[dict[str, Any]] = []
    for _, candidate in summary.iterrows():
        solution_index = int(candidate["solution_index"])
        comparison: dict[str, Any] = {
            "reference_solution_index": energy_solution,
            "candidate_solution_index": solution_index,
            "aep_gwh": float(candidate["aep_gwh"]),
            "aep_cost_vs_energy_gwh": float(energy_aep - candidate["aep_gwh"]),
            "aep_cost_vs_energy_pct": float(
                100.0 * (energy_aep - candidate["aep_gwh"]) / energy_aep
            ),
            "confirmation_E_enc": float(candidate["confirmation_E_enc"]),
            "confirmation_P_ge_1": float(candidate["confirmation_P_ge_1"]),
            "confirmation_mean_route_extension_m": float(
                candidate["confirmation_mean_route_extension_m"]
            ),
            "confirmation_N_enc_difference_vs_energy": int(
                candidate["confirmation_N_enc_farm"]
                - energy_row["confirmation_N_enc_farm"]
            ),
            "confirmation_trajectories_with_encounter_difference_vs_energy": int(
                round(
                    candidate["confirmation_P_ge_1"]
                    * candidate["confirmation_N_started"]
                    - energy_row["confirmation_P_ge_1"]
                    * energy_row["confirmation_N_started"]
                )
            ),
            "initial_validation_exact_nondominated": bool(
                candidate.get("initial_validation_exact_nondominated", False)
            ),
            "initial_validation_robustly_nondominated": bool(
                candidate.get("initial_validation_robustly_nondominated", False)
            ),
        }
        add_pairwise_fields(
            comparison,
            "E_enc",
            paired_reduction_statistics(
                e_reference,
                e_batches.loc[solution_index].to_numpy(dtype=float),
                critical_value=critical_value,
            ),
        )
        add_pairwise_fields(
            comparison,
            "P_ge_1",
            paired_reduction_statistics(
                p_reference,
                p_batches.loc[solution_index].to_numpy(dtype=float),
                critical_value=critical_value,
            ),
        )
        add_pairwise_fields(
            comparison,
            "route_extension",
            paired_reduction_statistics(
                route_reference,
                route_batches.loc[solution_index].to_numpy(dtype=float),
                critical_value=critical_value,
            ),
        )
        comparison["joint_E_enc_and_P_ge_1_reduction_supported"] = bool(
            comparison["E_enc_reduction_supported"]
            and comparison["P_ge_1_reduction_supported"]
        )
        pairwise_rows.append(comparison)
    pairwise = pd.DataFrame(pairwise_rows)

    robust_e = summary.loc[
        summary["confirmation_robustly_nondominated_E_enc"]
    ].copy()
    positions = select_anchor_positions(
        robust_e["aep_gwh"].to_numpy(dtype=float),
        robust_e["aep_loss_gwh"].to_numpy(dtype=float),
        robust_e["confirmation_E_enc"].to_numpy(dtype=float),
    )
    anchor_rows = []
    for label_name, position in positions.items():
        anchor_rows.append({"layout": label_name, **robust_e.iloc[position].to_dict()})
    anchors = pd.DataFrame(anchor_rows)

    if prior is not None and "fresh_is_nondominated" in previous.columns:
        primary_solutions = set(
            previous.loc[previous["fresh_is_nondominated"], "solution_index"]
            .astype(int)
            .tolist()
        )
    else:
        primary_solutions = {energy_solution}
    primary_solutions.add(energy_solution)
    primary = pairwise.loc[
        pairwise["candidate_solution_index"].isin(sorted(primary_solutions))
    ].copy()

    summary.to_csv(output_dir / "confirmation_layout_summary.csv", index=False)
    batches.to_csv(output_dir / "confirmation_batches.csv", index=False)
    per_turbine.to_csv(
        output_dir / "confirmation_per_turbine_encounters.csv", index=False
    )
    pairwise.to_csv(
        output_dir / "confirmation_pairwise_vs_energy.csv", index=False
    )
    primary.to_csv(
        output_dir / "confirmation_primary_contrasts.csv", index=False
    )
    summary.loc[
        summary["confirmation_exact_nondominated_E_enc"]
    ].to_csv(output_dir / "confirmation_exact_pareto_E_enc.csv", index=False)
    robust_e.to_csv(output_dir / "confirmation_robust_pareto_E_enc.csv", index=False)
    summary.loc[
        summary["confirmation_exact_nondominated_P_ge_1"]
    ].to_csv(output_dir / "confirmation_exact_pareto_P_ge_1.csv", index=False)
    summary.loc[
        summary["confirmation_robustly_nondominated_P_ge_1"]
    ].to_csv(output_dir / "confirmation_robust_pareto_P_ge_1.csv", index=False)
    anchors.to_csv(output_dir / "confirmation_anchor_summary.csv", index=False)
    plot_confirmation(
        summary,
        critical_value=critical_value,
        output_path=output_dir / "confirmation_pareto_fronts.png",
    )

    robust_e_solutions = robust_e["solution_index"].astype(int).tolist()
    robust_p_solutions = summary.loc[
        summary["confirmation_robustly_nondominated_P_ge_1"], "solution_index"
    ].astype(int).tolist()
    joint_supported = pairwise.loc[
        pairwise["joint_E_enc_and_P_ge_1_reduction_supported"],
        "candidate_solution_index",
    ].astype(int).tolist()
    if robust_e_solutions == [energy_solution] and robust_p_solutions == [
        energy_solution
    ]:
        result_class = "single_energy_endpoint_for_both_encounter_metrics"
    elif joint_supported:
        result_class = "confirmed_joint_AEP_encounter_tradeoff"
    else:
        result_class = "metric_specific_or_uncertain_tradeoff"

    runtime_seconds = float(time.perf_counter() - start_time)
    final_summary = {
        **settings,
        "runtime_seconds": runtime_seconds,
        "energy_solution_index": energy_solution,
        "number_of_exact_E_enc_pareto_layouts": int(
            summary["confirmation_exact_nondominated_E_enc"].sum()
        ),
        "number_of_robust_E_enc_pareto_layouts": int(len(robust_e_solutions)),
        "robust_E_enc_solution_indices": robust_e_solutions,
        "number_of_exact_P_ge_1_pareto_layouts": int(
            summary["confirmation_exact_nondominated_P_ge_1"].sum()
        ),
        "number_of_robust_P_ge_1_pareto_layouts": int(len(robust_p_solutions)),
        "robust_P_ge_1_solution_indices": robust_p_solutions,
        "jointly_supported_solution_indices_vs_energy": joint_supported,
        "primary_confirmation_solution_indices": sorted(primary_solutions),
        "maximum_failed_trajectories": int(
            summary["confirmation_N_failed"].max()
        ),
        "result_class": result_class,
        "interpretation_boundary": (
            "This confirmation concerns hypothetical two-dimensional encounter "
            "metrics, not calibrated flux, collision probability, or mortality."
        ),
    }
    (output_dir / "confirmation_summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    shutil.move(str(output_dir), str(final_output_dir))
    output_dir = final_output_dir

    print("\nPrimary independent confirmation contrasts:")
    display_columns = [
        "candidate_solution_index",
        "aep_cost_vs_energy_gwh",
        "E_enc_mean_reduction",
        "E_enc_interval_lower",
        "E_enc_interval_upper",
        "E_enc_reduction_supported",
        "P_ge_1_mean_reduction",
        "P_ge_1_interval_lower",
        "P_ge_1_interval_upper",
        "P_ge_1_reduction_supported",
    ]
    print(primary[display_columns].to_string(index=False))
    print(f"\nResult class: {result_class}")
    print(f"Saved Model 8 confirmation outputs to {output_dir}")


if __name__ == "__main__":
    main()
