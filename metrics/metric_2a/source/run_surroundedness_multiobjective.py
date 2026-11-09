import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from config import Scenario
from wake import AEPModel
from coupled_risk import SurroundednessRiskModel
from validation_problem import ValidationLayoutProblem

from run_validation import (
    GridSampling,
    HVCallback,
    min_spacing,
    count_spacing_violations,
    count_boundary_violations,
)


def find_anchor_indices(dataframe):
    """
    Identify the three main Pareto-front layouts:

    - maximum-AEP solution;
    - minimum-risk solution;
    - knee solution.

    The knee is selected as the point nearest the ideal
    location after scaling both raw objectives to [0, 1].
    """
    energy_index = int(
        dataframe["aep_gwh"].idxmax()
    )

    risk_index = int(
        dataframe["risk"].idxmin()
    )

    aep_values = dataframe[
        "aep_gwh"
    ].to_numpy(dtype=float)

    risk_values = dataframe[
        "risk"
    ].to_numpy(dtype=float)

    aep_range = (
        np.max(aep_values)
        - np.min(aep_values)
    )

    risk_range = (
        np.max(risk_values)
        - np.min(risk_values)
    )

    if aep_range <= 1e-12:
        aep_loss_scaled = np.zeros_like(
            aep_values
        )
    else:
        aep_loss_scaled = (
            np.max(aep_values)
            - aep_values
        ) / aep_range

    if risk_range <= 1e-12:
        risk_scaled = np.zeros_like(
            risk_values
        )
    else:
        risk_scaled = (
            risk_values
            - np.min(risk_values)
        ) / risk_range

    distance_to_ideal = np.sqrt(
        aep_loss_scaled**2
        + risk_scaled**2
    )

    knee_index = int(
        np.argmin(distance_to_ideal)
    )

    return {
        "energy_optimal": energy_index,
        "knee": knee_index,
        "risk_optimal": risk_index,
    }


def build_pareto_dataframe(
    X,
    F,
    scenario,
    aep_model,
    risk_model,
):
    """
    Recalculate and save raw objectives and geometric
    diagnostics for every Pareto layout.
    """
    rows = []

    required_spacing = (
        scenario.site.min_spacing_D
        * scenario.turbine.rotor_diameter_m
    )

    for layout_id, candidate in enumerate(X):
        xy = candidate.reshape(
            scenario.site.n_turbines,
            2,
        )

        raw_aep = float(
            aep_model.evaluate(xy)
        )

        diagnostics = risk_model.diagnostics(
            xy
        )

        rows.append(
            {
                "layout_id": layout_id,
                "aep_gwh": raw_aep,
                "risk": float(
                    diagnostics["total_risk"]
                ),
                "surroundedness": float(
                    diagnostics["surroundedness"]
                ),
                "target_x_m": float(
                    diagnostics["target_x_m"]
                ),
                "target_y_m": float(
                    diagnostics["target_y_m"]
                ),
                "target_distance_to_centroid_m": float(
                    diagnostics[
                        "distance_to_centroid_m"
                    ]
                ),
                "target_radial_rank": int(
                    diagnostics["radial_rank"]
                ),
                "neighbours_within_5D": int(
                    diagnostics[
                        "neighbour_count_within_5D"
                    ]
                ),
                "minimum_spacing_m": min_spacing(
                    xy
                ),
                "spacing_violations": (
                    count_spacing_violations(
                        xy,
                        required_spacing,
                    )
                ),
                "boundary_violations": (
                    count_boundary_violations(
                        xy,
                        scenario.site.width_m,
                        scenario.site.height_m,
                    )
                ),
                "normalised_aep_objective": float(
                    F[layout_id, 0]
                ),
                "normalised_risk_objective": float(
                    F[layout_id, 1]
                ),
            }
        )

    return pd.DataFrame(rows)


def save_anchor_layouts(
    X,
    dataframe,
    anchors,
    scenario,
    output_dir,
):
    """
    Save coordinates and summary values for the three
    selected Pareto layouts.
    """
    summary_rows = []

    energy_aep = float(
        dataframe.loc[
            anchors["energy_optimal"],
            "aep_gwh",
        ]
    )

    energy_risk = float(
        dataframe.loc[
            anchors["energy_optimal"],
            "risk",
        ]
    )

    for name, index in anchors.items():
        xy = X[index].reshape(
            scenario.site.n_turbines,
            2,
        )

        layout_dataframe = pd.DataFrame(
            {
                "turbine_id": np.arange(
                    scenario.site.n_turbines
                ),
                "x_m": xy[:, 0],
                "y_m": xy[:, 1],
                "is_target": (
                    np.arange(
                        scenario.site.n_turbines
                    )
                    == 0
                ),
            }
        )

        layout_dataframe.to_csv(
            output_dir
            / f"layout_{name}.csv",
            index=False,
        )

        aep = float(
            dataframe.loc[index, "aep_gwh"]
        )

        risk = float(
            dataframe.loc[index, "risk"]
        )

        aep_loss_percent = (
            100.0
            * (energy_aep - aep)
            / energy_aep
        )

        if abs(energy_risk) <= 1e-12:
            risk_reduction_percent = np.nan
        else:
            risk_reduction_percent = (
                100.0
                * (energy_risk - risk)
                / energy_risk
            )

        summary_rows.append(
            {
                "layout": name,
                "layout_id": index,
                "aep_gwh": aep,
                "risk": risk,
                "surroundedness": float(
                    dataframe.loc[
                        index,
                        "surroundedness",
                    ]
                ),
                "target_distance_to_centroid_m": float(
                    dataframe.loc[
                        index,
                        "target_distance_to_centroid_m",
                    ]
                ),
                "target_radial_rank": int(
                    dataframe.loc[
                        index,
                        "target_radial_rank",
                    ]
                ),
                "neighbours_within_5D": int(
                    dataframe.loc[
                        index,
                        "neighbours_within_5D",
                    ]
                ),
                "minimum_spacing_m": float(
                    dataframe.loc[
                        index,
                        "minimum_spacing_m",
                    ]
                ),
                "aep_loss_vs_energy_optimal_percent": (
                    aep_loss_percent
                ),
                "risk_reduction_vs_energy_optimal_percent": (
                    risk_reduction_percent
                ),
            }
        )

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    summary_dataframe.to_csv(
        output_dir / "anchor_summary.csv",
        index=False,
    )

    return summary_dataframe


def plot_pareto(
    dataframe,
    anchors,
    mode,
    output_path,
):
    """
    Plot the raw AEP-risk Pareto front.
    """
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        dataframe["risk"],
        dataframe["aep_gwh"],
        s=35,
    )

    marker_map = {
        "energy_optimal": "o",
        "knee": "s",
        "risk_optimal": "*",
    }

    for name, index in anchors.items():
        ax.scatter(
            dataframe.loc[index, "risk"],
            dataframe.loc[index, "aep_gwh"],
            s=150,
            marker=marker_map[name],
            label=name.replace("_", " ").title(),
        )

    ax.set_xlabel(
        "Artificial surroundedness risk"
    )

    ax.set_ylabel(
        "Annual energy production (GWh)"
    )

    ax.set_title(
        "Coupled AEP–risk Pareto front\n"
        f"Mode: {mode}"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def plot_anchor_layouts(
    X,
    anchors,
    scenario,
    target_index,
    mode,
    output_path,
):
    """
    Plot the three anchor layouts in one figure.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    for axis, (name, index) in zip(
        axes,
        anchors.items(),
    ):
        xy = X[index].reshape(
            scenario.site.n_turbines,
            2,
        )

        other_indices = [
            i
            for i in range(
                scenario.site.n_turbines
            )
            if i != target_index
        ]

        axis.scatter(
            xy[other_indices, 0],
            xy[other_indices, 1],
            s=55,
            marker="o",
        )

        axis.scatter(
            xy[target_index, 0],
            xy[target_index, 1],
            s=170,
            marker="*",
            label="Target turbine",
        )

        for turbine_index, (x, y) in enumerate(
            xy
        ):
            axis.annotate(
                str(turbine_index),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

        axis.set_xlim(
            0.0,
            scenario.site.width_m,
        )

        axis.set_ylim(
            0.0,
            scenario.site.height_m,
        )

        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        axis.set_title(
            name.replace("_", " ").title()
        )

        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(True, alpha=0.3)

    fig.suptitle(
        "Coupled surroundedness layouts\n"
        f"Mode: {mode}"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def plot_hypervolume(
    hypervolume_values,
    output_path,
):
    """
    Save the NSGA-II hypervolume history.
    """
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    generations = np.arange(
        1,
        len(hypervolume_values) + 1,
    )

    ax.plot(
        generations,
        hypervolume_values,
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(
        "Coupled optimisation convergence"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Jensen AEP plus coupled surroundedness "
            "risk using NSGA-II."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "interior_risky",
            "interior_safe",
        ],
        default="interior_risky",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--scale_D",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--pop",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--gen",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    scenario = Scenario()
    aep_model = AEPModel(scenario)

    risk_model = SurroundednessRiskModel(
        scenario=scenario,
        target_index=args.target,
        interaction_scale_D=args.scale_D,
        mode=args.mode,
    )

    ideal_aep = float(
        aep_model.ideal_aep(
            scenario.site.n_turbines
        )
    )

    aep_min = 0.5 * ideal_aep
    aep_max = ideal_aep

    risk_min = 0.0
    risk_max = risk_model.risk_scale()

    problem = ValidationLayoutProblem(
        scenario=scenario,
        aep_model=aep_model,
        risk_model=risk_model,
        aep_min=aep_min,
        aep_max=aep_max,
        risk_min=risk_min,
        risk_max=risk_max,
    )

    sampling = GridSampling(
        scenario=scenario,
        seed=args.seed,
        model_name="constant",
    )

    algorithm = NSGA2(
        pop_size=args.pop,
        sampling=sampling,
        eliminate_duplicates=True,
    )

    callback = HVCallback(
        ref_point=np.array(
            [1.2, 1.2]
        )
    )

    print("")
    print("Running coupled AEP-surroundedness optimisation")
    print("----------------------------------------------")
    print(f"Mode: {args.mode}")
    print(f"Target index: {args.target}")
    print(
        f"Interaction scale: "
        f"{args.scale_D:.2f}D"
    )
    print(f"Seed: {args.seed}")
    print(f"Population: {args.pop}")
    print(f"Generations: {args.gen}")
    print("")

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", args.gen),
        seed=args.seed,
        callback=callback,
        verbose=True,
    )

    if result.X is None:
        raise RuntimeError(
            "No feasible Pareto solutions were found."
        )

    X = np.atleast_2d(
        np.asarray(
            result.X,
            dtype=float,
        )
    )

    F = np.atleast_2d(
        np.asarray(
            result.F,
            dtype=float,
        )
    )

    dataframe = build_pareto_dataframe(
        X=X,
        F=F,
        scenario=scenario,
        aep_model=aep_model,
        risk_model=risk_model,
    )

    anchors = find_anchor_indices(
        dataframe
    )

    scale_label = str(
        args.scale_D
    ).replace(".", "p")

    output_dir = (
        Path("results")
        / "coupled_multiobjective"
        / (
            f"surroundedness_{args.mode}"
            f"_scale{scale_label}D"
            f"_seed{args.seed}"
            f"_pop{args.pop}"
            f"_gen{args.gen}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_dir / "pareto_points.csv",
        index=False,
    )

    anchor_summary = save_anchor_layouts(
        X=X,
        dataframe=dataframe,
        anchors=anchors,
        scenario=scenario,
        output_dir=output_dir,
    )

    hypervolume_dataframe = pd.DataFrame(
        {
            "generation": np.arange(
                1,
                len(callback.data["hv"]) + 1,
            ),
            "hypervolume": callback.data["hv"],
        }
    )

    hypervolume_dataframe.to_csv(
        output_dir / "hypervolume.csv",
        index=False,
    )

    plot_pareto(
        dataframe=dataframe,
        anchors=anchors,
        mode=args.mode,
        output_path=(
            output_dir
            / "pareto_front.png"
        ),
    )

    plot_anchor_layouts(
        X=X,
        anchors=anchors,
        scenario=scenario,
        target_index=args.target,
        mode=args.mode,
        output_path=(
            output_dir
            / "anchor_layouts.png"
        ),
    )

    plot_hypervolume(
        hypervolume_values=callback.data["hv"],
        output_path=(
            output_dir
            / "hypervolume.png"
        ),
    )

    summary = {
        "model": "surroundedness",
        "mode": args.mode,
        "target_index": args.target,
        "interaction_scale_D": args.scale_D,
        "seed": args.seed,
        "population": args.pop,
        "generations": args.gen,
        "number_of_pareto_layouts": int(
            len(dataframe)
        ),
        "minimum_spacing_in_pareto_m": float(
            dataframe["minimum_spacing_m"].min()
        ),
        "total_spacing_violations": int(
            dataframe[
                "spacing_violations"
            ].sum()
        ),
        "total_boundary_violations": int(
            dataframe[
                "boundary_violations"
            ].sum()
        ),
        "final_hypervolume": float(
            callback.data["hv"][-1]
        ),
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("")
    print("Coupled optimisation complete.")
    print(
        f"Pareto layouts: "
        f"{len(dataframe)}"
    )
    print(
        f"Minimum Pareto spacing: "
        f"{dataframe['minimum_spacing_m'].min():.2f} m"
    )
    print(
        "Spacing violations: "
        f"{dataframe['spacing_violations'].sum()}"
    )
    print(
        "Boundary violations: "
        f"{dataframe['boundary_violations'].sum()}"
    )
    print("")
    print("Anchor solutions:")
    print(
        anchor_summary.to_string(
            index=False
        )
    )
    print("")
    print(
        f"Saved outputs to: "
        f"{output_dir}"
    )
    print("")


if __name__ == "__main__":
    main()