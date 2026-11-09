from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

from config import Scenario
from directional_absorbing_risk import (
    DirectionalAbsorbingRiskModel,
)
from risk_only_problem import RiskOnlyLayoutProblem
from run_validation import (
    GridSampling,
    count_boundary_violations,
    count_spacing_violations,
    min_spacing,
)


class DirectionalRiskCallback(Callback):
    """
    Record the best feasible directional-risk solution in
    every generation.
    """

    def __init__(
        self,
        scenario,
        risk_model,
    ):
        super().__init__()

        self.scenario = scenario
        self.risk_model = risk_model

        self.data["generation"] = []
        self.data["best_risk"] = []
        self.data["minimum_spacing_m"] = []
        self.data["mean_exposure"] = []
        self.data["minimum_exposure"] = []
        self.data["effective_pairs"] = []
        self.data["maximum_pair_shielding"] = []

    def notify(self, algorithm):
        X = np.asarray(
            algorithm.pop.get("X"),
            dtype=float,
        )

        F = np.asarray(
            algorithm.pop.get("F"),
            dtype=float,
        ).reshape(-1)

        G = algorithm.pop.get("G")

        if G is None:
            feasible = np.ones(
                len(X),
                dtype=bool,
            )
        else:
            G = np.asarray(
                G,
                dtype=float,
            )

            if G.ndim == 1:
                G = G[:, None]

            feasible = np.all(
                G <= 1e-9,
                axis=1,
            )

        if not np.any(feasible):
            return

        feasible_indices = np.flatnonzero(
            feasible
        )

        best_index = feasible_indices[
            np.argmin(
                F[feasible_indices]
            )
        ]

        xy = X[best_index].reshape(
            self.scenario.site.n_turbines,
            2,
        )

        diagnostics = (
            self.risk_model.diagnostics(xy)
        )

        self.data["generation"].append(
            int(algorithm.n_gen)
        )

        self.data["best_risk"].append(
            float(
                diagnostics["total_risk"]
            )
        )

        self.data["minimum_spacing_m"].append(
            float(min_spacing(xy))
        )

        self.data["mean_exposure"].append(
            float(
                diagnostics["mean_exposure"]
            )
        )

        self.data["minimum_exposure"].append(
            float(
                diagnostics["minimum_exposure"]
            )
        )

        self.data["effective_pairs"].append(
            int(
                diagnostics[
                    "number_of_effective_pairs"
                ]
            )
        )

        self.data[
            "maximum_pair_shielding"
        ].append(
            float(
                diagnostics[
                    "maximum_pair_shielding"
                ]
            )
        )


def create_reference_layout(
    scenario,
    risk_model,
    aligned,
):
    """
    Create two feasible six-turbine lines.

    aligned=True:
        Lines parallel to bird travel.

    aligned=False:
        Lines perpendicular to bird travel.

    Both references use the same relative spacings and are
    centred within the site.
    """
    centre = np.array(
        [
            scenario.site.width_m / 2.0,
            scenario.site.height_m / 2.0,
        ],
        dtype=float,
    )

    along_unit, cross_unit = (
        risk_model.direction_vectors()
    )

    primary_positions = np.array(
        [
            -2500.0,
            -1500.0,
            -500.0,
            500.0,
            1500.0,
            2500.0,
        ],
        dtype=float,
    )

    secondary_positions = np.array(
        [
            -650.0,
            650.0,
        ],
        dtype=float,
    )

    points = []

    for secondary in secondary_positions:
        for primary in primary_positions:
            if aligned:
                along_position = primary
                cross_position = secondary
            else:
                along_position = secondary
                cross_position = primary

            point = (
                centre
                + along_position * along_unit
                + cross_position * cross_unit
            )

            points.append(point)

    return np.asarray(
        points,
        dtype=float,
    )


def estimate_file_groups(
    cross_coordinates,
    threshold_m,
):
    """
    Descriptive grouping based only on cross-flow
    coordinate proximity.

    This is not a formal clustering algorithm.
    """
    ordered = np.sort(
        np.asarray(
            cross_coordinates,
            dtype=float,
        )
    )

    if len(ordered) == 0:
        return []

    groups = [[float(ordered[0])]]

    for value in ordered[1:]:
        if (
            value
            - groups[-1][-1]
            <= threshold_m
        ):
            groups[-1].append(
                float(value)
            )
        else:
            groups.append(
                [float(value)]
            )

    return groups


def save_layout_plot(
    xy,
    scenario,
    risk_model,
    title,
    output_path,
):
    """
    Plot a layout in the original site coordinate frame.
    """
    fig, axis = plt.subplots(
        figsize=(7, 7)
    )

    axis.scatter(
        xy[:, 0],
        xy[:, 1],
        s=70,
    )

    for turbine_id, (
        x_coordinate,
        y_coordinate,
    ) in enumerate(xy):
        axis.annotate(
            str(turbine_id),
            (
                x_coordinate,
                y_coordinate,
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    along_unit, _ = (
        risk_model.direction_vectors()
    )

    arrow_start = np.array(
        [
            0.85 * scenario.site.width_m,
            0.85 * scenario.site.height_m,
        ]
    )

    arrow_length = 700.0

    axis.arrow(
        arrow_start[0],
        arrow_start[1],
        arrow_length * along_unit[0],
        arrow_length * along_unit[1],
        width=18.0,
        head_width=130.0,
        length_includes_head=True,
    )

    axis.text(
        arrow_start[0],
        arrow_start[1] + 150.0,
        "Bird travel",
        ha="center",
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

    axis.set_xlabel(
        "x coordinate (m)"
    )

    axis.set_ylabel(
        "y coordinate (m)"
    )

    axis.set_title(title)

    axis.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def save_bird_frame_plot(
    xy,
    risk_model,
    output_path,
):
    """
    Plot the layout in coordinates parallel and
    perpendicular to bird travel.

    Vertical groups in this plot represent directional
    turbine files.
    """
    along, cross = (
        risk_model.bird_frame_coordinates(
            xy
        )
    )

    fig, axis = plt.subplots(
        figsize=(7, 7)
    )

    axis.scatter(
        cross,
        along,
        s=70,
    )

    for turbine_id, (
        cross_coordinate,
        along_coordinate,
    ) in enumerate(
        zip(
            cross,
            along,
        )
    ):
        axis.annotate(
            str(turbine_id),
            (
                cross_coordinate,
                along_coordinate,
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_xlabel(
        "Cross-flow coordinate (m)"
    )

    axis.set_ylabel(
        "Along-flow coordinate (m)"
    )

    axis.set_title(
        "Optimised layout in bird-travel coordinates"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def save_reference_comparison(
    layouts,
    scenario,
    risk_model,
    output_path,
):
    """
    Compare the aligned reference, cross-flow reference and
    optimised layout.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    along_unit, _ = (
        risk_model.direction_vectors()
    )

    for axis, (
        name,
        xy,
    ) in zip(
        axes,
        layouts.items(),
    ):
        risk = risk_model.collisions(xy)

        axis.scatter(
            xy[:, 0],
            xy[:, 1],
            s=55,
        )

        arrow_start = np.array(
            [
                0.85 * scenario.site.width_m,
                0.85 * scenario.site.height_m,
            ]
        )

        axis.arrow(
            arrow_start[0],
            arrow_start[1],
            600.0 * along_unit[0],
            600.0 * along_unit[1],
            width=16.0,
            head_width=120.0,
            length_includes_head=True,
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

        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

        axis.set_title(
            f"{name}\nRisk = {risk:.6f}"
        )

        axis.grid(
            True,
            alpha=0.3,
        )

    fig.suptitle(
        "Directional absorbing-risk comparison"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def save_convergence_plot(
    convergence,
    output_path,
):
    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        convergence["generation"],
        convergence["best_risk"],
    )

    axis.set_xlabel("Generation")
    axis.set_ylabel(
        "Best total diagnostic risk"
    )

    axis.set_title(
        "Directional risk-only convergence"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Optimise the newly proposed directional "
            "absorbing diagnostic risk model."
        )
    )

    parser.add_argument(
        "--bird_deg",
        type=float,
        default=225.0,
    )

    parser.add_argument(
        "--downstream_scale_D",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--cross_scale_D",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--shielding",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--pop",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--gen",
        type=int,
        default=120,
    )

    args = parser.parse_args()

    scenario = Scenario()

    risk_model = (
        DirectionalAbsorbingRiskModel(
            scenario=scenario,
            bird_travel_deg=args.bird_deg,
            downstream_scale_D=(
                args.downstream_scale_D
            ),
            cross_scale_D=(
                args.cross_scale_D
            ),
            maximum_shielding=(
                args.shielding
            ),
        )
    )

    problem = RiskOnlyLayoutProblem(
        scenario=scenario,
        risk_model=risk_model,
    )

    sampling = GridSampling(
        scenario=scenario,
        seed=args.seed,
        model_name="constant",
    )

    algorithm = GA(
        pop_size=args.pop,
        sampling=sampling,
        eliminate_duplicates=True,
    )

    callback = DirectionalRiskCallback(
        scenario=scenario,
        risk_model=risk_model,
    )

    print("")
    print(
        "Running directional absorbing risk-only optimisation"
    )
    print(
        "----------------------------------------------------"
    )
    print(
        "Status: NEWLY PROPOSED / DIAGNOSTIC ONLY"
    )
    print(
        f"Bird travel direction: "
        f"{args.bird_deg:.2f} degrees"
    )
    print(
        f"Downstream scale: "
        f"{args.downstream_scale_D:.2f}D"
    )
    print(
        f"Cross-flow scale: "
        f"{args.cross_scale_D:.2f}D"
    )
    print(
        f"Maximum pair shielding: "
        f"{args.shielding:.3f}"
    )
    print(f"Seed: {args.seed}")
    print(f"Population: {args.pop}")
    print(f"Generations: {args.gen}")
    print("")

    result = minimize(
        problem,
        algorithm,
        termination=(
            "n_gen",
            args.gen,
        ),
        seed=args.seed,
        callback=callback,
        verbose=True,
    )

    if result.X is None:
        raise RuntimeError(
            "No feasible directional-risk solution found."
        )

    best_xy = np.asarray(
        result.X,
        dtype=float,
    ).reshape(
        scenario.site.n_turbines,
        2,
    )

    diagnostics = (
        risk_model.diagnostics(best_xy)
    )

    required_spacing_m = (
        scenario.site.min_spacing_D
        * scenario.turbine.rotor_diameter_m
    )

    spacing_violation_count = (
        count_spacing_violations(
            best_xy,
            required_spacing_m,
        )
    )

    boundary_violation_count = (
        count_boundary_violations(
            best_xy,
            scenario.site.width_m,
            scenario.site.height_m,
        )
    )

    aligned_reference = (
        create_reference_layout(
            scenario=scenario,
            risk_model=risk_model,
            aligned=True,
        )
    )

    cross_flow_reference = (
        create_reference_layout(
            scenario=scenario,
            risk_model=risk_model,
            aligned=False,
        )
    )

    aligned_reference_risk = float(
        risk_model.collisions(
            aligned_reference
        )
    )

    cross_flow_reference_risk = float(
        risk_model.collisions(
            cross_flow_reference
        )
    )

    baseline_risk = float(
        risk_model.risk_scale(
            scenario.site.n_turbines
        )
    )

    best_risk = float(
        diagnostics["total_risk"]
    )

    risk_reduction_percent = (
        100.0
        * (
            baseline_risk
            - best_risk
        )
        / baseline_risk
    )

    file_groups = estimate_file_groups(
        cross_coordinates=diagnostics[
            "cross_coordinates_m"
        ],
        threshold_m=(
            1.5
            * risk_model.cross_scale_m
        ),
    )

    file_group_sizes = [
        len(group)
        for group in file_groups
    ]

    angle_label = str(
        args.bird_deg
    ).replace(
        ".",
        "p",
    )

    downstream_label = str(
        args.downstream_scale_D
    ).replace(
        ".",
        "p",
    )

    cross_label = str(
        args.cross_scale_D
    ).replace(
        ".",
        "p",
    )

    output_dir = (
        Path("results")
        / "directional_absorbing_risk_only"
        / (
            f"angle{angle_label}"
            f"_down{downstream_label}D"
            f"_cross{cross_label}D"
            f"_seed{args.seed}"
            f"_pop{args.pop}"
            f"_gen{args.gen}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    layout_dataframe = pd.DataFrame(
        {
            "turbine_id": np.arange(
                scenario.site.n_turbines
            ),
            "x_m": best_xy[:, 0],
            "y_m": best_xy[:, 1],
            "along_coordinate_m": diagnostics[
                "along_coordinates_m"
            ],
            "cross_coordinate_m": diagnostics[
                "cross_coordinates_m"
            ],
            "exposure": diagnostics[
                "exposure_per_turbine"
            ],
            "diagnostic_risk": diagnostics[
                "per_turbine_risk"
            ],
            "upstream_interaction_count": diagnostics[
                "upstream_interaction_counts"
            ],
        }
    )

    layout_dataframe.to_csv(
        output_dir / "best_layout.csv",
        index=False,
    )

    pair_dataframe = pd.DataFrame(
        {
            "source_turbine": diagnostics[
                "source_indices"
            ],
            "receiver_turbine": diagnostics[
                "receiver_indices"
            ],
            "downstream_distance_m": diagnostics[
                "downstream_distances_m"
            ],
            "cross_distance_m": diagnostics[
                "cross_distances_m"
            ],
            "kernel": diagnostics[
                "kernels"
            ],
            "shielding_fraction": diagnostics[
                "shielding_fractions"
            ],
        }
    )

    pair_dataframe.to_csv(
        output_dir / "directional_pairs.csv",
        index=False,
    )

    convergence_dataframe = pd.DataFrame(
        callback.data
    )

    convergence_dataframe.to_csv(
        output_dir / "convergence.csv",
        index=False,
    )

    reference_dataframe = pd.DataFrame(
        [
            {
                "layout": (
                    "aligned_two_file_reference"
                ),
                "risk": aligned_reference_risk,
                "minimum_spacing_m": min_spacing(
                    aligned_reference
                ),
            },
            {
                "layout": (
                    "cross_flow_two_row_reference"
                ),
                "risk": cross_flow_reference_risk,
                "minimum_spacing_m": min_spacing(
                    cross_flow_reference
                ),
            },
            {
                "layout": "optimised",
                "risk": best_risk,
                "minimum_spacing_m": min_spacing(
                    best_xy
                ),
            },
        ]
    )

    reference_dataframe.to_csv(
        output_dir
        / "reference_comparison.csv",
        index=False,
    )

    summary = {
        "model_status": (
            "NEWLY PROPOSED / DIAGNOSTIC ONLY"
        ),
        "bird_travel_deg": (
            args.bird_deg
        ),
        "downstream_scale_D": (
            args.downstream_scale_D
        ),
        "cross_scale_D": (
            args.cross_scale_D
        ),
        "maximum_shielding": (
            args.shielding
        ),
        "seed": args.seed,
        "population": args.pop,
        "generations": args.gen,
        "baseline_total_risk": (
            baseline_risk
        ),
        "best_total_risk": best_risk,
        "risk_reduction_vs_unshielded_percent": (
            risk_reduction_percent
        ),
        "aligned_reference_risk": (
            aligned_reference_risk
        ),
        "cross_flow_reference_risk": (
            cross_flow_reference_risk
        ),
        "minimum_exposure": float(
            diagnostics["minimum_exposure"]
        ),
        "mean_exposure": float(
            diagnostics["mean_exposure"]
        ),
        "maximum_exposure": float(
            diagnostics["maximum_exposure"]
        ),
        "number_of_directional_pairs": int(
            diagnostics[
                "number_of_directional_pairs"
            ]
        ),
        "number_of_effective_pairs": int(
            diagnostics[
                "number_of_effective_pairs"
            ]
        ),
        "maximum_pair_shielding": float(
            diagnostics[
                "maximum_pair_shielding"
            ]
        ),
        "mean_pair_shielding": float(
            diagnostics[
                "mean_pair_shielding"
            ]
        ),
        "minimum_spacing_m": float(
            min_spacing(best_xy)
        ),
        "minimum_spacing_required_m": float(
            required_spacing_m
        ),
        "spacing_violations": int(
            spacing_violation_count
        ),
        "boundary_violations": int(
            boundary_violation_count
        ),
        "descriptive_file_group_threshold_m": float(
            1.5
            * risk_model.cross_scale_m
        ),
        "descriptive_number_of_file_groups": int(
            len(file_groups)
        ),
        "descriptive_file_group_sizes": (
            file_group_sizes
        ),
    }

    with (
        output_dir / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    save_layout_plot(
        xy=best_xy,
        scenario=scenario,
        risk_model=risk_model,
        title=(
            "Directional absorbing risk-only optimum\n"
            f"Risk = {best_risk:.6f}"
        ),
        output_path=(
            output_dir
            / "best_layout_site.png"
        ),
    )

    save_bird_frame_plot(
        xy=best_xy,
        risk_model=risk_model,
        output_path=(
            output_dir
            / "best_layout_bird_frame.png"
        ),
    )

    save_reference_comparison(
        layouts={
            "Aligned two-file reference": (
                aligned_reference
            ),
            "Cross-flow reference": (
                cross_flow_reference
            ),
            "Optimised layout": best_xy,
        },
        scenario=scenario,
        risk_model=risk_model,
        output_path=(
            output_dir
            / "reference_comparison.png"
        ),
    )

    save_convergence_plot(
        convergence=convergence_dataframe,
        output_path=(
            output_dir
            / "convergence.png"
        ),
    )

    print("")
    print(
        "Directional absorbing risk-only run complete."
    )
    print(
        f"Best total risk: "
        f"{best_risk:.9f}"
    )
    print(
        "Unshielded baseline risk: "
        f"{baseline_risk:.9f}"
    )
    print(
        "Risk reduction from baseline: "
        f"{risk_reduction_percent:.3f}%"
    )
    print("")
    print(
        f"Aligned reference risk: "
        f"{aligned_reference_risk:.9f}"
    )
    print(
        f"Cross-flow reference risk: "
        f"{cross_flow_reference_risk:.9f}"
    )
    print("")
    print(
        f"Minimum exposure: "
        f"{diagnostics['minimum_exposure']:.9f}"
    )
    print(
        f"Mean exposure: "
        f"{diagnostics['mean_exposure']:.9f}"
    )
    print(
        "Effective directional pairs: "
        f"{diagnostics['number_of_effective_pairs']}"
    )
    print(
        "Maximum pair shielding: "
        f"{diagnostics['maximum_pair_shielding']:.9f}"
    )
    print("")
    print(
        f"Estimated file groups: "
        f"{len(file_groups)}"
    )
    print(
        f"Estimated file group sizes: "
        f"{file_group_sizes}"
    )
    print("")
    print(
        f"Minimum spacing: "
        f"{min_spacing(best_xy):.2f} m"
    )
    print(
        f"Spacing violations: "
        f"{spacing_violation_count}"
    )
    print(
        f"Boundary violations: "
        f"{boundary_violation_count}"
    )
    print("")
    print(
        f"Saved outputs to: "
        f"{output_dir}"
    )
    print("")


if __name__ == "__main__":
    main()