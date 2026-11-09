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
from pairwise_risk import PairwiseGaussianRiskModel
from validation_problem import ValidationLayoutProblem

from run_validation import (
    GridSampling,
    HVCallback,
    count_boundary_violations,
    count_spacing_violations,
    min_spacing,
)


def load_layout_csv(layout_path, scenario):
    """
    Load a saved turbine layout.

    Required columns:
        x_m, y_m

    Rows are sorted by turbine_id when available.
    """
    layout_path = Path(layout_path)

    if not layout_path.exists():
        raise FileNotFoundError(
            f"Layout file not found: {layout_path}"
        )

    dataframe = pd.read_csv(layout_path)

    required_columns = {
        "x_m",
        "y_m",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{layout_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if "turbine_id" in dataframe.columns:
        dataframe = dataframe.sort_values(
            "turbine_id"
        )

    xy = dataframe[
        [
            "x_m",
            "y_m",
        ]
    ].to_numpy(dtype=float)

    expected_shape = (
        scenario.site.n_turbines,
        2,
    )

    if xy.shape != expected_shape:
        raise ValueError(
            f"Expected layout shape {expected_shape}, "
            f"received {xy.shape}."
        )

    if not np.all(np.isfinite(xy)):
        raise ValueError(
            f"Non-finite coordinates found in {layout_path}."
        )

    return xy


def find_anchor_indices(dataframe):
    """
    Identify:

    - maximum-AEP endpoint;
    - minimum-risk endpoint;
    - geometric knee.

    The knee is found after scaling the observed AEP loss
    and risk ranges to [0, 1]. It is a descriptive point,
    not a uniquely optimal design.
    """
    energy_index = int(
        dataframe["aep_gwh"].idxmax()
    )

    risk_index = int(
        dataframe["pairwise_risk"].idxmin()
    )

    aep_values = dataframe[
        "aep_gwh"
    ].to_numpy(dtype=float)

    risk_values = dataframe[
        "pairwise_risk"
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
        scaled_aep_loss = np.zeros_like(
            aep_values
        )
    else:
        scaled_aep_loss = (
            np.max(aep_values)
            - aep_values
        ) / aep_range

    if risk_range <= 1e-12:
        scaled_risk = np.zeros_like(
            risk_values
        )
    else:
        scaled_risk = (
            risk_values
            - np.min(risk_values)
        ) / risk_range

    distance_to_ideal = np.sqrt(
        scaled_aep_loss**2
        + scaled_risk**2
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
    Recalculate raw objectives and layout diagnostics for
    every non-dominated layout.
    """
    required_spacing_m = (
        scenario.site.min_spacing_D
        * scenario.turbine.rotor_diameter_m
    )

    rows = []

    for layout_id, candidate in enumerate(X):
        xy = candidate.reshape(
            scenario.site.n_turbines,
            2,
        )

        aep_gwh = float(
            aep_model.evaluate(xy)
        )

        diagnostics = risk_model.diagnostics(
            xy
        )

        rows.append(
            {
                "layout_id": layout_id,
                "aep_gwh": aep_gwh,
                "pairwise_risk": float(
                    diagnostics["total_risk"]
                ),
                "minimum_pair_distance_m": float(
                    diagnostics[
                        "minimum_pair_distance_m"
                    ]
                ),
                "mean_pair_distance_m": float(
                    diagnostics[
                        "mean_pair_distance_m"
                    ]
                ),
                "maximum_pair_distance_m": float(
                    diagnostics[
                        "maximum_pair_distance_m"
                    ]
                ),
                "maximum_pair_weight": float(
                    diagnostics[
                        "maximum_pair_weight"
                    ]
                ),
                "mean_pair_weight": float(
                    diagnostics[
                        "mean_pair_weight"
                    ]
                ),
                "minimum_spacing_m": float(
                    min_spacing(xy)
                ),
                "spacing_violations": int(
                    count_spacing_violations(
                        xy,
                        required_spacing_m,
                    )
                ),
                "boundary_violations": int(
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


def save_anchor_results(
    X,
    dataframe,
    anchors,
    scenario,
    risk_model,
    independent_aep_gwh,
    independent_aep_risk,
    output_dir,
):
    """
    Save coordinates and comparison metrics for the three
    selected Pareto layouts.
    """
    run_energy_index = anchors[
        "energy_optimal"
    ]

    run_energy_aep = float(
        dataframe.loc[
            run_energy_index,
            "aep_gwh",
        ]
    )

    run_energy_risk = float(
        dataframe.loc[
            run_energy_index,
            "pairwise_risk",
        ]
    )

    summary_rows = []

    for name, index in anchors.items():
        xy = X[index].reshape(
            scenario.site.n_turbines,
            2,
        )

        diagnostics = risk_model.diagnostics(
            xy
        )

        layout_dataframe = pd.DataFrame(
            {
                "turbine_id": np.arange(
                    scenario.site.n_turbines
                ),
                "x_m": xy[:, 0],
                "y_m": xy[:, 1],
                "per_turbine_risk": (
                    diagnostics[
                        "per_turbine_risk"
                    ]
                ),
            }
        )

        layout_dataframe.to_csv(
            output_dir
            / f"layout_{name}.csv",
            index=False,
        )

        aep_gwh = float(
            dataframe.loc[
                index,
                "aep_gwh",
            ]
        )

        pairwise_risk = float(
            dataframe.loc[
                index,
                "pairwise_risk",
            ]
        )

        aep_loss_vs_run_percent = (
            100.0
            * (
                run_energy_aep
                - aep_gwh
            )
            / run_energy_aep
        )

        aep_loss_vs_independent_percent = (
            100.0
            * (
                independent_aep_gwh
                - aep_gwh
            )
            / independent_aep_gwh
        )

        risk_reduction_vs_run_percent = (
            100.0
            * (
                run_energy_risk
                - pairwise_risk
            )
            / run_energy_risk
        )

        risk_reduction_vs_independent_percent = (
            100.0
            * (
                independent_aep_risk
                - pairwise_risk
            )
            / independent_aep_risk
        )

        summary_rows.append(
            {
                "layout": name,
                "layout_id": index,
                "aep_gwh": aep_gwh,
                "pairwise_risk": pairwise_risk,
                "minimum_pair_distance_m": float(
                    dataframe.loc[
                        index,
                        "minimum_pair_distance_m",
                    ]
                ),
                "mean_pair_distance_m": float(
                    dataframe.loc[
                        index,
                        "mean_pair_distance_m",
                    ]
                ),
                "maximum_pair_weight": float(
                    dataframe.loc[
                        index,
                        "maximum_pair_weight",
                    ]
                ),
                "aep_loss_vs_run_energy_percent": (
                    aep_loss_vs_run_percent
                ),
                "aep_loss_vs_independent_aep_percent": (
                    aep_loss_vs_independent_percent
                ),
                "risk_reduction_vs_run_energy_percent": (
                    risk_reduction_vs_run_percent
                ),
                "risk_reduction_vs_independent_aep_percent": (
                    risk_reduction_vs_independent_percent
                ),
            }
        )

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    summary_dataframe.to_csv(
        output_dir
        / "anchor_summary.csv",
        index=False,
    )

    return summary_dataframe


def plot_pareto_front(
    dataframe,
    anchors,
    independent_aep_point,
    risk_only_point,
    scale_D,
    output_path,
):
    """
    Plot the raw AEP-risk front together with the two
    independently obtained single-objective references.
    """
    fig, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.scatter(
        dataframe["pairwise_risk"],
        dataframe["aep_gwh"],
        s=35,
        label="NSGA-II Pareto layouts",
    )

    marker_map = {
        "energy_optimal": "o",
        "knee": "s",
        "risk_optimal": "*",
    }

    for name, index in anchors.items():
        axis.scatter(
            dataframe.loc[
                index,
                "pairwise_risk",
            ],
            dataframe.loc[
                index,
                "aep_gwh",
            ],
            s=160,
            marker=marker_map[name],
            label=name.replace(
                "_",
                " ",
            ).title(),
        )

    axis.scatter(
        independent_aep_point[
            "pairwise_risk"
        ],
        independent_aep_point[
            "aep_gwh"
        ],
        s=170,
        marker="^",
        label="Independent AEP-only reference",
    )

    axis.scatter(
        risk_only_point[
            "pairwise_risk"
        ],
        risk_only_point[
            "aep_gwh"
        ],
        s=170,
        marker="D",
        label="Independent risk-only reference",
    )

    axis.set_xlabel(
        "Artificial pairwise Gaussian risk"
    )

    axis.set_ylabel(
        "Annual energy production (GWh)"
    )

    axis.set_title(
        "Jensen AEP–pairwise-risk Pareto front\n"
        f"Interaction scale = {scale_D:.1f}D"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

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
    scale_D,
    output_path,
):
    """
    Plot the energy endpoint, knee and risk endpoint.
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

        axis.scatter(
            xy[:, 0],
            xy[:, 1],
            s=65,
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

        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

        axis.set_title(
            name.replace(
                "_",
                " ",
            ).title()
        )

        axis.grid(
            True,
            alpha=0.3,
        )

    fig.suptitle(
        "Permutation-invariant pairwise layouts\n"
        f"Interaction scale = {scale_D:.1f}D"
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
    Save hypervolume convergence history.
    """
    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    generations = np.arange(
        1,
        len(hypervolume_values) + 1,
    )

    axis.plot(
        generations,
        hypervolume_values,
    )

    axis.set_xlabel("Generation")
    axis.set_ylabel("Hypervolume")

    axis.set_title(
        "Pairwise multi-objective convergence"
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
            "Run Jensen AEP plus permutation-invariant "
            "pairwise Gaussian risk using NSGA-II."
        )
    )

    parser.add_argument(
        "--scale_D",
        type=float,
        default=8.0,
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

    parser.add_argument(
        "--aep_only_layout",
        type=Path,
        default=(
            Path("results")
            / "reference_experiment"
            / "aep_only"
            / "jensen_seed42_pop160_gen250"
            / "aep_only_layout.csv"
        ),
    )

    parser.add_argument(
        "--risk_only_layout",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    scenario = Scenario()
    aep_model = AEPModel(scenario)

    risk_model = PairwiseGaussianRiskModel(
        scenario=scenario,
        interaction_scale_D=args.scale_D,
    )

    scale_label = str(
        args.scale_D
    ).replace(
        ".",
        "p",
    )

    if args.risk_only_layout is None:
        args.risk_only_layout = (
            Path("results")
            / "pairwise_risk_only"
            / (
                f"scale{scale_label}D"
                "_seed42_pop80_gen100"
            )
            / "best_layout.csv"
        )

    independent_aep_xy = load_layout_csv(
        args.aep_only_layout,
        scenario,
    )

    risk_only_xy = load_layout_csv(
        args.risk_only_layout,
        scenario,
    )

    independent_aep_point = {
        "aep_gwh": float(
            aep_model.evaluate(
                independent_aep_xy
            )
        ),
        "pairwise_risk": float(
            risk_model.collisions(
                independent_aep_xy
            )
        ),
    }

    risk_only_point = {
        "aep_gwh": float(
            aep_model.evaluate(
                risk_only_xy
            )
        ),
        "pairwise_risk": float(
            risk_model.collisions(
                risk_only_xy
            )
        ),
    }

    ideal_aep = float(
        aep_model.ideal_aep(
            scenario.site.n_turbines
        )
    )

    problem = ValidationLayoutProblem(
        scenario=scenario,
        aep_model=aep_model,
        risk_model=risk_model,
        aep_min=0.5 * ideal_aep,
        aep_max=ideal_aep,
        risk_min=0.0,
        risk_max=risk_model.risk_scale(),
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
    print("Running AEP-pairwise-risk optimisation")
    print("--------------------------------------")
    print(
        f"Interaction scale: "
        f"{args.scale_D:.2f}D"
    )
    print(
        f"Sigma: "
        f"{risk_model.sigma_m:.2f} m"
    )
    print(f"Seed: {args.seed}")
    print(f"Population: {args.pop}")
    print(f"Generations: {args.gen}")
    print("")
    print("Independent AEP-only reference:")
    print(
        f"  AEP: "
        f"{independent_aep_point['aep_gwh']:.6f} GWh"
    )
    print(
        f"  Pairwise risk: "
        f"{independent_aep_point['pairwise_risk']:.6f}"
    )
    print("")
    print("Independent risk-only reference:")
    print(
        f"  AEP: "
        f"{risk_only_point['aep_gwh']:.6f} GWh"
    )
    print(
        f"  Pairwise risk: "
        f"{risk_only_point['pairwise_risk']:.6f}"
    )
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
            "No feasible Pareto layouts were found."
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

    output_dir = (
        Path("results")
        / "pairwise_multiobjective"
        / (
            f"scale{scale_label}D"
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
        output_dir
        / "pareto_points.csv",
        index=False,
    )

    anchor_summary = save_anchor_results(
        X=X,
        dataframe=dataframe,
        anchors=anchors,
        scenario=scenario,
        risk_model=risk_model,
        independent_aep_gwh=(
            independent_aep_point[
                "aep_gwh"
            ]
        ),
        independent_aep_risk=(
            independent_aep_point[
                "pairwise_risk"
            ]
        ),
        output_dir=output_dir,
    )

    hypervolume_dataframe = pd.DataFrame(
        {
            "generation": np.arange(
                1,
                len(callback.data["hv"]) + 1,
            ),
            "hypervolume": (
                callback.data["hv"]
            ),
        }
    )

    hypervolume_dataframe.to_csv(
        output_dir
        / "hypervolume.csv",
        index=False,
    )

    plot_pareto_front(
        dataframe=dataframe,
        anchors=anchors,
        independent_aep_point=(
            independent_aep_point
        ),
        risk_only_point=(
            risk_only_point
        ),
        scale_D=args.scale_D,
        output_path=(
            output_dir
            / "pareto_front.png"
        ),
    )

    plot_anchor_layouts(
        X=X,
        anchors=anchors,
        scenario=scenario,
        scale_D=args.scale_D,
        output_path=(
            output_dir
            / "anchor_layouts.png"
        ),
    )

    plot_hypervolume(
        hypervolume_values=(
            callback.data["hv"]
        ),
        output_path=(
            output_dir
            / "hypervolume.png"
        ),
    )

    energy_anchor = anchor_summary[
        anchor_summary["layout"]
        == "energy_optimal"
    ].iloc[0]

    risk_anchor = anchor_summary[
        anchor_summary["layout"]
        == "risk_optimal"
    ].iloc[0]

    summary = {
        "model": (
            "permutation_invariant_"
            "pairwise_gaussian"
        ),
        "interaction_scale_D": (
            args.scale_D
        ),
        "sigma_m": risk_model.sigma_m,
        "seed": args.seed,
        "population": args.pop,
        "generations": args.gen,
        "number_of_pareto_layouts": int(
            len(dataframe)
        ),
        "minimum_pareto_spacing_m": float(
            dataframe[
                "minimum_spacing_m"
            ].min()
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
        "independent_aep_reference": (
            independent_aep_point
        ),
        "independent_risk_reference": (
            risk_only_point
        ),
        "aep_endpoint_gap_gwh": float(
            independent_aep_point[
                "aep_gwh"
            ]
            - energy_anchor["aep_gwh"]
        ),
        "risk_endpoint_gap": float(
            risk_anchor["pairwise_risk"]
            - risk_only_point[
                "pairwise_risk"
            ]
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
    print("Pairwise multi-objective run complete.")
    print(
        f"Pareto layouts: "
        f"{len(dataframe)}"
    )
    print(
        f"Minimum Pareto spacing: "
        f"{dataframe['minimum_spacing_m'].min():.2f} m"
    )
    print(
        f"Spacing violations: "
        f"{dataframe['spacing_violations'].sum()}"
    )
    print(
        f"Boundary violations: "
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
        "Gap from independent AEP endpoint: "
        f"{summary['aep_endpoint_gap_gwh']:.6f} GWh"
    )
    print(
        "Gap from independent risk endpoint: "
        f"{summary['risk_endpoint_gap']:.6f}"
    )
    print("")
    print(
        f"Saved outputs to: "
        f"{output_dir}"
    )
    print("")


if __name__ == "__main__":
    main()