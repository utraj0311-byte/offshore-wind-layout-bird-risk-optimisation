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
from pairwise_risk import PairwiseGaussianRiskModel
from risk_only_problem import RiskOnlyLayoutProblem

from run_validation import (
    GridSampling,
    count_boundary_violations,
    count_spacing_violations,
    min_spacing,
)


class PairwiseRiskCallback(Callback):
    """
    Track convergence of the best feasible pairwise-risk
    layout after each generation.
    """

    def __init__(self, problem):
        super().__init__()

        self.problem = problem

        self.data["generation"] = []
        self.data["best_risk"] = []
        self.data["minimum_pair_distance_m"] = []
        self.data["mean_pair_distance_m"] = []

    def notify(self, algorithm):
        self.data["generation"].append(
            int(algorithm.n_gen)
        )

        if algorithm.opt is None:
            self.data["best_risk"].append(np.nan)
            self.data[
                "minimum_pair_distance_m"
            ].append(np.nan)
            self.data[
                "mean_pair_distance_m"
            ].append(np.nan)
            return

        candidate_matrix = algorithm.opt.get("X")

        if candidate_matrix is None:
            self.data["best_risk"].append(np.nan)
            self.data[
                "minimum_pair_distance_m"
            ].append(np.nan)
            self.data[
                "mean_pair_distance_m"
            ].append(np.nan)
            return

        candidates = np.atleast_2d(
            np.asarray(
                candidate_matrix,
                dtype=float,
            )
        )

        risks = []
        diagnostics_list = []

        for candidate in candidates:
            xy = candidate.reshape(
                self.problem.n_turbines,
                2,
            )

            diagnostics = (
                self.problem.risk_model
                .diagnostics(xy)
            )

            risks.append(
                diagnostics["total_risk"]
            )

            diagnostics_list.append(
                diagnostics
            )

        best_index = int(
            np.argmin(risks)
        )

        best_diagnostics = (
            diagnostics_list[best_index]
        )

        self.data["best_risk"].append(
            float(
                best_diagnostics[
                    "total_risk"
                ]
            )
        )

        self.data[
            "minimum_pair_distance_m"
        ].append(
            float(
                best_diagnostics[
                    "minimum_pair_distance_m"
                ]
            )
        )

        self.data[
            "mean_pair_distance_m"
        ].append(
            float(
                best_diagnostics[
                    "mean_pair_distance_m"
                ]
            )
        )


def save_layout_figure(
    xy,
    scenario,
    scale_D,
    risk,
    output_path,
):
    """
    Save the final pairwise risk-optimal layout.
    """
    fig, axis = plt.subplots(
        figsize=(7, 7)
    )

    axis.scatter(
        xy[:, 0],
        xy[:, 1],
        s=90,
    )

    for turbine_index, (x, y) in enumerate(xy):
        axis.annotate(
            str(turbine_index),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
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

    axis.set_xlabel("x coordinate (m)")
    axis.set_ylabel("y coordinate (m)")

    axis.set_title(
        "Pairwise Gaussian risk-only optimum\n"
        f"Interaction scale = {scale_D:.1f}D, "
        f"risk = {risk:.6f}"
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


def save_convergence_figure(
    convergence_dataframe,
    output_path,
):
    """
    Save the best-risk convergence history.
    """
    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        convergence_dataframe["generation"],
        convergence_dataframe["best_risk"],
    )

    axis.set_xlabel("Generation")
    axis.set_ylabel("Best pairwise risk")

    axis.set_title(
        "Pairwise risk-only convergence"
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


def save_pair_distance_histogram(
    pair_distances,
    scale_D,
    output_path,
):
    """
    Save the distribution of all 66 pair distances.
    """
    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.hist(
        pair_distances,
        bins=12,
    )

    axis.set_xlabel(
        "Pair distance (m)"
    )

    axis.set_ylabel(
        "Number of turbine pairs"
    )

    axis.set_title(
        "Pair-distance distribution\n"
        f"Interaction scale = {scale_D:.1f}D"
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
            "Run the 12-turbine permutation-invariant "
            "pairwise Gaussian risk-only optimisation."
        )
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

    risk_model = PairwiseGaussianRiskModel(
        scenario=scenario,
        interaction_scale_D=args.scale_D,
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

    callback = PairwiseRiskCallback(
        problem=problem
    )

    print("")
    print("Running 12-turbine pairwise risk-only optimisation")
    print("--------------------------------------------------")
    print(
        f"Interaction scale: "
        f"{args.scale_D:.2f}D"
    )
    print(
        f"Sigma: "
        f"{risk_model.sigma_m:.2f} m"
    )
    print(
        f"Number of turbine pairs: "
        f"{scenario.site.n_turbines * (scenario.site.n_turbines - 1) // 2}"
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
            "No feasible pairwise risk-only solution "
            "was found."
        )

    candidates = np.atleast_2d(
        np.asarray(
            result.X,
            dtype=float,
        )
    )

    candidate_risks = []

    for candidate in candidates:
        xy_candidate = candidate.reshape(
            scenario.site.n_turbines,
            2,
        )

        candidate_risks.append(
            risk_model.collisions(
                xy_candidate
            )
        )

    best_index = int(
        np.argmin(candidate_risks)
    )

    best_xy = candidates[
        best_index
    ].reshape(
        scenario.site.n_turbines,
        2,
    )

    diagnostics = risk_model.diagnostics(
        best_xy
    )

    required_spacing_m = (
        scenario.site.min_spacing_D
        * scenario.turbine.rotor_diameter_m
    )

    observed_spacing_m = min_spacing(
        best_xy
    )

    spacing_violations = (
        count_spacing_violations(
            best_xy,
            required_spacing_m,
        )
    )

    boundary_violations = (
        count_boundary_violations(
            best_xy,
            scenario.site.width_m,
            scenario.site.height_m,
        )
    )

    scale_label = str(
        args.scale_D
    ).replace(".", "p")

    output_dir = (
        Path("results")
        / "pairwise_risk_only"
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

    layout_dataframe = pd.DataFrame(
        {
            "turbine_id": np.arange(
                scenario.site.n_turbines
            ),
            "x_m": best_xy[:, 0],
            "y_m": best_xy[:, 1],
            "per_turbine_risk": (
                diagnostics[
                    "per_turbine_risk"
                ]
            ),
        }
    )

    layout_dataframe.to_csv(
        output_dir
        / "best_layout.csv",
        index=False,
    )

    pair_indices = diagnostics[
        "pair_indices"
    ]

    pair_dataframe = pd.DataFrame(
        {
            "turbine_i": (
                pair_indices[:, 0]
            ),
            "turbine_j": (
                pair_indices[:, 1]
            ),
            "distance_m": (
                diagnostics[
                    "pair_distances_m"
                ]
            ),
            "gaussian_weight": (
                diagnostics[
                    "pair_weights"
                ]
            ),
        }
    )

    pair_dataframe.to_csv(
        output_dir
        / "pair_data.csv",
        index=False,
    )

    convergence_dataframe = pd.DataFrame(
        {
            "generation": (
                callback.data["generation"]
            ),
            "best_risk": (
                callback.data["best_risk"]
            ),
            "minimum_pair_distance_m": (
                callback.data[
                    "minimum_pair_distance_m"
                ]
            ),
            "mean_pair_distance_m": (
                callback.data[
                    "mean_pair_distance_m"
                ]
            ),
        }
    )

    convergence_dataframe.to_csv(
        output_dir
        / "convergence.csv",
        index=False,
    )

    pair_distances = np.asarray(
        diagnostics[
            "pair_distances_m"
        ],
        dtype=float,
    )

    pair_weights = np.asarray(
        diagnostics[
            "pair_weights"
        ],
        dtype=float,
    )

    distance_quantiles = np.quantile(
        pair_distances,
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ],
    )

    summary = {
        "model": (
            "permutation_invariant_"
            "pairwise_gaussian"
        ),
        "interaction_scale_D": (
            args.scale_D
        ),
        "sigma_m": (
            risk_model.sigma_m
        ),
        "seed": args.seed,
        "population": args.pop,
        "generations": args.gen,
        "number_of_turbines": int(
            scenario.site.n_turbines
        ),
        "number_of_pairs": int(
            diagnostics[
                "number_of_pairs"
            ]
        ),
        "best_risk": float(
            diagnostics[
                "total_risk"
            ]
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
        "pair_distance_quantiles_m": {
            "q10": float(
                distance_quantiles[0]
            ),
            "q25": float(
                distance_quantiles[1]
            ),
            "q50": float(
                distance_quantiles[2]
            ),
            "q75": float(
                distance_quantiles[3]
            ),
            "q90": float(
                distance_quantiles[4]
            ),
        },
        "sum_pair_weights": float(
            np.sum(pair_weights)
        ),
        "minimum_spacing_required_m": (
            required_spacing_m
        ),
        "minimum_spacing_observed_m": (
            observed_spacing_m
        ),
        "spacing_violations": int(
            spacing_violations
        ),
        "boundary_violations": int(
            boundary_violations
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

    save_layout_figure(
        xy=best_xy,
        scenario=scenario,
        scale_D=args.scale_D,
        risk=diagnostics[
            "total_risk"
        ],
        output_path=(
            output_dir
            / "best_layout.png"
        ),
    )

    save_convergence_figure(
        convergence_dataframe=(
            convergence_dataframe
        ),
        output_path=(
            output_dir
            / "convergence.png"
        ),
    )

    save_pair_distance_histogram(
        pair_distances=pair_distances,
        scale_D=args.scale_D,
        output_path=(
            output_dir
            / "pair_distance_histogram.png"
        ),
    )

    print("")
    print("Pairwise risk-only optimisation complete.")
    print(
        f"Best total risk: "
        f"{diagnostics['total_risk']:.9f}"
    )
    print(
        f"Minimum pair distance: "
        f"{diagnostics['minimum_pair_distance_m']:.2f} m"
    )
    print(
        f"Mean pair distance: "
        f"{diagnostics['mean_pair_distance_m']:.2f} m"
    )
    print(
        f"Maximum pair distance: "
        f"{diagnostics['maximum_pair_distance_m']:.2f} m"
    )
    print(
        f"Maximum pair weight: "
        f"{diagnostics['maximum_pair_weight']:.9f}"
    )
    print(
        f"Minimum spacing required: "
        f"{required_spacing_m:.2f} m"
    )
    print(
        f"Spacing violations: "
        f"{spacing_violations}"
    )
    print(
        f"Boundary violations: "
        f"{boundary_violations}"
    )
    print(
        f"Saved outputs to: "
        f"{output_dir}"
    )
    print("")


if __name__ == "__main__":
    main()