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
from coupled_risk import SurroundednessRiskModel
from risk_only_problem import RiskOnlyLayoutProblem

from run_validation import (
    GridSampling,
    min_spacing,
    count_spacing_violations,
    count_boundary_violations,
)


class RiskConvergenceCallback(Callback):
    """
    Record the best feasible artificial risk found
    after each generation.
    """

    def __init__(self, problem):
        super().__init__()

        self.problem = problem

        self.data["generation"] = []
        self.data["best_risk"] = []
        self.data["best_surroundedness"] = []

    def notify(self, algorithm):
        generation = int(algorithm.n_gen)

        self.data["generation"].append(
            generation
        )

        if algorithm.opt is None:
            self.data["best_risk"].append(
                np.nan
            )
            self.data[
                "best_surroundedness"
            ].append(np.nan)
            return

        X = algorithm.opt.get("X")

        if X is None:
            self.data["best_risk"].append(
                np.nan
            )
            self.data[
                "best_surroundedness"
            ].append(np.nan)
            return

        candidates = np.atleast_2d(
            np.asarray(X, dtype=float)
        )

        risks = []
        surroundedness_values = []

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

            surroundedness_values.append(
                diagnostics["surroundedness"]
            )

        best_index = int(
            np.argmin(risks)
        )

        self.data["best_risk"].append(
            float(risks[best_index])
        )

        self.data[
            "best_surroundedness"
        ].append(
            float(
                surroundedness_values[
                    best_index
                ]
            )
        )


def save_layout_figure(
    xy,
    scenario,
    target_index,
    mode,
    output_path,
):
    """
    Save the final risk-optimal layout.

    The target turbine uses a star marker.
    """
    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    other_indices = [
        index
        for index in range(len(xy))
        if index != target_index
    ]

    ax.scatter(
        xy[other_indices, 0],
        xy[other_indices, 1],
        s=70,
        marker="o",
        label="Other turbines",
    )

    ax.scatter(
        xy[target_index, 0],
        xy[target_index, 1],
        s=180,
        marker="*",
        label="Target turbine",
    )

    for index, (x, y) in enumerate(xy):
        ax.annotate(
            str(index),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlim(
        0.0,
        scenario.site.width_m,
    )

    ax.set_ylim(
        0.0,
        scenario.site.height_m,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")

    ax.set_title(
        f"Surroundedness risk-only optimum\n"
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


def save_convergence_figure(
    callback,
    output_path,
):
    """
    Save the best-risk convergence history.
    """
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        callback.data["generation"],
        callback.data["best_risk"],
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best artificial risk")
    ax.set_title(
        "Surroundedness risk-only convergence"
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
            "Run the designated-turbine "
            "surroundedness risk-only validation."
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
        default=40,
    )

    parser.add_argument(
        "--gen",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    scenario = Scenario()

    risk_model = SurroundednessRiskModel(
        scenario=scenario,
        target_index=args.target,
        interaction_scale_D=args.scale_D,
        mode=args.mode,
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

    callback = RiskConvergenceCallback(
        problem
    )

    print("")
    print("Running surroundedness risk-only validation")
    print("-------------------------------------------")
    print(f"Mode: {args.mode}")
    print(f"Target turbine index: {args.target}")
    print(
        f"Interaction scale: "
        f"{args.scale_D:.2f}D"
    )
    print(
        f"Interaction sigma: "
        f"{risk_model.sigma_m:.2f} m"
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
            "No feasible risk-only solution was found."
        )

    candidates = np.atleast_2d(
        np.asarray(result.X, dtype=float)
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
        / "coupled_validation"
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

    layout_df = pd.DataFrame(
        {
            "turbine_id": np.arange(
                scenario.site.n_turbines
            ),
            "x_m": best_xy[:, 0],
            "y_m": best_xy[:, 1],
            "is_target": (
                np.arange(
                    scenario.site.n_turbines
                )
                == args.target
            ),
        }
    )

    layout_df.to_csv(
        output_dir / "best_layout.csv",
        index=False,
    )

    convergence_df = pd.DataFrame(
        {
            "generation": (
                callback.data["generation"]
            ),
            "best_risk": (
                callback.data["best_risk"]
            ),
            "best_surroundedness": (
                callback.data[
                    "best_surroundedness"
                ]
            ),
        }
    )

    convergence_df.to_csv(
        output_dir / "convergence.csv",
        index=False,
    )

    summary = {
        "model": "surroundedness",
        "mode": args.mode,
        "target_index": args.target,
        "interaction_scale_D": (
            args.scale_D
        ),
        "interaction_sigma_m": (
            risk_model.sigma_m
        ),
        "seed": args.seed,
        "population": args.pop,
        "generations": args.gen,
        "best_risk": float(
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
        "distance_to_centroid_m": float(
            diagnostics[
                "distance_to_centroid_m"
            ]
        ),
        "radial_rank": int(
            diagnostics["radial_rank"]
        ),
        "neighbour_count_within_5D": int(
            diagnostics[
                "neighbour_count_within_5D"
            ]
        ),
        "minimum_spacing_required_m": (
            required_spacing_m
        ),
        "minimum_spacing_observed_m": (
            observed_spacing_m
        ),
        "spacing_violations": (
            spacing_violations
        ),
        "boundary_violations": (
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
        target_index=args.target,
        mode=args.mode,
        output_path=(
            output_dir
            / "best_layout.png"
        ),
    )

    save_convergence_figure(
        callback=callback,
        output_path=(
            output_dir
            / "convergence.png"
        ),
    )

    print("")
    print("Risk-only validation complete.")
    print(
        f"Best risk: "
        f"{diagnostics['total_risk']:.6f}"
    )
    print(
        f"Surroundedness: "
        f"{diagnostics['surroundedness']:.6f}"
    )
    print(
        f"Target distance to centroid: "
        f"{diagnostics['distance_to_centroid_m']:.2f} m"
    )
    print(
        f"Target radial rank: "
        f"{diagnostics['radial_rank']} "
        f"of {scenario.site.n_turbines - 1}"
    )
    print(
        f"Neighbours within 5D: "
        f"{diagnostics['neighbour_count_within_5D']}"
    )
    print(
        f"Minimum spacing: "
        f"{observed_spacing_m:.2f} m"
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