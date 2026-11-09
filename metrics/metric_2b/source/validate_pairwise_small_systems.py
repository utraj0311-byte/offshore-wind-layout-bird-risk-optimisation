import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from config import Scenario
from pairwise_risk import PairwiseGaussianRiskModel


class SmallPairwiseProblem(Problem):
    """
    Single-objective pairwise-risk optimisation for a small
    number of turbines.

    Objective:
        minimise total pairwise Gaussian interaction.

    Constraints:
        minimum pairwise turbine spacing.

    The rectangular site boundary is enforced through the
    coordinate bounds.
    """

    def __init__(
        self,
        n_turbines,
        site_width_m,
        site_height_m,
        minimum_spacing_m,
        risk_model,
    ):
        self.n_turbines = int(n_turbines)
        self.site_width_m = float(site_width_m)
        self.site_height_m = float(site_height_m)
        self.minimum_spacing_m = float(
            minimum_spacing_m
        )
        self.risk_model = risk_model

        if self.n_turbines < 2:
            raise ValueError(
                "At least two turbines are required."
            )

        number_of_constraints = (
            self.n_turbines
            * (self.n_turbines - 1)
            // 2
        )

        lower_bounds = np.tile(
            [0.0, 0.0],
            self.n_turbines,
        )

        upper_bounds = np.tile(
            [
                self.site_width_m,
                self.site_height_m,
            ],
            self.n_turbines,
        )

        super().__init__(
            n_var=2 * self.n_turbines,
            n_obj=1,
            n_ieq_constr=number_of_constraints,
            xl=lower_bounds,
            xu=upper_bounds,
        )

    def _evaluate(
        self,
        X,
        out,
        *args,
        **kwargs,
    ):
        X = np.atleast_2d(
            np.asarray(X, dtype=float)
        )

        objectives = np.zeros(
            (len(X), 1),
            dtype=float,
        )

        constraints = np.zeros(
            (
                len(X),
                self.n_ieq_constr,
            ),
            dtype=float,
        )

        for population_index, candidate in enumerate(X):
            xy = candidate.reshape(
                self.n_turbines,
                2,
            )

            objectives[
                population_index,
                0,
            ] = self.risk_model.collisions(xy)

            constraint_index = 0

            for i in range(self.n_turbines):
                for j in range(
                    i + 1,
                    self.n_turbines,
                ):
                    distance = np.linalg.norm(
                        xy[i] - xy[j]
                    )

                    constraints[
                        population_index,
                        constraint_index,
                    ] = (
                        self.minimum_spacing_m
                        - distance
                    )

                    constraint_index += 1

        out["F"] = objectives
        out["G"] = constraints


class RiskConvergenceCallback(Callback):
    """
    Record the best objective value after each generation.
    """

    def __init__(self):
        super().__init__()

        self.data["generation"] = []
        self.data["best_risk"] = []

    def notify(self, algorithm):
        self.data["generation"].append(
            int(algorithm.n_gen)
        )

        if algorithm.opt is None:
            self.data["best_risk"].append(
                np.nan
            )
            return

        objective_values = algorithm.opt.get(
            "F"
        )

        if objective_values is None:
            self.data["best_risk"].append(
                np.nan
            )
            return

        best_risk = float(
            np.min(
                np.asarray(
                    objective_values,
                    dtype=float,
                )
            )
        )

        self.data["best_risk"].append(
            best_risk
        )


def pairwise_distances(xy):
    """
    Return all unique pairwise distances.
    """
    xy = np.asarray(
        xy,
        dtype=float,
    )

    distances = []

    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            distances.append(
                np.linalg.norm(
                    xy[i] - xy[j]
                )
            )

    return np.asarray(
        distances,
        dtype=float,
    )


def optimise_system(
    n_turbines,
    scenario,
    risk_model,
    population,
    generations,
    seed,
):
    """
    Run the small-system GA and return its best solution.
    """
    minimum_spacing_m = (
        scenario.site.min_spacing_D
        * scenario.turbine.rotor_diameter_m
    )

    problem = SmallPairwiseProblem(
        n_turbines=n_turbines,
        site_width_m=scenario.site.width_m,
        site_height_m=scenario.site.height_m,
        minimum_spacing_m=minimum_spacing_m,
        risk_model=risk_model,
    )

    algorithm = GA(
        pop_size=population,
        eliminate_duplicates=True,
    )

    callback = RiskConvergenceCallback()

    result = minimize(
        problem,
        algorithm,
        termination=(
            "n_gen",
            generations,
        ),
        seed=seed,
        callback=callback,
        verbose=True,
    )

    if result.X is None:
        raise RuntimeError(
            f"No feasible solution found for "
            f"{n_turbines} turbines."
        )

    candidates = np.atleast_2d(
        np.asarray(
            result.X,
            dtype=float,
        )
    )

    risks = []

    for candidate in candidates:
        xy = candidate.reshape(
            n_turbines,
            2,
        )

        risks.append(
            risk_model.collisions(xy)
        )

    best_index = int(
        np.argmin(risks)
    )

    best_xy = candidates[
        best_index
    ].reshape(
        n_turbines,
        2,
    )

    return (
        best_xy,
        float(risks[best_index]),
        callback,
    )


def save_layout_csv(
    xy,
    output_path,
):
    """
    Save turbine coordinates to CSV.
    """
    dataframe = pd.DataFrame(
        {
            "turbine_id": np.arange(
                len(xy)
            ),
            "x_m": xy[:, 0],
            "y_m": xy[:, 1],
        }
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )


def save_layout_figure(
    xy,
    title,
    site_width_m,
    site_height_m,
    output_path,
):
    """
    Save a plan-view turbine-layout figure.
    """
    fig, axis = plt.subplots(
        figsize=(7, 7)
    )

    axis.scatter(
        xy[:, 0],
        xy[:, 1],
        s=100,
    )

    for index, (x, y) in enumerate(xy):
        axis.annotate(
            str(index),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
        )

    axis.set_xlim(
        0.0,
        site_width_m,
    )

    axis.set_ylim(
        0.0,
        site_height_m,
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
    axis.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def save_convergence(
    callback,
    csv_path,
    figure_path,
    title,
):
    """
    Save convergence history as CSV and PNG.
    """
    dataframe = pd.DataFrame(
        {
            "generation": (
                callback.data["generation"]
            ),
            "best_risk": (
                callback.data["best_risk"]
            ),
        }
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        dataframe["generation"],
        dataframe["best_risk"],
    )

    axis.set_xlabel(
        "Generation"
    )

    axis.set_ylabel(
        "Best pairwise risk"
    )

    axis.set_title(title)
    axis.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=200,
    )

    plt.close(fig)


def two_turbine_reference(
    site_width_m,
    site_height_m,
    risk_model,
):
    """
    Opposite site corners give the maximum possible
    separation for two turbines.
    """
    xy = np.array(
        [
            [0.0, 0.0],
            [
                site_width_m,
                site_height_m,
            ],
        ],
        dtype=float,
    )

    return (
        xy,
        risk_model.collisions(xy),
    )


def three_turbine_reference(
    site_width_m,
    site_height_m,
    risk_model,
):
    """
    Symmetric three-turbine reference for the square site.

    Two turbines occupy adjacent corners of one site edge.
    The third occupies the midpoint of the opposite edge.

    For the selected Gaussian interaction scale, this
    balances the three pair distances better than placing
    turbines at three corners.
    """
    xy = np.array(
        [
            [0.0, site_height_m],
            [
                site_width_m,
                site_height_m,
            ],
            [
                0.5 * site_width_m,
                0.0,
            ],
        ],
        dtype=float,
    )

    return (
        xy,
        risk_model.collisions(xy),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the pairwise Gaussian objective "
            "using two- and three-turbine optimisation."
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
        "--pop2",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--gen2",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--pop3",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--gen3",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    scenario = Scenario()

    risk_model = PairwiseGaussianRiskModel(
        scenario=scenario,
        interaction_scale_D=args.scale_D,
    )

    scale_label = str(
        args.scale_D
    ).replace(".", "p")

    output_dir = (
        Path("results")
        / "pairwise_small_system"
        / (
            f"scale{scale_label}D"
            f"_seed{args.seed}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("")
    print("Small-system pairwise optimisation")
    print("----------------------------------")
    print(
        f"Interaction scale: "
        f"{args.scale_D:.2f}D"
    )
    print(
        f"Sigma: "
        f"{risk_model.sigma_m:.2f} m"
    )
    print("")

    # =================================================
    # Two-turbine optimisation
    # =================================================

    print("Optimising two turbines...")

    (
        two_xy,
        two_risk,
        two_callback,
    ) = optimise_system(
        n_turbines=2,
        scenario=scenario,
        risk_model=risk_model,
        population=args.pop2,
        generations=args.gen2,
        seed=args.seed,
    )

    (
        two_reference_xy,
        two_reference_risk,
    ) = two_turbine_reference(
        site_width_m=(
            scenario.site.width_m
        ),
        site_height_m=(
            scenario.site.height_m
        ),
        risk_model=risk_model,
    )

    two_distances = pairwise_distances(
        two_xy
    )

    two_reference_distance = float(
        np.hypot(
            scenario.site.width_m,
            scenario.site.height_m,
        )
    )

    two_distance_ratio = float(
        two_distances[0]
        / two_reference_distance
    )

    # Require at least 99% of the maximum possible
    # site-diagonal separation.
    assert two_distance_ratio >= 0.99, (
        "Two-turbine optimiser did not recover "
        "near-opposite-corner separation."
    )

    # =================================================
    # Three-turbine optimisation
    # =================================================

    print("")
    print("Optimising three turbines...")

    (
        three_xy,
        three_risk,
        three_callback,
    ) = optimise_system(
        n_turbines=3,
        scenario=scenario,
        risk_model=risk_model,
        population=args.pop3,
        generations=args.gen3,
        seed=args.seed,
    )

    (
        three_reference_xy,
        three_reference_risk,
    ) = three_turbine_reference(
        site_width_m=(
            scenario.site.width_m
        ),
        site_height_m=(
            scenario.site.height_m
        ),
        risk_model=risk_model,
    )

    three_distances = np.sort(
        pairwise_distances(
            three_xy
        )
    )

    three_reference_distances = np.sort(
        pairwise_distances(
            three_reference_xy
        )
    )

    # The optimiser should recover a risk value at least
    # as good as the symmetric edge-midpoint reference,
    # within a small numerical tolerance.
    assert three_risk <= (
        three_reference_risk
        * 1.001
    ), (
        "Three-turbine optimiser did not recover the "
        "symmetric low-risk reference objective."
    )

    # Compare unordered pair-distance structures so that
    # rotations, reflections and turbine relabelling do
    # not affect the validation.
    assert np.allclose(
        three_distances,
        three_reference_distances,
        rtol=0.02,
        atol=20.0,
    ), (
        "Three-turbine optimiser found an unexpected "
        "pair-distance structure."
    )

    # =================================================
    # Save layouts and convergence results
    # =================================================

    save_layout_csv(
        two_xy,
        output_dir
        / "two_turbine_layout.csv",
    )

    save_layout_csv(
        two_reference_xy,
        output_dir
        / "two_turbine_reference_layout.csv",
    )

    save_layout_csv(
        three_xy,
        output_dir
        / "three_turbine_layout.csv",
    )

    save_layout_csv(
        three_reference_xy,
        output_dir
        / "three_turbine_reference_layout.csv",
    )

    save_layout_figure(
        xy=two_xy,
        title=(
            "Pairwise-risk optimum: "
            "two turbines"
        ),
        site_width_m=(
            scenario.site.width_m
        ),
        site_height_m=(
            scenario.site.height_m
        ),
        output_path=(
            output_dir
            / "two_turbine_layout.png"
        ),
    )

    save_layout_figure(
        xy=three_xy,
        title=(
            "Pairwise-risk optimum: "
            "three turbines"
        ),
        site_width_m=(
            scenario.site.width_m
        ),
        site_height_m=(
            scenario.site.height_m
        ),
        output_path=(
            output_dir
            / "three_turbine_layout.png"
        ),
    )

    save_convergence(
        callback=two_callback,
        csv_path=(
            output_dir
            / "two_turbine_convergence.csv"
        ),
        figure_path=(
            output_dir
            / "two_turbine_convergence.png"
        ),
        title=(
            "Two-turbine pairwise-risk convergence"
        ),
    )

    save_convergence(
        callback=three_callback,
        csv_path=(
            output_dir
            / "three_turbine_convergence.csv"
        ),
        figure_path=(
            output_dir
            / "three_turbine_convergence.png"
        ),
        title=(
            "Three-turbine pairwise-risk convergence"
        ),
    )

    summary = {
        "interaction_scale_D": args.scale_D,
        "sigma_m": risk_model.sigma_m,
        "seed": args.seed,
        "two_turbine": {
            "population": args.pop2,
            "generations": args.gen2,
            "optimised_risk": two_risk,
            "reference_corner_risk": (
                two_reference_risk
            ),
            "optimised_distance_m": float(
                two_distances[0]
            ),
            "maximum_distance_m": float(
                two_reference_distance
            ),
            "distance_ratio": (
                two_distance_ratio
            ),
        },
        "three_turbine": {
            "population": args.pop3,
            "generations": args.gen3,
            "optimised_risk": three_risk,
            "symmetric_reference_risk": (
                three_reference_risk
            ),
            "optimised_pair_distances_m": (
                three_distances.tolist()
            ),
            "reference_pair_distances_m": (
                three_reference_distances.tolist()
            ),
            "risk_ratio_optimised_to_reference": float(
                three_risk
                / three_reference_risk
            ),
        },
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

    # =================================================
    # Terminal summary
    # =================================================

    print("")
    print("Small-system validation complete.")
    print("")

    print("Two turbines:")
    print(
        f"  Optimised risk: "
        f"{two_risk:.12f}"
    )
    print(
        f"  Opposite-corner reference risk: "
        f"{two_reference_risk:.12f}"
    )
    print(
        f"  Optimised separation: "
        f"{two_distances[0]:.2f} m"
    )
    print(
        f"  Maximum possible separation: "
        f"{two_reference_distance:.2f} m"
    )
    print(
        f"  Distance recovery: "
        f"{100.0 * two_distance_ratio:.4f}%"
    )

    print("")
    print("Three turbines:")
    print(
        f"  Optimised risk: "
        f"{three_risk:.12f}"
    )
    print(
        f"  Symmetric reference risk: "
        f"{three_reference_risk:.12f}"
    )
    print(
        "  Optimised pair distances: "
        f"{np.round(three_distances, 2)}"
    )
    print(
        "  Reference pair distances: "
        f"{np.round(three_reference_distances, 2)}"
    )
    print(
        "  Optimised/reference risk ratio: "
        f"{three_risk / three_reference_risk:.6f}"
    )

    print("")
    print(
        "All small-system pairwise "
        "optimisation checks passed."
    )
    print(
        f"Saved outputs to: "
        f"{output_dir}"
    )
    print("")


if __name__ == "__main__":
    main()