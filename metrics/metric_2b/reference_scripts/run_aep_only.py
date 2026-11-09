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
from wake import AEPModel
from aep_only_problem import AEPOnlyLayoutProblem

# Reuse the same feasible sampling and checks as the validated runner.
from run_validation import (
    GridSampling,
    min_spacing,
    count_spacing_violations,
    count_boundary_violations,
)


class AEPConvergenceCallback(Callback):
    """
    Record the best feasible raw AEP found after each generation.
    """

    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.data["generation"] = []
        self.data["best_aep_gwh"] = []
        self.data["best_objective"] = []

    def notify(self, algorithm):
        generation = int(algorithm.n_gen)

        if algorithm.opt is None:
            self.data["generation"].append(generation)
            self.data["best_aep_gwh"].append(np.nan)
            self.data["best_objective"].append(np.nan)
            return

        X = algorithm.opt.get("X")
        F = algorithm.opt.get("F")

        if X is None or len(np.atleast_2d(X)) == 0:
            self.data["generation"].append(generation)
            self.data["best_aep_gwh"].append(np.nan)
            self.data["best_objective"].append(np.nan)
            return

        X = np.atleast_2d(np.asarray(X, dtype=float))
        F = np.atleast_2d(np.asarray(F, dtype=float))

        raw_aep_values = []

        for candidate in X:
            xy = candidate.reshape(
                self.problem.n_turbines,
                2,
            )

            raw_aep_values.append(
                float(
                    self.problem.aep_model.evaluate(xy)
                )
            )

        best_index = int(np.argmax(raw_aep_values))

        self.data["generation"].append(generation)
        self.data["best_aep_gwh"].append(
            raw_aep_values[best_index]
        )
        self.data["best_objective"].append(
            float(F[best_index, 0])
        )


def save_layout_figure(xy, scenario, output_path):
    """
    Save the AEP-optimal layout as a simple site plot.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=80,
    )

    for index, (x, y) in enumerate(xy, start=1):
        ax.annotate(
            str(index),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlim(0.0, scenario.site.width_m)
    ax.set_ylim(0.0, scenario.site.height_m)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")
    ax.set_title("AEP-only optimum — Jensen reference model")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_convergence_figure(callback, output_path):
    """
    Plot the best raw AEP found per generation.
    """
    generations = callback.data["generation"]
    best_aep = callback.data["best_aep_gwh"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(generations, best_aep)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best AEP (GWh)")
    ax.set_title("AEP-only optimisation convergence")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_final_population(
    result,
    scenario,
    aep_model,
    output_path,
):
    """
    Save the objective and feasibility of every layout
    in the final population.
    """
    population = result.algorithm.pop

    X = np.asarray(population.get("X"), dtype=float)
    F = np.asarray(population.get("F"), dtype=float)
    CV = np.asarray(population.get("CV"), dtype=float).reshape(-1)

    rows = []

    min_spacing_m = (
        scenario.site.min_spacing_D
        * scenario.turbine.rotor_diameter_m
    )

    for index, candidate in enumerate(X):
        xy = candidate.reshape(
            scenario.site.n_turbines,
            2,
        )

        rows.append(
            {
                "candidate": index,
                "aep_gwh": float(
                    aep_model.evaluate(xy)
                ),
                "normalised_objective": float(
                    np.atleast_1d(F[index])[0]
                ),
                "constraint_violation": float(CV[index]),
                "minimum_spacing_m": min_spacing(xy),
                "spacing_violations": count_spacing_violations(
                    xy,
                    min_spacing_m,
                ),
                "boundary_violations": count_boundary_violations(
                    xy,
                    scenario.site.width_m,
                    scenario.site.height_m,
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the independent AEP-only reference "
            "layout optimisation."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--pop",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--gen",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    scenario = Scenario()
    aep_model = AEPModel(scenario)

    ideal_aep = float(
        aep_model.ideal_aep(
            scenario.site.n_turbines
        )
    )

    if not np.isfinite(ideal_aep):
        raise ValueError(
            "The wake-free AEP reference is not finite."
        )

    if ideal_aep <= 0.0:
        raise ValueError(
            "The wake-free AEP reference must be positive."
        )

    # Use the corrected wake-free AEP as the upper
    # normalisation bound.
    aep_max = ideal_aep
    aep_min = 0.5 * ideal_aep

    problem = AEPOnlyLayoutProblem(
        scenario=scenario,
        aep_model=aep_model,
        aep_min=aep_min,
        aep_max=aep_max,
    )

    # Keep the same AEP normalisation as the
    # existing validation framework.
    aep_max = ideal_aep
    aep_min = 0.5 * ideal_aep

    problem = AEPOnlyLayoutProblem(
        scenario=scenario,
        aep_model=aep_model,
        aep_min=aep_min,
        aep_max=aep_max,
    )

    # "constant" uses the neutral mixture of
    # random feasible and grid-like layouts.
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

    callback = AEPConvergenceCallback(problem)

    print("")
    print("Running AEP-only reference optimisation")
    print("---------------------------------------")
    print(f"Energy model: simplified Jensen")
    print(f"Turbines: {scenario.site.n_turbines}")
    print(
        "Minimum spacing: "
        f"{problem.min_spacing_m:.1f} m"
    )
    print(f"Seed: {args.seed}")
    print(f"Population size: {args.pop}")
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
            "No feasible AEP-only solution was found."
        )

    candidate_layouts = np.atleast_2d(
        np.asarray(result.X, dtype=float)
    )

    raw_aep_values = []

    for candidate in candidate_layouts:
        xy_candidate = candidate.reshape(
            scenario.site.n_turbines,
            2,
        )

        raw_aep_values.append(
            float(
                aep_model.evaluate(xy_candidate)
            )
        )

    best_index = int(np.argmax(raw_aep_values))
    best_aep = float(raw_aep_values[best_index])
    if best_aep > ideal_aep + 1e-6:
         raise RuntimeError(
        "The optimised AEP exceeds the wake-free AEP reference. "
        "Check the AEP model or ideal_aep implementation."
    )

    best_xy = candidate_layouts[best_index].reshape(
        scenario.site.n_turbines,
        2,
    )

    min_spacing_m = (
        scenario.site.min_spacing_D
        * scenario.turbine.rotor_diameter_m
    )

    spacing_violations = count_spacing_violations(
        best_xy,
        min_spacing_m,
    )

    boundary_violations = count_boundary_violations(
        best_xy,
        scenario.site.width_m,
        scenario.site.height_m,
    )

    output_dir = (
        Path("results")
        / "reference_experiment"
        / "aep_only"
        / (
            f"jensen_seed{args.seed}"
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
                1,
                scenario.site.n_turbines + 1,
            ),
            "x_m": best_xy[:, 0],
            "y_m": best_xy[:, 1],
        }
    )

    layout_df.to_csv(
        output_dir / "aep_only_layout.csv",
        index=False,
    )

    convergence_df = pd.DataFrame(
        {
            "generation": callback.data["generation"],
            "best_aep_gwh": callback.data["best_aep_gwh"],
            "best_objective": callback.data[
                "best_objective"
            ],
        }
    )

    convergence_df.to_csv(
        output_dir / "aep_convergence.csv",
        index=False,
    )

    save_final_population(
        result=result,
        scenario=scenario,
        aep_model=aep_model,
        output_path=(
            output_dir
            / "final_population_summary.csv"
        ),
    )

    save_layout_figure(
        best_xy,
        scenario,
        output_dir / "aep_only_layout.png",
    )

    save_convergence_figure(
        callback,
        output_dir / "aep_convergence.png",
    )

    summary = {
        "experiment": "AEP-only reference optimisation",
        "energy_model": "simplified_jensen",
        "seed": args.seed,
        "population_size": args.pop,
        "generations": args.gen,
        "number_of_turbines": (
            scenario.site.n_turbines
        ),
        "site_width_m": scenario.site.width_m,
        "site_height_m": scenario.site.height_m,
        "minimum_spacing_required_m": min_spacing_m,
        "best_aep_gwh": best_aep,
        "ideal_wake_free_aep_gwh": ideal_aep,
        "wake_loss_fraction": (
            (ideal_aep - best_aep)
            / ideal_aep
        ),
        "minimum_spacing_observed_m": min_spacing(
            best_xy
        ),
        "spacing_violations": spacing_violations,
        "boundary_violations": boundary_violations,
        "layout_file": "aep_only_layout.csv",
        "convergence_file": "aep_convergence.csv",
    }

    with open(
        output_dir / "aep_only_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("")
    print("AEP-only optimisation complete.")
    print(f"Best AEP: {best_aep:.6f} GWh")
    print(
        "Minimum observed spacing: "
        f"{min_spacing(best_xy):.2f} m"
    )
    print(
        f"Spacing violations: {spacing_violations}"
    )
    print(
        f"Boundary violations: {boundary_violations}"
    )
    print(f"Saved outputs to: {output_dir}")
    print("")


if __name__ == "__main__":
    main()