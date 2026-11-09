from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.sampling import Sampling
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
from run_directional_absorbing_risk_only import (
    DirectionalRiskCallback,
    create_reference_layout,
    save_bird_frame_plot,
    save_convergence_plot,
    save_layout_plot,
)


class ReferenceSeededSampling(Sampling):
    """
    Generate the normal initial population, then replace
    its first member with the known aligned two-file
    reference layout.

    All other population members remain unchanged.
    """

    def __init__(
        self,
        base_sampling,
        reference_xy,
    ):
        super().__init__()

        self.base_sampling = base_sampling

        self.reference_vector = np.asarray(
            reference_xy,
            dtype=float,
        ).reshape(-1)

    def _do(
        self,
        problem,
        n_samples,
        **kwargs,
    ):
        X = self.base_sampling._do(
            problem,
            n_samples,
            **kwargs,
        )

        X = np.asarray(
            X,
            dtype=float,
        ).copy()

        if X.ndim != 2:
            raise ValueError(
                "Base sampling did not return a "
                "two-dimensional population."
            )

        if X.shape[1] != len(
            self.reference_vector
        ):
            raise ValueError(
                "Reference layout has the wrong "
                "number of decision variables."
            )

        X[0] = self.reference_vector

        return X


def save_layout_csv(
    xy,
    risk_model,
    output_path,
):
    diagnostics = risk_model.diagnostics(
        xy
    )

    dataframe = pd.DataFrame(
        {
            "turbine_id": np.arange(
                len(xy)
            ),
            "x_m": xy[:, 0],
            "y_m": xy[:, 1],
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

    dataframe.to_csv(
        output_path,
        index=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a directional absorbing-risk optimisation "
            "seeded with the known aligned two-file layout."
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

    reference_xy = create_reference_layout(
        scenario=scenario,
        risk_model=risk_model,
        aligned=True,
    )

    required_spacing_m = (
        scenario.site.min_spacing_D
        * scenario.turbine.rotor_diameter_m
    )

    reference_spacing_violations = (
        count_spacing_violations(
            reference_xy,
            required_spacing_m,
        )
    )

    reference_boundary_violations = (
        count_boundary_violations(
            reference_xy,
            scenario.site.width_m,
            scenario.site.height_m,
        )
    )

    if reference_spacing_violations != 0:
        raise RuntimeError(
            "The aligned reference contains spacing "
            "violations and cannot be used as a seed."
        )

    if reference_boundary_violations != 0:
        raise RuntimeError(
            "The aligned reference contains boundary "
            "violations and cannot be used as a seed."
        )

    reference_risk = float(
        risk_model.collisions(
            reference_xy
        )
    )

    problem = RiskOnlyLayoutProblem(
        scenario=scenario,
        risk_model=risk_model,
    )

    base_sampling = GridSampling(
        scenario=scenario,
        seed=args.seed,
        model_name="constant",
    )

    seeded_sampling = (
        ReferenceSeededSampling(
            base_sampling=base_sampling,
            reference_xy=reference_xy,
        )
    )

    algorithm = GA(
        pop_size=args.pop,
        sampling=seeded_sampling,
        eliminate_duplicates=True,
    )

    callback = DirectionalRiskCallback(
        scenario=scenario,
        risk_model=risk_model,
    )

    print("")
    print(
        "Running reference-seeded directional optimisation"
    )
    print(
        "-------------------------------------------------"
    )
    print(
        "Only the population initialisation differs from "
        "the previous unseeded run."
    )
    print("")
    print(
        f"Reference risk: "
        f"{reference_risk:.12f}"
    )
    print(
        f"Reference minimum spacing: "
        f"{min_spacing(reference_xy):.6f} m"
    )
    print(
        f"Bird direction: "
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
        f"Population: {args.pop}"
    )
    print(
        f"Generations: {args.gen}"
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
            "No feasible solution was returned."
        )

    best_xy = np.asarray(
        result.X,
        dtype=float,
    ).reshape(
        scenario.site.n_turbines,
        2,
    )

    best_diagnostics = (
        risk_model.diagnostics(
            best_xy
        )
    )

    best_risk = float(
        best_diagnostics["total_risk"]
    )

    best_spacing_violations = (
        count_spacing_violations(
            best_xy,
            required_spacing_m,
        )
    )

    best_boundary_violations = (
        count_boundary_violations(
            best_xy,
            scenario.site.width_m,
            scenario.site.height_m,
        )
    )

    convergence_dataframe = (
        pd.DataFrame(
            callback.data
        )
    )

    if len(convergence_dataframe) == 0:
        raise RuntimeError(
            "No convergence records were produced."
        )

    generation_1_best_risk = float(
        convergence_dataframe[
            "best_risk"
        ].iloc[0]
    )

    improvement = (
        reference_risk
        - best_risk
    )

    improvement_percent = (
        100.0
        * improvement
        / reference_risk
    )

    tolerance = 1e-8

    if best_risk < (
        reference_risk
        - tolerance
    ):
        outcome = (
            "REFERENCE IMPROVED"
        )
    elif best_risk <= (
        reference_risk
        + tolerance
    ):
        outcome = (
            "REFERENCE RETAINED"
        )
    else:
        outcome = (
            "REFERENCE LOST OR DEGRADED"
        )

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
        / "directional_absorbing_seeded"
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

    save_layout_csv(
        xy=reference_xy,
        risk_model=risk_model,
        output_path=(
            output_dir
            / "reference_layout.csv"
        ),
    )

    save_layout_csv(
        xy=best_xy,
        risk_model=risk_model,
        output_path=(
            output_dir
            / "best_layout.csv"
        ),
    )

    convergence_dataframe.to_csv(
        output_dir
        / "convergence.csv",
        index=False,
    )

    save_layout_plot(
        xy=reference_xy,
        scenario=scenario,
        risk_model=risk_model,
        title=(
            "Initial aligned two-file reference\n"
            f"Risk = {reference_risk:.6f}"
        ),
        output_path=(
            output_dir
            / "reference_layout_site.png"
        ),
    )

    save_bird_frame_plot(
        xy=reference_xy,
        risk_model=risk_model,
        output_path=(
            output_dir
            / "reference_layout_bird_frame.png"
        ),
    )

    save_layout_plot(
        xy=best_xy,
        scenario=scenario,
        risk_model=risk_model,
        title=(
            "Best reference-seeded layout\n"
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

    save_convergence_plot(
        convergence=convergence_dataframe,
        output_path=(
            output_dir
            / "convergence.png"
        ),
    )

    summary = {
        "test_purpose": (
            "Determine whether GA retains or improves "
            "the known aligned two-file basin when it is "
            "included in the initial population."
        ),
        "model_status": (
            "NEWLY PROPOSED / DIAGNOSTIC ONLY"
        ),
        "seeded_reference_injected": True,
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
        "reference_risk": (
            reference_risk
        ),
        "generation_1_best_risk": (
            generation_1_best_risk
        ),
        "final_best_risk": (
            best_risk
        ),
        "absolute_improvement_from_reference": (
            improvement
        ),
        "improvement_from_reference_percent": (
            improvement_percent
        ),
        "outcome": outcome,
        "reference_minimum_spacing_m": float(
            min_spacing(reference_xy)
        ),
        "final_minimum_spacing_m": float(
            min_spacing(best_xy)
        ),
        "reference_spacing_violations": int(
            reference_spacing_violations
        ),
        "reference_boundary_violations": int(
            reference_boundary_violations
        ),
        "final_spacing_violations": int(
            best_spacing_violations
        ),
        "final_boundary_violations": int(
            best_boundary_violations
        ),
        "final_minimum_exposure": float(
            best_diagnostics[
                "minimum_exposure"
            ]
        ),
        "final_mean_exposure": float(
            best_diagnostics[
                "mean_exposure"
            ]
        ),
        "final_maximum_exposure": float(
            best_diagnostics[
                "maximum_exposure"
            ]
        ),
        "final_effective_pairs": int(
            best_diagnostics[
                "number_of_effective_pairs"
            ]
        ),
        "final_maximum_pair_shielding": float(
            best_diagnostics[
                "maximum_pair_shielding"
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
    print(
        "Reference-seeded run complete."
    )
    print(
        "------------------------------"
    )
    print(
        f"Reference risk: "
        f"{reference_risk:.12f}"
    )
    print(
        f"Generation-1 best risk: "
        f"{generation_1_best_risk:.12f}"
    )
    print(
        f"Final best risk: "
        f"{best_risk:.12f}"
    )
    print(
        f"Improvement from reference: "
        f"{improvement:.12f}"
    )
    print(
        f"Improvement percentage: "
        f"{improvement_percent:.6f}%"
    )
    print(
        f"Outcome: {outcome}"
    )
    print("")
    print(
        f"Final minimum spacing: "
        f"{min_spacing(best_xy):.6f} m"
    )
    print(
        f"Spacing violations: "
        f"{best_spacing_violations}"
    )
    print(
        f"Boundary violations: "
        f"{best_boundary_violations}"
    )
    print("")
    print(
        f"Saved outputs to: "
        f"{output_dir}"
    )
    print("")


if __name__ == "__main__":
    main()