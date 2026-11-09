"""Run the required pre-optimisation validation for replacement Model 8.

The script compares aligned, staggered, and wider-gap 12-turbine layouts using
the same incoming trajectories and the same response random numbers.  The
behavioural parameters are hypothetical scenario assumptions, not calibrated
species parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model8_meso_response import (
    BehaviourScenario,
    MesoResponseModel,
    TrajectoryPopulation,
    cross_unit_vectors,
    heading_unit_vectors,
)


def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    delta = points[:, None, :] - points[None, :, :]
    matrix = np.sqrt(np.sum(delta * delta, axis=2))
    return matrix[np.triu_indices(len(points), k=1)]


def validate_layout(
    xy: np.ndarray,
    *,
    site_width_m: float,
    site_height_m: float,
    minimum_spacing_m: float,
) -> None:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    if points.shape != (12, 2):
        raise ValueError(f"Expected a (12, 2) layout, received {points.shape}.")
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] > site_width_m)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] > site_height_m)
    ):
        raise ValueError("A fixed validation layout lies outside the site.")
    minimum_actual = float(np.min(pairwise_distances(points)))
    if minimum_actual + 1.0e-9 < minimum_spacing_m:
        raise ValueError(
            "A fixed validation layout violates minimum spacing: "
            f"{minimum_actual:.6f} < {minimum_spacing_m:.6f} m."
        )


def make_fixed_layouts(
    *,
    site_width_m: float,
    site_height_m: float,
    bird_heading_deg_toward: float,
    minimum_spacing_m: float,
) -> dict[str, np.ndarray]:
    """Create three controlled, feasible layouts in the bird reference frame."""
    centre = np.array([site_width_m / 2.0, site_height_m / 2.0])
    along = heading_unit_vectors(bird_heading_deg_toward)
    cross = cross_unit_vectors(bird_heading_deg_toward)
    spacing = 1.08 * minimum_spacing_m

    def convert(along_cross: list[tuple[float, float]]) -> np.ndarray:
        coordinates = np.asarray(
            [centre + a * along + c * cross for a, c in along_cross],
            dtype=float,
        )
        validate_layout(
            coordinates,
            site_width_m=site_width_m,
            site_height_m=site_height_m,
            minimum_spacing_m=minimum_spacing_m,
        )
        return coordinates

    aligned_points = [
        (a * spacing, c * spacing)
        for c in (-1.0, 0.0, 1.0)
        for a in (-1.5, -0.5, 0.5, 1.5)
    ]

    staggered_points: list[tuple[float, float]] = []
    for row_index, c in enumerate((-1.0, 0.0, 1.0)):
        offset = (-0.25 if row_index % 2 == 0 else 0.25) * spacing
        for a in (-1.5, -0.5, 0.5, 1.5):
            staggered_points.append((a * spacing + offset, c * spacing))

    wider_gap_points = [
        (a * spacing, c * spacing)
        for c in (-1.75, -0.75, 0.75, 1.75)
        for a in (-1.0, 0.0, 1.0)
    ]

    return {
        "aligned": convert(aligned_points),
        "staggered": convert(staggered_points),
        "wider_gap": convert(wider_gap_points),
    }


def save_layouts(layouts: dict[str, np.ndarray], output_dir: Path) -> None:
    layout_dir = output_dir / "layouts"
    layout_dir.mkdir(parents=True, exist_ok=True)
    for name, xy in layouts.items():
        pd.DataFrame(
            {
                "turbine_id": np.arange(len(xy), dtype=int),
                "x_m": xy[:, 0],
                "y_m": xy[:, 1],
            }
        ).to_csv(layout_dir / f"{name}.csv", index=False)


def plot_fixed_layouts(
    layouts: dict[str, np.ndarray],
    *,
    output_path: Path,
    site_width_m: float,
    site_height_m: float,
    heading_deg: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    direction = heading_unit_vectors(heading_deg)
    for axis, (name, xy) in zip(axes, layouts.items()):
        axis.scatter(xy[:, 0], xy[:, 1], s=45, color="#1f4e79", zorder=3)
        for turbine_index, point in enumerate(xy):
            axis.annotate(str(turbine_index), point, xytext=(4, 4), textcoords="offset points")
        start = np.array([0.16 * site_width_m, 0.84 * site_height_m])
        axis.arrow(
            start[0],
            start[1],
            700.0 * direction[0],
            700.0 * direction[1],
            width=18.0,
            head_width=120.0,
            color="#c44e52",
            length_includes_head=True,
        )
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlim(0.0, site_width_m)
        axis.set_ylim(0.0, site_height_m)
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for layout_name, frame in summary.groupby("layout", sort=False):
        ordered = frame.sort_values("p_response")
        axes[0].plot(
            ordered["p_response"],
            ordered["E_enc"],
            marker="o",
            label=layout_name.replace("_", " "),
        )
        axes[1].plot(
            ordered["p_response"],
            ordered["P_ge_1"],
            marker="o",
            label=layout_name.replace("_", " "),
        )
    axes[0].set_ylabel("Encounter intensity, E_enc")
    axes[1].set_ylabel("Encounter prevalence, P>=1")
    for axis in axes:
        axis.set_xlabel("Response probability scenario")
        axis.set_xlim(-0.02, 0.92)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_paths(
    *,
    layout_xy: np.ndarray,
    result,
    behaviour: BehaviourScenario,
    output_path: Path,
    site_width_m: float,
    site_height_m: float,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for path in result.stored_paths_m.values():
        axis.plot(path[:, 0], path[:, 1], color="#4c78a8", alpha=0.35, linewidth=0.9)
    for point in layout_xy:
        response_circle = plt.Circle(
            point,
            behaviour.response_radius_m,
            fill=False,
            color="#f58518",
            alpha=0.18,
            linewidth=0.7,
        )
        encounter_circle = plt.Circle(
            point,
            behaviour.encounter_radius_m,
            fill=False,
            color="#e45756",
            alpha=0.35,
            linewidth=0.8,
        )
        axis.add_patch(response_circle)
        axis.add_patch(encounter_circle)
    axis.scatter(layout_xy[:, 0], layout_xy[:, 1], s=35, color="black", zorder=4)
    margin = behaviour.response_radius_m + 300.0
    axis.set_xlim(-margin, site_width_m + margin)
    axis.set_ylim(-margin, site_height_m + margin)
    axis.set_aspect("equal")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(
        f"Stored trajectories: wider-gap layout, p={behaviour.response_probability:g}"
    )
    axis.grid(alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate scenario-based Model 8 before optimisation."
    )
    parser.add_argument("--output", type=Path, default=Path("results/model8_validation"))
    parser.add_argument("--n_trajectories", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bird_heading_deg", type=float, default=225.0)
    parser.add_argument("--site_width_m", type=float, default=6000.0)
    parser.add_argument("--site_height_m", type=float, default=6000.0)
    parser.add_argument("--rotor_diameter_m", type=float, default=242.24)
    parser.add_argument("--minimum_spacing_D", type=float, default=4.0)
    parser.add_argument("--response_radius_D", type=float, default=2.0)
    parser.add_argument("--encounter_radius_D", type=float, default=1.0)
    parser.add_argument("--deflection_angle_deg", type=float, default=45.0)
    parser.add_argument("--return_length_D", type=float, default=8.0)
    parser.add_argument("--step_size_D", type=float, default=0.20)
    parser.add_argument("--path_count", type=int, default=40)
    parser.add_argument("--robustness_seed_count", type=int, default=3)
    args = parser.parse_args()

    if args.n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive.")
    if args.path_count < 0:
        raise ValueError("path_count must be non-negative.")
    if args.robustness_seed_count <= 0:
        raise ValueError("robustness_seed_count must be positive.")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    minimum_spacing_m = args.minimum_spacing_D * args.rotor_diameter_m
    layouts = make_fixed_layouts(
        site_width_m=args.site_width_m,
        site_height_m=args.site_height_m,
        bird_heading_deg_toward=args.bird_heading_deg,
        minimum_spacing_m=minimum_spacing_m,
    )
    save_layouts(layouts, output_dir)
    plot_fixed_layouts(
        layouts,
        output_path=output_dir / "fixed_layouts.png",
        site_width_m=args.site_width_m,
        site_height_m=args.site_height_m,
        heading_deg=args.bird_heading_deg,
    )

    reference_behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=args.rotor_diameter_m,
        response_probability=0.0,
        response_radius_D=args.response_radius_D,
        encounter_radius_D=args.encounter_radius_D,
        deflection_angle_deg=args.deflection_angle_deg,
        return_length_D=args.return_length_D,
        step_size_D=args.step_size_D,
    )
    longitudinal_margin_m = (
        reference_behaviour.response_radius_m + reference_behaviour.return_length_m
    )
    population = TrajectoryPopulation.generate_parallel(
        site_width_m=args.site_width_m,
        site_height_m=args.site_height_m,
        heading_deg_toward=args.bird_heading_deg,
        number_of_trajectories=args.n_trajectories,
        maximum_response_events=12,
        longitudinal_margin_m=longitudinal_margin_m,
        seed=args.seed,
    )

    response_probabilities = (0.0, 0.3, 0.6, 0.9)
    summary_rows: list[dict] = []
    turbine_rows: list[dict] = []
    path_indices = range(min(args.path_count, args.n_trajectories))

    for probability in response_probabilities:
        scenario = BehaviourScenario.from_rotor_diameters(
            rotor_diameter_m=args.rotor_diameter_m,
            response_probability=probability,
            response_radius_D=args.response_radius_D,
            encounter_radius_D=args.encounter_radius_D,
            deflection_angle_deg=args.deflection_angle_deg,
            return_length_D=args.return_length_D,
            step_size_D=args.step_size_D,
        )
        model = MesoResponseModel(scenario)
        for layout_name, layout_xy in layouts.items():
            should_store = layout_name == "wider_gap" and probability in (0.0, 0.6, 0.9)
            result = model.simulate(
                layout_xy,
                population,
                store_path_indices=path_indices if should_store else (),
                path_stride=2,
            )
            row = {
                "layout": layout_name,
                "p_response": probability,
                **result.summary(),
                "minimum_spacing_m": float(np.min(pairwise_distances(layout_xy))),
            }
            summary_rows.append(row)
            for turbine_id, encounter_count in enumerate(result.per_turbine_encounters):
                turbine_rows.append(
                    {
                        "layout": layout_name,
                        "p_response": probability,
                        "turbine_id": turbine_id,
                        "N_enc_i": int(encounter_count),
                    }
                )
            if probability == 0.0:
                if result.total_response_events != 0:
                    raise RuntimeError("No-response validation produced a response event.")
                if not np.allclose(result.route_extensions_m, 0.0, atol=1.0e-9):
                    raise RuntimeError("No-response validation did not remain straight.")
            if result.number_failed != 0:
                raise RuntimeError(
                    f"Trajectory accounting failed for {layout_name}, p={probability}."
                )
            if should_store:
                plot_paths(
                    layout_xy=layout_xy,
                    result=result,
                    behaviour=scenario,
                    output_path=output_dir
                    / f"trajectories_wider_gap_p{str(probability).replace('.', 'p')}.png",
                    site_width_m=args.site_width_m,
                    site_height_m=args.site_height_m,
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "fixed_layout_summary.csv", index=False)
    pd.DataFrame(turbine_rows).to_csv(
        output_dir / "per_turbine_encounters.csv", index=False
    )
    plot_sensitivity(summary, output_dir / "response_sensitivity.png")

    convergence_sizes = sorted(
        {
            min(args.n_trajectories, value)
            for value in (250, 500, 1000, 2000, 5000, args.n_trajectories)
            if min(args.n_trajectories, value) > 0
        }
    )
    convergence_behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=args.rotor_diameter_m,
        response_probability=0.6,
        response_radius_D=args.response_radius_D,
        encounter_radius_D=args.encounter_radius_D,
        deflection_angle_deg=args.deflection_angle_deg,
        return_length_D=args.return_length_D,
        step_size_D=args.step_size_D,
    )
    convergence_model = MesoResponseModel(convergence_behaviour)
    convergence_rows = []
    for sample_size in convergence_sizes:
        result = convergence_model.simulate(
            layouts["aligned"], population.subset(sample_size)
        )
        convergence_rows.append(
            {
                "N_in": sample_size,
                "E_enc": result.encounter_intensity,
                "P_ge_1": result.encounter_prevalence,
                "N_failed": result.number_failed,
            }
        )
    pd.DataFrame(convergence_rows).to_csv(
        output_dir / "monte_carlo_convergence.csv", index=False
    )

    robustness_rows = []
    robustness_sample_size = min(args.n_trajectories, 1000)
    for seed_offset in range(args.robustness_seed_count):
        robustness_population = TrajectoryPopulation.generate_parallel(
            site_width_m=args.site_width_m,
            site_height_m=args.site_height_m,
            heading_deg_toward=args.bird_heading_deg,
            number_of_trajectories=robustness_sample_size,
            maximum_response_events=12,
            longitudinal_margin_m=longitudinal_margin_m,
            seed=args.seed + seed_offset,
        )
        for layout_name, layout_xy in layouts.items():
            result = convergence_model.simulate(layout_xy, robustness_population)
            robustness_rows.append(
                {
                    "seed": args.seed + seed_offset,
                    "N_in": robustness_sample_size,
                    "layout": layout_name,
                    "p_response": 0.6,
                    "E_enc": result.encounter_intensity,
                    "P_ge_1": result.encounter_prevalence,
                    "N_failed": result.number_failed,
                }
            )
    pd.DataFrame(robustness_rows).to_csv(
        output_dir / "random_seed_robustness.csv", index=False
    )

    step_rows = []
    step_population = population.subset(min(args.n_trajectories, 500))
    for step_size_D in (0.40, 0.20, 0.10):
        step_behaviour = BehaviourScenario.from_rotor_diameters(
            rotor_diameter_m=args.rotor_diameter_m,
            response_probability=0.6,
            response_radius_D=args.response_radius_D,
            encounter_radius_D=args.encounter_radius_D,
            deflection_angle_deg=args.deflection_angle_deg,
            return_length_D=args.return_length_D,
            step_size_D=step_size_D,
        )
        result = MesoResponseModel(step_behaviour).simulate(
            layouts["aligned"], step_population
        )
        step_rows.append(
            {
                "step_size_D": step_size_D,
                "step_size_m": step_behaviour.step_size_m,
                "N_in": step_population.number_of_trajectories,
                "E_enc": result.encounter_intensity,
                "P_ge_1": result.encounter_prevalence,
                "mean_route_extension_m": result.mean_route_extension_m,
                "N_failed": result.number_failed,
            }
        )
    pd.DataFrame(step_rows).to_csv(
        output_dir / "step_size_convergence.csv", index=False
    )

    run_settings = {
        "model": "Model 8 scenario-based meso-response and turbine encounters",
        "scientific_status": "hypothetical behavioural sensitivity model",
        "direction_convention": "TOWARD mathematical degrees (0 east, 90 north)",
        "site_width_m": args.site_width_m,
        "site_height_m": args.site_height_m,
        "number_of_turbines": 12,
        "minimum_spacing_D": args.minimum_spacing_D,
        "minimum_spacing_m": minimum_spacing_m,
        "bird_heading_deg_toward": args.bird_heading_deg,
        "N_in": args.n_trajectories,
        "common_random_number_seed": args.seed,
        "random_seed_robustness_count": args.robustness_seed_count,
        "response_probability_scenarios": list(response_probabilities),
        "behaviour_geometry": reference_behaviour.settings(args.rotor_diameter_m),
        "layout_classes": list(layouts),
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(run_settings, indent=2), encoding="utf-8"
    )

    display_columns = [
        "layout",
        "p_response",
        "N_response_events",
        "N_enc_farm",
        "E_enc",
        "P_ge_1",
        "mean_route_extension_m",
        "N_failed",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"\nSaved Model 8 validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
