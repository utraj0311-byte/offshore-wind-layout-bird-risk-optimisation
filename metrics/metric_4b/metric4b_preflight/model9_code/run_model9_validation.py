"""Run Model 9 pre-optimisation validation on three controlled layouts.

The script answers the required diagnostic question by comparing Model 8
encounter intensity (E_enc) with both sampled Model 9 rotor-transit intensity
(E_RT) and its lower-variance conditional-expectation estimator (E_RT_CE).
It requires an explicit, traceable flight-height CSV and never substitutes a
synthetic or uniform biological distribution.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from model8_meso_response import (
    BehaviourScenario,
    cross_unit_vectors,
    heading_unit_vectors,
)
from model9_batching import summarise_model9_batches
from model9_rotor_transit import (
    DiscreteFlightHeightDistribution,
    Model9Population,
    RotorGeometry,
    RotorTransitModel,
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
    actual = float(np.min(pairwise_distances(points)))
    if actual + 1.0e-9 < minimum_spacing_m:
        raise ValueError(
            "A fixed validation layout violates minimum spacing: "
            f"{actual:.6f} < {minimum_spacing_m:.6f} m."
        )


def make_fixed_layouts(
    *,
    site_width_m: float,
    site_height_m: float,
    bird_heading_deg_toward: float,
    minimum_spacing_m: float,
) -> dict[str, np.ndarray]:
    """Recreate Model 8's aligned, staggered, and wider-gap layouts."""
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

    aligned = [
        (a * spacing, c * spacing)
        for c in (-1.0, 0.0, 1.0)
        for a in (-1.5, -0.5, 0.5, 1.5)
    ]
    staggered: list[tuple[float, float]] = []
    for row_index, c in enumerate((-1.0, 0.0, 1.0)):
        offset = (-0.25 if row_index % 2 == 0 else 0.25) * spacing
        for a in (-1.5, -0.5, 0.5, 1.5):
            staggered.append((a * spacing + offset, c * spacing))
    wider_gap = [
        (a * spacing, c * spacing)
        for c in (-1.75, -0.75, 0.75, 1.75)
        for a in (-1.0, 0.0, 1.0)
    ]
    return {
        "aligned": convert(aligned),
        "staggered": convert(staggered),
        "wider_gap": convert(wider_gap),
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


def plot_encounter_vs_transit(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    batch_counts = summary["objective_batch_count"].to_numpy(dtype=int)
    critical = np.asarray(
        [
            0.0 if count <= 1 else student_t.ppf(0.975, df=count - 1)
            for count in batch_counts
        ],
        dtype=float,
    )
    axis.errorbar(
        summary["E_enc"],
        summary["E_RT"],
        yerr=critical * summary["E_RT_batch_se"],
        fmt="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="#9c5f35",
        color="#9c5f35",
        ecolor="#c6a087",
        elinewidth=1.2,
        capsize=3,
        label="Sampled E_RT",
    )
    primary_metric = "E_RT"
    if "E_RT_CE" in summary.columns and summary["E_RT_CE"].notna().all():
        primary_metric = "E_RT_CE"
        axis.errorbar(
            summary["E_enc"],
            summary["E_RT_CE"],
            xerr=critical * summary["E_enc_batch_se"],
            yerr=critical * summary["E_RT_CE_batch_se"],
            fmt="D",
            markersize=6,
            color="#315f73",
            ecolor="#7898a6",
            elinewidth=1.2,
            capsize=3,
            label="Conditional E_RT_CE",
        )
    for row in summary.itertuples(index=False):
        axis.annotate(
            str(row.layout).replace("_", " "),
            (row.E_enc, getattr(row, primary_metric)),
            xytext=(6, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("Model 8 encounter intensity, E_enc")
    axis.set_ylabel("Model 9 rotor-transit intensity, E_RT")
    axis.set_title("Sampled and conditional rotor-transit exposure")
    axis.text(
        0.02,
        0.98,
        "Error bars: marginal 95% Student-t intervals across batches",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#4d4d4d",
    )
    axis.legend(frameon=False, loc="best")
    axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_populations(
    *,
    distribution: DiscreteFlightHeightDistribution,
    batch_count: int,
    trajectories_per_batch: int,
    first_trajectory_seed: int,
    first_flight_height_seed: int,
    site_width_m: float,
    site_height_m: float,
    heading_deg_toward: float,
    longitudinal_margin_m: float,
    maximum_response_events: int,
) -> tuple[Model9Population, ...]:
    return tuple(
        Model9Population.generate_parallel(
            flight_height_distribution=distribution,
            flight_height_seed=first_flight_height_seed + batch_index,
            site_width_m=site_width_m,
            site_height_m=site_height_m,
            heading_deg_toward=heading_deg_toward,
            number_of_trajectories=trajectories_per_batch,
            maximum_response_events=maximum_response_events,
            longitudinal_margin_m=longitudinal_margin_m,
            trajectory_seed=first_trajectory_seed + batch_index,
        )
        for batch_index in range(batch_count)
    )


def ranking_order(frame: pd.DataFrame, column: str) -> list[str]:
    return frame.sort_values([column, "layout"], kind="stable")["layout"].tolist()


def paired_layout_differences(batch_results: pd.DataFrame) -> pd.DataFrame:
    """Estimate paired layout differences across common-random-number batches."""
    required = {"layout", "batch_index", "E_enc", "E_RT"}
    if not required.issubset(batch_results.columns):
        raise ValueError(f"Batch results are missing columns: {sorted(required)}")

    metrics = ["E_enc", "E_RT"]
    if "E_RT_CE" in batch_results.columns and batch_results["E_RT_CE"].notna().all():
        metrics.append("E_RT_CE")
    selected_columns = ["batch_index", *metrics]
    rows: list[dict[str, Any]] = []
    layouts = sorted(batch_results["layout"].unique())
    for first, second in itertools.combinations(layouts, 2):
        left = batch_results.loc[
            batch_results["layout"] == first,
            selected_columns,
        ]
        right = batch_results.loc[
            batch_results["layout"] == second,
            selected_columns,
        ]
        paired = left.merge(
            right,
            on="batch_index",
            suffixes=("_first", "_second"),
            validate="one_to_one",
        ).sort_values("batch_index")
        if len(paired) != len(left) or len(paired) != len(right):
            raise ValueError("Every layout must use the same batch indices.")

        for metric in metrics:
            differences = (
                paired[f"{metric}_first"].to_numpy(dtype=float)
                - paired[f"{metric}_second"].to_numpy(dtype=float)
            )
            count = len(differences)
            mean = float(np.mean(differences))
            sample_sd = (
                0.0 if count == 1 else float(np.std(differences, ddof=1))
            )
            standard_error = float(sample_sd / np.sqrt(count))
            if count <= 1:
                lower = upper = mean
                critical = None
            else:
                critical = float(student_t.ppf(0.975, df=count - 1))
                half_width = critical * standard_error
                lower = mean - half_width
                upper = mean + half_width
            rows.append(
                {
                    "layout_first": first,
                    "layout_second": second,
                    "metric": metric,
                    "batch_count": count,
                    "mean_first_minus_second": mean,
                    "paired_batch_sd": sample_sd,
                    "paired_batch_se": standard_error,
                    "t_critical_95pct": critical,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "ci95_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                }
            )
    return pd.DataFrame(rows)


def estimator_precision_comparison(batch_results: pd.DataFrame) -> pd.DataFrame:
    """Compare sampled E_RT with its conditional-expectation estimator."""
    required = {"layout", "batch_index", "E_RT", "E_RT_CE"}
    if not required.issubset(batch_results.columns):
        raise ValueError(f"Batch results are missing columns: {sorted(required)}")
    if batch_results["E_RT_CE"].isna().any():
        raise ValueError("E_RT_CE must be available for every validation batch.")

    rows: list[dict[str, Any]] = []
    for layout, group in batch_results.groupby("layout", sort=True):
        ordered = group.sort_values("batch_index")
        sampled = ordered["E_RT"].to_numpy(dtype=float)
        conditional = ordered["E_RT_CE"].to_numpy(dtype=float)
        differences = sampled - conditional
        count = len(differences)
        sampled_sd = 0.0 if count == 1 else float(np.std(sampled, ddof=1))
        conditional_sd = (
            0.0 if count == 1 else float(np.std(conditional, ddof=1))
        )
        difference_sd = (
            0.0 if count == 1 else float(np.std(differences, ddof=1))
        )
        difference_se = float(difference_sd / np.sqrt(count))
        mean_difference = float(np.mean(differences))
        if count <= 1:
            critical = None
            lower = upper = mean_difference
        else:
            critical = float(student_t.ppf(0.975, df=count - 1))
            half_width = critical * difference_se
            lower = mean_difference - half_width
            upper = mean_difference + half_width
        rows.append(
            {
                "layout": str(layout),
                "batch_count": count,
                "mean_E_RT": float(np.mean(sampled)),
                "mean_E_RT_CE": float(np.mean(conditional)),
                "mean_sampled_minus_conditional": mean_difference,
                "sampled_batch_sd": sampled_sd,
                "conditional_batch_sd": conditional_sd,
                "sd_reduction_factor": (
                    float(sampled_sd / conditional_sd)
                    if conditional_sd > 0.0
                    else None
                ),
                "variance_reduction_factor": (
                    float((sampled_sd / conditional_sd) ** 2)
                    if conditional_sd > 0.0
                    else None
                ),
                "difference_batch_se": difference_se,
                "t_critical_95pct": critical,
                "difference_ci95_lower": lower,
                "difference_ci95_upper": upper,
                "difference_ci95_excludes_zero": bool(
                    lower > 0.0 or upper < 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def validation_diagnostic(
    summary: pd.DataFrame,
    batch_results: pd.DataFrame,
    paired_differences: pd.DataFrame,
    estimator_precision: pd.DataFrame | None = None,
) -> dict[str, Any]:
    encounter_order = ranking_order(summary, "E_enc")
    transit_order = ranking_order(summary, "E_RT")
    batch_transit_orders = [
        ranking_order(group, "E_RT")
        for _, group in batch_results.groupby("batch_index", sort=True)
    ]
    transit_pairs = paired_differences.loc[
        paired_differences["metric"] == "E_RT"
    ]
    paired_transit_records = [
        {
            "layout_first": str(row.layout_first),
            "layout_second": str(row.layout_second),
            "mean_first_minus_second": float(row.mean_first_minus_second),
            "ci95_lower": float(row.ci95_lower),
            "ci95_upper": float(row.ci95_upper),
            "ci95_excludes_zero": bool(row.ci95_excludes_zero),
        }
        for row in transit_pairs.itertuples(index=False)
    ]
    all_transit_intervals_exclude_zero = bool(
        len(transit_pairs) > 0 and transit_pairs["ci95_excludes_zero"].all()
    )
    has_conditional = (
        "E_RT_CE" in summary.columns and summary["E_RT_CE"].notna().all()
    )
    conditional_order = (
        ranking_order(summary, "E_RT_CE") if has_conditional else None
    )
    conditional_batch_orders = (
        [
            ranking_order(group, "E_RT_CE")
            for _, group in batch_results.groupby("batch_index", sort=True)
        ]
        if has_conditional
        else None
    )
    conditional_pairs = (
        paired_differences.loc[paired_differences["metric"] == "E_RT_CE"]
        if has_conditional
        else pd.DataFrame()
    )
    all_conditional_intervals_exclude_zero = bool(
        has_conditional
        and len(conditional_pairs) > 0
        and conditional_pairs["ci95_excludes_zero"].all()
    )
    ratios = summary["RT_to_encounter_ratio"].to_numpy(dtype=float)
    finite_ratios = ratios[np.isfinite(ratios)]
    diagnostic = {
        "encounter_ranking_low_to_high": encounter_order,
        "rotor_transit_ranking_low_to_high": transit_order,
        "ranking_changed": encounter_order != transit_order,
        "point_estimate_ranking_changed": encounter_order != transit_order,
        "rotor_transit_batch_rankings_low_to_high": batch_transit_orders,
        "rotor_transit_ranking_identical_across_batches": bool(
            batch_transit_orders
            and all(order == batch_transit_orders[0] for order in batch_transit_orders)
        ),
        "all_pairwise_E_RT_95pct_intervals_exclude_zero": (
            all_transit_intervals_exclude_zero
        ),
        "paired_E_RT_differences_95pct": paired_transit_records,
        "N_RT_farm_by_layout": {
            str(row.layout): int(row.N_RT_farm)
            for row in summary.itertuples(index=False)
        },
        "ranking_resolution": (
            "all tested layout pairs are separated by paired 95% Student-t "
            "intervals across independent batches"
            if all_transit_intervals_exclude_zero
            else "point-estimate ranking is provisional because at least one "
            "paired 95% Student-t interval includes zero"
        ),
        "minimum_RT_to_encounter_ratio": (
            None if len(finite_ratios) == 0 else float(np.min(finite_ratios))
        ),
        "maximum_RT_to_encounter_ratio": (
            None if len(finite_ratios) == 0 else float(np.max(finite_ratios))
        ),
        "ratio_range": (
            None
            if len(finite_ratios) == 0
            else float(np.max(finite_ratios) - np.min(finite_ratios))
        ),
        "interpretation_rule": (
            "Interpret point ranks together with paired batch intervals. A "
            "changed point ranking alone is not evidence of robust new layout "
            "discrimination when uncertainty intervals include zero."
        ),
    }
    if has_conditional:
        conditional_records = [
            {
                "layout_first": str(row.layout_first),
                "layout_second": str(row.layout_second),
                "mean_first_minus_second": float(row.mean_first_minus_second),
                "ci95_lower": float(row.ci95_lower),
                "ci95_upper": float(row.ci95_upper),
                "ci95_excludes_zero": bool(row.ci95_excludes_zero),
            }
            for row in conditional_pairs.itertuples(index=False)
        ]
        diagnostic.update(
            {
                "optimisation_estimator": "E_RT_CE",
                "conditional_point_estimate_ranking_changed": (
                    encounter_order != conditional_order
                ),
                "conditional_rotor_transit_ranking_low_to_high": conditional_order,
                "conditional_rotor_transit_batch_rankings_low_to_high": (
                    conditional_batch_orders
                ),
                "conditional_ranking_identical_across_batches": bool(
                    conditional_batch_orders
                    and all(
                        order == conditional_batch_orders[0]
                        for order in conditional_batch_orders
                    )
                ),
                "all_pairwise_E_RT_CE_95pct_intervals_exclude_zero": (
                    all_conditional_intervals_exclude_zero
                ),
                "paired_E_RT_CE_differences_95pct": conditional_records,
                "N_RT_CE_farm_by_layout": {
                    str(row.layout): float(row.N_RT_CE_farm)
                    for row in summary.itertuples(index=False)
                },
                "minimum_RT_CE_to_encounter_ratio": float(
                    summary["RT_CE_to_encounter_ratio"].min()
                ),
                "maximum_RT_CE_to_encounter_ratio": float(
                    summary["RT_CE_to_encounter_ratio"].max()
                ),
                "RT_CE_to_encounter_ratio_range": float(
                    summary["RT_CE_to_encounter_ratio"].max()
                    - summary["RT_CE_to_encounter_ratio"].min()
                ),
                "conditional_ranking_resolution": (
                    "all tested layout pairs are separated by paired 95% "
                    "Student-t intervals across independent batches"
                    if all_conditional_intervals_exclude_zero
                    else "at least one paired E_RT_CE 95% Student-t interval "
                    "includes zero"
                ),
            }
        )
        if estimator_precision is not None:
            agreement = bool(
                not estimator_precision["difference_ci95_excludes_zero"].any()
            )
            diagnostic.update(
                {
                    "sampled_and_conditional_estimators_agree_at_95pct": agreement,
                    "variance_reduction_factor_by_layout": {
                        str(row.layout): (
                            None
                            if pd.isna(row.variance_reduction_factor)
                            else float(row.variance_reduction_factor)
                        )
                        for row in estimator_precision.itertuples(index=False)
                    },
                }
            )
    return diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Model 9 encounter-to-rotor-transit conversion."
    )
    parser.add_argument("--flight_height_csv", type=Path, required=True)
    parser.add_argument("--fhd_name", type=str, required=True)
    parser.add_argument("--fhd_source", type=str, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/model9_validation")
    )
    parser.add_argument("--trajectories_per_batch", type=int, default=2000)
    parser.add_argument("--batch_count", type=int, default=3)
    parser.add_argument("--trajectory_seed", type=int, default=42)
    parser.add_argument("--flight_height_seed", type=int, default=10042)
    parser.add_argument("--bird_heading_deg", type=float, default=225.0)
    parser.add_argument("--rotor_plane_heading_deg", type=float, default=None)
    parser.add_argument("--site_width_m", type=float, default=6000.0)
    parser.add_argument("--site_height_m", type=float, default=6000.0)
    parser.add_argument("--rotor_diameter_m", type=float, default=242.24)
    parser.add_argument("--hub_height_m", type=float, default=150.0)
    parser.add_argument("--minimum_spacing_D", type=float, default=4.0)
    parser.add_argument("--p_response", type=float, default=0.6)
    parser.add_argument("--response_radius_D", type=float, default=2.0)
    parser.add_argument("--encounter_radius_D", type=float, default=1.0)
    parser.add_argument("--deflection_angle_deg", type=float, default=45.0)
    parser.add_argument("--return_length_D", type=float, default=8.0)
    parser.add_argument("--step_size_D", type=float, default=0.20)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    if args.trajectories_per_batch <= 0:
        raise ValueError("trajectories_per_batch must be positive.")
    if args.batch_count <= 0:
        raise ValueError("batch_count must be positive.")

    distribution = DiscreteFlightHeightDistribution.from_csv(
        args.flight_height_csv,
        name=args.fhd_name,
        source=args.fhd_source,
    )
    rotor = RotorGeometry(
        rotor_diameter_m=args.rotor_diameter_m,
        hub_height_m=args.hub_height_m,
    )
    behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=args.rotor_diameter_m,
        response_probability=args.p_response,
        response_radius_D=args.response_radius_D,
        encounter_radius_D=args.encounter_radius_D,
        deflection_angle_deg=args.deflection_angle_deg,
        return_length_D=args.return_length_D,
        step_size_D=args.step_size_D,
    )
    plane_heading = (
        args.bird_heading_deg
        if args.rotor_plane_heading_deg is None
        else args.rotor_plane_heading_deg
    )
    model = RotorTransitModel(
        behaviour,
        rotor,
        rotor_plane_heading_deg_toward=plane_heading,
        flight_height_distribution=distribution,
    )

    minimum_spacing_m = args.minimum_spacing_D * args.rotor_diameter_m
    layouts = make_fixed_layouts(
        site_width_m=args.site_width_m,
        site_height_m=args.site_height_m,
        bird_heading_deg_toward=args.bird_heading_deg,
        minimum_spacing_m=minimum_spacing_m,
    )
    populations = build_populations(
        distribution=distribution,
        batch_count=args.batch_count,
        trajectories_per_batch=args.trajectories_per_batch,
        first_trajectory_seed=args.trajectory_seed,
        first_flight_height_seed=args.flight_height_seed,
        site_width_m=args.site_width_m,
        site_height_m=args.site_height_m,
        heading_deg_toward=args.bird_heading_deg,
        longitudinal_margin_m=(
            behaviour.response_radius_m + behaviour.return_length_m
        ),
        maximum_response_events=12,
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    save_layouts(layouts, output_dir)

    summary_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    turbine_rows: list[dict[str, Any]] = []
    for layout_name, layout_xy in layouts.items():
        combined = summarise_model9_batches(model, layout_xy, populations)
        if combined["N_failed"] != 0:
            raise RuntimeError(f"Trajectory failures occurred for {layout_name}.")
        summary_rows.append(
            {
                key: value
                for key, value in {"layout": layout_name, **combined}.items()
                if key
                not in {
                    "batch_summaries",
                    "per_turbine_encounters",
                    "per_turbine_transits",
                    "per_turbine_expected_transits",
                }
            }
        )
        for row in combined["batch_summaries"]:
            batch_rows.append({"layout": layout_name, **row})
        for turbine_id, (encounters, transits, expected_transits) in enumerate(
            zip(
                combined["per_turbine_encounters"],
                combined["per_turbine_transits"],
                combined["per_turbine_expected_transits"],
            )
        ):
            turbine_rows.append(
                {
                    "layout": layout_name,
                    "turbine_id": turbine_id,
                    "N_enc_i": int(encounters),
                    "N_RT_i": int(transits),
                    "N_RT_CE_i": float(expected_transits),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary["rank_E_enc"] = summary["E_enc"].rank(method="min").astype(int)
    summary["rank_E_RT"] = summary["E_RT"].rank(method="min").astype(int)
    summary["rank_E_RT_CE"] = summary["E_RT_CE"].rank(method="min").astype(int)
    summary["rank_changed"] = summary["rank_E_enc"] != summary["rank_E_RT"]
    summary["rank_changed_CE"] = (
        summary["rank_E_enc"] != summary["rank_E_RT_CE"]
    )
    summary.to_csv(output_dir / "layout_encounter_transit_comparison.csv", index=False)
    batch_results = pd.DataFrame(batch_rows)
    batch_results.to_csv(
        output_dir / "independent_batch_results.csv", index=False
    )
    pd.DataFrame(turbine_rows).to_csv(
        output_dir / "per_turbine_encounters_and_transits.csv", index=False
    )
    paired_differences = paired_layout_differences(batch_results)
    paired_differences.to_csv(
        output_dir / "paired_layout_differences.csv", index=False
    )
    estimator_precision = estimator_precision_comparison(batch_results)
    estimator_precision.to_csv(
        output_dir / "estimator_precision_comparison.csv", index=False
    )
    diagnostic = validation_diagnostic(
        summary,
        batch_results,
        paired_differences,
        estimator_precision,
    )
    (output_dir / "validation_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8"
    )
    settings = {
        "model": "Model 9 rotor-transit exposure model",
        "endpoint": "rotor transit, not blade collision",
        "scientific_status": (
            "methodological extension with inherited hypothetical Model 8 "
            "meso-response parameters"
        ),
        "flight_height_assumption": (
            "one sampled height per trajectory, constant through farm crossing"
        ),
        "variance_reduction_estimator": (
            "E_RT_CE integrates the same discrete FHD over each circular "
            "rotor-crossing height window conditional on the horizontal path; "
            "sampled E_RT is retained as an audit metric"
        ),
        "recommended_optimisation_bird_metric": "E_RT_CE",
        "rotor_plane_convention": (
            "fixed plane through each turbine, perpendicular to the stated "
            "rotor-plane heading"
        ),
        "site_width_m": args.site_width_m,
        "site_height_m": args.site_height_m,
        "number_of_turbines": 12,
        "minimum_spacing_D": args.minimum_spacing_D,
        "minimum_spacing_m": minimum_spacing_m,
        "bird_heading_deg_toward": args.bird_heading_deg,
        "rotor_plane_heading_deg_toward": plane_heading,
        "batch_count": args.batch_count,
        "trajectories_per_batch": args.trajectories_per_batch,
        "total_trajectories_per_layout": (
            args.batch_count * args.trajectories_per_batch
        ),
        "trajectory_seeds": [
            args.trajectory_seed + index for index in range(args.batch_count)
        ],
        "flight_height_seeds": [
            args.flight_height_seed + index for index in range(args.batch_count)
        ],
        "flight_height_distribution": distribution.settings(),
        "rotor_geometry": rotor.settings(),
        "behaviour": behaviour.settings(args.rotor_diameter_m),
        "layout_classes": list(layouts),
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    if not args.no_plot:
        plot_encounter_vs_transit(
            summary,
            output_dir / "encounter_vs_rotor_transit.png",
        )

    display = [
        "layout",
        "E_enc",
        "E_RT",
        "E_RT_CE",
        "P_RT",
        "P_RT_CE",
        "RT_to_encounter_ratio",
        "RT_CE_to_encounter_ratio",
        "rank_E_enc",
        "rank_E_RT",
        "rank_E_RT_CE",
        "N_failed",
    ]
    print(summary[display].to_string(index=False))
    print(json.dumps(diagnostic, indent=2))
    print(f"\nSaved Model 9 validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
