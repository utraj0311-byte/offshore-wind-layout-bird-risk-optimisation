"""Run deterministic pre-optimisation checks for the Shielding Model.

The validation is intentionally small: the three exact two-turbine cases from
the model definition, four interpretable 12-turbine layouts including the
two-lane geometric minimum, one direction check, and (optionally) the supplied
energy-endpoint layout.  No bird trajectories or random sampling are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shielding_model import (
    N_EFF_DECIMALS,
    crossflight_unit_vector,
    evaluate_shielding,
    heading_unit_vector,
)


def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    delta = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    return distances[np.triu_indices(len(points), k=1)]


def validate_layout(
    xy: np.ndarray,
    *,
    number_of_turbines: int,
    site_width_m: float,
    site_height_m: float,
    minimum_spacing_m: float,
) -> None:
    points = np.asarray(xy, dtype=float)
    if points.shape != (number_of_turbines, 2):
        raise ValueError(
            f"Expected {(number_of_turbines, 2)} coordinates, received {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("Layout coordinates must be finite.")
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] > site_width_m)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] > site_height_m)
    ):
        raise ValueError("A validation layout lies outside the site boundary.")
    minimum_actual = float(np.min(pairwise_distances(points)))
    if minimum_actual + 1.0e-9 < minimum_spacing_m:
        raise ValueError(
            "A validation layout violates minimum spacing: "
            f"{minimum_actual:.6f} < {minimum_spacing_m:.6f} m."
        )


def load_layout_csv(path: Path, number_of_turbines: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Layout file not found: {path}")
    dataframe = pd.read_csv(path)
    if "turbine_id" in dataframe.columns:
        dataframe = dataframe.sort_values("turbine_id")
    if not {"x_m", "y_m"}.issubset(dataframe.columns):
        raise ValueError(f"{path} must contain x_m and y_m columns.")
    coordinates = dataframe[["x_m", "y_m"]].to_numpy(dtype=float)
    if coordinates.shape != (number_of_turbines, 2):
        raise ValueError(
            f"Expected {(number_of_turbines, 2)} coordinates, "
            f"received {coordinates.shape}."
        )
    return coordinates


def make_fixed_layouts(
    *,
    site_width_m: float,
    site_height_m: float,
    bird_heading_deg_toward: float,
    minimum_spacing_m: float,
) -> dict[str, np.ndarray]:
    """Create four feasible 12-turbine layouts in the bird reference frame.

    ``two_aligned_lanes`` is a feasibility witness for the locked-case
    geometric minimum ``N_eff = 2``.  ``three_aligned_lanes`` and
    ``along_flight_stagger`` differ only in their downstream positions and
    must therefore have identical shielding.  ``four_separate_lanes`` has one
    additional exposed lane.
    """

    centre = np.array([site_width_m / 2.0, site_height_m / 2.0], dtype=float)
    along = heading_unit_vector(bird_heading_deg_toward)
    cross = crossflight_unit_vector(bird_heading_deg_toward)
    spacing = 1.08 * minimum_spacing_m

    def convert(along_cross: list[tuple[float, float]]) -> np.ndarray:
        coordinates = np.asarray(
            [centre + along_m * along + cross_m * cross for along_m, cross_m in along_cross],
            dtype=float,
        )
        validate_layout(
            coordinates,
            number_of_turbines=12,
            site_width_m=site_width_m,
            site_height_m=site_height_m,
            minimum_spacing_m=minimum_spacing_m,
        )
        return coordinates

    two_aligned_lanes = [
        (along_index * spacing, cross_index * spacing)
        for cross_index in (-0.5, 0.5)
        for along_index in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
    ]

    three_aligned_lanes = [
        (along_index * spacing, cross_index * spacing)
        for cross_index in (-1.0, 0.0, 1.0)
        for along_index in (-1.5, -0.5, 0.5, 1.5)
    ]

    along_flight_stagger: list[tuple[float, float]] = []
    for lane_index, cross_index in enumerate((-1.0, 0.0, 1.0)):
        offset = (-0.25 if lane_index % 2 == 0 else 0.25) * spacing
        for along_index in (-1.5, -0.5, 0.5, 1.5):
            along_flight_stagger.append(
                (along_index * spacing + offset, cross_index * spacing)
            )

    four_separate_lanes = [
        (along_index * spacing, cross_index * spacing)
        for cross_index in (-1.75, -0.75, 0.75, 1.75)
        for along_index in (-1.0, 0.0, 1.0)
    ]

    return {
        "two_aligned_lanes": convert(two_aligned_lanes),
        "three_aligned_lanes": convert(three_aligned_lanes),
        "along_flight_stagger": convert(along_flight_stagger),
        "four_separate_lanes": convert(four_separate_lanes),
    }


def _result_row(
    *,
    check: str,
    layout: str,
    result,
    expected_N_eff: float | None,
    minimum_spacing_m: float | None,
) -> dict:
    passed = (
        None
        if expected_N_eff is None
        else bool(np.isclose(result.N_eff, expected_N_eff, rtol=0.0, atol=1.0e-9))
    )
    return {
        "check": check,
        "layout": layout,
        **result.summary(),
        "expected_N_eff": expected_N_eff,
        "check_passed": passed,
        "minimum_pair_spacing_m": minimum_spacing_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the deterministic directional Shielding Model."
    )
    parser.add_argument("--energy_seed_layout", type=Path)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/shielding_validation")
    )
    parser.add_argument("--bird_heading_deg", type=float, default=225.0)
    parser.add_argument("--rotor_diameter_m", type=float, default=242.24)
    parser.add_argument("--corridor_width_m", type=float, default=6000.0)
    parser.add_argument("--site_width_m", type=float, default=6000.0)
    parser.add_argument("--site_height_m", type=float, default=6000.0)
    parser.add_argument("--minimum_spacing_D", type=float, default=4.0)
    args = parser.parse_args()

    if args.site_width_m <= 0.0 or args.site_height_m <= 0.0:
        raise ValueError("Site dimensions must be positive.")
    if args.minimum_spacing_D <= 0.0:
        raise ValueError("minimum_spacing_D must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    diameter = float(args.rotor_diameter_m)
    minimum_spacing = float(args.minimum_spacing_D * diameter)
    direction = heading_unit_vector(args.bird_heading_deg)
    crossflight = crossflight_unit_vector(args.bird_heading_deg)

    rows: list[dict] = []

    # Exact two-turbine cases.  A 4D along-flight separation keeps all three
    # layouts feasible while the cross-flight offset controls their shielding.
    for name, cross_offset_D, expected in (
        ("perfect_alignment", 0.0, 1.0),
        ("half_diameter_offset", 0.5, 1.5),
        ("one_diameter_offset", 1.0, 2.0),
    ):
        coordinates = np.vstack(
            (
                np.zeros(2, dtype=float),
                4.0 * diameter * direction + cross_offset_D * diameter * crossflight,
            )
        )
        result = evaluate_shielding(
            coordinates,
            rotor_diameter_m=diameter,
            bird_heading_deg_toward=args.bird_heading_deg,
            corridor_width_m=args.corridor_width_m,
        )
        rows.append(
            _result_row(
                check="exact_two_turbine_geometry",
                layout=name,
                result=result,
                expected_N_eff=expected,
                minimum_spacing_m=float(np.min(pairwise_distances(coordinates))),
            )
        )

    fixed_layouts = make_fixed_layouts(
        site_width_m=args.site_width_m,
        site_height_m=args.site_height_m,
        bird_heading_deg_toward=args.bird_heading_deg,
        minimum_spacing_m=minimum_spacing,
    )
    fixed_expected = {
        "two_aligned_lanes": 2.0,
        "three_aligned_lanes": 3.0,
        "along_flight_stagger": 3.0,
        "four_separate_lanes": 4.0,
    }
    coordinate_rows: list[dict] = []
    for layout_name, coordinates in fixed_layouts.items():
        result = evaluate_shielding(
            coordinates,
            rotor_diameter_m=diameter,
            bird_heading_deg_toward=args.bird_heading_deg,
            corridor_width_m=args.corridor_width_m,
        )
        rows.append(
            _result_row(
                check="fixed_12_turbine_layout",
                layout=layout_name,
                result=result,
                expected_N_eff=fixed_expected[layout_name],
                minimum_spacing_m=float(np.min(pairwise_distances(coordinates))),
            )
        )
        for turbine_id, (point, projected) in enumerate(
            zip(coordinates, result.projected_centres_m)
        ):
            coordinate_rows.append(
                {
                    "layout": layout_name,
                    "turbine_id": turbine_id,
                    "x_m": point[0],
                    "y_m": point[1],
                    "projected_crossflight_m": projected,
                }
            )

    direction_layout = fixed_layouts["three_aligned_lanes"]
    perpendicular_result = evaluate_shielding(
        direction_layout,
        rotor_diameter_m=diameter,
        bird_heading_deg_toward=args.bird_heading_deg + 90.0,
        corridor_width_m=args.corridor_width_m,
    )
    rows.append(
        _result_row(
            check="direction_dependence",
            layout="three_aligned_lanes_at_heading_plus_90_deg",
            result=perpendicular_result,
            expected_N_eff=4.0,
            minimum_spacing_m=float(np.min(pairwise_distances(direction_layout))),
        )
    )

    energy_summary = None
    if args.energy_seed_layout is not None:
        energy_xy = load_layout_csv(args.energy_seed_layout, number_of_turbines=12)
        validate_layout(
            energy_xy,
            number_of_turbines=12,
            site_width_m=args.site_width_m,
            site_height_m=args.site_height_m,
            minimum_spacing_m=minimum_spacing,
        )
        energy_result = evaluate_shielding(
            energy_xy,
            rotor_diameter_m=diameter,
            bird_heading_deg_toward=args.bird_heading_deg,
            corridor_width_m=args.corridor_width_m,
        )
        energy_summary = energy_result.summary()
        rows.append(
            _result_row(
                check="supplied_layout",
                layout="energy_endpoint",
                result=energy_result,
                expected_N_eff=None,
                minimum_spacing_m=float(np.min(pairwise_distances(energy_xy))),
            )
        )
        pd.DataFrame(
            {
                "turbine_id": np.arange(12, dtype=int),
                "x_m": energy_xy[:, 0],
                "y_m": energy_xy[:, 1],
                "projected_crossflight_m": energy_result.projected_centres_m,
                "interval_start_m": energy_result.rotor_intervals_m[:, 0],
                "interval_end_m": energy_result.rotor_intervals_m[:, 1],
            }
        ).sort_values("projected_crossflight_m").to_csv(
            output_dir / "energy_endpoint_projection.csv", index=False
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "shielding_validation_summary.csv", index=False)
    pd.DataFrame(coordinate_rows).to_csv(
        output_dir / "fixed_layout_coordinates.csv", index=False
    )

    required_checks = summary["check_passed"].dropna().astype(bool)
    if not bool(required_checks.all()):
        failed = summary.loc[summary["check_passed"] == False, ["layout", "N_eff"]]
        raise AssertionError(f"Shielding validation failed:\n{failed.to_string(index=False)}")

    settings = {
        "model": "Directional analytical Shielding Model",
        "scientific_status": "literature-informed geometric surrogate developed in this study",
        "primary_metric": "N_eff = L_exp / D",
        "secondary_metric": "P_enc = L_exp / W",
        "bird_heading_convention": "TOWARD; 0=east (+x), 90=north (+y)",
        "bird_heading_deg_toward": float(args.bird_heading_deg) % 360.0,
        "rotor_diameter_m": diameter,
        "corridor_width_m": float(args.corridor_width_m),
        "site_width_m": float(args.site_width_m),
        "site_height_m": float(args.site_height_m),
        "minimum_spacing_D": float(args.minimum_spacing_D),
        "minimum_spacing_m": minimum_spacing,
        "locked_case_validated_minimum_N_eff": 2.0,
        "N_eff_numerical_decimals": N_EFF_DECIMALS,
        "random_sampling_used": False,
        "all_required_checks_passed": True,
        "energy_seed_layout": (
            None if args.energy_seed_layout is None else str(args.energy_seed_layout)
        ),
        "energy_endpoint_result": energy_summary,
        "not_modelled": [
            "bird response or avoidance",
            "individual trajectories",
            "encounter intensity",
            "flight height",
            "rotor transit",
            "collision probability",
            "mortality",
        ],
    }
    (output_dir / "run_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    display_columns = [
        "check",
        "layout",
        "N_eff",
        "unique_exposed_width_m",
        "P_enc",
        "P_free",
        "expected_N_eff",
        "check_passed",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"\nAll required deterministic checks passed. Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
