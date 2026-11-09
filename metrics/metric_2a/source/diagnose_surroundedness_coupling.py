import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import Scenario
from coupled_risk import SurroundednessRiskModel

from run_validation import (
    min_spacing,
    count_spacing_violations,
    count_boundary_violations,
)


def load_layout(layout_path, scenario):
    """
    Load one saved turbine layout.

    The file must contain x_m and y_m. Turbine rows are
    sorted by turbine_id when that column is available.
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
            f"Expected shape {expected_shape}, "
            f"received {xy.shape} from {layout_path}."
        )

    if not np.all(np.isfinite(xy)):
        raise ValueError(
            f"Non-finite coordinates found in {layout_path}."
        )

    return xy


def nearest_neighbour_distance(
    xy,
    target_index,
):
    """
    Return the nearest distance from the target turbine
    to any other turbine.
    """
    displacement = (
        xy
        - xy[target_index]
    )

    distances = np.linalg.norm(
        displacement,
        axis=1,
    )

    distances = np.delete(
        distances,
        target_index,
    )

    return float(
        np.min(distances)
    )


def evaluate_target(
    xy,
    scenario,
    target_index=0,
    interaction_scale_D=4.0,
):
    """
    Evaluate surroundedness and geometric diagnostics
    for one target turbine.
    """
    model = SurroundednessRiskModel(
        scenario=scenario,
        target_index=target_index,
        interaction_scale_D=interaction_scale_D,
        mode="interior_risky",
    )

    diagnostics = model.diagnostics(xy)

    required_spacing_m = (
        scenario.site.min_spacing_D
        * scenario.turbine.rotor_diameter_m
    )

    return {
        "target_index": target_index,
        "surroundedness": float(
            diagnostics["surroundedness"]
        ),
        "risk": float(
            diagnostics["total_risk"]
        ),
        "distance_to_centroid_m": float(
            diagnostics[
                "distance_to_centroid_m"
            ]
        ),
        "radial_rank": int(
            diagnostics["radial_rank"]
        ),
        "nearest_neighbour_distance_m": (
            nearest_neighbour_distance(
                xy,
                target_index,
            )
        ),
        "neighbours_within_5D": int(
            diagnostics[
                "neighbour_count_within_5D"
            ]
        ),
        "minimum_layout_spacing_m": float(
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
    }


def all_turbine_metrics(
    xy,
    scenario,
    interaction_scale_D,
):
    """
    Evaluate surroundedness and geometric position for
    every turbine in one layout.
    """
    rows = []

    for turbine_index in range(
        scenario.site.n_turbines
    ):
        row = evaluate_target(
            xy=xy,
            scenario=scenario,
            target_index=turbine_index,
            interaction_scale_D=(
                interaction_scale_D
            ),
        )

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    return dataframe.sort_values(
        "surroundedness",
        ascending=True,
    ).reset_index(drop=True)


def save_layout(
    xy,
    output_path,
):
    """
    Save a generated counterfactual layout.
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether surroundedness-risk reduction "
            "comes from target movement, neighbour "
            "reorganisation, or both."
        )
    )

    parser.add_argument(
        "--energy_layout",
        type=Path,
        default=(
            Path("results")
            / "coupled_multiobjective"
            / (
                "surroundedness_interior_risky"
                "_scale4p0D_seed42_pop60_gen60"
            )
            / "layout_energy_optimal.csv"
        ),
    )

    parser.add_argument(
        "--risk_layout",
        type=Path,
        default=(
            Path("results")
            / "coupled_multiobjective"
            / (
                "surroundedness_interior_risky"
                "_scale4p0D_seed42_pop60_gen60"
            )
            / "layout_risk_optimal.csv"
        ),
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
        "--target",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--scale_D",
        type=float,
        default=4.0,
    )

    args = parser.parse_args()

    scenario = Scenario()

    energy_xy = load_layout(
        args.energy_layout,
        scenario,
    )

    risk_xy = load_layout(
        args.risk_layout,
        scenario,
    )

    aep_only_xy = load_layout(
        args.aep_only_layout,
        scenario,
    )

    # -------------------------------------------------
    # Four factorial/counterfactual combinations
    # -------------------------------------------------

    # Energy target + energy neighbours.
    energy_full = energy_xy.copy()

    # Risk target + risk neighbours.
    risk_full = risk_xy.copy()

    # Risk-endpoint target position combined with
    # energy-endpoint neighbours.
    risk_target_energy_neighbours = (
        energy_xy.copy()
    )

    risk_target_energy_neighbours[
        args.target
    ] = risk_xy[args.target]

    # Energy-endpoint target position combined with
    # risk-endpoint neighbours.
    energy_target_risk_neighbours = (
        risk_xy.copy()
    )

    energy_target_risk_neighbours[
        args.target
    ] = energy_xy[args.target]

    counterfactual_layouts = {
        "energy_target_energy_neighbours": (
            energy_full
        ),
        "risk_target_energy_neighbours": (
            risk_target_energy_neighbours
        ),
        "energy_target_risk_neighbours": (
            energy_target_risk_neighbours
        ),
        "risk_target_risk_neighbours": (
            risk_full
        ),
    }

    counterfactual_rows = []

    for name, xy in (
        counterfactual_layouts.items()
    ):
        result = evaluate_target(
            xy=xy,
            scenario=scenario,
            target_index=args.target,
            interaction_scale_D=args.scale_D,
        )

        result["layout_combination"] = name

        counterfactual_rows.append(
            result
        )

    counterfactual_dataframe = pd.DataFrame(
        counterfactual_rows
    )

    # -------------------------------------------------
    # AEP-only relabelling null
    # -------------------------------------------------

    aep_only_metrics = all_turbine_metrics(
        xy=aep_only_xy,
        scenario=scenario,
        interaction_scale_D=args.scale_D,
    )

    designated_aep_row = (
        aep_only_metrics[
            aep_only_metrics["target_index"]
            == args.target
        ]
        .iloc[0]
    )

    minimum_aep_row = (
        aep_only_metrics.iloc[0]
    )

    # -------------------------------------------------
    # Output folder
    # -------------------------------------------------

    scale_label = str(
        args.scale_D
    ).replace(".", "p")

    output_dir = (
        Path("results")
        / "coupled_diagnostics"
        / (
            "surroundedness_counterfactual"
            f"_scale{scale_label}D"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    counterfactual_dataframe.to_csv(
        output_dir
        / "counterfactual_results.csv",
        index=False,
    )

    aep_only_metrics.to_csv(
        output_dir
        / "aep_only_all_turbine_metrics.csv",
        index=False,
    )

    for name, xy in (
        counterfactual_layouts.items()
    ):
        save_layout(
            xy,
            output_dir
            / f"{name}.csv",
        )

    # -------------------------------------------------
    # Simple sequential comparisons
    # -------------------------------------------------

    result_lookup = {
        row["layout_combination"]: row
        for row in counterfactual_rows
    }

    s_ee = result_lookup[
        "energy_target_energy_neighbours"
    ]["surroundedness"]

    s_re = result_lookup[
        "risk_target_energy_neighbours"
    ]["surroundedness"]

    s_er = result_lookup[
        "energy_target_risk_neighbours"
    ]["surroundedness"]

    s_rr = result_lookup[
        "risk_target_risk_neighbours"
    ]["surroundedness"]

    total_reduction = (
        s_ee
        - s_rr
    )

    summary = {
        "target_index": args.target,
        "interaction_scale_D": args.scale_D,
        "energy_full_surroundedness": s_ee,
        "risk_target_energy_neighbours_surroundedness": (
            s_re
        ),
        "energy_target_risk_neighbours_surroundedness": (
            s_er
        ),
        "risk_full_surroundedness": s_rr,
        "total_surroundedness_reduction": (
            total_reduction
        ),
        "target_move_reduction_with_energy_neighbours": (
            s_ee
            - s_re
        ),
        "neighbour_reorganisation_reduction_after_target_move": (
            s_re
            - s_rr
        ),
        "neighbour_reorganisation_reduction_with_energy_target": (
            s_ee
            - s_er
        ),
        "target_move_reduction_after_neighbour_reorganisation": (
            s_er
            - s_rr
        ),
        "aep_only_designated_target_surroundedness": float(
            designated_aep_row["surroundedness"]
        ),
        "aep_only_minimum_surroundedness": float(
            minimum_aep_row["surroundedness"]
        ),
        "aep_only_minimum_surroundedness_turbine": int(
            minimum_aep_row["target_index"]
        ),
        "aep_only_designated_target_radial_rank": int(
            designated_aep_row["radial_rank"]
        ),
        "aep_only_minimum_risk_turbine_radial_rank": int(
            minimum_aep_row["radial_rank"]
        ),
        "warning": (
            "Counterfactual effects are path-dependent. "
            "The two orderings are reported separately."
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

    # -------------------------------------------------
    # Terminal report
    # -------------------------------------------------

    print("")
    print("Surroundedness coupling diagnostic")
    print("----------------------------------")
    print("")

    print("Counterfactual combinations:")
    print(
        counterfactual_dataframe[
            [
                "layout_combination",
                "surroundedness",
                "distance_to_centroid_m",
                "nearest_neighbour_distance_m",
                "minimum_layout_spacing_m",
                "spacing_violations",
            ]
        ].to_string(index=False)
    )

    print("")
    print("Sequential decomposition:")
    print(
        "Full reduction: "
        f"{total_reduction:.6f}"
    )
    print(
        "Target move with energy neighbours: "
        f"{s_ee - s_re:.6f}"
    )
    print(
        "Additional neighbour effect after target move: "
        f"{s_re - s_rr:.6f}"
    )
    print(
        "Neighbour effect with energy target: "
        f"{s_ee - s_er:.6f}"
    )
    print(
        "Additional target effect after neighbour move: "
        f"{s_er - s_rr:.6f}"
    )

    print("")
    print("AEP-only relabelling null:")
    print(
        "Designated turbine 0 surroundedness: "
        f"{designated_aep_row['surroundedness']:.6f}"
    )
    print(
        "Minimum surroundedness in AEP-only layout: "
        f"{minimum_aep_row['surroundedness']:.6f}"
    )
    print(
        "Turbine with minimum surroundedness: "
        f"{int(minimum_aep_row['target_index'])}"
    )
    print(
        "Designated turbine radial rank: "
        f"{int(designated_aep_row['radial_rank'])}"
    )

    print("")
    print(
        f"Saved outputs to: {output_dir}"
    )
    print("")


if __name__ == "__main__":
    main()