"""Re-evaluate saved Metric 3b layouts with a finer wind-direction grid.

This is a diagnostic only. It reads saved Pareto layouts and calls the existing
``FLORISAEPModel``; it does not run NSGA-II or alter the bird metric.

The default check evaluates saved solutions 46, 25 and 0. It first reproduces
their original 12-sector AEP values and aborts if that consistency check fails.
It then changes only ``scenario.wind.n_sectors`` from 12 to 72, corresponding
to 5-degree direction spacing. The original ``config.py`` therefore continues
to provide the directional weighting and the seven wind-speed bins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOLUTIONS = (46, 25, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate AEP for saved Metric 3b layouts with 5-degree wind "
            "directions; no optimisation is performed."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="bird_layout_opt root containing config.py and floris_aep.py",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Extracted heading225_W6000_seed48_pop160_gen250 directory",
    )
    parser.add_argument(
        "--floris-config",
        type=Path,
        required=True,
        help="Original floris_config/gch_iea15mw.yaml",
    )
    parser.add_argument(
        "--solutions",
        nargs="+",
        type=int,
        default=list(DEFAULT_SOLUTIONS),
        help="Saved solution_index values to evaluate (default: 46 25 0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every saved Pareto layout instead of --solutions",
    )
    parser.add_argument(
        "--direction-step-deg",
        type=float,
        default=5.0,
        help="Fine wind-direction spacing in degrees (default: 5)",
    )
    parser.add_argument(
        "--baseline-tolerance-gwh",
        type=float,
        default=0.01,
        help="Maximum allowed 12-sector reproduction error (default: 0.01 GWh)",
    )
    parser.add_argument(
        "--skip-12-check",
        action="store_true",
        help=(
            "Skip re-evaluation with 12 sectors. Use only after the default "
            "three-layout consistency check has passed in the same environment."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: RUN_DIR/aep_5deg_check.csv)",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def load_saved_front(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    points_path = require_file(run_dir / "pareto_points.csv", "Pareto CSV")
    vectors_path = require_file(
        run_dir / "pareto_decision_vectors.npy", "Pareto decision vectors"
    )
    settings_path = require_file(run_dir / "run_settings.json", "Run settings")

    points = pd.read_csv(points_path)
    vectors = np.load(vectors_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))

    required_columns = {
        "solution_index",
        "aep_gwh",
        "N_eff",
        "number_of_turbines",
        "number_of_merged_intervals",
        "source_evaluation_index",
    }
    missing = required_columns.difference(points.columns)
    if missing:
        raise ValueError(f"pareto_points.csv is missing columns: {sorted(missing)}")
    if len(points) != len(vectors):
        raise ValueError(
            "Pareto row mismatch: "
            f"{len(points)} CSV rows versus {len(vectors)} decision vectors"
        )
    if points["solution_index"].duplicated().any():
        raise ValueError("solution_index must be unique")

    n_turbines = points["number_of_turbines"].astype(int).unique()
    if len(n_turbines) != 1:
        raise ValueError(f"Expected one turbine count, found {n_turbines.tolist()}")
    expected_variables = 2 * int(n_turbines[0])
    if vectors.ndim != 2 or vectors.shape[1] != expected_variables:
        raise ValueError(
            f"Expected decision-vector shape (n, {expected_variables}), "
            f"found {vectors.shape}"
        )
    return points, vectors, settings


def select_rows(
    points: pd.DataFrame,
    vectors: np.ndarray,
    requested: list[int],
    use_all: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    if use_all:
        positions = np.arange(len(points), dtype=int)
    else:
        lookup = {
            int(solution_index): position
            for position, solution_index in enumerate(points["solution_index"])
        }
        unknown = [value for value in requested if value not in lookup]
        if unknown:
            raise ValueError(
                f"Unknown solution_index value(s): {unknown}; available range is "
                f"{min(lookup)} to {max(lookup)}"
            )
        positions = np.asarray([lookup[value] for value in requested], dtype=int)
    return points.iloc[positions].copy().reset_index(drop=True), vectors[positions].copy()


def build_aep_model(
    *,
    Scenario,
    FLORISAEPModel,
    settings: dict,
    floris_config: Path,
    n_turbines: int,
    n_sectors: int,
):
    scenario = Scenario()
    scenario.site.n_turbines = n_turbines
    scenario.site.width_m = float(settings.get("site_width_m", 6000.0))
    scenario.site.height_m = float(settings.get("site_height_m", 6000.0))
    scenario.site.min_spacing_D = float(
        settings.get("problem", {}).get("minimum_spacing_D", 4.0)
    )
    dominant_direction = float(
        settings.get("aep_model", {}).get("dominant_wind_from_deg", 225.0)
    )
    scenario.site.wind_from_deg = dominant_direction
    scenario.wind.n_sectors = int(n_sectors)

    speed_bins = np.asarray(scenario.wind.speed_bins, dtype=float)
    expected_speed_bins = int(
        settings.get("aep_model", {}).get("number_of_speed_bins", len(speed_bins))
    )
    if len(speed_bins) != expected_speed_bins:
        raise ValueError(
            "The active config.py does not reproduce the saved speed-bin count: "
            f"found {len(speed_bins)}, expected {expected_speed_bins}."
        )

    model = FLORISAEPModel(scenario, config_path=floris_config)
    if hasattr(model, "settings_summary"):
        summary = model.settings_summary()
        actual_sectors = int(summary.get("number_of_direction_sectors", n_sectors))
        actual_speeds = int(summary.get("number_of_speed_bins", len(speed_bins)))
        if actual_sectors != n_sectors or actual_speeds != len(speed_bins):
            raise ValueError(
                "FLORISAEPModel settings do not match the requested grid: "
                f"{actual_sectors} directions and {actual_speeds} speeds."
            )
    return model


def evaluate_layouts(model, vectors: np.ndarray, n_turbines: int, label: str) -> np.ndarray:
    values: list[float] = []
    total = len(vectors)
    for position, vector in enumerate(vectors, start=1):
        print(f"[{label}] layout {position}/{total}", flush=True)
        xy = np.asarray(vector, dtype=float).reshape(n_turbines, 2)
        values.append(float(model.evaluate(xy)))
    return np.asarray(values, dtype=float)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    floris_config = require_file(args.floris_config, "FLORIS configuration")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_dir}")
    for required_module in ("config.py", "floris_aep.py"):
        require_file(project_root / required_module, required_module)

    step = float(args.direction_step_deg)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("--direction-step-deg must be positive and finite")
    fine_sector_count_float = 360.0 / step
    fine_sector_count = int(round(fine_sector_count_float))
    if not np.isclose(fine_sector_count, fine_sector_count_float, atol=1.0e-12):
        raise ValueError("--direction-step-deg must divide 360 exactly")

    sys.path.insert(0, str(project_root))
    from config import Scenario
    from floris_aep import FLORISAEPModel

    points, vectors, settings = load_saved_front(run_dir)
    selected, selected_vectors = select_rows(
        points, vectors, list(args.solutions), args.all
    )
    n_turbines = int(selected["number_of_turbines"].iloc[0])

    saved_sector_count = int(
        settings.get("aep_model", {}).get("number_of_direction_sectors", 12)
    )
    if saved_sector_count != 12:
        raise ValueError(
            f"Expected a 12-sector saved run, found {saved_sector_count} sectors"
        )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "aep_5deg_check.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    saved_aep = selected["aep_gwh"].to_numpy(dtype=float)
    if args.skip_12_check:
        recomputed_12 = np.full(len(selected), np.nan)
        baseline_error = np.full(len(selected), np.nan)
    else:
        model_12 = build_aep_model(
            Scenario=Scenario,
            FLORISAEPModel=FLORISAEPModel,
            settings=settings,
            floris_config=floris_config,
            n_turbines=n_turbines,
            n_sectors=saved_sector_count,
        )
        recomputed_12 = evaluate_layouts(
            model_12, selected_vectors, n_turbines, "12 sectors"
        )
        baseline_error = recomputed_12 - saved_aep
        maximum_error = float(np.max(np.abs(baseline_error)))
        print(f"Maximum 12-sector reproduction error: {maximum_error:.9f} GWh")
        if maximum_error > args.baseline_tolerance_gwh:
            failed = selected[
                [
                    "solution_index",
                    "N_eff",
                    "number_of_merged_intervals",
                    "source_evaluation_index",
                ]
            ].copy()
            failed["aep_saved_12sector_gwh"] = saved_aep
            failed["aep_recomputed_12sector_gwh"] = recomputed_12
            failed["baseline_error_gwh"] = baseline_error
            failed["aep_5deg_gwh"] = np.nan
            failed.to_csv(output, index=False)
            raise RuntimeError(
                "The original 12-sector AEP was not reproduced within "
                f"{args.baseline_tolerance_gwh:g} GWh. The 5-degree run was "
                f"not started. Diagnostic rows were written to {output}."
            )

    model_fine = build_aep_model(
        Scenario=Scenario,
        FLORISAEPModel=FLORISAEPModel,
        settings=settings,
        floris_config=floris_config,
        n_turbines=n_turbines,
        n_sectors=fine_sector_count,
    )
    fine_aep = evaluate_layouts(
        model_fine, selected_vectors, n_turbines, f"{step:g}-degree"
    )

    result = selected[
        [
            "solution_index",
            "N_eff",
            "number_of_merged_intervals",
            "source_evaluation_index",
        ]
    ].copy()
    result["aep_saved_12sector_gwh"] = saved_aep
    result["aep_recomputed_12sector_gwh"] = recomputed_12
    result["baseline_error_gwh"] = baseline_error
    result["fine_direction_step_deg"] = step
    result["fine_direction_sector_count"] = fine_sector_count
    result["aep_5deg_gwh"] = fine_aep
    result["aep_change_5deg_minus_saved_gwh"] = fine_aep - saved_aep
    result["aep_change_percent"] = 100.0 * (fine_aep - saved_aep) / saved_aep

    reference_position = int(np.argmax(saved_aep))
    reference_saved = float(saved_aep[reference_position])
    reference_fine = float(fine_aep[reference_position])
    result["aep_loss_vs_selected_energy_12sector_percent"] = (
        100.0 * (reference_saved - saved_aep) / reference_saved
    )
    result["aep_loss_vs_selected_energy_5deg_percent"] = (
        100.0 * (reference_fine - fine_aep) / reference_fine
    )
    result["change_in_tradeoff_percentage_points"] = (
        result["aep_loss_vs_selected_energy_5deg_percent"]
        - result["aep_loss_vs_selected_energy_12sector_percent"]
    )
    result["aep_rank_5deg"] = (
        pd.Series(fine_aep).rank(method="min", ascending=False).astype(int)
    )
    result.to_csv(output, index=False)

    display_columns = [
        "solution_index",
        "N_eff",
        "aep_saved_12sector_gwh",
        "aep_5deg_gwh",
        "aep_change_5deg_minus_saved_gwh",
        "aep_loss_vs_selected_energy_12sector_percent",
        "aep_loss_vs_selected_energy_5deg_percent",
    ]
    print("\nCompleted saved-layout diagnostic; no optimisation was run.")
    print(result[display_columns].to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
