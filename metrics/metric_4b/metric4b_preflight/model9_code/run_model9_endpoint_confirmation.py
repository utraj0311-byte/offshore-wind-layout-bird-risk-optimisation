"""Run the frozen high-sample confirmation of the Model 9 full-run endpoints.

This is not another optimisation.  It evaluates three layouts selected before
the confirmation data are generated, using identical new trajectory
populations within every batch:

* the original Model 8/energy layout (benchmark),
* full-run ``pareto_032`` (highest-AEP reference), and
* full-run ``pareto_037`` (high-energy apparent lower-risk candidate).

The primary gate is a paired 95% Student-t interval for
E_RT_CE(pareto_037) - E_RT_CE(pareto_032).  A trade-off is confirmed only when
the entire interval is below zero.  Rotor transit remains an exposure endpoint,
not a blade-collision or mortality estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import Scenario
from floris_aep import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_TURBINE_REFERENCE_PATH,
    FLORISAEPModel,
)
from model8_meso_response import BehaviourScenario
from model9_batching import summarise_model9_batches
from model9_confirmation import (
    PRIMARY_METRIC,
    candidate_summary_frame,
    confirmation_decision,
)
from model9_optimisation_utils import load_layout_csv, serialisable
from model9_rotor_transit import (
    DiscreteFlightHeightDistribution,
    Model9Population,
    RotorGeometry,
    RotorTransitModel,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = MODULE_DIR / "data" / "model9_confirmation_manifest.json"
DEFAULT_FHD_PATH = MODULE_DIR / "data" / "northern_gannet_Johnston2014_FHD.csv"
DEFAULT_FHD_METADATA_PATH = (
    MODULE_DIR / "data" / "northern_gannet_Johnston2014_FHD_metadata.json"
)
DEFAULT_OUTPUT_PATH = (
    MODULE_DIR / "results" / "model9_endpoint_confirmation_seed30142"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in (
        "floris",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "PyYAML",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not reported"
    return versions


def prepare_output_dir(path: Path, *, make_archive: bool) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {resolved}. Choose a new --output path."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    archive_path = resolved.with_suffix(".zip")
    if make_archive and archive_path.exists():
        raise FileExistsError(
            f"Result archive already exists: {archive_path}. Move it or choose "
            "a new --output path."
        )
    return resolved


def expand_seed_ranges(ranges: list[list[int]]) -> set[int]:
    result: set[int] = set()
    for pair in ranges:
        if len(pair) != 2 or int(pair[1]) < int(pair[0]):
            raise ValueError(f"Invalid forbidden seed range: {pair}")
        result.update(range(int(pair[0]), int(pair[1]) + 1))
    return result


def validate_new_seeds(
    manifest: dict[str, Any],
    *,
    batch_count: int,
    first_trajectory_seed: int,
    first_flight_height_seed: int,
) -> tuple[list[int], list[int]]:
    trajectory_seeds = [first_trajectory_seed + i for i in range(batch_count)]
    height_seeds = [first_flight_height_seed + i for i in range(batch_count)]
    forbidden = manifest["forbidden_reused_seed_ranges"]
    reused_trajectory = sorted(
        set(trajectory_seeds)
        & expand_seed_ranges(forbidden["trajectory_seeds"])
    )
    reused_height = sorted(
        set(height_seeds)
        & expand_seed_ranges(forbidden["flight_height_seeds"])
    )
    if reused_trajectory or reused_height:
        raise ValueError(
            "Confirmation seeds overlap earlier Model 9 data. "
            f"Trajectory overlap={reused_trajectory}; height overlap={reused_height}."
        )
    return trajectory_seeds, height_seeds


def load_locked_layouts(
    manifest: dict[str, Any],
    scenario: Scenario,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    minimum_spacing_m = (
        scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m
    )
    layouts: dict[str, np.ndarray] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for definition in manifest["candidates"]:
        candidate_id = str(definition["candidate_id"])
        path = MODULE_DIR / str(definition["layout_csv"])
        actual_hash = sha256(path)
        if actual_hash != str(definition["layout_sha256"]):
            raise ValueError(
                f"Locked layout checksum changed for {candidate_id}: {actual_hash}."
            )
        layouts[candidate_id] = load_layout_csv(
            path,
            number_of_turbines=scenario.site.n_turbines,
            site_width_m=scenario.site.width_m,
            site_height_m=scenario.site.height_m,
            minimum_spacing_m=minimum_spacing_m,
        )
        definitions[candidate_id] = dict(definition)
    required = {"original_energy_layout", "pareto_032", "pareto_037"}
    if set(layouts) != required:
        raise ValueError(
            f"Confirmation manifest must contain exactly {sorted(required)}."
        )
    return layouts, definitions


def paired_difference_frame(batch_frame: pd.DataFrame) -> pd.DataFrame:
    pivot = batch_frame.pivot(
        index="batch_index", columns="candidate_id", values=PRIMARY_METRIC
    ).sort_index()
    seeds = (
        batch_frame.groupby("batch_index", as_index=True)[
            ["trajectory_seed", "flight_height_seed"]
        ]
        .first()
        .reindex(pivot.index)
    )
    return pd.DataFrame(
        {
            "batch_index": pivot.index.astype(int),
            "trajectory_seed": seeds["trajectory_seed"].astype(int).to_numpy(),
            "flight_height_seed": seeds["flight_height_seed"].astype(int).to_numpy(),
            "E_RT_CE_original_energy_layout": pivot[
                "original_energy_layout"
            ].to_numpy(),
            "E_RT_CE_pareto_032": pivot["pareto_032"].to_numpy(),
            "E_RT_CE_pareto_037": pivot["pareto_037"].to_numpy(),
            "primary_pareto_037_minus_pareto_032": (
                pivot["pareto_037"] - pivot["pareto_032"]
            ).to_numpy(),
            "secondary_pareto_032_minus_original": (
                pivot["pareto_032"] - pivot["original_energy_layout"]
            ).to_numpy(),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm the frozen Model 9 full-run endpoint comparison."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch_count", type=int, default=15)
    parser.add_argument("--trajectories_per_batch", type=int, default=100000)
    parser.add_argument("--trajectory_seed", type=int, default=30142)
    parser.add_argument("--flight_height_seed", type=int, default=40142)
    parser.add_argument("--flight_height_csv", type=Path, default=DEFAULT_FHD_PATH)
    parser.add_argument("--floris_config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--turbine_reference", type=Path, default=DEFAULT_TURBINE_REFERENCE_PATH
    )
    parser.add_argument("--no_zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_count < 15:
        raise ValueError(
            "The pre-specified confirmation requires at least 15 batches."
        )
    if args.trajectories_per_batch < 100000:
        raise ValueError(
            "The pre-specified confirmation requires at least 100,000 "
            "trajectories per batch."
        )
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    trajectory_seeds, height_seeds = validate_new_seeds(
        manifest,
        batch_count=args.batch_count,
        first_trajectory_seed=args.trajectory_seed,
        first_flight_height_seed=args.flight_height_seed,
    )
    output_dir = prepare_output_dir(args.output, make_archive=not args.no_zip)
    started = time.time()
    scenario = Scenario()
    layouts, definitions = load_locked_layouts(manifest, scenario)

    aep_model = FLORISAEPModel(
        scenario,
        args.floris_config,
        turbine_reference_path=args.turbine_reference,
    )
    if aep_model.number_of_conditions != 84:
        raise RuntimeError(
            f"Expected 84 inherited FLORIS conditions, got {aep_model.number_of_conditions}."
        )
    aep_gwh: dict[str, float] = {}
    for candidate_id, layout in layouts.items():
        calculated = aep_model.evaluate(layout)
        expected = float(definitions[candidate_id]["expected_aep_gwh"])
        if not np.isclose(calculated, expected, rtol=0.0, atol=0.01):
            raise RuntimeError(
                f"AEP for {candidate_id} changed from the frozen source run: "
                f"{calculated:.9f} versus {expected:.9f} GWh/year."
            )
        aep_gwh[candidate_id] = calculated
    if not (
        aep_gwh["pareto_032"]
        > aep_gwh["pareto_037"]
        > aep_gwh["original_energy_layout"]
    ):
        raise RuntimeError("The frozen candidate AEP ordering has changed.")

    metadata = json.loads(DEFAULT_FHD_METADATA_PATH.read_text(encoding="utf-8"))
    distribution = DiscreteFlightHeightDistribution.from_csv(
        args.flight_height_csv,
        name="northern_gannet_Johnston2014_Gannet.est",
        source=metadata["primary_citation"],
    )
    if distribution.input_sha256 != metadata["sha256"]:
        raise ValueError(
            "The flight-height CSV does not match the frozen northern-gannet input."
        )
    rotor = RotorGeometry(
        rotor_diameter_m=aep_model.floris_rotor_diameter_m,
        hub_height_m=aep_model.floris_hub_height_m,
    )
    birds = scenario.birds
    behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=rotor.rotor_diameter_m,
        response_probability=birds.response_probability,
        response_radius_D=birds.response_radius_D,
        encounter_radius_D=birds.encounter_radius_D,
        deflection_angle_deg=birds.deflection_angle_deg,
        return_length_D=birds.return_length_D,
        step_size_D=birds.step_size_D,
    )
    transit_model = RotorTransitModel(
        behaviour,
        rotor,
        rotor_plane_heading_deg_toward=birds.heading_deg_toward,
        flight_height_distribution=distribution,
    )

    candidate_ids = [str(item["candidate_id"]) for item in manifest["candidates"]]
    settings: dict[str, Any] = {
        "model": "Model 9 pre-specified endpoint confirmation",
        "endpoint": "rotor-swept-area transit exposure, not collision or mortality",
        "run_status": "confirmation_started",
        "analysis_is_pre_specified": True,
        "manifest": manifest,
        "manifest_sha256": sha256(DEFAULT_MANIFEST_PATH),
        "runtime_versions": runtime_versions(),
        "batch_count": args.batch_count,
        "trajectories_per_batch": args.trajectories_per_batch,
        "total_trajectories_per_layout": (
            args.batch_count * args.trajectories_per_batch
        ),
        "trajectory_seeds": trajectory_seeds,
        "flight_height_seeds": height_seeds,
        "paired_common_random_numbers": (
            "Within every batch, all three layouts use the same entry positions, "
            "meso-response draws, and flight-height draws."
        ),
        "candidate_aep_gwh": aep_gwh,
        "aep_model": aep_model.settings_summary(),
        "flight_height_distribution": distribution.settings(),
        "rotor_geometry": rotor.settings(),
        "behaviour": behaviour.settings(rotor.rotor_diameter_m),
    }
    settings_path = output_dir / "run_settings.json"
    settings_path.write_text(
        json.dumps(serialisable(settings), indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "confirmation_batches_checkpoint.csv"
    for batch_index, (trajectory_seed, flight_height_seed) in enumerate(
        zip(trajectory_seeds, height_seeds)
    ):
        print(
            f"Confirmation batch {batch_index + 1}/{args.batch_count}: "
            f"trajectory seed={trajectory_seed}, "
            f"N={args.trajectories_per_batch:,} per layout.",
            flush=True,
        )
        population = Model9Population.generate_parallel(
            flight_height_distribution=distribution,
            flight_height_seed=flight_height_seed,
            site_width_m=scenario.site.width_m,
            site_height_m=scenario.site.height_m,
            heading_deg_toward=birds.heading_deg_toward,
            number_of_trajectories=args.trajectories_per_batch,
            maximum_response_events=birds.maximum_response_events,
            longitudinal_margin_m=(
                behaviour.response_radius_m + behaviour.return_length_m
            ),
            trajectory_seed=trajectory_seed,
        )
        for candidate_id in candidate_ids:
            combined = summarise_model9_batches(
                transit_model,
                layouts[candidate_id],
                [population],
            )
            if int(combined["N_failed"]) != 0:
                raise RuntimeError(
                    f"Trajectory failures occurred for {candidate_id} in "
                    f"batch {batch_index}."
                )
            batch = dict(combined["batch_summaries"][0])
            batch["batch_index"] = batch_index
            rows.append({"candidate_id": candidate_id, **batch})
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    batch_frame = pd.DataFrame(rows)
    summary = candidate_summary_frame(
        batch_frame,
        candidate_ids=candidate_ids,
        aep_gwh=aep_gwh,
    )
    decision = confirmation_decision(batch_frame, aep_gwh=aep_gwh)
    paired = paired_difference_frame(batch_frame)

    final_batch_path = output_dir / "confirmation_batches.csv"
    batch_frame.to_csv(final_batch_path, index=False)
    summary.to_csv(output_dir / "confirmation_candidate_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_batch_differences.csv", index=False)
    (output_dir / "confirmation_decision.json").write_text(
        json.dumps(serialisable(decision), indent=2), encoding="utf-8"
    )

    elapsed = float(time.time() - started)
    settings.update(
        {
            "run_status": "completed",
            "elapsed_seconds": elapsed,
            "primary_status": decision["primary_status"],
            "primary_tradeoff_gate_passed": decision[
                "primary_tradeoff_gate_passed"
            ],
            "secondary_status": decision["secondary_status"],
        }
    )
    settings_path.write_text(
        json.dumps(serialisable(settings), indent=2), encoding="utf-8"
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    archive_path: Path | None = None
    if not args.no_zip:
        archive_path = output_dir.with_suffix(".zip")
        shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )

    primary = decision["primary_comparison"]
    print("\nPrimary confirmation result")
    print(f"Status: {decision['primary_status']}")
    print(
        "Mean E_RT_CE difference (pareto_037 - pareto_032): "
        f"{primary['mean_first_minus_second']:.9g}"
    )
    print(
        "Paired 95% Student-t interval: "
        f"[{primary['ci_lower']:.9g}, {primary['ci_upper']:.9g}]"
    )
    print(f"Completed in {elapsed / 60.0:.2f} minutes.")
    print(f"Results: {output_dir}")
    if archive_path is not None:
        print(f"Upload this archive for review: {archive_path}")


if __name__ == "__main__":
    main()

