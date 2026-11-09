"""Run only the targeted p038 extension to the completed Model 9 confirmation.

The original confirmation already evaluated pareto_032 on 15 batches of
100,000 trajectories.  This runner regenerates those exact common-random-
number populations, evaluates only pareto_038, and pairs the new values with
the frozen p032 rows.  It therefore adds 1.5 million, rather than repeating all
4.5 million, layout-trajectory simulations.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from model9_confirmation import candidate_summary_frame
from model9_optimisation_utils import load_layout_csv, serialisable
from model9_p038_extension import (
    combine_p032_p038_batches,
    extension_decision,
    load_prior_confirmation_archive,
    paired_difference_frame,
    sha256,
    validate_prior_confirmation,
)
from model9_rotor_transit import (
    DiscreteFlightHeightDistribution,
    Model9Population,
    RotorGeometry,
    RotorTransitModel,
)
from run_model9_endpoint_confirmation import runtime_versions


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = MODULE_DIR / "data" / "model9_p038_extension_manifest.json"
DEFAULT_PRIOR_CONFIRMATION_PATH = (
    MODULE_DIR / "data" / "model9_endpoint_confirmation_seed30142.zip"
)
DEFAULT_FHD_PATH = MODULE_DIR / "data" / "northern_gannet_Johnston2014_FHD.csv"
DEFAULT_FHD_METADATA_PATH = (
    MODULE_DIR / "data" / "northern_gannet_Johnston2014_FHD_metadata.json"
)
DEFAULT_OUTPUT_PATH = (
    MODULE_DIR / "results" / "model9_p038_confirmation_seed30142"
)


def prepare_output_dir(path: Path, *, make_archive: bool, resume: bool) -> Path:
    resolved = path.resolve()
    archive_path = resolved.with_suffix(".zip")
    if archive_path.exists():
        raise FileExistsError(
            f"Result archive already exists: {archive_path}. Move it or choose "
            "a new --output path."
        )
    if resume:
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Cannot resume because the output directory is absent: {resolved}"
            )
    else:
        if resolved.exists() and any(resolved.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {resolved}. Choose a new "
                "--output path or use --resume after an interrupted run."
            )
        resolved.mkdir(parents=True, exist_ok=True)
    if make_archive and archive_path.exists():
        raise FileExistsError(f"Result archive already exists: {archive_path}")
    return resolved


def load_checkpoint(
    checkpoint_path: Path,
    *,
    resume: bool,
    trajectory_seeds: list[int],
    height_seeds: list[int],
    trajectories_per_batch: int,
) -> list[dict[str, Any]]:
    if not resume:
        return []
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume because checkpoint is absent: {checkpoint_path}"
        )
    frame = pd.read_csv(checkpoint_path)
    required = {
        "candidate_id",
        "batch_index",
        "trajectory_seed",
        "flight_height_seed",
        "N_started",
        "N_completed",
        "N_failed",
        "E_RT_CE",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Checkpoint is missing columns: {missing}")
    if set(frame["candidate_id"].astype(str)) != {"pareto_038"}:
        raise ValueError("Checkpoint contains an unexpected candidate.")
    frame = frame.sort_values("batch_index")
    indices = frame["batch_index"].astype(int).tolist()
    if indices != list(range(len(indices))):
        raise ValueError("Checkpoint batches must be a contiguous prefix from zero.")
    if len(indices) > len(trajectory_seeds):
        raise ValueError("Checkpoint contains too many batches.")
    for position, row in frame.iterrows():
        batch_index = int(row["batch_index"])
        if int(row["trajectory_seed"]) != trajectory_seeds[batch_index]:
            raise ValueError("Checkpoint trajectory seed changed.")
        if int(row["flight_height_seed"]) != height_seeds[batch_index]:
            raise ValueError("Checkpoint flight-height seed changed.")
        if int(row["N_started"]) != trajectories_per_batch:
            raise ValueError("Checkpoint trajectory count changed.")
        if int(row["N_completed"]) != trajectories_per_batch:
            raise ValueError("Checkpoint contains incomplete trajectories.")
        if int(row["N_failed"]) != 0:
            raise ValueError("Checkpoint contains failed trajectories.")
    return frame.to_dict(orient="records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate only pareto_038 on the frozen seed-30142 confirmation "
            "batches and compare it with the archived pareto_032 results."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--prior_confirmation",
        type=Path,
        default=DEFAULT_PRIOR_CONFIRMATION_PATH,
    )
    parser.add_argument("--flight_height_csv", type=Path, default=DEFAULT_FHD_PATH)
    parser.add_argument("--floris_config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--turbine_reference", type=Path, default=DEFAULT_TURBINE_REFERENCE_PATH
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from p038_confirmation_batches_checkpoint.csv.",
    )
    parser.add_argument("--no_zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    prior_spec = manifest["prior_confirmation"]
    prior_batches, prior_settings, prior_decision = load_prior_confirmation_archive(
        args.prior_confirmation,
        expected_sha256=str(prior_spec["archive_sha256"]),
    )
    p032 = validate_prior_confirmation(
        prior_batches,
        prior_settings,
        prior_decision,
        manifest,
    )

    required_numpy = str(prior_spec["required_numpy_version"])
    if np.__version__ != required_numpy:
        raise RuntimeError(
            "This extension must regenerate the exact previous random populations. "
            f"Use NumPy {required_numpy}; the current version is {np.__version__}."
        )

    trajectory_seeds = [int(x) for x in prior_spec["trajectory_seeds"]]
    height_seeds = [int(x) for x in prior_spec["flight_height_seeds"]]
    batch_count = int(prior_spec["batch_count"])
    trajectories_per_batch = int(prior_spec["trajectories_per_batch"])
    output_dir = prepare_output_dir(
        args.output,
        make_archive=not args.no_zip,
        resume=args.resume,
    )
    checkpoint_path = output_dir / "p038_confirmation_batches_checkpoint.csv"
    rows = load_checkpoint(
        checkpoint_path,
        resume=args.resume,
        trajectory_seeds=trajectory_seeds,
        height_seeds=height_seeds,
        trajectories_per_batch=trajectories_per_batch,
    )
    started = time.time()

    scenario = Scenario()
    p038_spec = manifest["pareto_038"]
    layout_path = MODULE_DIR / str(p038_spec["layout_csv"])
    if sha256(layout_path) != str(p038_spec["layout_sha256"]):
        raise ValueError("The frozen pareto_038 layout checksum changed.")
    minimum_spacing_m = (
        scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m
    )
    p038_layout = load_layout_csv(
        layout_path,
        number_of_turbines=scenario.site.n_turbines,
        site_width_m=scenario.site.width_m,
        site_height_m=scenario.site.height_m,
        minimum_spacing_m=minimum_spacing_m,
    )

    aep_model = FLORISAEPModel(
        scenario,
        args.floris_config,
        turbine_reference_path=args.turbine_reference,
    )
    aep_settings = aep_model.settings_summary()
    if aep_model.number_of_conditions != 84:
        raise RuntimeError(
            f"Expected 84 inherited FLORIS conditions, got {aep_model.number_of_conditions}."
        )
    if str(aep_settings["config_sha256"]) != str(
        prior_spec["required_floris_config_sha256"]
    ):
        raise RuntimeError("The FLORIS configuration differs from the prior run.")
    p038_aep = float(aep_model.evaluate(p038_layout))
    if not np.isclose(
        p038_aep,
        float(p038_spec["expected_aep_gwh"]),
        rtol=0.0,
        atol=0.01,
    ):
        raise RuntimeError(
            "AEP for pareto_038 changed from the frozen full run: "
            f"{p038_aep:.9f} versus {p038_spec['expected_aep_gwh']:.9f} GWh/year."
        )
    p032_aep = float(prior_spec["pareto_032_aep_gwh"])
    if not p038_aep < p032_aep:
        raise RuntimeError("The frozen p038-versus-p032 AEP ordering changed.")

    metadata = json.loads(DEFAULT_FHD_METADATA_PATH.read_text(encoding="utf-8"))
    distribution = DiscreteFlightHeightDistribution.from_csv(
        args.flight_height_csv,
        name="northern_gannet_Johnston2014_Gannet.est",
        source=metadata["primary_citation"],
    )
    if distribution.input_sha256 != str(prior_spec["required_fhd_sha256"]):
        raise ValueError(
            "The flight-height CSV differs from the completed confirmation."
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

    settings: dict[str, Any] = {
        "model": "Model 9 targeted p038 endpoint confirmation extension",
        "endpoint": "rotor-swept-area transit exposure, not collision or mortality",
        "run_status": "extension_started",
        "manifest": manifest,
        "manifest_sha256": sha256(DEFAULT_MANIFEST_PATH),
        "prior_confirmation_archive": str(args.prior_confirmation.resolve()),
        "prior_confirmation_archive_sha256": sha256(args.prior_confirmation),
        "runtime_versions": runtime_versions(),
        "batch_count": batch_count,
        "trajectories_per_batch": trajectories_per_batch,
        "new_layout_trajectory_simulations": batch_count * trajectories_per_batch,
        "trajectory_seeds": trajectory_seeds,
        "flight_height_seeds": height_seeds,
        "paired_common_random_numbers": (
            "pareto_038 regenerates the exact population seeds used by the "
            "archived pareto_032 rows; identical NumPy and scientific inputs "
            "are enforced before simulation."
        ),
        "candidate_aep_gwh": {
            "pareto_032": p032_aep,
            "pareto_038": p038_aep,
        },
        "aep_model": aep_settings,
        "flight_height_distribution": distribution.settings(),
        "rotor_geometry": rotor.settings(),
        "behaviour": behaviour.settings(rotor.rotor_diameter_m),
        "resumed_from_batches": len(rows),
    }
    settings_path = output_dir / "run_settings.json"
    settings_path.write_text(
        json.dumps(serialisable(settings), indent=2), encoding="utf-8"
    )

    for batch_index in range(len(rows), batch_count):
        trajectory_seed = trajectory_seeds[batch_index]
        flight_height_seed = height_seeds[batch_index]
        print(
            f"p038 batch {batch_index + 1}/{batch_count}: "
            f"trajectory seed={trajectory_seed}, "
            f"N={trajectories_per_batch:,}.",
            flush=True,
        )
        population = Model9Population.generate_parallel(
            flight_height_distribution=distribution,
            flight_height_seed=flight_height_seed,
            site_width_m=scenario.site.width_m,
            site_height_m=scenario.site.height_m,
            heading_deg_toward=birds.heading_deg_toward,
            number_of_trajectories=trajectories_per_batch,
            maximum_response_events=birds.maximum_response_events,
            longitudinal_margin_m=(
                behaviour.response_radius_m + behaviour.return_length_m
            ),
            trajectory_seed=trajectory_seed,
        )
        combined_summary = summarise_model9_batches(
            transit_model,
            p038_layout,
            [population],
        )
        if int(combined_summary["N_failed"]) != 0:
            raise RuntimeError(
                f"Trajectory failures occurred for pareto_038 in batch {batch_index}."
            )
        batch = dict(combined_summary["batch_summaries"][0])
        batch["batch_index"] = batch_index
        rows.append({"candidate_id": "pareto_038", **batch})
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    p038 = pd.DataFrame(rows).sort_values("batch_index").reset_index(drop=True)
    combined = combine_p032_p038_batches(p032, p038)
    aep_gwh = {"pareto_032": p032_aep, "pareto_038": p038_aep}
    summary = candidate_summary_frame(
        combined,
        candidate_ids=["pareto_032", "pareto_038"],
        aep_gwh=aep_gwh,
    )
    paired = paired_difference_frame(combined)
    decision = extension_decision(
        combined,
        aep_gwh=aep_gwh,
        prior_decision=prior_decision,
    )

    p038.to_csv(output_dir / "p038_confirmation_batches.csv", index=False)
    combined.to_csv(output_dir / "combined_p032_p038_batches.csv", index=False)
    summary.to_csv(output_dir / "p038_candidate_summary.csv", index=False)
    paired.to_csv(output_dir / "p038_paired_differences.csv", index=False)
    (output_dir / "p038_confirmation_decision.json").write_text(
        json.dumps(serialisable(decision), indent=2), encoding="utf-8"
    )

    elapsed = float(time.time() - started)
    settings.update(
        {
            "run_status": "completed",
            "elapsed_seconds_this_invocation": elapsed,
            "extension_status": decision["extension_status"],
            "extension_tradeoff_gate_passed": decision[
                "extension_tradeoff_gate_passed"
            ],
            "overall_model9_status": decision["overall_model9_status"],
        }
    )
    settings_path.write_text(
        json.dumps(serialisable(settings), indent=2), encoding="utf-8"
    )

    archive_path: Path | None = None
    if not args.no_zip:
        archive_path = output_dir.with_suffix(".zip")
        shutil.make_archive(
            str(output_dir),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )

    comparison = decision["extension_comparison"]
    print("\np038 extension result")
    print(f"Status: {decision['extension_status']}")
    print(
        "Mean E_RT_CE difference (pareto_038 - pareto_032): "
        f"{comparison['mean_first_minus_second']:.9g}"
    )
    print(
        "Paired 95% Student-t interval: "
        f"[{comparison['ci_lower']:.9g}, {comparison['ci_upper']:.9g}]"
    )
    print(f"Completed this invocation in {elapsed / 60.0:.2f} minutes.")
    print(f"Results: {output_dir}")
    if archive_path is not None:
        print(f"Upload this archive for review: {archive_path}")


if __name__ == "__main__":
    main()
