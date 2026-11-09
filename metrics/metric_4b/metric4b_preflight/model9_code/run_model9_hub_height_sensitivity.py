"""Run the frozen fixed-hub-height Model 9 geometry sensitivity.

This is not a replacement bird model and it is not another optimisation.
Every simulated trajectory is assigned the 150 m hub height, where the rotor's
horizontal cross-section is widest.  Four already-frozen layouts are then
evaluated on identical trajectory populations.  The single primary comparison
is the original energy layout minus pareto_032; every other comparison is
descriptive.
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
from model9_hub_height_sensitivity import (
    all_pairwise_comparisons,
    fixed_hub_height_distribution,
    hub_height_decision,
    paired_difference_frame,
    validate_hub_height_batches,
)
from model9_optimisation_utils import load_layout_csv, serialisable
from model9_rotor_transit import Model9Population, RotorGeometry, RotorTransitModel
from run_model9_endpoint_confirmation import runtime_versions, sha256


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = (
    MODULE_DIR / "data" / "model9_hub_height_sensitivity_manifest.json"
)
DEFAULT_OUTPUT_PATH = (
    MODULE_DIR / "results" / "model9_hub_height_sensitivity_seed90142"
)
CHECKPOINT_NAME = "hub_height_batches_checkpoint.csv"


def prepare_output_dir(path: Path, *, make_archive: bool, resume: bool) -> Path:
    """Create a new output directory or safely reopen an interrupted run."""
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


def locked_design(
    manifest: dict[str, Any],
) -> tuple[int, int, list[int], list[int]]:
    """Read and cross-check the immutable final sampling design."""
    design = manifest["final_design"]
    batch_count = int(design["batch_count"])
    trajectories_per_batch = int(design["trajectories_per_batch"])
    trajectory_seeds = [int(value) for value in design["trajectory_seeds"]]
    height_seeds = [int(value) for value in design["flight_height_seeds"]]
    if batch_count < 2 or trajectories_per_batch <= 0:
        raise ValueError("The locked sampling design is invalid.")
    if len(trajectory_seeds) != batch_count or len(height_seeds) != batch_count:
        raise ValueError("Locked seed lists must contain one seed per batch.")
    if len(set(trajectory_seeds)) != batch_count:
        raise ValueError("Locked trajectory seeds must be unique.")
    if len(set(height_seeds)) != batch_count:
        raise ValueError("Locked flight-height seeds must be unique.")
    if set(trajectory_seeds) & set(height_seeds):
        raise ValueError("Trajectory and flight-height seed namespaces must differ.")
    if int(design["total_trajectories_per_layout"]) != (
        batch_count * trajectories_per_batch
    ):
        raise ValueError("Locked per-layout trajectory total is inconsistent.")
    candidate_count = len(manifest["candidates"])
    if int(design["total_layout_trajectory_simulations"]) != (
        batch_count * trajectories_per_batch * candidate_count
    ):
        raise ValueError("Locked total layout-trajectory count is inconsistent.")
    return batch_count, trajectories_per_batch, trajectory_seeds, height_seeds


def load_locked_layouts(
    manifest: dict[str, Any],
    scenario: Scenario,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Load checksum-locked layouts and validate their feasibility."""
    minimum_spacing_m = (
        scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m
    )
    layouts: dict[str, np.ndarray] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for definition in manifest["candidates"]:
        candidate_id = str(definition["candidate_id"])
        if candidate_id in layouts:
            raise ValueError(f"Duplicate candidate in manifest: {candidate_id}")
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
    required = {
        "original_energy_layout",
        "pareto_032",
        "pareto_037",
        "pareto_038",
    }
    if set(layouts) != required:
        raise ValueError(
            f"Sensitivity manifest must contain exactly {sorted(required)}."
        )
    return layouts, definitions


def _validate_checkpoint_rows(
    frame: pd.DataFrame,
    *,
    candidate_ids: list[str],
    trajectory_seeds: list[int],
    height_seeds: list[int],
    trajectories_per_batch: int,
    hub_height_m: float,
) -> None:
    required = {
        "candidate_id",
        "batch_index",
        "trajectory_seed",
        "flight_height_seed",
        "N_started",
        "N_completed",
        "N_failed",
        "E_RT",
        "E_RT_CE",
        "P_RT",
        "P_RT_CE",
        "N_RT_farm",
        "N_RT_CE_farm",
        "P_height_band",
        "mean_flight_height_m",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Checkpoint is missing columns: {missing}")
    if len(frame) % len(candidate_ids) != 0:
        raise ValueError("Checkpoint ends with a partial layout batch.")
    completed_batches = len(frame) // len(candidate_ids)
    if completed_batches > len(trajectory_seeds):
        raise ValueError("Checkpoint contains too many batches.")
    indices = sorted(frame["batch_index"].astype(int).unique().tolist())
    if indices != list(range(completed_batches)):
        raise ValueError("Checkpoint batches must be a contiguous prefix from zero.")
    for batch_index in indices:
        group = frame.loc[frame["batch_index"].astype(int) == batch_index]
        if sorted(group["candidate_id"].astype(str).tolist()) != sorted(candidate_ids):
            raise ValueError("Every checkpoint batch must contain all frozen layouts.")
        if not bool((group["trajectory_seed"] == trajectory_seeds[batch_index]).all()):
            raise ValueError("Checkpoint trajectory seed changed.")
        if not bool((group["flight_height_seed"] == height_seeds[batch_index]).all()):
            raise ValueError("Checkpoint flight-height seed changed.")
    if not bool((frame["N_started"] == trajectories_per_batch).all()):
        raise ValueError("Checkpoint trajectory count changed.")
    if not bool((frame["N_completed"] == trajectories_per_batch).all()):
        raise ValueError("Checkpoint contains incomplete trajectories.")
    if int(frame["N_failed"].sum()) != 0:
        raise ValueError("Checkpoint contains failed trajectories.")
    if not np.allclose(frame["mean_flight_height_m"], hub_height_m, atol=1e-12):
        raise ValueError("Checkpoint does not use the locked hub height.")
    if not np.allclose(frame["P_height_band"], 1.0, atol=1e-12):
        raise ValueError("Checkpoint contains trajectories outside the rotor band.")
    for sampled, conditional in (
        ("E_RT", "E_RT_CE"),
        ("P_RT", "P_RT_CE"),
        ("N_RT_farm", "N_RT_CE_farm"),
    ):
        if not np.allclose(frame[sampled], frame[conditional], atol=1e-12):
            raise ValueError(
                f"Checkpoint {sampled} and {conditional} values do not match."
            )


def load_checkpoint(
    checkpoint_path: Path,
    *,
    resume: bool,
    candidate_ids: list[str],
    trajectory_seeds: list[int],
    height_seeds: list[int],
    trajectories_per_batch: int,
    hub_height_m: float,
) -> list[dict[str, Any]]:
    """Load only a complete, design-compatible checkpoint."""
    if not resume:
        return []
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume because checkpoint is absent: {checkpoint_path}"
        )
    frame = pd.read_csv(checkpoint_path)
    _validate_checkpoint_rows(
        frame,
        candidate_ids=candidate_ids,
        trajectory_seeds=trajectory_seeds,
        height_seeds=height_seeds,
        trajectories_per_batch=trajectories_per_batch,
        hub_height_m=hub_height_m,
    )
    return frame.sort_values(["batch_index", "candidate_id"]).to_dict(
        orient="records"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen four-layout Model 9 fixed-hub-height sensitivity."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--floris_config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--turbine_reference", type=Path, default=DEFAULT_TURBINE_REFERENCE_PATH
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Continue an interrupted run from {CHECKPOINT_NAME}.",
    )
    parser.add_argument("--no_zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    (
        batch_count,
        trajectories_per_batch,
        trajectory_seeds,
        height_seeds,
    ) = locked_design(manifest)
    scenario = Scenario()
    scenario_spec = manifest["scenario"]
    hub_height_m = float(scenario_spec["hub_height_m"])
    rotor_diameter_m = float(scenario_spec["rotor_diameter_m"])
    if not np.isclose(scenario.turbine.hub_height_m, hub_height_m, atol=1e-12):
        raise RuntimeError("The configured turbine hub height changed.")
    if not np.isclose(
        scenario.turbine.rotor_diameter_m, rotor_diameter_m, atol=1e-12
    ):
        raise RuntimeError("The configured turbine rotor diameter changed.")
    if not np.isclose(
        scenario.birds.heading_deg_toward,
        float(scenario_spec["rotor_plane_heading_deg_toward"]),
        atol=1e-12,
    ):
        raise RuntimeError("The configured approach heading changed.")

    candidate_ids = [
        str(definition["candidate_id"]) for definition in manifest["candidates"]
    ]
    output_dir = prepare_output_dir(
        args.output,
        make_archive=not args.no_zip,
        resume=args.resume,
    )
    checkpoint_path = output_dir / CHECKPOINT_NAME
    rows = load_checkpoint(
        checkpoint_path,
        resume=args.resume,
        candidate_ids=candidate_ids,
        trajectory_seeds=trajectory_seeds,
        height_seeds=height_seeds,
        trajectories_per_batch=trajectories_per_batch,
        hub_height_m=hub_height_m,
    )
    completed_batches = len(rows) // len(candidate_ids)
    started = time.time()

    layouts, definitions = load_locked_layouts(manifest, scenario)
    required_inputs = manifest["required_inputs"]
    if sha256(args.floris_config) != str(required_inputs["floris_config_sha256"]):
        raise ValueError("The locked FLORIS configuration checksum changed.")
    if sha256(args.turbine_reference) != str(
        required_inputs["turbine_reference_sha256"]
    ):
        raise ValueError("The locked turbine-reference checksum changed.")
    aep_model = FLORISAEPModel(
        scenario,
        args.floris_config,
        turbine_reference_path=args.turbine_reference,
    )
    if aep_model.number_of_conditions != int(
        required_inputs["floris_condition_count"]
    ):
        raise RuntimeError("The inherited FLORIS wind-condition count changed.")
    if not np.isclose(
        aep_model.floris_rotor_diameter_m, rotor_diameter_m, atol=1e-12
    ):
        raise RuntimeError("FLORIS rotor diameter differs from the locked scenario.")
    if not np.isclose(
        aep_model.floris_hub_height_m, hub_height_m, atol=1e-12
    ):
        raise RuntimeError("FLORIS hub height differs from the locked scenario.")

    aep_gwh: dict[str, float] = {}
    for candidate_id in candidate_ids:
        calculated = float(aep_model.evaluate(layouts[candidate_id]))
        expected = float(definitions[candidate_id]["expected_aep_gwh"])
        if not np.isclose(calculated, expected, rtol=0.0, atol=0.01):
            raise RuntimeError(
                f"AEP for {candidate_id} changed: {calculated:.9f} versus "
                f"{expected:.9f} GWh/year."
            )
        aep_gwh[candidate_id] = calculated
    if not (
        aep_gwh["pareto_032"]
        > aep_gwh["pareto_037"]
        > aep_gwh["original_energy_layout"]
        > aep_gwh["pareto_038"]
    ):
        raise RuntimeError("The frozen candidate AEP ordering changed.")

    distribution = fixed_hub_height_distribution(hub_height_m)
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
        "model": "Model 9 controlled fixed-hub-height geometry sensitivity",
        "endpoint": "rotor-swept-area transit exposure, not collision or mortality",
        "run_status": "sensitivity_started",
        "analysis_is_species_specific": False,
        "analysis_plan_frozen_before_final_sensitivity_run": True,
        "development_pilot_included_in_final_inference": False,
        "manifest": manifest,
        "manifest_sha256": sha256(DEFAULT_MANIFEST_PATH),
        "runtime_versions": runtime_versions(),
        "batch_count": batch_count,
        "trajectories_per_batch": trajectories_per_batch,
        "total_trajectories_per_layout": batch_count * trajectories_per_batch,
        "total_layout_trajectory_simulations": (
            batch_count * trajectories_per_batch * len(candidate_ids)
        ),
        "trajectory_seeds": trajectory_seeds,
        "flight_height_seeds": height_seeds,
        "paired_common_random_numbers": (
            "Within every batch, all four layouts use identical entry positions, "
            "meso-response draws, and deterministic 150 m flight heights."
        ),
        "candidate_aep_gwh": aep_gwh,
        "aep_model": aep_model.settings_summary(),
        "flight_height_scenario": distribution.settings(),
        "rotor_geometry": rotor.settings(),
        "behaviour": behaviour.settings(rotor.rotor_diameter_m),
        "resumed_from_batches": completed_batches,
    }
    settings_path = output_dir / "run_settings.json"
    settings_path.write_text(
        json.dumps(serialisable(settings), indent=2), encoding="utf-8"
    )

    for batch_index in range(completed_batches, batch_count):
        trajectory_seed = trajectory_seeds[batch_index]
        flight_height_seed = height_seeds[batch_index]
        print(
            f"Hub-height batch {batch_index + 1}/{batch_count}: "
            f"trajectory seed={trajectory_seed}, N={trajectories_per_batch:,} "
            "per layout.",
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
        if not np.all(population.flight_heights_m == hub_height_m):
            raise RuntimeError("Generated population violated the fixed-height rule.")
        new_rows: list[dict[str, Any]] = []
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
            new_rows.append({"candidate_id": candidate_id, **batch})
        rows.extend(new_rows)
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    batch_frame = (
        pd.DataFrame(rows)
        .sort_values(["batch_index", "candidate_id"])
        .reset_index(drop=True)
    )
    validate_hub_height_batches(
        batch_frame,
        candidate_ids=candidate_ids,
        hub_height_m=hub_height_m,
    )
    summary = candidate_summary_frame(
        batch_frame,
        candidate_ids=candidate_ids,
        aep_gwh=aep_gwh,
    )
    paired = paired_difference_frame(
        batch_frame,
        candidate_ids=candidate_ids,
        hub_height_m=hub_height_m,
    )
    comparisons = all_pairwise_comparisons(
        batch_frame,
        candidate_ids=candidate_ids,
        aep_gwh=aep_gwh,
    )
    decision = hub_height_decision(batch_frame, aep_gwh=aep_gwh)

    batch_frame.to_csv(output_dir / "hub_height_batches.csv", index=False)
    summary.to_csv(output_dir / "hub_height_candidate_summary.csv", index=False)
    paired.to_csv(output_dir / "hub_height_paired_differences.csv", index=False)
    comparisons.to_csv(
        output_dir / "hub_height_pairwise_comparisons.csv", index=False
    )
    (output_dir / "hub_height_decision.json").write_text(
        json.dumps(serialisable(decision), indent=2), encoding="utf-8"
    )

    elapsed = float(time.time() - started)
    settings.update(
        {
            "run_status": "completed",
            "elapsed_seconds_this_invocation": elapsed,
            "primary_status": decision["primary_status"],
            "primary_tradeoff_gate_passed": decision[
                "primary_tradeoff_gate_passed"
            ],
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
    print("\nFixed-hub-height sensitivity result")
    print(f"Status: {decision['primary_status']}")
    print(
        "Mean E_RT_CE difference (original - pareto_032): "
        f"{primary['mean_first_minus_second']:.9g}"
    )
    print(
        "Paired 95% Student-t interval: "
        f"[{primary['ci_lower']:.9g}, {primary['ci_upper']:.9g}]"
    )
    print(f"Completed this invocation in {elapsed / 60.0:.2f} minutes.")
    print(f"Results: {output_dir}")
    if archive_path is not None:
        print(f"Upload this archive for review: {archive_path}")


if __name__ == "__main__":
    main()
