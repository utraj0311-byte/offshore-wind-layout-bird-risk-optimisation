"""Validation and statistics for the targeted Model 9 p038 extension.

The extension evaluates only ``pareto_038`` and pairs those new results with
the already completed ``pareto_032`` batches from the frozen seed-30142
confirmation.  Reuse is valid only when the prior archive, batch sizes, seeds,
runtime RNG version, and scientific inputs match the locked manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model9_confirmation import (
    PRIMARY_METRIC,
    comparison_record,
    validate_paired_batches,
)


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one archive member ending in {suffix!r}; "
            f"found {matches}."
        )
    return matches[0]


def load_prior_confirmation_archive(
    archive_path: Path,
    *,
    expected_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load and checksum the completed three-layout confirmation archive."""
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Prior confirmation archive not found: {archive_path}")
    actual_sha256 = sha256(archive_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Prior confirmation archive checksum changed: "
            f"{actual_sha256}; expected {expected_sha256}."
        )
    with zipfile.ZipFile(archive_path) as archive:
        batch_member = _unique_member(archive, "/confirmation_batches.csv")
        settings_member = _unique_member(archive, "/run_settings.json")
        decision_member = _unique_member(archive, "/confirmation_decision.json")
        batch_frame = pd.read_csv(io.BytesIO(archive.read(batch_member)))
        settings = json.loads(archive.read(settings_member).decode("utf-8"))
        decision = json.loads(archive.read(decision_member).decode("utf-8"))
    return batch_frame, settings, decision


def validate_prior_confirmation(
    batch_frame: pd.DataFrame,
    settings: dict[str, Any],
    decision: dict[str, Any],
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """Validate that the archived p032 rows are safe to reuse as paired data."""
    prior = manifest["prior_confirmation"]
    candidate_ids = [str(value) for value in prior["required_candidate_ids"]]
    batches = validate_paired_batches(batch_frame, candidate_ids)
    if settings.get("run_status") != "completed":
        raise ValueError("The prior confirmation archive is not marked completed.")
    if decision.get("primary_status") != prior["required_primary_status"]:
        raise ValueError("The prior p037 comparison status changed unexpectedly.")
    if bool(decision.get("primary_tradeoff_gate_passed")):
        raise ValueError("The prior archive unexpectedly reports a passed p037 gate.")

    expected_count = int(prior["batch_count"])
    expected_n = int(prior["trajectories_per_batch"])
    if batches != list(range(expected_count)):
        raise ValueError(
            f"Prior confirmation batch indices changed: {batches}."
        )
    if int(settings.get("batch_count", -1)) != expected_count:
        raise ValueError("Prior confirmation batch count changed.")
    if int(settings.get("trajectories_per_batch", -1)) != expected_n:
        raise ValueError("Prior confirmation trajectory count changed.")

    expected_trajectory_seeds = [int(x) for x in prior["trajectory_seeds"]]
    expected_height_seeds = [int(x) for x in prior["flight_height_seeds"]]
    seed_frame = (
        batch_frame.groupby("batch_index", as_index=True)[
            ["trajectory_seed", "flight_height_seed"]
        ]
        .first()
        .sort_index()
    )
    if seed_frame["trajectory_seed"].astype(int).tolist() != expected_trajectory_seeds:
        raise ValueError("Prior confirmation trajectory seeds changed.")
    if seed_frame["flight_height_seed"].astype(int).tolist() != expected_height_seeds:
        raise ValueError("Prior confirmation flight-height seeds changed.")
    if not bool((batch_frame["N_started"].astype(int) == expected_n).all()):
        raise ValueError("Prior confirmation batch sizes changed.")

    prior_numpy = str(settings.get("runtime_versions", {}).get("numpy", ""))
    if prior_numpy != str(prior["required_numpy_version"]):
        raise ValueError(
            f"Prior NumPy version is {prior_numpy}, not the locked version "
            f"{prior['required_numpy_version']}."
        )
    fhd_hash = str(
        settings.get("flight_height_distribution", {}).get("input_sha256", "")
    )
    if fhd_hash != str(prior["required_fhd_sha256"]):
        raise ValueError("Prior confirmation flight-height input changed.")
    floris_hash = str(settings.get("aep_model", {}).get("config_sha256", ""))
    if floris_hash != str(prior["required_floris_config_sha256"]):
        raise ValueError("Prior confirmation FLORIS configuration changed.")
    prior_p032_aep = float(
        settings.get("candidate_aep_gwh", {}).get("pareto_032", float("nan"))
    )
    if not np.isclose(
        prior_p032_aep,
        float(prior["pareto_032_aep_gwh"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Prior pareto_032 AEP changed.")

    p032 = batch_frame.loc[
        batch_frame["candidate_id"].astype(str) == "pareto_032"
    ].sort_values("batch_index").copy()
    pooled = float(p032["N_RT_CE_farm"].sum() / p032["N_started"].sum())
    if not np.isclose(
        pooled,
        float(prior["expected_p032_E_RT_CE"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            f"Prior p032 E_RT_CE changed: {pooled:.16g}."
        )
    return p032


def combine_p032_p038_batches(
    p032: pd.DataFrame,
    p038: pd.DataFrame,
) -> pd.DataFrame:
    """Return a strictly validated two-layout paired batch table."""
    combined = pd.concat([p032, p038], ignore_index=True, sort=False)
    validate_paired_batches(combined, ["pareto_032", "pareto_038"])
    return combined.sort_values(["batch_index", "candidate_id"]).reset_index(drop=True)


def paired_difference_frame(combined: pd.DataFrame) -> pd.DataFrame:
    """Return one transparent p038-p032 record per matched batch."""
    validate_paired_batches(combined, ["pareto_032", "pareto_038"])
    pivot = combined.pivot(
        index="batch_index", columns="candidate_id", values=PRIMARY_METRIC
    ).sort_index()
    seeds = (
        combined.groupby("batch_index", as_index=True)[
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
            "E_RT_CE_pareto_032": pivot["pareto_032"].to_numpy(),
            "E_RT_CE_pareto_038": pivot["pareto_038"].to_numpy(),
            "pareto_038_minus_pareto_032": (
                pivot["pareto_038"] - pivot["pareto_032"]
            ).to_numpy(),
        }
    )


def extension_decision(
    combined: pd.DataFrame,
    *,
    aep_gwh: dict[str, float],
    prior_decision: dict[str, Any],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Apply the single locked p038-versus-p032 extension gate."""
    comparison = comparison_record(
        combined,
        first_candidate="pareto_038",
        second_candidate="pareto_032",
        aep_gwh=aep_gwh,
        confidence_level=confidence_level,
    )
    if (
        comparison["interval_direction"] == "first_lower"
        and float(comparison["aep_difference_gwh"]) < 0.0
    ):
        status = "confirmed_extreme_aep_rotor_transit_tradeoff"
        gate_passed = True
        overall = "wide_tradeoff_confirmed_but_high_energy_p037_not_confirmed"
    elif comparison["interval_direction"] == "first_higher":
        status = "pareto_038_has_higher_rotor_transit"
        gate_passed = False
        overall = "no_confirmed_rotor_transit_tradeoff_in_tested_candidates"
    else:
        status = "pareto_038_difference_inconclusive"
        gate_passed = False
        overall = "no_confirmed_rotor_transit_tradeoff_in_tested_candidates"
    return {
        "analysis_plan_frozen_before_p038_simulation": True,
        "original_v5_primary_test": False,
        "primary_metric": PRIMARY_METRIC,
        "extension_status": status,
        "extension_tradeoff_gate_passed": gate_passed,
        "extension_comparison": comparison,
        "prior_p037_status": prior_decision["primary_status"],
        "prior_p037_tradeoff_gate_passed": bool(
            prior_decision["primary_tradeoff_gate_passed"]
        ),
        "overall_model9_status": overall,
        "interpretation_boundary": (
            "Rotor transit is an exposure endpoint, not blade collision, "
            "fatality, or mortality."
        ),
    }
