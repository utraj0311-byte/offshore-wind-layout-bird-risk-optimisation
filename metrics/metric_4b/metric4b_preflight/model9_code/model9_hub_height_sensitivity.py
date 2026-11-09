"""Helpers for the controlled fixed-hub-height Model 9 sensitivity test.

This scenario is deliberately not a species model.  Every trajectory is set to
the 150 m hub height so the analysis isolates horizontal layout and circular
rotor-plane geometry when vertical exposure is guaranteed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from model9_confirmation import (
    PRIMARY_METRIC,
    comparison_record,
    validate_paired_batches,
)
from model9_rotor_transit import DiscreteFlightHeightDistribution


def fixed_hub_height_distribution(
    hub_height_m: float,
) -> DiscreteFlightHeightDistribution:
    """Return a deterministic one-point height scenario at rotor centre."""
    if not np.isfinite(hub_height_m) or hub_height_m <= 0.0:
        raise ValueError("hub_height_m must be finite and positive.")
    return DiscreteFlightHeightDistribution(
        name=f"controlled_fixed_hub_height_{float(hub_height_m):g}m",
        heights_m=np.asarray([hub_height_m], dtype=float),
        probability_mass=np.asarray([1.0], dtype=float),
        source=(
            "Controlled maximum-rotor-cross-section sensitivity assumption; "
            "not an empirical species flight-height distribution."
        ),
        input_schema="single deterministic height",
    )


def validate_hub_height_batches(
    batch_frame: pd.DataFrame,
    *,
    candidate_ids: Sequence[str],
    hub_height_m: float,
) -> list[int]:
    """Validate paired accounting and fixed-height invariants."""
    batches = validate_paired_batches(batch_frame, candidate_ids)
    required = {
        "E_RT",
        "P_RT",
        "N_RT_farm",
        "N_RT_CE_farm",
        "P_RT_CE",
        "P_height_band",
        "mean_flight_height_m",
    }
    missing = sorted(required - set(batch_frame.columns))
    if missing:
        raise ValueError(f"Hub-height batch data are missing columns: {missing}")
    if not np.allclose(
        batch_frame["mean_flight_height_m"].to_numpy(dtype=float),
        float(hub_height_m),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Not every trajectory is at the locked hub height.")
    if not np.allclose(
        batch_frame["P_height_band"].to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Fixed hub-height trajectories must all be in the rotor band.")
    for sampled, conditional in (
        ("E_RT", "E_RT_CE"),
        ("P_RT", "P_RT_CE"),
        ("N_RT_farm", "N_RT_CE_farm"),
    ):
        if not np.allclose(
            batch_frame[sampled].to_numpy(dtype=float),
            batch_frame[conditional].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{sampled} and {conditional} must match at deterministic hub height."
            )
    return batches


def paired_difference_frame(
    batch_frame: pd.DataFrame,
    *,
    candidate_ids: Sequence[str],
    hub_height_m: float,
) -> pd.DataFrame:
    """Return one transparent row per common-random-number batch."""
    batches = validate_hub_height_batches(
        batch_frame,
        candidate_ids=candidate_ids,
        hub_height_m=hub_height_m,
    )
    pivot = batch_frame.pivot(
        index="batch_index", columns="candidate_id", values=PRIMARY_METRIC
    ).reindex(batches)
    seeds = (
        batch_frame.groupby("batch_index", as_index=True)[
            ["trajectory_seed", "flight_height_seed"]
        ]
        .first()
        .reindex(batches)
    )
    result = pd.DataFrame(
        {
            "batch_index": batches,
            "trajectory_seed": seeds["trajectory_seed"].astype(int).to_numpy(),
            "flight_height_seed": seeds["flight_height_seed"].astype(int).to_numpy(),
        }
    )
    for candidate_id in candidate_ids:
        result[f"E_RT_CE_{candidate_id}"] = pivot[candidate_id].to_numpy()
    result["primary_original_minus_pareto_032"] = (
        pivot["original_energy_layout"] - pivot["pareto_032"]
    ).to_numpy()
    result["descriptive_pareto_037_minus_pareto_032"] = (
        pivot["pareto_037"] - pivot["pareto_032"]
    ).to_numpy()
    result["descriptive_pareto_038_minus_pareto_032"] = (
        pivot["pareto_038"] - pivot["pareto_032"]
    ).to_numpy()
    return result


def all_pairwise_comparisons(
    batch_frame: pd.DataFrame,
    *,
    candidate_ids: Sequence[str],
    aep_gwh: Mapping[str, float],
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Return descriptive paired comparisons for every frozen layout pair."""
    rows: list[dict[str, Any]] = []
    for first, second in combinations(candidate_ids, 2):
        record = comparison_record(
            batch_frame,
            first_candidate=first,
            second_candidate=second,
            aep_gwh=aep_gwh,
            confidence_level=confidence_level,
        )
        rows.append(
            {
                "first_candidate": first,
                "second_candidate": second,
                "difference_definition": record["difference_definition"],
                "mean_E_RT_CE_difference": record["mean_first_minus_second"],
                "relative_difference_vs_second": record[
                    "relative_difference_vs_second"
                ],
                "ci95_lower": record["ci_lower"],
                "ci95_upper": record["ci_upper"],
                "two_sided_p_value": record["two_sided_p_value"],
                "interval_direction": record["interval_direction"],
                "aep_difference_gwh": record["aep_difference_gwh"],
                "relative_aep_difference_vs_second": record[
                    "relative_aep_difference_vs_second"
                ],
                "statistical_role": (
                    "primary"
                    if first == "original_energy_layout"
                    and second == "pareto_032"
                    else "descriptive_secondary"
                ),
            }
        )
    return pd.DataFrame(rows)


def hub_height_decision(
    batch_frame: pd.DataFrame,
    *,
    aep_gwh: Mapping[str, float],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Apply the single pilot-informed original-versus-p032 sensitivity gate."""
    primary = comparison_record(
        batch_frame,
        first_candidate="original_energy_layout",
        second_candidate="pareto_032",
        aep_gwh=aep_gwh,
        confidence_level=confidence_level,
    )
    if (
        primary["interval_direction"] == "first_lower"
        and float(primary["aep_difference_gwh"]) < 0.0
    ):
        status = "confirmed_aep_hub_height_exposure_tradeoff"
        gate_passed = True
    elif primary["interval_direction"] == "first_higher":
        status = "original_layout_has_higher_hub_height_exposure"
        gate_passed = False
    else:
        status = "hub_height_exposure_difference_inconclusive"
        gate_passed = False

    p037 = comparison_record(
        batch_frame,
        first_candidate="pareto_037",
        second_candidate="pareto_032",
        aep_gwh=aep_gwh,
        confidence_level=confidence_level,
    )
    p038 = comparison_record(
        batch_frame,
        first_candidate="pareto_038",
        second_candidate="pareto_032",
        aep_gwh=aep_gwh,
        confidence_level=confidence_level,
    )
    return {
        "analysis_plan_frozen_before_final_sensitivity_run": True,
        "analysis_is_species_specific": False,
        "scenario": "all trajectories fixed at 150 m hub height",
        "primary_metric": PRIMARY_METRIC,
        "primary_status": status,
        "primary_tradeoff_gate_passed": gate_passed,
        "primary_comparison": primary,
        "descriptive_secondary_comparisons": {
            "pareto_037_minus_pareto_032": p037,
            "pareto_038_minus_pareto_032": p038,
        },
        "interpretation_boundary": (
            "This is a controlled maximum-cross-section geometry sensitivity, "
            "not a bird-species prediction, collision probability, fatality, "
            "or mortality estimate."
        ),
    }
