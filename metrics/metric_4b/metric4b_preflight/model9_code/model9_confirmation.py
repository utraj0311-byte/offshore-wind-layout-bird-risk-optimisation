"""Statistical helpers for the pre-specified Model 9 endpoint confirmation.

The primary analysis compares matched batches.  Every layout in one batch is
simulated with the same trajectory population, so uncertainty is calculated
from within-batch layout differences rather than from unrelated marginal
confidence intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


PRIMARY_METRIC = "E_RT_CE"


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) < 2:
        raise ValueError(f"{name} must contain at least two one-dimensional values.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def student_t_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    """Return a two-sided Student-t interval for an independent-batch mean."""
    sample = _finite_vector(values, name="values")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    count = int(len(sample))
    degrees_of_freedom = count - 1
    mean = float(np.mean(sample))
    standard_deviation = float(np.std(sample, ddof=1))
    standard_error = float(standard_deviation / np.sqrt(count))
    critical = float(
        student_t.ppf(0.5 + 0.5 * confidence_level, degrees_of_freedom)
    )
    half_width = float(critical * standard_error)
    return {
        "batch_count": count,
        "degrees_of_freedom": degrees_of_freedom,
        "confidence_level": float(confidence_level),
        "mean": mean,
        "sample_sd": standard_deviation,
        "standard_error": standard_error,
        "critical_t": critical,
        "ci_lower": float(mean - half_width),
        "ci_upper": float(mean + half_width),
    }


def paired_student_t_interval(
    first: Sequence[float],
    second: Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Analyse ``first - second`` across matched independent batches."""
    first_values = _finite_vector(first, name="first")
    second_values = _finite_vector(second, name="second")
    if first_values.shape != second_values.shape:
        raise ValueError("Paired samples must have the same shape.")
    differences = first_values - second_values
    interval = student_t_mean_interval(
        differences,
        confidence_level=confidence_level,
    )
    standard_error = float(interval["standard_error"])
    mean = float(interval["mean"])
    if standard_error == 0.0:
        if mean == 0.0:
            t_statistic = 0.0
            p_value = 1.0
        else:
            t_statistic = float(np.copysign(np.inf, mean))
            p_value = 0.0
    else:
        t_statistic = float(mean / standard_error)
        p_value = float(
            2.0
            * student_t.sf(
                abs(t_statistic),
                df=int(interval["degrees_of_freedom"]),
            )
        )
    lower = float(interval["ci_lower"])
    upper = float(interval["ci_upper"])
    if upper < 0.0:
        direction = "first_lower"
    elif lower > 0.0:
        direction = "first_higher"
    else:
        direction = "inconclusive"
    return {
        **interval,
        "mean_first": float(np.mean(first_values)),
        "mean_second": float(np.mean(second_values)),
        "mean_first_minus_second": mean,
        "relative_difference_vs_second": (
            float(mean / np.mean(second_values))
            if float(np.mean(second_values)) != 0.0
            else float("nan")
        ),
        "t_statistic": t_statistic,
        "two_sided_p_value": p_value,
        "interval_direction": direction,
        "batch_differences": differences.tolist(),
    }


def validate_paired_batches(
    batch_frame: pd.DataFrame,
    candidate_ids: Sequence[str],
) -> list[int]:
    """Verify complete matched batches and common random-number seeds."""
    required = {
        "candidate_id",
        "batch_index",
        "trajectory_seed",
        "flight_height_seed",
        PRIMARY_METRIC,
        "N_started",
        "N_completed",
        "N_failed",
    }
    missing = sorted(required - set(batch_frame.columns))
    if missing:
        raise ValueError(f"Confirmation batch data are missing columns: {missing}")
    ids = [str(candidate_id) for candidate_id in candidate_ids]
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise ValueError("At least two unique candidate IDs are required.")
    unknown = sorted(set(batch_frame["candidate_id"].astype(str)) - set(ids))
    if unknown:
        raise ValueError(f"Unexpected candidates in confirmation data: {unknown}")
    counts = batch_frame.groupby(["candidate_id", "batch_index"]).size()
    if not bool((counts == 1).all()):
        raise ValueError("Each candidate must appear exactly once in every batch.")
    expected_batches: set[int] | None = None
    for candidate_id in ids:
        candidate_batches = set(
            batch_frame.loc[
                batch_frame["candidate_id"] == candidate_id,
                "batch_index",
            ].astype(int)
        )
        if expected_batches is None:
            expected_batches = candidate_batches
        elif candidate_batches != expected_batches:
            raise ValueError("All candidates must contain the same batch indices.")
    assert expected_batches is not None
    if len(expected_batches) < 2:
        raise ValueError("At least two complete paired batches are required.")
    for column in ("trajectory_seed", "flight_height_seed", "N_started"):
        unique_by_batch = batch_frame.groupby("batch_index")[column].nunique()
        if not bool((unique_by_batch == 1).all()):
            raise ValueError(
                f"All candidates in one batch must share the same {column}."
            )
    if int(batch_frame["N_failed"].sum()) != 0:
        raise RuntimeError("Confirmation data contain failed trajectories.")
    if not bool((batch_frame["N_started"] == batch_frame["N_completed"]).all()):
        raise RuntimeError("Started and completed trajectory counts differ.")
    metric = batch_frame[PRIMARY_METRIC].to_numpy(dtype=float)
    if not np.all(np.isfinite(metric)):
        raise ValueError(f"{PRIMARY_METRIC} values must be finite.")
    return sorted(expected_batches)


def candidate_summary_frame(
    batch_frame: pd.DataFrame,
    *,
    candidate_ids: Sequence[str],
    aep_gwh: Mapping[str, float],
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Aggregate raw counts and uncertainty without discarding batch structure."""
    validate_paired_batches(batch_frame, candidate_ids)
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        group = batch_frame.loc[
            batch_frame["candidate_id"] == candidate_id
        ].sort_values("batch_index")
        started = int(group["N_started"].sum())
        interval = student_t_mean_interval(
            group[PRIMARY_METRIC].to_numpy(dtype=float),
            confidence_level=confidence_level,
        )

        def summed(column: str) -> float:
            return float(group[column].sum())

        def weighted(column: str) -> float:
            return float(
                np.average(
                    group[column].to_numpy(dtype=float),
                    weights=group["N_started"].to_numpy(dtype=float),
                )
            )

        pooled_conditional = summed("N_RT_CE_farm") / started
        rows.append(
            {
                "candidate_id": candidate_id,
                "aep_gwh": float(aep_gwh[candidate_id]),
                "batch_count": int(len(group)),
                "trajectories_per_batch": int(group["N_started"].iloc[0]),
                "N_started": started,
                "N_completed": int(group["N_completed"].sum()),
                "N_failed": int(group["N_failed"].sum()),
                "N_enc_farm": int(group["N_enc_farm"].sum()),
                "E_enc": summed("N_enc_farm") / started,
                "P_ge_1": weighted("P_ge_1"),
                "N_RT_farm": int(group["N_RT_farm"].sum()),
                "E_RT": summed("N_RT_farm") / started,
                "P_RT": weighted("P_RT"),
                "N_RT_CE_farm": summed("N_RT_CE_farm"),
                PRIMARY_METRIC: pooled_conditional,
                "P_RT_CE": weighted("P_RT_CE"),
                "P_height_band": weighted("P_height_band"),
                "mean_route_extension_m": weighted("mean_route_extension_m"),
                f"{PRIMARY_METRIC}_batch_sd": float(interval["sample_sd"]),
                f"{PRIMARY_METRIC}_batch_se": float(interval["standard_error"]),
                f"{PRIMARY_METRIC}_ci95_lower": float(interval["ci_lower"]),
                f"{PRIMARY_METRIC}_ci95_upper": float(interval["ci_upper"]),
            }
        )
    return pd.DataFrame(rows)


def comparison_record(
    batch_frame: pd.DataFrame,
    *,
    first_candidate: str,
    second_candidate: str,
    aep_gwh: Mapping[str, float],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return a paired risk comparison plus its deterministic AEP difference."""
    batches = validate_paired_batches(
        batch_frame,
        sorted(set(batch_frame["candidate_id"].astype(str))),
    )
    pivot = batch_frame.pivot(
        index="batch_index",
        columns="candidate_id",
        values=PRIMARY_METRIC,
    ).reindex(batches)
    if first_candidate not in pivot or second_candidate not in pivot:
        raise ValueError("Requested comparison candidate is absent from batch data.")
    result = paired_student_t_interval(
        pivot[first_candidate].to_numpy(dtype=float),
        pivot[second_candidate].to_numpy(dtype=float),
        confidence_level=confidence_level,
    )
    first_aep = float(aep_gwh[first_candidate])
    second_aep = float(aep_gwh[second_candidate])
    return {
        "first_candidate": first_candidate,
        "second_candidate": second_candidate,
        "metric": PRIMARY_METRIC,
        "difference_definition": f"{first_candidate} - {second_candidate}",
        "first_aep_gwh": first_aep,
        "second_aep_gwh": second_aep,
        "aep_difference_gwh": float(first_aep - second_aep),
        "relative_aep_difference_vs_second": (
            float((first_aep - second_aep) / second_aep)
            if second_aep != 0.0
            else float("nan")
        ),
        **result,
    }


def confirmation_decision(
    batch_frame: pd.DataFrame,
    *,
    aep_gwh: Mapping[str, float],
    tradeoff_candidate: str = "pareto_037",
    energy_candidate: str = "pareto_032",
    baseline_candidate: str = "original_energy_layout",
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Apply the frozen primary gate and report a secondary baseline contrast."""
    primary = comparison_record(
        batch_frame,
        first_candidate=tradeoff_candidate,
        second_candidate=energy_candidate,
        aep_gwh=aep_gwh,
        confidence_level=confidence_level,
    )
    secondary = comparison_record(
        batch_frame,
        first_candidate=energy_candidate,
        second_candidate=baseline_candidate,
        aep_gwh=aep_gwh,
        confidence_level=confidence_level,
    )
    if (
        primary["interval_direction"] == "first_lower"
        and float(primary["aep_difference_gwh"]) < 0.0
    ):
        primary_status = "confirmed_aep_rotor_transit_tradeoff"
        gate_passed = True
    elif primary["interval_direction"] == "first_higher":
        primary_status = "tradeoff_candidate_has_higher_rotor_transit"
        gate_passed = False
    else:
        primary_status = "risk_difference_inconclusive"
        gate_passed = False

    if float(secondary["aep_difference_gwh"]) <= 0.0:
        secondary_status = "energy_candidate_does_not_improve_aep"
    elif secondary["interval_direction"] == "first_lower":
        secondary_status = "higher_aep_and_lower_rotor_transit_confirmed"
    elif secondary["interval_direction"] == "first_higher":
        secondary_status = "higher_aep_but_higher_rotor_transit"
    else:
        secondary_status = "higher_aep_with_inconclusive_rotor_transit_difference"

    return {
        "analysis_is_pre_specified": True,
        "primary_metric": PRIMARY_METRIC,
        "primary_status": primary_status,
        "primary_tradeoff_gate_passed": gate_passed,
        "primary_comparison": primary,
        "secondary_status": secondary_status,
        "secondary_comparison": secondary,
        "interpretation_boundary": (
            "Rotor transit is an exposure endpoint, not blade collision, "
            "fatality, or mortality."
        ),
    }

