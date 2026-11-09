"""Batch aggregation and Pareto-validation helpers for Model 8.

This module deliberately has no FLORIS or pymoo dependency.  The optimisation
problem and the post-optimisation validator both use the same aggregation
rules, and the rules can therefore be unit-tested in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from model8_meso_response import MesoResponseModel, TrajectoryPopulation


def summarise_meso_batches(
    model: MesoResponseModel,
    turbine_xy_m: np.ndarray,
    populations: Sequence[TrajectoryPopulation],
) -> dict[str, Any]:
    """Simulate and combine independent common-random-number batches.

    Counts are summed before ratios are formed.  This remains correct if batch
    sizes differ and avoids averaging rounded batch ratios.  Between-batch
    standard deviations and standard errors are retained as noise diagnostics.
    """

    batches = tuple(populations)
    if not batches:
        raise ValueError("At least one trajectory population is required.")

    batch_rows: list[dict[str, Any]] = []
    per_turbine_encounters: np.ndarray | None = None
    route_extension_sum_m = 0.0
    number_with_encounter = 0

    for batch_index, population in enumerate(batches):
        result = model.simulate(turbine_xy_m, population)
        result.validate_accounting()
        if per_turbine_encounters is None:
            per_turbine_encounters = np.zeros_like(
                result.per_turbine_encounters, dtype=np.int64
            )
        if result.per_turbine_encounters.shape != per_turbine_encounters.shape:
            raise ValueError("All batches must use the same turbine layout.")

        per_turbine_encounters += result.per_turbine_encounters.astype(
            np.int64, copy=False
        )
        route_extension_sum_m += float(np.sum(result.route_extensions_m))
        number_with_encounter += int(
            np.count_nonzero(result.per_trajectory_encounters >= 1)
        )
        batch_rows.append(
            {
                "batch_index": int(batch_index),
                "seed": int(population.seed),
                "N_started": result.number_started,
                "N_completed": result.number_completed,
                "N_failed": result.number_failed,
                "N_approach_events": result.total_approach_events,
                "N_response_events": result.total_response_events,
                "N_enc_farm": result.total_encounters,
                "E_enc": result.encounter_intensity,
                "P_ge_1": result.encounter_prevalence,
                "mean_route_extension_m": result.mean_route_extension_m,
            }
        )

    number_started = int(sum(row["N_started"] for row in batch_rows))
    number_completed = int(sum(row["N_completed"] for row in batch_rows))
    number_failed = int(sum(row["N_failed"] for row in batch_rows))
    total_encounters = int(sum(row["N_enc_farm"] for row in batch_rows))
    e_values = np.asarray([row["E_enc"] for row in batch_rows], dtype=float)
    p_values = np.asarray([row["P_ge_1"] for row in batch_rows], dtype=float)

    if number_started <= 0:
        raise ValueError("The combined trajectory count must be positive.")
    if number_started != number_completed + number_failed:
        raise RuntimeError("Combined trajectory accounting is inconsistent.")

    def sample_sd(values: np.ndarray) -> float:
        return 0.0 if len(values) == 1 else float(np.std(values, ddof=1))

    e_sd = sample_sd(e_values)
    p_sd = sample_sd(p_values)
    batch_count = len(batch_rows)
    assert per_turbine_encounters is not None
    return {
        "objective_batch_count": batch_count,
        "batch_summaries": batch_rows,
        "N_started": number_started,
        "N_completed": number_completed,
        "N_failed": number_failed,
        "N_approach_events": int(
            sum(row["N_approach_events"] for row in batch_rows)
        ),
        "N_response_events": int(
            sum(row["N_response_events"] for row in batch_rows)
        ),
        "N_enc_farm": total_encounters,
        "E_enc": float(total_encounters / number_started),
        "P_ge_1": float(number_with_encounter / number_started),
        "mean_route_extension_m": float(route_extension_sum_m / number_started),
        "E_enc_batch_sd": e_sd,
        "E_enc_batch_se": float(e_sd / np.sqrt(batch_count)),
        "E_enc_batch_min": float(np.min(e_values)),
        "E_enc_batch_max": float(np.max(e_values)),
        "P_ge_1_batch_sd": p_sd,
        "P_ge_1_batch_se": float(p_sd / np.sqrt(batch_count)),
        "per_turbine_encounters": per_turbine_encounters,
    }


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Return the minimisation non-dominated mask for a two-column array."""

    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("objectives must have shape (n, 2).")
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    if not np.all(np.isfinite(values)):
        raise ValueError("objectives must be finite.")

    keep = np.ones(len(values), dtype=bool)
    for index, candidate in enumerate(values):
        dominates = np.all(values <= candidate, axis=1) & np.any(
            values < candidate, axis=1
        )
        dominates[index] = False
        keep[index] = not bool(np.any(dominates))
    return keep


def robust_nondominated_mask(
    aep_loss_gwh: np.ndarray,
    encounter_batches: np.ndarray,
    *,
    z_value: float = 1.96,
) -> np.ndarray:
    """Filter a front using paired fresh-batch encounter uncertainty.

    A lower-energy-loss layout rejects a higher-loss layout when its encounter
    result is not significantly worse on the same fresh batches.  Paired batch
    differences preserve the variance reduction from common random numbers.
    This prevents negligible seed-specific differences from becoming reported
    trade-offs.
    """

    loss = np.asarray(aep_loss_gwh, dtype=float)
    batches = np.asarray(encounter_batches, dtype=float)
    if loss.ndim != 1 or batches.ndim != 2 or len(loss) != len(batches):
        raise ValueError(
            "aep_loss_gwh must have shape (n,) and encounter_batches (n, b)."
        )
    if len(loss) == 0 or batches.shape[1] == 0:
        raise ValueError("At least one layout and one fresh batch are required.")
    if z_value < 0.0 or not np.isfinite(z_value):
        raise ValueError("z_value must be finite and non-negative.")
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(batches)):
        raise ValueError("Robust Pareto inputs must be finite.")

    keep = np.ones(len(loss), dtype=bool)
    tolerance = 1.0e-12
    for candidate_index in range(len(loss)):
        for comparator_index in range(len(loss)):
            if comparator_index == candidate_index:
                continue
            if loss[comparator_index] > loss[candidate_index] + tolerance:
                continue

            paired_difference = (
                batches[comparator_index] - batches[candidate_index]
            )
            mean_difference = float(np.mean(paired_difference))
            if batches.shape[1] == 1:
                difference_se = 0.0
            else:
                difference_se = float(
                    np.std(paired_difference, ddof=1)
                    / np.sqrt(batches.shape[1])
                )
            comparator_not_worse = (
                mean_difference <= z_value * difference_se + tolerance
            )
            strictly_better_energy = (
                loss[comparator_index] < loss[candidate_index] - tolerance
            )
            strictly_better_encounters = mean_difference < -tolerance
            if comparator_not_worse and (
                strictly_better_energy or strictly_better_encounters
            ):
                keep[candidate_index] = False
                break
    return keep


def select_anchor_positions(
    aep_gwh: np.ndarray,
    aep_loss_gwh: np.ndarray,
    encounters: np.ndarray,
) -> dict[str, int]:
    """Select energy, compromise, and encounter anchors by row position."""

    aep = np.asarray(aep_gwh, dtype=float)
    loss = np.asarray(aep_loss_gwh, dtype=float)
    risk = np.asarray(encounters, dtype=float)
    if not (aep.ndim == loss.ndim == risk.ndim == 1):
        raise ValueError("Anchor inputs must be one-dimensional.")
    if not (len(aep) == len(loss) == len(risk)) or len(aep) == 0:
        raise ValueError("Anchor inputs must have the same non-zero length.")
    if not np.all(np.isfinite(np.column_stack([aep, loss, risk]))):
        raise ValueError("Anchor inputs must be finite.")

    def normalise(values: np.ndarray) -> np.ndarray:
        span = float(np.max(values) - np.min(values))
        if span <= 0.0:
            return np.zeros_like(values)
        return (values - np.min(values)) / span

    distance = np.hypot(normalise(loss), normalise(risk))
    return {
        "energy_endpoint": int(np.argmax(aep)),
        "knee": int(np.argmin(distance)),
        "encounter_endpoint": int(np.argmin(risk)),
    }
