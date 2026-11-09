"""Independent-batch aggregation for Model 9 rotor-transit exposure."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from model9_rotor_transit import Model9Population, RotorTransitModel


def summarise_model9_batches(
    model: RotorTransitModel,
    turbine_xy_m: np.ndarray,
    populations: Sequence[Model9Population],
) -> dict[str, Any]:
    """Simulate independent batches and form metrics from summed counts.

    The same population objects should be reused for every candidate layout.
    This preserves common entry positions, meso-response random numbers, and
    flight-height quantiles across layouts.
    """
    batches = tuple(populations)
    if not batches:
        raise ValueError("At least one Model 9 population is required.")

    batch_rows: list[dict[str, Any]] = []
    per_turbine_encounters: np.ndarray | None = None
    per_turbine_transits: np.ndarray | None = None
    per_turbine_expected_transits: np.ndarray | None = None
    conditional_estimator_available: bool | None = None
    route_extension_sum_m = 0.0
    number_with_encounter = 0
    number_with_transit = 0
    number_in_height_band = 0
    flight_height_sum_m = 0.0
    minimum_flight_height_m = np.inf
    maximum_flight_height_m = -np.inf

    for batch_index, population in enumerate(batches):
        result = model.simulate(turbine_xy_m, population)
        result.validate_accounting()
        meso = result.meso_result
        if per_turbine_encounters is None:
            per_turbine_encounters = np.zeros_like(
                meso.per_turbine_encounters, dtype=np.int64
            )
            per_turbine_transits = np.zeros_like(
                result.per_turbine_transits, dtype=np.int64
            )
            conditional_estimator_available = result.has_conditional_estimator
            if result.has_conditional_estimator:
                per_turbine_expected_transits = np.zeros_like(
                    result.per_turbine_expected_transits,
                    dtype=float,
                )
        assert per_turbine_transits is not None
        if result.has_conditional_estimator != conditional_estimator_available:
            raise ValueError(
                "Conditional-estimator availability must match across batches."
            )
        if (
            meso.per_turbine_encounters.shape != per_turbine_encounters.shape
            or result.per_turbine_transits.shape != per_turbine_transits.shape
        ):
            raise ValueError("All batches must use the same turbine layout.")

        per_turbine_encounters += meso.per_turbine_encounters.astype(
            np.int64, copy=False
        )
        per_turbine_transits += result.per_turbine_transits.astype(
            np.int64, copy=False
        )
        if result.has_conditional_estimator:
            assert per_turbine_expected_transits is not None
            per_turbine_expected_transits += result.per_turbine_expected_transits
        route_extension_sum_m += float(np.sum(meso.route_extensions_m))
        number_with_encounter += int(
            np.count_nonzero(meso.per_trajectory_encounters >= 1)
        )
        number_with_transit += int(
            np.count_nonzero(result.per_trajectory_transits >= 1)
        )
        in_height_band = (
            (population.flight_heights_m >= model.rotor_geometry.lower_tip_height_m)
            & (population.flight_heights_m <= model.rotor_geometry.upper_tip_height_m)
        )
        number_in_height_band += int(np.count_nonzero(in_height_band))
        flight_height_sum_m += float(np.sum(population.flight_heights_m))
        minimum_flight_height_m = min(
            minimum_flight_height_m,
            float(np.min(population.flight_heights_m)),
        )
        maximum_flight_height_m = max(
            maximum_flight_height_m,
            float(np.max(population.flight_heights_m)),
        )

        batch_rows.append(
            {
                "batch_index": int(batch_index),
                "trajectory_seed": population.trajectory_seed,
                "flight_height_seed": int(population.flight_height_seed),
                "N_started": result.number_started,
                "N_completed": result.number_completed,
                "N_failed": result.number_failed,
                "N_approach_events": meso.total_approach_events,
                "N_response_events": meso.total_response_events,
                "N_enc_farm": meso.total_encounters,
                "E_enc": meso.encounter_intensity,
                "P_ge_1": meso.encounter_prevalence,
                "N_RT_farm": result.total_transits,
                "E_RT": result.rotor_transit_intensity,
                "P_RT": result.rotor_transit_prevalence,
                "N_RT_CE_farm": (
                    result.expected_total_transits
                    if result.has_conditional_estimator
                    else None
                ),
                "E_RT_CE": (
                    result.conditional_rotor_transit_intensity
                    if result.has_conditional_estimator
                    else None
                ),
                "P_RT_CE": (
                    result.conditional_rotor_transit_prevalence
                    if result.has_conditional_estimator
                    else None
                ),
                "P_height_band": result.rotor_height_band_prevalence,
                "mean_flight_height_m": float(
                    np.mean(population.flight_heights_m)
                ),
                "mean_route_extension_m": meso.mean_route_extension_m,
            }
        )

    number_started = int(sum(row["N_started"] for row in batch_rows))
    number_completed = int(sum(row["N_completed"] for row in batch_rows))
    number_failed = int(sum(row["N_failed"] for row in batch_rows))
    total_encounters = int(sum(row["N_enc_farm"] for row in batch_rows))
    total_transits = int(sum(row["N_RT_farm"] for row in batch_rows))
    if number_started <= 0:
        raise ValueError("The combined trajectory count must be positive.")
    if number_started != number_completed + number_failed:
        raise RuntimeError("Combined trajectory accounting is inconsistent.")

    def values(column: str) -> np.ndarray:
        return np.asarray([row[column] for row in batch_rows], dtype=float)

    def sample_sd(sample: np.ndarray) -> float:
        return 0.0 if len(sample) == 1 else float(np.std(sample, ddof=1))

    def diagnostics(column: str) -> dict[str, float]:
        sample = values(column)
        standard_deviation = sample_sd(sample)
        return {
            f"{column}_batch_sd": standard_deviation,
            f"{column}_batch_se": float(
                standard_deviation / np.sqrt(len(batch_rows))
            ),
            f"{column}_batch_min": float(np.min(sample)),
            f"{column}_batch_max": float(np.max(sample)),
        }

    assert per_turbine_encounters is not None
    assert per_turbine_transits is not None
    encounter_intensity = float(total_encounters / number_started)
    rotor_transit_intensity = float(total_transits / number_started)
    summary = {
        "objective_batch_count": len(batch_rows),
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
        "E_enc": encounter_intensity,
        "P_ge_1": float(number_with_encounter / number_started),
        "N_RT_farm": total_transits,
        "E_RT": rotor_transit_intensity,
        "P_RT": float(number_with_transit / number_started),
        "P_height_band": float(number_in_height_band / number_started),
        "RT_to_encounter_ratio": (
            float(rotor_transit_intensity / encounter_intensity)
            if encounter_intensity > 0.0
            else float("nan")
        ),
        "mean_flight_height_m": float(flight_height_sum_m / number_started),
        "minimum_flight_height_m": float(minimum_flight_height_m),
        "maximum_flight_height_m": float(maximum_flight_height_m),
        "mean_route_extension_m": float(route_extension_sum_m / number_started),
        "per_turbine_encounters": per_turbine_encounters,
        "per_turbine_transits": per_turbine_transits,
        **diagnostics("E_enc"),
        **diagnostics("P_ge_1"),
        **diagnostics("E_RT"),
        **diagnostics("P_RT"),
    }
    if conditional_estimator_available:
        assert per_turbine_expected_transits is not None
        expected_total_transits = float(
            sum(float(row["N_RT_CE_farm"]) for row in batch_rows)
        )
        conditional_intensity = float(expected_total_transits / number_started)
        conditional_prevalence = float(
            sum(float(row["P_RT_CE"]) * int(row["N_started"]) for row in batch_rows)
            / number_started
        )
        summary.update(
            {
                "N_RT_CE_farm": expected_total_transits,
                "E_RT_CE": conditional_intensity,
                "P_RT_CE": conditional_prevalence,
                "RT_CE_to_encounter_ratio": (
                    float(conditional_intensity / encounter_intensity)
                    if encounter_intensity > 0.0
                    else float("nan")
                ),
                "per_turbine_expected_transits": per_turbine_expected_transits,
                **diagnostics("E_RT_CE"),
                **diagnostics("P_RT_CE"),
            }
        )
    else:
        summary.update(
            {
                "N_RT_CE_farm": None,
                "E_RT_CE": None,
                "P_RT_CE": None,
                "RT_CE_to_encounter_ratio": None,
                "per_turbine_expected_transits": None,
            }
        )
    return summary
