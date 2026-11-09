"""Input validation and output helpers for the Model 9 optimiser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def coordinate_columns(number_of_turbines: int) -> list[str]:
    if number_of_turbines <= 0:
        raise ValueError("number_of_turbines must be positive.")
    return [
        name
        for turbine_id in range(number_of_turbines)
        for name in (f"x_{turbine_id}_m", f"y_{turbine_id}_m")
    ]


def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return np.empty(0, dtype=float)
    delta = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    return distances[np.triu_indices(len(points), k=1)]


def minimum_pairwise_spacing(xy: np.ndarray) -> float:
    distances = pairwise_distances(xy)
    return float("inf") if len(distances) == 0 else float(np.min(distances))


def validate_layout(
    xy: np.ndarray,
    *,
    number_of_turbines: int,
    site_width_m: float,
    site_height_m: float,
    minimum_spacing_m: float,
    tolerance_m: float = 1.0e-6,
) -> dict[str, float | int]:
    points = np.asarray(xy, dtype=float)
    if points.shape != (number_of_turbines, 2):
        raise ValueError(
            f"Expected layout shape ({number_of_turbines}, 2), got {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("Layout coordinates must be finite.")
    if tolerance_m < 0.0:
        raise ValueError("tolerance_m cannot be negative.")
    outside = (
        (points[:, 0] < -tolerance_m)
        | (points[:, 0] > site_width_m + tolerance_m)
        | (points[:, 1] < -tolerance_m)
        | (points[:, 1] > site_height_m + tolerance_m)
    )
    if np.any(outside):
        ids = np.flatnonzero(outside).tolist()
        raise ValueError(f"Turbines outside the site boundary: {ids}.")
    distances = pairwise_distances(points)
    violations = int(np.count_nonzero(distances + tolerance_m < minimum_spacing_m))
    actual_minimum = minimum_pairwise_spacing(points)
    if violations:
        raise ValueError(
            "Layout violates minimum spacing: "
            f"{actual_minimum:.6f} < {minimum_spacing_m:.6f} m."
        )
    return {
        "minimum_pairwise_spacing_m": actual_minimum,
        "spacing_violation_count": violations,
    }


def load_layout_csv(
    path: str | Path,
    *,
    number_of_turbines: int,
    site_width_m: float,
    site_height_m: float,
    minimum_spacing_m: float,
) -> np.ndarray:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    if not {"x_m", "y_m"}.issubset(frame.columns):
        raise ValueError(f"Layout CSV must contain x_m and y_m columns: {csv_path}")
    if "turbine_id" in frame.columns:
        frame = frame.sort_values("turbine_id", kind="stable")
    coordinates = frame[["x_m", "y_m"]].to_numpy(dtype=float)
    validate_layout(
        coordinates,
        number_of_turbines=number_of_turbines,
        site_width_m=site_width_m,
        site_height_m=site_height_m,
        minimum_spacing_m=minimum_spacing_m,
    )
    return coordinates


def load_initial_population(
    path: str | Path,
    *,
    population_size: int,
    seed: int,
    number_of_turbines: int,
    site_width_m: float,
    site_height_m: float,
    minimum_spacing_m: float,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Load a deterministic subset of the validated Model 8 population.

    The four exact anchors occupy rows 0--3 and are always retained.  Smaller
    smoke and pilot profiles draw their remaining rows without replacement
    from the other supplied layouts using the optimiser seed.
    """
    if population_size < 4:
        raise ValueError("population_size must be at least four to retain all anchors.")
    frame = pd.read_csv(Path(path))
    columns = coordinate_columns(number_of_turbines)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Initial-population CSV is missing columns: {missing}")
    if population_size > len(frame):
        raise ValueError(
            f"population_size {population_size} exceeds the {len(frame)} supplied "
            "Model 8 layouts."
        )

    all_vectors = frame[columns].to_numpy(dtype=float)
    for row_index, vector in enumerate(all_vectors):
        try:
            validate_layout(
                vector.reshape(number_of_turbines, 2),
                number_of_turbines=number_of_turbines,
                site_width_m=site_width_m,
                site_height_m=site_height_m,
                minimum_spacing_m=minimum_spacing_m,
            )
        except ValueError as exc:
            raise ValueError(
                f"Initial-population row {row_index} is invalid: {exc}"
            ) from exc

    if population_size == len(frame):
        selected = np.arange(len(frame), dtype=int)
    else:
        rng = np.random.default_rng(int(seed))
        remaining = rng.permutation(np.arange(4, len(frame), dtype=int))
        selected = np.concatenate(
            (np.arange(4, dtype=int), remaining[: population_size - 4])
        )
    labels = (
        frame.iloc[selected]["seed_source"].astype(str).tolist()
        if "seed_source" in frame.columns
        else ["supplied_model8_population"] * len(selected)
    )
    return all_vectors[selected].copy(), labels, selected.astype(int).tolist()


def layout_frame(xy: np.ndarray) -> pd.DataFrame:
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    return pd.DataFrame(
        {
            "turbine_id": np.arange(len(points), dtype=int),
            "x_m": points[:, 0],
            "y_m": points[:, 1],
        }
    )


def serialisable(value: Any) -> Any:
    """Recursively convert NumPy/Path values for JSON output."""
    if isinstance(value, dict):
        return {str(key): serialisable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialisable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def select_anchor_positions(
    aep_gwh: np.ndarray,
    aep_loss_gwh: np.ndarray,
    rotor_transit: np.ndarray,
) -> dict[str, int]:
    """Select descriptive endpoints and a scale-normalised reference point."""
    aep = np.asarray(aep_gwh, dtype=float)
    loss = np.asarray(aep_loss_gwh, dtype=float)
    risk = np.asarray(rotor_transit, dtype=float)
    if not (aep.ndim == loss.ndim == risk.ndim == 1):
        raise ValueError("Anchor inputs must be one-dimensional.")
    if len(aep) == 0 or not (len(aep) == len(loss) == len(risk)):
        raise ValueError("Anchor inputs must have the same non-zero length.")
    if not np.all(np.isfinite(np.column_stack((aep, loss, risk)))):
        raise ValueError("Anchor inputs must be finite.")

    def normalise(values: np.ndarray) -> np.ndarray:
        span = float(np.ptp(values))
        return np.zeros_like(values) if span <= 0.0 else (values - values.min()) / span

    balanced_distance = np.hypot(normalise(loss), normalise(risk))
    return {
        "energy_endpoint": int(np.argmax(aep)),
        "balanced_reference": int(np.argmin(balanced_distance)),
        "rotor_transit_endpoint": int(np.argmin(risk)),
    }
