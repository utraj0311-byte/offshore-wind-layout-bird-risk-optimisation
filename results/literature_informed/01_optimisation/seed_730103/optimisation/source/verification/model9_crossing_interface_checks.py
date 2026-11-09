"""Pre-edit snapshot and focused regression checks for Model 9's crossing hook.

The Band equation used below is a verification fixture with explicit example
parameters. It is not a production Band model or a selected collision baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "upload"
SNAPSHOT = Path(__file__).with_name("model9_crossing_pre_edit.json")


def imported(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    meso = imported("model8_meso_response", UPLOAD / "model8_meso_response(2).py")
    sys.path.insert(0, str(UPLOAD))
    rotor = imported("model9_rotor_transit", UPLOAD / "model9_rotor_transit.py")
    validation = imported("run_model8_validation", UPLOAD / "run_model8_validation(1).py")
    return meso, rotor, validation


def signature(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return {
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(k): signature(v) for k, v in sorted(value.items())}
    if value is None:
        return None
    raise TypeError(type(value))


def result_signature(result):
    result.validate_accounting()
    meso = result.meso_result
    return {
        "summary": result.summary(),
        "model8_fields": {
            f.name: signature(getattr(meso, f.name)) for f in fields(meso)
        },
        "model9_arrays": {
            name: signature(getattr(result, name))
            for name in (
                "flight_heights_m",
                "transit_matrix",
                "expected_transit_probability_matrix",
            )
        },
    }


def baseline_case(meso, rotor, validation):
    diameter = 242.24
    site = 6000.0
    layout = validation.make_fixed_layouts(
        site_width_m=site,
        site_height_m=site,
        bird_heading_deg_toward=225.0,
        minimum_spacing_m=4.0 * diameter,
    )["staggered"]
    fhd = rotor.DiscreteFlightHeightDistribution.from_csv(
        UPLOAD / "northern_gannet_Johnston2014_FHD(3).csv",
        name="J2014_crossing_interface_check",
    )
    pop = rotor.Model9Population.generate_parallel(
        flight_height_distribution=fhd,
        flight_height_seed=701,
        site_width_m=site,
        site_height_m=site,
        heading_deg_toward=225.0,
        number_of_trajectories=64,
        maximum_response_events=len(layout),
        longitudinal_margin_m=10.0 * diameter,
        trajectory_seed=42,
        stratified_entries=True,
    )
    behaviour = meso.BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=diameter,
        response_probability=0.6,
        response_radius_D=2.0,
        encounter_radius_D=1.0,
        deflection_angle_deg=45.0,
        return_length_D=8.0,
        step_size_D=0.20,
    )
    model = rotor.RotorTransitModel(
        behaviour,
        rotor.RotorGeometry(rotor_diameter_m=diameter, hub_height_m=150.0),
        rotor_plane_heading_deg_toward=225.0,
        flight_height_distribution=fhd,
    )
    return model, layout, pop


def store_pre_edit(meso, rotor, validation):
    model, layout, pop = baseline_case(meso, rotor, validation)
    record = {
        "model8_sha256": hashlib.sha256(
            (UPLOAD / "model8_meso_response(2).py").read_bytes()
        ).hexdigest(),
        "model9_sha256": hashlib.sha256(
            (UPLOAD / "model9_rotor_transit.py").read_bytes()
        ).hexdigest(),
        "result": result_signature(model.simulate(layout, pop, store_path_indices=(0, 13))),
    }
    SNAPSHOT.write_text(json.dumps(record, indent=2) + "\n")
    print("Pre-edit snapshot saved:", SNAPSHOT)
    print("Pre-edit metrics:", json.dumps(record["result"]["summary"], sort_keys=True))


def fixed_case(meso, rotor, *, all_heights_outside=False, far_lateral=False):
    # x=25 is exactly on two adjacent movement segments: intentional duplicate.
    layout = np.array([[0.0, 0.0], [25.0, 5.0], [60.0, 0.0], [25.0, 50.0]])
    y = [100.0, 105.0] if far_lateral else [0.0, 5.0]
    trajectories = meso.TrajectoryPopulation(
        start_positions_m=np.array([[-15.0, yi] for yi in y]),
        headings_deg_toward=np.array([0.0, 0.0]),
        downstream_distance_m=np.array([90.0, 90.0]),
        response_uniforms=np.full((2, 4), 0.8),
        side_uniforms=np.full((2, 4), 0.2),
        seed=8,
    )
    heights = [300.0, 301.0] if all_heights_outside else [140.0, 150.0, 160.0]
    mass = [0.5, 0.5] if all_heights_outside else [0.25, 0.50, 0.25]
    fhd = rotor.DiscreteFlightHeightDistribution(
        name="fixed_height_test", heights_m=np.array(heights), probability_mass=np.array(mass)
    )
    population = rotor.Model9Population(
        trajectories=trajectories,
        flight_heights_m=np.array([heights[0], heights[1]]),
        height_uniforms=np.array([0.1, 0.6]),
        flight_height_seed=9,
        flight_height_distribution_name=fhd.name,
    )
    behaviour = meso.BehaviourScenario(
        response_probability=0.0,
        response_radius_m=10.0,
        encounter_radius_m=10.0,
        deflection_angle_deg=45.0,
        return_length_m=80.0,
        step_size_m=10.0,
    )
    model = rotor.RotorTransitModel(
        behaviour,
        rotor.RotorGeometry(rotor_diameter_m=20.0, hub_height_m=150.0),
        rotor_plane_heading_deg_toward=0.0,
        flight_height_distribution=fhd,
    )
    return model, layout, population


def fixture_band_probability(*, radius_m: float, upwind: bool) -> float:
    """Band (2012) eq. 3 at an illustrative flapping crossing, not model code."""
    omega = 7.55 * 2.0 * np.pi / 60.0
    speed = 14.9
    chord = 4.0
    pitch = np.deg2rad(15.0)
    alpha = speed / (radius_m * omega)
    sign = 1 if upwind else -1
    strike_width = abs(sign * chord * np.sin(pitch) + alpha * chord * np.cos(pitch))
    bird_clearance = max(0.94, 1.72 * alpha)
    return min(1.0, 3.0 * omega / (2.0 * np.pi * speed) * (strike_width + bird_clearance))


def verify(meso, rotor, validation):
    before = json.loads(SNAPSHOT.read_text())
    assert before["model8_sha256"] == hashlib.sha256(
        (UPLOAD / "model8_meso_response(2).py").read_bytes()
    ).hexdigest(), "Model 8 source was modified"
    model, layout, pop = baseline_case(meso, rotor, validation)
    without = model.simulate(layout, pop, store_path_indices=(0, 13))
    assert result_signature(without) == before["result"], "Callback-absent regression"

    retained_batches = []

    def audit_callback(batch):
        retained_batches.append(batch)
        # Even a caller who re-enables writing can only corrupt its own copies.
        for name in (
            "trajectory_indices", "turbine_indices", "turbine_xy_m",
            "crossing_xy_m", "lateral_offset_m", "crossing_direction_sign",
            "segment_heading_deg_toward", "sampled_flight_heights_m",
        ):
            values = getattr(batch, name)
            assert not values.flags.writeable
            if values.size:
                values.setflags(write=True)
                values.flat[0] = 0
        assert isinstance(batch.height_bins_m, (tuple, type(None)))

    np.random.seed(55821)
    state_before = np.random.get_state()
    with_callback = model.simulate(
        layout, pop, store_path_indices=(0, 13), crossing_callback=audit_callback
    )
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    assert retained_batches, "The 12-turbine callback did not receive a crossing"
    assert result_signature(with_callback) == before["result"], "Callback-attached regression"

    # Controlled geometry and event order: x=25 falls on both segment endpoints.
    fixed, turbines, birds = fixed_case(meso, rotor)
    events = []
    weighted = np.zeros((2, len(turbines)))

    def crossing_callback(batch):
        assert batch.rotor_plane_heading_deg_toward == 0.0
        assert batch.rotor_radius_m == 10.0 and batch.hub_height_m == 150.0
        for i, j, txy, xy, d, direction, h in zip(
            batch.trajectory_indices, batch.turbine_indices,
            batch.turbine_xy_m, batch.crossing_xy_m,
            batch.lateral_offset_m, batch.crossing_direction_sign,
            batch.sampled_flight_heights_m,
        ):
            i, j = int(i), int(j)
            assert np.isclose(xy[0], txy[0])
            assert np.isclose(xy[1] - txy[1], d)
            assert direction == 1 and batch.segment_heading_deg_toward[np.where(batch.trajectory_indices == i)[0][0]] == 0.0
            assert h == birds.flight_heights_m[i]
            # Test-only local p=1 for every height bin inside the same circular disc.
            inside = [
                p for height, p in zip(batch.height_bins_m, batch.height_probability_mass)
                if d * d + (height - batch.hub_height_m) ** 2
                <= batch.rotor_radius_m ** 2
            ]
            weighted[i, j] = max(weighted[i, j], sum(inside))
            events.append((i, j, tuple(np.round(xy, 8)), float(d)))

    fixed_result = fixed.simulate(turbines, birds, crossing_callback=crossing_callback)
    assert events == [
        (0, 0, (0.0, 0.0), 0.0),
        (1, 0, (0.0, 5.0), 5.0),
        (0, 1, (25.0, 0.0), -5.0),
        (1, 1, (25.0, 5.0), 0.0),
        (0, 1, (25.0, 0.0), -5.0),
        (1, 1, (25.0, 5.0), 0.0),
        (0, 2, (60.0, 0.0), 0.0),
        (1, 2, (60.0, 5.0), 5.0),
    ], events
    np.testing.assert_allclose(weighted, fixed_result.expected_transit_probability_matrix, atol=1e-15)
    assert np.count_nonzero(fixed_result.transit_matrix) == 5
    assert np.count_nonzero(weighted) == 6
    assert fixed_result.meso_result.total_response_events == 0
    # Use the bird-1/turbine-0 crossing's own lateral offset and sampled height.
    # Upwind/downwind labels are explicit inputs, never inferred from heading.
    controlled_radius = float(np.hypot(
        events[1][3], birds.flight_heights_m[events[1][0]] - fixed.rotor_geometry.hub_height_m
    ))
    assert controlled_radius == 5.0
    p_up = fixture_band_probability(radius_m=controlled_radius, upwind=True)
    p_down = fixture_band_probability(radius_m=controlled_radius, upwind=False)
    assert 0.0 < p_down < p_up < 1.0, (p_up, p_down)

    no_crossing, distant, distant_birds = fixed_case(meso, rotor, far_lateral=True)
    no_events = []
    no_result = no_crossing.simulate(
        distant, distant_birds, crossing_callback=no_events.append
    )
    assert len(no_events) == 0
    assert no_result.conditional_rotor_transit_intensity == 0.0
    no_heights, same_turbines, outside_birds = fixed_case(
        meso, rotor, all_heights_outside=True
    )
    no_height_result = no_heights.simulate(
        same_turbines, outside_birds, crossing_callback=lambda batch: None
    )
    assert no_height_result.conditional_rotor_transit_intensity == 0.0

    # Also compare p=1 with the full, 301-bin FHD, using independent bin sums.
    full_matrix = np.zeros_like(without.expected_transit_probability_matrix)

    def unit_strike(batch):
        for i, j, offset in zip(
            batch.trajectory_indices, batch.turbine_indices, batch.lateral_offset_m
        ):
            mass = sum(
                p for height, p in zip(batch.height_bins_m, batch.height_probability_mass)
                if offset * offset + (height - batch.hub_height_m) ** 2
                <= batch.rotor_radius_m ** 2
            )
            full_matrix[i, j] = max(full_matrix[i, j], mass)

    same = model.simulate(layout, pop, crossing_callback=unit_strike)
    np.testing.assert_allclose(full_matrix, same.expected_transit_probability_matrix, atol=1e-12, rtol=0.0)
    assert np.isclose(
        float(full_matrix.sum() / pop.number_of_trajectories),
        same.conditional_rotor_transit_intensity, atol=1e-12, rtol=0.0
    )
    print(json.dumps({
        "status": "all checks passed",
        "baseline_birds": pop.number_of_trajectories,
        "baseline_turbines": len(layout),
        "unchanged_model8_sha256": before["model8_sha256"],
        "baseline_e_rt_ce": same.conditional_rotor_transit_intensity,
        "max_p1_matrix_error": float(np.max(abs(full_matrix - same.expected_transit_probability_matrix))),
        "controlled_event_count": len(events),
        "controlled_unique_bird_turbine_pairs": int(np.count_nonzero(weighted)),
        "upwind_fixture_probability": p_up,
        "downwind_fixture_probability": p_down,
        "zero_crossing_event_count": len(no_events),
        "zero_height_e_rt_ce": no_height_result.conditional_rotor_transit_intensity,
        "global_rng_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "verify"), required=True)
    args = parser.parse_args()
    M, R, V = modules()
    if args.mode == "baseline":
        store_pre_edit(M, R, V)
    else:
        verify(M, R, V)
