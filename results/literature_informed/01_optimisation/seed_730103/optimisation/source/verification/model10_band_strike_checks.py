"""Deterministic source-geometry, Band, passive-callback and exposure tests.

All RPM/pitch/bird inputs in this file are test fixtures, not the final
operating baseline. Run: python verification/model10_band_strike_checks.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "upload"
sys.path.insert(0, str(ROOT))

from model10_band_strike import (  # noqa: E402
    BandCrossingAccumulator, BandScenario, ProjectedBladeStations,
    local_strike_probability,
)


def imported(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = imported("model8_meso_response", UPLOAD / "model8_meso_response(2).py")
R = imported("model9_rotor_transit", UPLOAD / "model9_rotor_transit.py")
V = imported("run_model8_validation", UPLOAD / "run_model8_validation(1).py")
PREVIOUS = imported("model9_crossing_interface_checks", Path(__file__).with_name("model9_crossing_interface_checks.py"))

SOURCE = UPLOAD / "IEA15MW_Band_input_sources(1).zip"
BLADE = ProjectedBladeStations.from_source_zip(SOURCE)
FIXTURE = BandScenario(
    rotor_rpm=7.55, collective_pitch_deg=0.0,
    bird_speed_m_s=14.9,
    bird_length_m=0.94, bird_wingspan_m=1.72,
    flight_style="flapping", wind_heading_deg_toward=0.0,
)


def one_crossing(offset_m: float, *, heights=(150.0,), mass=(1.0,), wind=0.0):
    """Horizontal x-crossing with a turbine at y=-offset and a fixed FHD."""
    layout = np.array([[0.0, -offset_m]])
    trajectories = M.TrajectoryPopulation(
        start_positions_m=np.array([[-20.0, 0.0]]),
        headings_deg_toward=np.array([0.0]),
        downstream_distance_m=np.array([40.0]),
        response_uniforms=np.array([[0.99]]),
        side_uniforms=np.array([[0.25]]),
        seed=121,
    )
    fhd = R.DiscreteFlightHeightDistribution(
        name="model10_fixed", heights_m=np.array(heights), probability_mass=np.array(mass),
    )
    population = R.Model9Population(
        trajectories=trajectories,
        flight_heights_m=np.array([heights[0]]),
        height_uniforms=np.array([0.0]),
        flight_height_seed=122,
        flight_height_distribution_name=fhd.name,
    )
    behaviour = M.BehaviourScenario(
        response_probability=0.0, response_radius_m=10.0,
        encounter_radius_m=10.0, deflection_angle_deg=45.0,
        return_length_m=80.0, step_size_m=10.0,
    )
    model = R.RotorTransitModel(
        behaviour, R.RotorGeometry(rotor_diameter_m=242.24, hub_height_m=150.0),
        rotor_plane_heading_deg_toward=0.0, flight_height_distribution=fhd,
    )
    observer = BandCrossingAccumulator(
        blade=BLADE, scenario=replace(FIXTURE, wind_heading_deg_toward=wind),
        number_of_trajectories=1, number_of_turbines=1,
    )
    batches = []

    def report(batch):
        batches.append(batch)
        observer(batch)

    result = model.simulate(layout, population, crossing_callback=report)
    return result, observer, batches


def geometry_checks():
    assert len(BLADE.radius_m) == 50
    assert BLADE.hub_radius_m == 3.97
    assert BLADE.tip_radius_m == 120.97
    assert BLADE.precone_deg == -4.0
    assert BLADE.number_of_blades == 3
    assert BLADE.blade_sha256 == "ed8eaaa11813946594bc791e9c808eba18d8078bf9409956814a00ba3c3ccef0"
    assert BLADE.elastodyn_sha256 == "be58f41b85580db71c66fc2de1eaafc768e0cb96b206d95c7a61aeaf3c48e934"
    precone = np.cos(np.deg2rad(4.0))
    assert np.isclose(BLADE.radius_m[0], 3.97 * precone, atol=1e-13)
    assert np.isclose(BLADE.radius_m[-1], 120.9699315223028 * precone, atol=1e-12)
    assert np.isclose(2 * BLADE.tip_radius_m * precone, 241.35064632, atol=1e-8)
    assert BLADE.radius_m[-1] < BLADE.tip_radius_m * precone < 121.12
    assert BLADE.at_radius(0.0) is None
    assert BLADE.at_radius(BLADE.radius_m[0] - 1e-8) is None
    assert BLADE.at_radius(BLADE.radius_m[-1] + 1e-8) is None
    assert BLADE.at_radius(BLADE.tip_radius_m * precone) is None
    assert BLADE.at_radius(121.12) is None
    assert BLADE.at_radius(BLADE.radius_m[0]) == (BLADE.chord_m[0], BLADE.twist_deg[0])
    assert BLADE.at_radius(BLADE.radius_m[-1]) == (BLADE.chord_m[-1], BLADE.twist_deg[-1])
    try:
        BLADE.at_radius(-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative physical radius was accepted")
    midway = (BLADE.radius_m[0] + BLADE.radius_m[1]) / 2
    assert np.isclose(BLADE.at_radius(midway)[0],
                      (BLADE.chord_m[0] + BLADE.chord_m[1]) / 2)
    assert np.isclose(BLADE.at_radius(midway)[1],
                      (BLADE.twist_deg[0] + BLADE.twist_deg[1]) / 2)


def kernel_checks():
    r = 50.0
    up = local_strike_probability(radius_m=r, vertical_from_hub_m=r,
                                  blade=BLADE, scenario=FIXTURE, approach="upwind")
    down = local_strike_probability(radius_m=r, vertical_from_hub_m=r,
                                    blade=BLADE, scenario=FIXTURE, approach="downwind")
    assert 0.0 < down < up < 1.0
    # Explicit independent reference calculation of Band equation (3).
    omega = FIXTURE.rotor_rpm * 2 * np.pi / 60
    alpha = FIXTURE.bird_speed_m_s / (r * omega)
    chord, twist = BLADE.at_radius(r)
    gamma = np.deg2rad(twist + FIXTURE.collective_pitch_deg)
    up_reference = 3 * omega / (2 * np.pi * 14.9) * (
        abs(chord * np.sin(gamma) + alpha * chord * np.cos(gamma))
        + max(0.94, 1.72 * alpha)
    )
    assert np.isclose(up, up_reference, atol=1e-15)
    g = replace(FIXTURE, flight_style="gliding")
    gliding_r = 10.0  # wingspan term dominates bird length at this radius
    top = local_strike_probability(radius_m=gliding_r, vertical_from_hub_m=gliding_r,
                                   blade=BLADE, scenario=g, approach="upwind")
    bottom = local_strike_probability(radius_m=gliding_r, vertical_from_hub_m=-gliding_r,
                                      blade=BLADE, scenario=g, approach="upwind")
    side = local_strike_probability(radius_m=gliding_r, vertical_from_hub_m=0,
                                    blade=BLADE, scenario=g, approach="upwind")
    assert top == bottom and 0 < side < top
    assert local_strike_probability(radius_m=0, vertical_from_hub_m=0,
                                    blade=BLADE, scenario=FIXTURE, approach="upwind") == 0
    assert local_strike_probability(radius_m=121.12, vertical_from_hub_m=0,
                                    blade=BLADE, scenario=FIXTURE, approach="upwind") == 0
    for name in ("rotor_rpm", "bird_speed_m_s",
                 "bird_length_m", "bird_wingspan_m"):
        for invalid in (0, -1, float("nan")):
            try:
                replace(FIXTURE, **{name: invalid})
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted invalid {name}={invalid}")


def fixed_crossing_checks():
    root, _, _ = one_crossing(0.0)
    assert root.conditional_rotor_transit_intensity == 1.0
    # Both inclusive crossing endpoints are reported; the callback keeps one pair.
    inside, in_obs, batches = one_crossing(50.0)
    assert len(batches) == 2
    assert [float(b.crossing_xy_m[0, 0]) for b in batches] == [0.0, 0.0]
    assert [float(b.crossing_xy_m[0, 1]) for b in batches] == [0.0, 0.0]
    assert [float(b.lateral_offset_m[0]) for b in batches] == [50.0, 50.0]
    assert all(int(b.turbine_indices[0]) == 0 for b in batches)
    assert all(int(b.trajectory_indices[0]) == 0 for b in batches)
    assert inside.conditional_rotor_transit_intensity == 1.0
    assert np.isclose(in_obs.expected_potential_strikes_per_entrant,
                      local_strike_probability(radius_m=50.0, vertical_from_hub_m=0.0,
                                               blade=BLADE, scenario=FIXTURE,
                                               approach="downwind"))
    _, reversed_obs, _ = one_crossing(50.0, wind=180.0)
    assert reversed_obs.expected_potential_strikes_per_entrant > in_obs.expected_potential_strikes_per_entrant
    # Strict station endpoints, projected physical tip, FLORIS nominal disc edge.
    for offset, expected_positive in (
        (BLADE.radius_m[0], True),
        (BLADE.radius_m[-1], True),
        (BLADE.tip_radius_m * np.cos(np.deg2rad(4.0)), False),
        (BLADE.radius_m[-1] + 0.01, False),
        (121.12, False),
    ):
        exposed, band, _ = one_crossing(offset)
        assert np.isclose(exposed.conditional_rotor_transit_intensity, 1.0, atol=1e-12), offset
        assert (band.expected_potential_strikes_per_entrant > 0.0) == expected_positive, offset
    # Entire conditional FHD outside disc, despite horizontal plane crossings.
    no_heights, masked, _ = one_crossing(50.0, heights=(300.0,), mass=(1.0,))
    assert no_heights.conditional_rotor_transit_intensity == 0.0
    assert masked.expected_potential_strikes_per_entrant == 0.0
    no_lateral, off_disc, events = one_crossing(121.13)
    assert no_lateral.conditional_rotor_transit_intensity == 0.0
    assert off_disc.expected_potential_strikes_per_entrant == 0.0
    assert len(events) == 0

    # Exact FHD-weighting in the same disc: half at valid height, half outside.
    mixed, weighted, _ = one_crossing(50.0, heights=(150.0, 300.0), mass=(0.5, 0.5))
    assert mixed.conditional_rotor_transit_intensity == 0.5
    assert np.isclose(weighted.expected_potential_strikes_per_entrant,
                      0.5 * in_obs.expected_potential_strikes_per_entrant)
    return in_obs.expected_potential_strikes_per_entrant, reversed_obs.expected_potential_strikes_per_entrant


def full_disc_p1_and_regression():
    before = json.loads(PREVIOUS.SNAPSHOT.read_text())
    model, layout, population = PREVIOUS.baseline_case(M, R, V)
    unobserved = model.simulate(layout, population, store_path_indices=(0, 13))
    assert PREVIOUS.result_signature(unobserved) == before["result"]
    observer = BandCrossingAccumulator(
        blade=BLADE, scenario=FIXTURE,
        number_of_trajectories=population.number_of_trajectories,
        number_of_turbines=len(layout),
    )
    observed = model.simulate(layout, population, store_path_indices=(0, 13),
                              crossing_callback=observer)
    assert PREVIOUS.result_signature(observed) == before["result"]
    assert observer.expected_potential_strikes_per_entrant >= 0.0
    assert observer.expected_potential_strikes_per_entrant <= observed.conditional_rotor_transit_intensity

    # TEST-ONLY UNIT PROBABILITY: deliberately bypass the blade station mask,
    # retaining *every* FHD mass inside the full nominal Model 9 disc.
    unit = np.zeros_like(unobserved.expected_transit_probability_matrix)

    def p1_on_full_disc(batch):
        for i, j, d in zip(batch.trajectory_indices, batch.turbine_indices,
                           batch.lateral_offset_m):
            conditional_mass = sum(
                m for z, m in zip(batch.height_bins_m, batch.height_probability_mass)
                if d*d + (z - batch.hub_height_m)**2 <= batch.rotor_radius_m**2
            )
            unit[i, j] = max(unit[i, j], conditional_mass)

    p1_result = model.simulate(
        layout, population, store_path_indices=(0, 13),
        crossing_callback=p1_on_full_disc,
    )
    np.testing.assert_allclose(unit, p1_result.expected_transit_probability_matrix,
                               rtol=0.0, atol=1e-12)
    assert np.isclose(unit.sum()/population.number_of_trajectories,
                      p1_result.conditional_rotor_transit_intensity, atol=1e-12)
    assert PREVIOUS.result_signature(p1_result) == before["result"]
    return (p1_result.conditional_rotor_transit_intensity,
            observer.expected_potential_strikes_per_entrant,
            float(np.max(np.abs(unit - p1_result.expected_transit_probability_matrix))))


def downstream_and_label_checks():
    fixed, turbines, birds = PREVIOUS.fixed_case(M, R)
    first = BandCrossingAccumulator(
        blade=BLADE, scenario=FIXTURE,
        number_of_trajectories=birds.number_of_trajectories,
        number_of_turbines=len(turbines),
    )
    a = fixed.simulate(turbines, birds, crossing_callback=first)
    assert first.contribution_matrix[0, 0] > 0.0
    assert first.contribution_matrix[0, 2] > 0.0  # second downstream turbine
    permutation = np.array([2, 0, 3, 1])
    second = BandCrossingAccumulator(
        blade=BLADE, scenario=FIXTURE,
        number_of_trajectories=birds.number_of_trajectories,
        number_of_turbines=len(turbines),
    )
    b = fixed.simulate(turbines[permutation], birds, crossing_callback=second)
    np.testing.assert_allclose(
        second.contribution_matrix, first.contribution_matrix[:, permutation],
        rtol=0.0, atol=1e-15,
    )
    np.testing.assert_allclose(
        b.expected_transit_probability_matrix,
        a.expected_transit_probability_matrix[:, permutation],
        rtol=0.0, atol=1e-15,
    )
    assert a.conditional_rotor_transit_intensity == b.conditional_rotor_transit_intensity
    return float(first.expected_potential_strikes_per_entrant)


if __name__ == "__main__":
    geometry_checks()
    kernel_checks()
    down, up = fixed_crossing_checks()
    downstream = downstream_and_label_checks()
    exposure, weighted, p1_error = full_disc_p1_and_regression()
    assert hashlib.sha256((UPLOAD / "model8_meso_response(2).py").read_bytes()).hexdigest() == "2d9985f6c165a83fac5ed1493cefa1a65e8082a39d17e80e3776be12257e823d"
    assert hashlib.sha256((UPLOAD / "model9_rotor_transit.py").read_bytes()).hexdigest() == "3c821d5f62de8b292bc00190ae71ae1379be64ad4cd11c67dc631186b5650f09"
    print(json.dumps({
        "status": "all deterministic checks passed",
        "station_count": len(BLADE.radius_m),
        "projected_root_m": float(BLADE.radius_m[0]),
        "projected_last_station_m": float(BLADE.radius_m[-1]),
        "projected_physical_tip_m": float(BLADE.tip_radius_m*np.cos(np.deg2rad(4.0))),
        "nominal_model9_disc_m": 121.12,
        "fixed_downwind_fixture_p": down,
        "fixed_upwind_fixture_p": up,
        "full_baseline_e_rt_ce": exposure,
        "illustrative_collision_weighted_contribution": weighted,
        "downstream_two_bird_fixture_contribution": downstream,
        "p1_full_disc_max_matrix_error": p1_error,
        "model8_model9_snapshots_unchanged": True,
    }, indent=2))
