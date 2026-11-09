"""Final Model 10 gate: actual 12-turbine AEP, Model 8/9 and Band estimator.

Run with the locked FLORIS/pymoo dependencies on PYTHONPATH, for example:
 PYTHONPATH=verification/_runtime_deps python verification/model10_integrated_fixed_layout_check.py
Uses the archived 12-turbine energy endpoint and exact source-checked inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upload"), str(ROOT / "verification/_runtime_source")]

from config import Scenario  # noqa: E402
from floris_aep import FLORISAEPModel  # noqa: E402
from model8_meso_response import BehaviourScenario, TrajectoryPopulation  # noqa: E402
from model9_rotor_transit import (  # noqa: E402
    DiscreteFlightHeightDistribution, Model9Population, RotorGeometry,
    RotorTransitModel,
)
from model10_aep_collision_problem import Model10AEPCollisionProblem  # noqa: E402
from model10_band_strike import (  # noqa: E402
    BandCrossingAccumulator, BandScenario, ProjectedBladeStations,
)

from model10_band_strike_checks import one_crossing  # noqa: E402

FLORIS_CONFIG = ROOT / "verification/library_inputs/thesis/gch_iea15mw(2).yaml"
LAYOUT = ROOT / "verification/runtime_inputs/energy_baseline.csv"
FHD_CSV = ROOT / "upload/northern_gannet_Johnston2014_FHD(3).csv"
SOURCE_ZIP = ROOT / "upload/IEA15MW_Band_input_sources(1).zip"
REPORT = ROOT / "verification/model10_integrated_fixed_layout_report.json"


def checked_inputs():
    import floris
    if hashlib.sha256(FLORIS_CONFIG.read_bytes()).hexdigest() != (
        "f6b788df9e38f006dfb6c68ddfb80ab090dad9cd555982a8d5c5e4b0c2f2d364"
    ):
        raise AssertionError("FLORIS configuration differs from Model 9 archive")
    turbine = Path(floris.__file__).parent / "turbine_library/iea_15MW.yaml"
    if hashlib.sha256(turbine.read_bytes()).hexdigest() != (
        "93d2bcfcb619b1819efe8149ce9e8231656e51a29f2ae5c1079a46e339f02feb"
    ):
        raise AssertionError("FLORIS turbine data differ from Model 9 archive")
    if hashlib.sha256(FHD_CSV.read_bytes()).hexdigest() != (
        "adc9ea903db9c8e34bfe5b88c382381b737c2e676d79af437f66d198208a0f2c"
    ):
        raise AssertionError("Gannet FHD differs from Model 9 archive")
    blade = ProjectedBladeStations.from_source_zip(SOURCE_ZIP)
    band = BandScenario(
        rotor_rpm=7.517553314698912,
        collective_pitch_deg=6.760740271637748,
        bird_speed_m_s=14.9,
        bird_length_m=0.94, bird_wingspan_m=1.72,
        flight_style="gliding",
        # The existing site wind-from 225° (meteorological) moves toward
        # mathematical 45°, opposite the bird's fixed 225° reference heading.
        wind_heading_deg_toward=45.0,
    )
    with LAYOUT.open(newline="") as stream:
        xy = np.array([[float(row["x_m"]), float(row["y_m"])]
                       for row in csv.DictReader(stream)])
    return blade, band, xy


def check_controlled_crossings(blade, band):
    # With a possible disk transit but r=50 m, one bird gives positive risk.
    positive, _, positive_batches = one_crossing(30.0, heights=(190.0,))
    observer = BandCrossingAccumulator(blade, band, 1, 1)
    for event in positive_batches:
        observer(event)
    assert positive.conditional_rotor_transit_intensity == 1
    assert len(positive_batches) == 2  # inclusive endpoint duplication
    p = observer.expected_potential_strikes_per_entrant
    assert 0.0 < p <= 1 and math.isfinite(p)
    # Plane intersection beyond the rotor's horizontal extent: no valid transit.
    zero, _, zero_batches = one_crossing(130.0, heights=(150.0,))
    observer_zero = BandCrossingAccumulator(blade, band, 1, 1)
    for event in zero_batches:
        observer_zero(event)
    assert zero.total_transits == 0
    assert zero.conditional_rotor_transit_intensity == 0
    assert observer_zero.expected_potential_strikes_per_entrant == 0

    # A single bird passes three distinct turbines on the same axis. Each
    # inclusive endpoint is reported twice, but each valid turbine once.
    repeated_xy = np.array([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]])
    traj = TrajectoryPopulation(
        start_positions_m=np.array([[-20.0, 0.0]]),
        headings_deg_toward=np.array([0.0]),
        downstream_distance_m=np.array([2040.0]),
        response_uniforms=np.full((1, 3), 0.99),
        side_uniforms=np.full((1, 3), 0.25),
        seed=50,
    )
    fhd = DiscreteFlightHeightDistribution(
        "controlled_repeat", np.array([190.0]), np.array([1.0])
    )
    pop = Model9Population(traj, np.array([190.0]), np.array([0.5]), 51, fhd.name)
    behaviour = BehaviourScenario(0.0, 484.48, 242.24, 45.0, 1937.92, 10.0)
    model = RotorTransitModel(
        behaviour, RotorGeometry(242.24, 150.0),
        rotor_plane_heading_deg_toward=0.0, flight_height_distribution=fhd,
    )
    observer_repeat = BandCrossingAccumulator(blade, band, 1, 3)
    events = []

    def report(event):
        events.extend(event.turbine_indices.tolist())
        observer_repeat(event)

    run = model.simulate(repeated_xy, pop, crossing_callback=report)
    assert np.array_equal(run.transit_matrix, [[True, True, True]])
    assert np.allclose(run.expected_transit_probability_matrix, [[1.0, 1.0, 1.0]])
    assert run.total_transits == 3 and run.conditional_rotor_transit_intensity == 3
    assert np.count_nonzero(observer_repeat.contribution_matrix) == 3
    assert math.isclose(observer_repeat.expected_potential_strikes_per_entrant,
                        float(np.sum(observer_repeat.contribution_matrix)))
    assert sorted(set(events)) == [0, 1, 2]
    assert len(events) > 3  # verifies duplicate notifications were deduplicated
    return {
        "one_positive_transit_probability": p,
        "zero_disk_transit_probability": observer_zero.expected_potential_strikes_per_entrant,
        "downstream_distinct_transits": run.total_transits,
        "downstream_accepted_bird_turbine_pairs": int(np.count_nonzero(observer_repeat.contribution_matrix)),
        "downstream_callback_notifications": len(events),
        "downstream_sum_of_accepted_strikes": observer_repeat.expected_potential_strikes_per_entrant,
    }


def main():
    blade, band, xy = checked_inputs()
    assert xy.shape == (12, 2)
    scenario = Scenario()
    scenario.turbine.rotor_diameter_m = 242.24
    aep = FLORISAEPModel(scenario, FLORIS_CONFIG)
    behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=242.24, response_probability=0.6,
        response_radius_D=2.0, encounter_radius_D=1.0,
        deflection_angle_deg=45.0, return_length_D=8.0, step_size_D=0.2,
    )
    fhd = DiscreteFlightHeightDistribution.from_csv(FHD_CSV)
    model = RotorTransitModel(
        behaviour, RotorGeometry(242.24, 150.0),
        rotor_plane_heading_deg_toward=225.0,
        flight_height_distribution=fhd,
    )
    pop = Model9Population.generate_parallel(
        flight_height_distribution=fhd, flight_height_seed=10142,
        site_width_m=6000.0, site_height_m=6000.0,
        heading_deg_toward=225.0, number_of_trajectories=512,
        maximum_response_events=12,
        longitudinal_margin_m=behaviour.response_radius_m + behaviour.return_length_m,
        trajectory_seed=142,
    )
    original_aep = aep.evaluate(xy)
    assert original_aep == 847.555657223372  # archived Model 9 energy endpoint
    original = model.simulate(xy, pop)
    observer = BandCrossingAccumulator(blade, band, 512, 12)
    observed = model.simulate(xy, pop, crossing_callback=observer)
    assert np.array_equal(original.meso_result.encounter_matrix, observed.meso_result.encounter_matrix)
    assert np.array_equal(original.meso_result.response_matrix, observed.meso_result.response_matrix)
    assert np.array_equal(original.meso_result.per_trajectory_encounters,
                          observed.meso_result.per_trajectory_encounters)
    assert np.array_equal(original.transit_matrix, observed.transit_matrix)
    assert np.array_equal(original.expected_transit_probability_matrix,
                          observed.expected_transit_probability_matrix)
    assert original.conditional_rotor_transit_intensity == observed.conditional_rotor_transit_intensity

    reference = observer.expected_potential_strikes_per_entrant
    assert reference > 0
    problem = Model10AEPCollisionProblem(
        scenario=scenario, aep_model=aep, rotor_model=model,
        populations=(pop,), blade=blade, band_scenario=band,
        aep_ceiling_gwh=854.4956585617186,
        aep_loss_reference_gwh=137.96914764866568,
        strike_reference=reference,
    )
    raw = problem.evaluate_raw_layout(xy)
    assert raw["aep_gwh"] == original_aep == aep.evaluate(xy)
    assert raw["E_RT_CE"] == original.conditional_rotor_transit_intensity
    assert raw["E_potential_strike"] == reference
    assert math.isclose(raw["N_potential_strikes"],
                        float(np.sum(raw["per_turbine_potential_strikes"])), abs_tol=1e-13)
    assert raw["accepted_bird_turbine_pairs"] == np.count_nonzero(observer.contribution_matrix)
    assert np.all(np.isfinite(observer.contribution_matrix))
    assert np.all(observer.contribution_matrix >= 0)
    assert np.all(observer.contribution_matrix <= observed.expected_transit_probability_matrix + 1e-12)
    objectives = problem.evaluate(xy.flatten(), return_values_of=["F", "G"])
    f, g = objectives
    assert math.isclose(f[0], raw["aep_loss_gwh"] / 137.96914764866568, abs_tol=1e-12)
    assert math.isclose(f[1], 1.0, abs_tol=1e-12)
    assert len(g) == 66 and np.max(g) < 0
    summary = {
        "status": "PASS",
        "model8_sha256": hashlib.sha256((ROOT / "upload/model8_meso_response(2).py").read_bytes()).hexdigest(),
        "model9_sha256": hashlib.sha256((ROOT / "upload/model9_rotor_transit.py").read_bytes()).hexdigest(),
        "band_sha256": hashlib.sha256((ROOT / "model10_band_strike.py").read_bytes()).hexdigest(),
        "floris_config_sha256": hashlib.sha256(FLORIS_CONFIG.read_bytes()).hexdigest(),
        "population_seed": 142, "flight_height_seed": 10142,
        "n_turbines": 12, "n_birds": 512,
        "aep_without_band_gwh": original_aep,
        "aep_with_band_gwh": raw["aep_gwh"],
        "original_E_RT_CE": original.conditional_rotor_transit_intensity,
        "with_band_E_RT_CE": raw["E_RT_CE"],
        "sampled_rotor_transits": raw["N_RT_farm"],
        "total_expected_potential_strikes": raw["N_potential_strikes"],
        "E_potential_strike_per_entrant": raw["E_potential_strike"],
        "accepted_bird_turbine_pairs": raw["accepted_bird_turbine_pairs"],
        "sum_per_turbine_strikes": float(np.sum(raw["per_turbine_potential_strikes"])),
        "f0": float(f[0]), "f1": float(f[1]),
        "maximum_spacing_constraint": float(np.max(g)),
        "controlled_crossings": check_controlled_crossings(blade, band),
    }
    REPORT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
