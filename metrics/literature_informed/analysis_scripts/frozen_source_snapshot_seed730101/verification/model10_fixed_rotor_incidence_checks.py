"""Controlled fixed-plane Band incidence checks. No optimisation or final inputs.

Run: python verification/model10_fixed_rotor_incidence_checks.py
The selected IEA row and guidance gannet speed are *candidate test inputs*.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import replace

import numpy as np

import model10_band_strike_checks as old
from model10_band_strike import (
    BandCrossingAccumulator, BandScenario, local_strike_probability,
)


def candidate_source_row() -> dict[str, float]:
    """Read A/B/F from the same original Rotor Performance workbook row 34."""
    with zipfile.ZipFile(old.SOURCE) as archive:
        raw = archive.read("IEA-15-240-RWT_tabular.xlsx")
    assert hashlib.sha256(raw).hexdigest() == (
        "9ed6ee219a97c443d0b900498e3ef4fefa12d855d9a7c9e4c5b9bc0702b9078a"
    )
    with zipfile.ZipFile(io.BytesIO(raw)) as workbook:
        names = workbook.read("xl/workbook.xml").decode()
        assert 'name="Rotor Performance"' in names
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet18.xml"))
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    wind_rows = []
    for source_row in sheet.findall(".//s:row", namespace):
        for cell in source_row.findall("s:c", namespace):
            if cell.attrib["r"] == f"A{source_row.attrib['r']}":
                value = cell.find("s:v", namespace)
                if value is not None:
                    wind_rows.append((int(source_row.attrib["r"]), float(value.text)))
    assert min(wind_rows, key=lambda entry: abs(entry[1] - 12.0))[0] == 34
    values = {}
    for cell in sheet.findall(".//s:row[@r='34']/s:c", namespace):
        if cell.attrib["r"] in ("A34", "B34", "F34"):
            values[cell.attrib["r"]] = float(cell.find("s:v", namespace).text)
    assert set(values) == {"A34", "B34", "F34"}
    assert math.isclose(values["A34"], 12.2589068261207, abs_tol=1e-12)
    assert math.isclose(values["B34"], 6.760740271637748, abs_tol=1e-12)
    assert math.isclose(values["F34"], 7.517553314698912, abs_tol=1e-12)
    return values


def scenario_from_row(row: dict[str, float]) -> BandScenario:
    return BandScenario(
        rotor_rpm=row["F34"], collective_pitch_deg=row["B34"],
        bird_speed_m_s=14.9, bird_length_m=0.94, bird_wingspan_m=1.72,
        flight_style="gliding", wind_heading_deg_toward=0.0,
    )


def independent_eq3(
    scenario: BandScenario, *, incidence_deg: float, approach: str,
) -> float:
    """Independent direct substitution for the 30 m lateral, +40 m height point."""
    omega = scenario.rotor_rpm * 2.0 * math.pi / 60.0
    v_normal = scenario.bird_speed_m_s * abs(math.cos(math.radians(incidence_deg)))
    chord, twist = old.BLADE.at_radius(50.0)
    blade_pitch = math.radians(twist + scenario.collective_pitch_deg)
    alpha = v_normal / (50.0 * omega)
    sign = 1.0 if approach == "upwind" else -1.0
    width = abs(sign * chord * math.sin(blade_pitch)
                + alpha * chord * math.cos(blade_pitch))
    bird_extent = max(scenario.bird_length_m, scenario.bird_wingspan_m * alpha * 0.8)
    return min(1.0, 3 * omega / (2.0 * math.pi * v_normal)
               * (width + bird_extent))


def candidate_fixed_crossings(row: dict[str, float]) -> dict:
    # One actual Model 9 crossing and FHD at 190 m; r=sqrt(30²+40²)=50 m.
    model9, _, batches = old.one_crossing(30.0, heights=(190.0,))
    assert model9.conditional_rotor_transit_intensity == 1.0
    assert len(batches) == 2  # duplicate inclusive step endpoint
    batch = batches[0]
    assert batch.rotor_plane_heading_deg_toward == 0.0
    assert np.array_equal(batch.crossing_xy_m, [[0.0, 0.0]])
    assert np.array_equal(batch.lateral_offset_m, [30.0])

    # Three actual line geometries cross the *same fixed* x=0 rotor plane at
    # (0,0), with identical turbine and horizontal offset. Check Model 9's
    # own intersection function before swapping their heading into the batch.
    starts = np.array([[-10., 0.], [-10., -10.], [-10., 10.], [10., 0.]])
    ends = np.array([[10., 0.], [10., 10.], [10., -10.], [-10., 0.]])
    mask, offset = old.R._segment_rotor_plane_crossings(
        starts, ends, np.array([[0., -30.]]), 0.0,
    )
    assert np.all(mask) and np.array_equal(offset[:, 0], [30., 30., 30., 30.])

    candidate = scenario_from_row(row)

    def contribution(local_heading: float, inputs: BandScenario) -> float:
        event = replace(
            batch,
            segment_heading_deg_toward=np.array([local_heading]),
            crossing_direction_sign=np.array([1 if math.cos(math.radians(local_heading)) > 0 else -1]),
        )
        assert np.array_equal(event.crossing_xy_m, batch.crossing_xy_m)
        assert np.array_equal(event.lateral_offset_m, batch.lateral_offset_m)
        observer = BandCrossingAccumulator(
            old.BLADE, inputs, number_of_trajectories=1, number_of_turbines=1
        )
        observer(event)
        observer(event)  # inclusive-endpoint duplicate must not add contribution
        return observer.expected_potential_strikes_per_entrant

    down_normal = contribution(0.0, candidate)
    down_plus = contribution(45.0, candidate)
    down_minus = contribution(-45.0, candidate)
    up_normal = contribution(180.0, candidate)
    up_plus = contribution(135.0, candidate)
    up_minus = contribution(225.0, candidate)
    up_reversed_wind = contribution(0.0, replace(candidate, wind_heading_deg_toward=180.0))
    down_slower = contribution(0.0, replace(candidate, bird_speed_m_s=14.01))

    assert math.isclose(down_plus, down_minus, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(up_plus, up_minus, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(up_normal, up_reversed_wind, rel_tol=0, abs_tol=1e-15)
    assert 0.0 < down_normal < up_normal < 1.0
    assert down_plus != down_normal and up_plus != up_normal
    assert down_slower != down_normal
    for value, delta, side in (
        (down_normal, 0, "downwind"), (down_plus, 45, "downwind"),
        (down_minus, -45, "downwind"), (up_normal, 0, "upwind"),
        (up_plus, 45, "upwind"), (up_minus, -45, "upwind"),
    ):
        assert math.isclose(value, independent_eq3(candidate, incidence_deg=delta,
                                                    approach=side), abs_tol=1e-15)
    assert math.isclose(down_slower, independent_eq3(
        replace(candidate, bird_speed_m_s=14.01), incidence_deg=0,
        approach="downwind"), abs_tol=1e-15)
    assert math.isclose(down_normal, local_strike_probability(
        radius_m=50.0, vertical_from_hub_m=40.0, blade=old.BLADE,
        scenario=candidate, approach="downwind"), abs_tol=1e-15)
    for invalid in (float("nan"), 0.0, -0.5, 1.001):
        try:
            local_strike_probability(
                radius_m=50., vertical_from_hub_m=40., blade=old.BLADE,
                scenario=candidate, approach="downwind", incidence_cosine=invalid,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid incidence was accepted")

    return {
        "reference_bird_speed_m_s": candidate.bird_speed_m_s,
        "turned_normal_speed_m_s": candidate.bird_speed_m_s * math.cos(math.radians(45)),
        "fixed_crossing_ertce_all_cases": model9.conditional_rotor_transit_intensity,
        "downwind_p_normal": down_normal,
        "downwind_p_plus45": down_plus,
        "downwind_p_minus45": down_minus,
        "upwind_p_normal": up_normal,
        "upwind_p_plus45": up_plus,
        "upwind_p_minus45": up_minus,
        "downwind_p_normal_speed14_01": down_slower,
        "downwind_turn45_pct": 100 * (down_plus/down_normal - 1),
        "upwind_turn45_pct": 100 * (up_plus/up_normal - 1),
        "downwind_speed14_01_pct": 100 * (down_slower/down_normal - 1),
    }


def model8_model9_regression(row: dict[str, float]) -> dict:
    before = json.loads(old.PREVIOUS.SNAPSHOT.read_text())
    model, layout, population = old.PREVIOUS.baseline_case(old.M, old.R, old.V)
    unobserved = model.simulate(layout, population, store_path_indices=(0, 13))
    assert old.PREVIOUS.result_signature(unobserved) == before["result"]
    scenario = replace(scenario_from_row(row), wind_heading_deg_toward=225.0)
    band = BandCrossingAccumulator(old.BLADE, scenario, len(population.flight_heights_m), len(layout))
    observed_deltas = []

    def callback(batch):
        assert batch.rotor_plane_heading_deg_toward == 225.0
        observed_deltas.extend(
            ((h - batch.rotor_plane_heading_deg_toward + 180) % 360) - 180
            for h in batch.segment_heading_deg_toward
        )
        band(batch)

    with_callback = model.simulate(layout, population, store_path_indices=(0, 13),
                                   crossing_callback=callback)
    assert old.PREVIOUS.result_signature(with_callback) == before["result"]
    assert any(0 < abs(delta) <= 45.0000001 for delta in observed_deltas)
    assert all(abs(delta) <= 45.0000001 for delta in observed_deltas)
    assert 0 <= band.expected_potential_strikes_per_entrant <= (
        with_callback.conditional_rotor_transit_intensity
    )
    return {
        "actual_crossing_events": len(observed_deltas),
        "turned_crossing_events": int(sum(abs(d) > 1e-8 for d in observed_deltas)),
        "same_saved_model8_model9_result_with_without_callback": True,
    }


if __name__ == "__main__":
    row = candidate_source_row()
    print(json.dumps({
        "status": "all fixed-rotor controlled checks passed; conventions remain provisional",
        "source_workbook_sheet_row": "Rotor Performance!A34/B34/F34, IEA v1.1.18",
        "source_wind_m_s": row["A34"],
        "source_pitch_deg": row["B34"],
        "source_rpm": row["F34"],
        **candidate_fixed_crossings(row),
        **model8_model9_regression(row),
    }, indent=2))
