"""Run the archived Model 10 checks across Python/NumPy platforms.

The archived Model 8/9 regression fixture hashes every coordinate in two
optional stored path traces. Different libm implementations can change their
last floating-point bits without changing any encounter, response, exposure,
summary, or Band-callback result. Only those two traces may use a numerical
comparison against the original archived coordinates. Every other archived
field remains subject to the original exact equality check.

This is an independent verification wrapper; frozen model modules and their
original test files are not edited. Do not use this test as model calibration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "verification"),
                str(ROOT / "verification/_runtime_source"), str(ROOT / "upload")]
PATHS = ROOT / "verification/model10_preedit_path_coordinates.npz"
PATHS_SHA256 = "ab31299acfa1919fda8c78897a03dae6c1f459e628f760acdc04b02a3fd68c6d"
PATH_TOLERANCE_M = 1e-8


def imported(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load existing check: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.report_dir.is_dir():
        raise FileNotFoundError("The new verification stage is missing")
    if hashlib.sha256(PATHS.read_bytes()).hexdigest() != PATHS_SHA256:
        raise AssertionError("Archived path-coordinate fixture was modified")

    band = imported("model10_band_strike_checks",
                    ROOT / "verification/model10_band_strike_checks.py")
    old = band.PREVIOUS
    original = json.loads(old.SNAPSHOT.read_text(encoding="utf-8"))
    model, layout, population = old.baseline_case(band.M, band.R, band.V)
    plain = model.simulate(layout, population, store_path_indices=(0, 13))
    current = old.result_signature(plain)
    archived = original["result"]
    if archived["summary"] != current["summary"]:
        raise AssertionError("Archived Model 8/9 summary metrics differ")
    for section in ("model8_fields", "model9_arrays"):
        if set(current[section]) != set(archived[section]):
            raise AssertionError(f"Archived {section} field names changed")
        for name, prior in archived[section].items():
            if section == "model8_fields" and name == "stored_paths_m":
                continue
            if current[section][name] != prior:
                raise AssertionError(f"Archived {section}.{name} changed")

    previous_paths = archived["model8_fields"]["stored_paths_m"]
    current_paths = plain.meso_result.stored_paths_m
    if set(current_paths) != {0, 13} or set(previous_paths) != {"0", "13"}:
        raise AssertionError("Expected exactly the two archived optional paths")
    differences = {}
    with np.load(PATHS, allow_pickle=False) as reference:
        if set(reference.files) != {"bird_0", "bird_13"}:
            raise AssertionError("Archived path coordinates have wrong keys")
        for bird_index in (0, 13):
            expected = reference[f"bird_{bird_index}"]
            actual = current_paths[bird_index]
            fingerprint = previous_paths[str(bird_index)]
            if (hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()
                    != fingerprint["sha256"]):
                raise AssertionError("Numerical fixture differs from archived path hash")
            if (actual.shape != expected.shape or actual.dtype != expected.dtype
                    or actual.ndim != 2 or actual.shape[1] != 2
                    or not np.all(np.isfinite(actual))):
                raise AssertionError("Path shape, coordinates or dtype changed")
            max_delta = float(np.max(np.abs(actual - expected)))
            if max_delta > PATH_TOLERANCE_M:
                raise AssertionError(
                    f"Bird {bird_index} path coordinates differ by {max_delta:.12g} m"
                )
            differences[f"bird_{bird_index}_maximum_coordinate_difference_m"] = max_delta

    # The original existing checks read SNAPSHOT, so give them a clearly
    # identified copy with only the two portable path hashes updated. They
    # continue to require exact equality for every other archived field.
    portable = copy.deepcopy(original)
    portable["result"]["model8_fields"]["stored_paths_m"] = current["model8_fields"]["stored_paths_m"]
    portability_snapshot = args.report_dir / "portable_regression_snapshot.json"
    portability_snapshot.write_text(json.dumps(portable, indent=2) + "\n", encoding="utf-8")
    old.SNAPSHOT = portability_snapshot
    band.geometry_checks()
    band.kernel_checks()
    down, up = band.fixed_crossing_checks()
    downstream = band.downstream_and_label_checks()
    exposure, weighted, unit_error = band.full_disc_p1_and_regression()

    rotor = imported("model10_fixed_rotor_incidence_checks",
                     ROOT / "verification/model10_fixed_rotor_incidence_checks.py")
    selected_row = rotor.candidate_source_row()
    rotor.candidate_fixed_crossings(selected_row)
    rotor.model8_model9_regression(selected_row)

    report = {
        "status": "PASS", "exact_archived_snapshot": current == archived,
        "archived_scalar_and_all_other_array_checks": "exact match",
        "stored_path_coordinate_tolerance_m": PATH_TOLERANCE_M,
        "archived_stored_path_numeric_comparison": differences,
        "existing_band_checks": "PASS",
        "existing_fixed_rotor_incidence_checks": "PASS",
        "band_downwind_fixture_probability": float(down),
        "band_upwind_fixture_probability": float(up),
        "conditional_disc_exposure_fixture": float(exposure),
        "band_weighted_fixture_contribution": float(weighted),
        "unit_strike_matrix_max_error": float(unit_error),
        "downstream_two_bird_fixture_contribution": float(downstream),
        "original_snapshot_sha256": hashlib.sha256(
            (ROOT / "verification/model9_crossing_pre_edit.json").read_bytes()).hexdigest(),
        "reference_path_fixture_sha256": PATHS_SHA256,
        "portable_regression_snapshot_sha256": hashlib.sha256(
            portability_snapshot.read_bytes()).hexdigest(),
    }
    (args.report_dir / "portable_checks_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
