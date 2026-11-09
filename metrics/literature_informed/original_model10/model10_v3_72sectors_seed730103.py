"""Model 10: 72-sector rerun of the reference-seeded northern-gannet FHD optimisation.

Stages (use the same --output-dir for all stages):
    python -u model10_fhd_robust_seeded_50k.py verify
    python -u model10_fhd_robust_seeded_50k.py benchmark
    python -u model10_fhd_robust_seeded_50k.py smoke
    python -u model10_fhd_robust_seeded_50k.py optimise

The final stage is intentionally separate: review the benchmark before running
it. The holdout script must be run only after the training-selected coordinates
are frozen. None of the frozen Model 8/9/10, Band, or FLORIS code is changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "upload"), str(ROOT / "verification"),
                str(ROOT / "verification/_runtime_source")]

from config import Scenario  # noqa: E402
from floris_aep import FLORISAEPModel  # noqa: E402
from model8_meso_response import BehaviourScenario  # noqa: E402
from model9_rotor_transit import (  # noqa: E402
    DiscreteFlightHeightDistribution, RotorGeometry, RotorTransitModel,
)
from model10_band_strike import BandScenario, ProjectedBladeStations  # noqa: E402
from model10_aep_collision_problem import Model10AEPCollisionProblem  # noqa: E402
from model10_run_baseline_nsga2 import (  # noqa: E402
    Progress, load_layout, populations, verify_file, write_csv,
)
from run_model8_validation import make_fixed_layouts  # noqa: E402

CONFIG = ROOT / "verification/library_inputs/thesis/gch_iea15mw(2).yaml"
REFERENCE = ROOT / "verification/runtime_inputs/energy_baseline.csv"
FHD_CSV = ROOT / "upload/northern_gannet_Johnston2014_FHD(3).csv"
BLADE_ZIP = ROOT / "upload/IEA15MW_Band_input_sources(1).zip"
LOSS_REFERENCE_GWH = 137.96914764866568
RISK_REFERENCE = 3.20618950134584e-5
N_DIRECTION_SECTORS = 72
POPULATION = 24
GENERATIONS = 20
TRAIN_BATCHES = 5
TRAIN_BIRDS = 10_000
CONFIRM_BATCHES = 10
CONFIRM_BIRDS = 50_000
MINIMUM_IMPROVEMENT = 0.02
OPTIMISER_SEED = 730103
INITIALISATION_SEED = 730301
TRAIN_TRAJECTORY_SEED = 731101
TRAIN_HEIGHT_SEED = 732101
SMOKE_TRAJECTORY_SEED = 733101
SMOKE_HEIGHT_SEED = 734101
CONFIRM_TRAJECTORY_SEED = 735101
CONFIRM_HEIGHT_SEED = 736101
VERIFY_TRAJECTORY_SEED = 737101
VERIFY_HEIGHT_SEED = 738101
SMOKE_OPTIMISER_SEED = 739101
HYPERVOLUME_REFERENCE = (2.0, 2.0)  # minimised, dimensionless objectives

SOURCE_FILES = {
    "runner": "model10_v3_72sectors_seed730103.py",
    "confirmation": "model10_fhd_robust_confirm.py",
    "model10_objective": "model10_aep_collision_problem.py",
    "model10_band": "model10_band_strike.py",
    "model10_baseline_runner": "model10_run_baseline_nsga2.py",
    "model8": "upload/model8_meso_response(2).py",
    "model8_sampler": "upload/run_model8_multiobjective(1).py",
    "model8_validation": "upload/run_model8_validation(1).py",
    "model9": "upload/model9_rotor_transit.py",
    "config": "upload/config(7).py",
    "floris_wrapper": "upload/floris_aep(6).py",
    "floris_config": "verification/library_inputs/thesis/gch_iea15mw(2).yaml",
    "gannet_FHD": "upload/northern_gannet_Johnston2014_FHD(3).csv",
    "blade_source": "upload/IEA15MW_Band_input_sources(1).zip",
    "reference_layout": "verification/runtime_inputs/energy_baseline.csv",
    "band_checks": "verification/model10_band_strike_checks.py",
    "rotor_incidence_checks": "verification/model10_fixed_rotor_incidence_checks.py",
    "integration_checks": "verification/model10_integrated_fixed_layout_check.py",
    "model9_checks": "verification/model9_crossing_interface_checks.py",
    "preedit_snapshot": "verification/model9_crossing_pre_edit.json",
    "preedit_path_coordinates": "verification/model10_preedit_path_coordinates.npz",
    "portable_snapshot_checks": "verification/model10_portable_snapshot_checks.py",
}

EXPECTED_HASHES = {
    "model8": "2d9985f6c165a83fac5ed1493cefa1a65e8082a39d17e80e3776be12257e823d",
    "model9": "3c821d5f62de8b292bc00190ae71ae1379be64ad4cd11c67dc631186b5650f09",
    "model10_band": "2c095e33563323fb60a8e7d70cb3ab7b5a9672adbf0df98da32d006fa7ed96e0",
    "floris_config": "f6b788df9e38f006dfb6c68ddfb80ab090dad9cd555982a8d5c5e4b0c2f2d364",
    "gannet_FHD": "adc9ea903db9c8e34bfe5b88c382381b737c2e676d79af437f66d198208a0f2c",
    "floris_turbine": "93d2bcfcb619b1819efe8149ce9e8231656e51a29f2ae5c1079a46e339f02feb",
    "preedit_path_coordinates": "ab31299acfa1919fda8c78897a03dae6c1f459e628f760acdc04b02a3fd68c6d",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    import floris

    hashes = {name: digest(ROOT / path) for name, path in SOURCE_FILES.items()}
    for name, expected in EXPECTED_HASHES.items():
        if name == "floris_turbine":
            path = Path(floris.__file__).parent / "turbine_library/iea_15MW.yaml"
        else:
            path = ROOT / SOURCE_FILES[name]
        verify_file(path, expected)
        hashes[name] = expected
    return hashes


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_stage(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing stage: {path}")
    path.mkdir(parents=True)
    return path


def model_components():
    hashes = source_hashes()
    scenario = Scenario()
    scenario.wind.n_sectors = N_DIRECTION_SECTORS
    aep = FLORISAEPModel(scenario, CONFIG)
    diameter = float(aep.floris_rotor_diameter_m)
    if (scenario.site.n_turbines != 12 or scenario.site.width_m != 6000
            or scenario.site.height_m != 6000 or scenario.site.min_spacing_D != 4.0
            or scenario.site.wind_from_deg != 225.0 or diameter != 242.24
            or float(aep.floris_hub_height_m) != 150.0):
        raise AssertionError("Site, wind or turbine settings changed")
    scenario.turbine.rotor_diameter_m = diameter
    behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=diameter, response_probability=0.6,
        response_radius_D=2.0, encounter_radius_D=1.0,
        deflection_angle_deg=45.0, return_length_D=8.0, step_size_D=0.2,
    )
    fhd = DiscreteFlightHeightDistribution.from_csv(FHD_CSV)
    assert len(fhd.heights_m) == 300 and np.array_equal(fhd.heights_m, np.arange(300))
    assert math.isclose(float(np.sum(fhd.probability_mass)), 1.0, abs_tol=1e-12)
    assert fhd.input_sha256 == EXPECTED_HASHES["gannet_FHD"]
    assert not (len(fhd.heights_m) == 1 and fhd.heights_m[0] == 150)
    rotor = RotorTransitModel(
        behaviour, RotorGeometry(diameter, 150.0),
        rotor_plane_heading_deg_toward=225.0,
        flight_height_distribution=fhd,
    )
    band = BandScenario(
        rotor_rpm=7.517553314698912,
        collective_pitch_deg=6.760740271637748,
        bird_speed_m_s=14.9, bird_length_m=0.94,
        bird_wingspan_m=1.72, flight_style="gliding",
        wind_heading_deg_toward=45.0,
    )
    blade = ProjectedBladeStations.from_source_zip(BLADE_ZIP)
    reference = load_layout(REFERENCE)
    reference_aep = float(aep.evaluate(reference))
    if not math.isclose(reference_aep, 804.390138, abs_tol=1e-3):
        raise AssertionError(
            f"72-sector reference AEP check failed: {reference_aep:.12f} GWh/yr"
        )
    ceiling = float(aep.ideal_aep(12))
    if not math.isclose(ceiling, 854.4956585617186, abs_tol=1e-7):
        raise AssertionError("FLORIS ideal AEP changed")
    return hashes, scenario, aep, behaviour, fhd, rotor, band, blade, reference, ceiling


def create_problem(components, birds, strike_reference=1.0, *, audit=False):
    _, scenario, aep, _, _, rotor, band, blade, _, ceiling = components
    cls = AuditedProblem if audit else Model10AEPCollisionProblem
    return cls(
        scenario=scenario, aep_model=aep, rotor_model=rotor,
        populations=birds, blade=blade, band_scenario=band,
        aep_ceiling_gwh=ceiling, aep_loss_reference_gwh=LOSS_REFERENCE_GWH,
        strike_reference=strike_reference, record_evaluations=False,
    )


def make_birds(components, count, batches, trajectory_seed, height_seed):
    birds = populations(
        fhd=components[4], behaviour=components[3], count=count,
        batches=batches, seed=trajectory_seed, flight_seed=height_seed,
    )
    assert all(b.number_of_trajectories == count for b in birds)
    assert [b.trajectory_seed for b in birds] == [trajectory_seed + i for i in range(batches)]
    assert [b.flight_height_seed for b in birds] == [height_seed + i for i in range(batches)]
    return birds


def xy_columns(xy):
    return {f"{axis}_{i:02d}_m": float(xy[i, j])
            for i in range(12) for j, axis in enumerate(("x", "y"))}


def row_xy(row):
    return np.asarray([[float(row[f"x_{i:02d}_m"]), float(row[f"y_{i:02d}_m"])]
                       for i in range(12)])


def check_layout(xy, minimum):
    points = np.asarray(xy, dtype=float).reshape(12, 2)
    if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 6000):
        raise AssertionError("A layout is outside the site")
    shortest = min(float(np.linalg.norm(points[i] - points[j]))
                   for i in range(12) for j in range(i + 1, 12))
    if shortest + 1e-8 < minimum:
        raise AssertionError(f"Turbines closer than 4D: {shortest:.8f} < {minimum}")
    return shortest


class AuditedProblem(Model10AEPCollisionProblem):
    """Record full candidate geometry and the five unaltered per-batch scores."""

    def _evaluate(self, X, out, *args, **kwargs):
        objective, constraints = [], []
        for vector in np.atleast_2d(np.asarray(X, dtype=float)):
            xy = vector.reshape(12, 2)
            raw = self.evaluate_raw_layout(xy)
            g = self.spacing_constraints(xy)
            f0 = raw["aep_loss_gwh"] / self.aep_loss_reference_gwh
            f1 = raw["E_potential_strike"] / self.strike_reference
            if (not np.isclose(float(np.mean([b["E_potential_strike"] for b in raw["batch_rows"]])),
                               raw["E_potential_strike"], rtol=0, atol=1e-15)):
                raise AssertionError("Bird objective is not the five-batch mean")
            objective.append((f0, f1))
            constraints.append(g)
            self.evaluation_rows.append({
                "evaluation_index": len(self.evaluation_rows),
                "aep_gwh": raw["aep_gwh"],
                "risk_per_entrant": raw["E_potential_strike"],
                "conditional_rotor_transits_per_entrant": raw["E_RT_CE"],
                "N_started": raw["N_started"],
                "accepted_bird_turbine_pairs": raw["accepted_bird_turbine_pairs"],
                "f0_normalised_aep_loss": f0,
                "f1_normalised_risk": f1,
                "max_spacing_constraint_m": float(np.max(g)),
                "minimum_spacing_m": check_layout(xy, self.minimum_spacing_m)
                if np.max(g) <= 0 and np.all(xy >= 0) and np.all(xy <= 6000) else None,
                "max_boundary_violation_m": float(max(0.0, -np.min(xy), np.max(xy) - 6000)),
                **{f"batch_{i:02d}_risk": float(b["E_potential_strike"])
                   for i, b in enumerate(raw["batch_rows"])},
                **xy_columns(xy),
            })
        out["F"] = np.asarray(objective, dtype=float)
        out["G"] = np.asarray(constraints, dtype=float)


def sampler_module():
    path = ROOT / SOURCE_FILES["model8_sampler"]
    spec = importlib.util.spec_from_file_location("frozen_model8_fhd_sampler", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference_variants(reference, module, minimum):
    """Small geometry changes in coordinates aligned with flight at 225°."""
    along = np.array([math.cos(math.radians(225)), math.sin(math.radians(225))])
    across = np.array([-along[1], along[0]])
    centre = np.mean(reference, axis=0)
    s = (reference - centre) @ along
    t = (reference - centre) @ across
    order = np.argsort(s)
    variants = {"reference": reference}

    def add_backed_off(name, generate, steps):
        for step in steps:
            xy = generate(step)
            if (module.layout_is_feasible(xy, width_m=6000.0, height_m=6000.0,
                                          minimum_spacing_m=minimum)
                    and all(np.max(np.abs(xy - old)) > 1e-7 for old in variants.values())):
                variants[name] = xy
                return
        raise AssertionError(f"Could not build a distinct feasible reference variation: {name}")

    for group, indices in (("entry", order[:4]), ("exit", order[-4:])):
        for side in (-1, 1):
            def move(d, selected=indices, direction=side):
                xy = reference.copy()
                xy[selected] += direction * d * across
                return xy
            add_backed_off(f"{group}_cross_{side:+d}", move,
                           (60., 45., 30., 15., 7.5, 3.75))
    for sign in (-1, 1):
        add_backed_off(f"rotate_{sign:+d}",
                       lambda angle, direction=sign:
                       centre + (reference - centre) @ np.array([
                           [math.cos(direction * angle), math.sin(direction * angle)],
                           [-math.sin(direction * angle), math.cos(direction * angle)]]),
                       (math.radians(2.0), math.radians(1.0), math.radians(.5)))
    for axis, component in (("along", s), ("across", t)):
        for sign in (-1, 1):
            add_backed_off(f"{axis}_scale_{sign:+d}",
                           lambda amount, a=axis, direction=sign:
                           centre + ((s * (1.0 + direction * amount) if a == "along" else s)[:, None] * along
                                     + (t * (1.0 + direction * amount) if a == "across" else t)[:, None] * across),
                           (.012, .01, .006, .003))
    for group, indices in (("entry", order[:4]), ("exit", order[-4:])):
        def shift(d, selected=indices):
            xy = reference.copy()
            xy[selected] += d * along
            return xy
        add_backed_off(f"{group}_along_shift", shift, (25., 15., 8., -8., -15.))
    return variants


def initial_population(components, problem, n, seed, random_reserved):
    module = sampler_module()
    ref = components[8]
    controls = make_fixed_layouts(site_width_m=6000, site_height_m=6000,
                                  bird_heading_deg_toward=225,
                                  minimum_spacing_m=problem.minimum_spacing_m)
    candidates = {**reference_variants(ref, module, problem.minimum_spacing_m), **controls}
    if n < 5 or random_reserved < 1 or random_reserved >= n:
        raise ValueError("Need reference, structured layouts, and random diversity")
    exact = dict(list(candidates.items())[:n - random_reserved])
    if "reference" not in exact:
        raise AssertionError("The exact external reference must be first")
    seed_sampler = module.FeasibleSeededSampling(
        scenario=components[1], rotor_diameter_m=242.24,
        exact_seed_layouts=exact, seed=seed,
    )
    initial = list(seed_sampler._do(problem, n - random_reserved))
    labels = list(seed_sampler.last_labels)
    rng = np.random.default_rng(seed + 777)
    attempts = 0
    while len(initial) < n:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Failed to add distinct random feasible layouts")
        xy = module.random_feasible_layout(
            rng=rng, number_of_turbines=12, width_m=6000, height_m=6000,
            minimum_spacing_m=problem.minimum_spacing_m)
        flat = xy.ravel()
        if not any(np.max(np.abs(flat - item)) <= 1e-7 for item in initial):
            initial.append(flat)
            labels.append("random_reserved")
    arr = np.asarray(initial, dtype=float).reshape(n, 12, 2)
    if not np.array_equal(arr[0], ref):
        raise AssertionError("The reference is not the first initial layout")
    for xy in arr:
        check_layout(xy, problem.minimum_spacing_m)
    if not any("jitter" in label for label in labels) and n >= 24:
        raise AssertionError("Missing limited coordinate jitter")
    if sum(label == "random_reserved" for label in labels) != random_reserved:
        raise AssertionError("Missing distinct random layouts")
    return arr, labels


class AuditedProgress(Progress):
    def __init__(self, destination, problem):
        super().__init__(destination / "progress.jsonl")
        self.destination_dir = destination
        self.problem = problem
        self.hypervolume = HV(ref_point=np.asarray(HYPERVOLUME_REFERENCE))

    def notify(self, algorithm):
        f = np.asarray(algorithm.pop.get("F"), dtype=float)
        cv = np.asarray(algorithm.pop.get("CV"), dtype=float).reshape(-1)
        feasible = f[cv <= 1e-9]
        record = {
            "generation": int(algorithm.n_gen),
            "n_evaluations": int(algorithm.evaluator.n_eval),
            "feasible_count": len(feasible),
            "min_normalised_aep_loss": float(np.min(feasible[:, 0])) if len(feasible) else None,
            "min_normalised_risk": float(np.min(feasible[:, 1])) if len(feasible) else None,
            "normalised_hypervolume": float(self.hypervolume(feasible)) if len(feasible) else None,
            "hypervolume_reference": HYPERVOLUME_REFERENCE,
            "elapsed_s": round(time.perf_counter() - self.start, 3),
        }
        with self.destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        # Checkpoint all candidate coordinates and five batch scores each generation.
        write_csv(self.destination_dir / "all_evaluated_candidates.csv",
                  self.problem.evaluation_rows)
        print(json.dumps(record), flush=True)


def saved_sources(destination):
    for path in SOURCE_FILES.values():
        target = destination / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)


def run_existing_checks(destination):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), str(ROOT / "upload"), str(ROOT / "verification/_runtime_source"),
         env.get("PYTHONPATH", "")))
    # The existing archived fixture hashes optional plotting paths byte for
    # byte. Its tiny cross-platform float changes are checked numerically
    # against the archived coordinates; all summary and physical array hashes
    # must still match the original snapshot exactly.
    cmd = [sys.executable, str(ROOT / SOURCE_FILES["portable_snapshot_checks"]),
           "--report-dir", str(destination.resolve())]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Existing Model 10 checks failed:\n"
                           + result.stdout + "\n" + result.stderr)
    checks = json.loads((destination / "portable_checks_report.json").read_text())
    if checks["status"] != "PASS":
        raise AssertionError("Portable checks did not pass")
    # This existing check writes a report; redirect it into this new stage.
    path = ROOT / SOURCE_FILES["integration_checks"]
    spec = importlib.util.spec_from_file_location("fhd_unchanged_integration_check", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.REPORT = destination / "existing_integration_check.json"
    module.main()
    (destination / "existing_checks_stdout.txt").write_text(
        result.stdout, encoding="utf-8")
    return ["band_checks", "rotor_incidence_checks", "integration_checks"]


def verify(args):
    components = model_components()
    birds = make_birds(components, 256, 2, VERIFY_TRAJECTORY_SEED, VERIFY_HEIGHT_SEED)
    problem = create_problem(components, birds)
    reference = components[8]
    minimum = check_layout(reference, problem.minimum_spacing_m)
    first = problem.evaluate_raw_layout(reference)
    second = problem.evaluate_raw_layout(reference)
    for key in ("aep_gwh", "E_RT_CE", "E_potential_strike", "N_potential_strikes"):
        if first[key] != second[key]:
            raise AssertionError(f"Repeated reference evaluation changed: {key}")
    if first["batch_rows"] != second["batch_rows"]:
        raise AssertionError("The exact paired bird batches changed")
    stage = new_stage(args.output_dir / "verify")
    checks = run_existing_checks(stage)
    report = {
        "status": "PASS", "scenario": "northern-gannet FHD", "FHD": components[4].settings(),
        "delta_150m_active": False, "source_hashes": components[0],
        "reference_aep_gwh": first["aep_gwh"],
        "reference_minimum_pairwise_spacing_m": minimum,
        "same_seed_reference_evaluations_exactly_equal": True,
        "verification_trajectory_seeds": [VERIFY_TRAJECTORY_SEED + i for i in range(2)],
        "verification_height_seeds": [VERIFY_HEIGHT_SEED + i for i in range(2)],
        "existing_checks_passed": checks,
        "portable_snapshot_checks": json.loads(
            (stage / "portable_checks_report.json").read_text()),
    }
    write_json(stage / "report.json", report)
    print("Gannet FHD, reference, reproducibility and Model 10 checks PASS", flush=True)


def required_verification(args):
    path = args.output_dir / "verify/report.json"
    if not path.exists():
        raise RuntimeError("Run verify first")
    prior = json.loads(path.read_text(encoding="utf-8"))
    if prior["status"] != "PASS" or prior["source_hashes"] != source_hashes():
        raise RuntimeError("Verification failed or source inputs changed")


def benchmark(args):
    required_verification(args)
    components = model_components()
    start_birds = time.perf_counter()
    birds = make_birds(components, TRAIN_BIRDS, TRAIN_BATCHES,
                       TRAIN_TRAJECTORY_SEED, TRAIN_HEIGHT_SEED)
    birds_seconds = time.perf_counter() - start_birds
    problem = create_problem(components, birds)
    start_evaluation = time.perf_counter()
    raw = problem.evaluate_raw_layout(components[8])
    seconds = time.perf_counter() - start_evaluation
    if raw["N_started"] != TRAIN_BATCHES * TRAIN_BIRDS:
        raise AssertionError("Not all 50,000 birds were evaluated")
    stage = new_stage(args.output_dir / "benchmark")
    write_json(stage / "report.json", {
        "status": "PASS", "birds_per_batch": TRAIN_BIRDS, "batches": TRAIN_BATCHES,
        "N_started": raw["N_started"],
        "trajectory_seeds": [TRAIN_TRAJECTORY_SEED + i for i in range(TRAIN_BATCHES)],
        "height_seeds": [TRAIN_HEIGHT_SEED + i for i in range(TRAIN_BATCHES)],
        "bird_generation_seconds": birds_seconds, "candidate_seconds": seconds,
        "480_evaluations_hours_excluding_overheads": seconds * POPULATION * GENERATIONS / 3600,
        "reference_aep_gwh": raw["aep_gwh"],
        "reference_risk_by_batch": raw["batch_rows"],
        "reference_training_mean_risk": raw["E_potential_strike"],
        "source_hashes": components[0],
    })
    print(f"One complete 50,000-bird reference layout: {seconds:.2f} s; "
          f"480 evaluations: ~{seconds * 480 / 3600:.2f} h plus overhead", flush=True)


def smoke(args):
    required_verification(args)
    components = model_components()
    birds = make_birds(components, 64, 2, SMOKE_TRAJECTORY_SEED,
                       SMOKE_HEIGHT_SEED)
    baseline = create_problem(components, birds).evaluate_raw_layout(components[8])
    if baseline["E_potential_strike"] <= 0:
        # A short smoke panel can miss every crossing. Scaling only affects
        # objective units; no training or holdout result uses these birds.
        normaliser = 1.0
    else:
        normaliser = baseline["E_potential_strike"]
    problem = create_problem(components, birds, normaliser, audit=True)
    initial, labels = initial_population(components, problem, 8,
                                         SMOKE_OPTIMISER_SEED, random_reserved=2)
    stage = new_stage(args.output_dir / "smoke")
    algorithm = NSGA2(pop_size=8, sampling=initial.reshape(8, 24),
                      eliminate_duplicates=True)
    result = minimize(problem, algorithm, termination=("n_gen", 2),
                      seed=SMOKE_OPTIMISER_SEED,
                      callback=AuditedProgress(stage, problem), verbose=False)
    for xy in np.asarray(result.pop.get("X")).reshape(-1, 12, 2):
        check_layout(xy, problem.minimum_spacing_m)
    if len(problem.evaluation_rows) < 8 or result.X is None:
        raise AssertionError("Small NSGA-II smoke run did not evaluate candidates")
    write_json(stage / "report.json", {
        "status": "PASS", "generations": 2, "population": 8,
        "evaluations": len(problem.evaluation_rows), "seed_sources": labels,
        "trajectory_seeds": [SMOKE_TRAJECTORY_SEED + i for i in range(2)],
        "height_seeds": [SMOKE_HEIGHT_SEED + i for i in range(2)],
        "optimiser_seed": SMOKE_OPTIMISER_SEED,
        "reference_aep_gwh": baseline["aep_gwh"],
        "source_hashes": components[0],
    })
    print("Two-generation gannet FHD optimisation smoke run PASS", flush=True)


def training_nondominated(rows, reference_aep, reference_risk):
    feasible = [r for r in rows
                if r["max_spacing_constraint_m"] <= 0
                and r["max_boundary_violation_m"] <= 0]
    if not feasible:
        raise RuntimeError("No feasible layouts were evaluated")
    ref = {"evaluation_index": -1, "aep_gwh": reference_aep,
           "risk_per_entrant": reference_risk}
    pool = [ref] + feasible
    nondominated = []
    for row in pool:
        if not any(
                (other["aep_gwh"] >= row["aep_gwh"]
                 and other["risk_per_entrant"] <= row["risk_per_entrant"])
                and (other["aep_gwh"] > row["aep_gwh"]
                     or other["risk_per_entrant"] < row["risk_per_entrant"])
                for other in pool):
            nondominated.append(row)
    # One geometry per exact objective pair, preserving the first occurrence.
    unique = {}
    for row in nondominated:
        key = (row["aep_gwh"], row["risk_per_entrant"])
        unique.setdefault(key, row)
    return sorted(unique.values(), key=lambda r: (-r["aep_gwh"], r["risk_per_entrant"])), feasible


def select_candidates(front, feasible, reference_aep, reference_risk):
    without_reference = [row for row in front if row["evaluation_index"] >= 0]
    reference_dominates_all = all(
        reference_aep >= r["aep_gwh"] and reference_risk <= r["risk_per_entrant"]
        for r in feasible)
    if reference_dominates_all:
        # Documented fixed scales: archived energy loss and training reference risk.
        def competitor_distance(r):
            return math.hypot((r["aep_gwh"] - reference_aep) / LOSS_REFERENCE_GWH,
                              (r["risk_per_entrant"] - reference_risk) / reference_risk)
        nearest = min(feasible, key=lambda r: (competitor_distance(r), r["evaluation_index"]))
        return [("nearest_competitor", nearest)], True
    if not without_reference:
        return [], False
    aeps = np.asarray([r["aep_gwh"] for r in front])
    risks = np.asarray([r["risk_per_entrant"] for r in front])
    aep_range = float(np.ptp(aeps))
    risk_range = float(np.ptp(risks))
    if aep_range == 0 or risk_range == 0:
        raise RuntimeError("Training front has zero objective range; knee is undefined")
    def knee_distance(r):
        return math.hypot((max(aeps) - r["aep_gwh"]) / aep_range,
                          (r["risk_per_entrant"] - min(risks)) / risk_range)
    rankings = (
        ("highest_aep", sorted(without_reference,
                               key=lambda r: (-r["aep_gwh"], r["risk_per_entrant"]))),
        ("lowest_risk", sorted(without_reference,
                              key=lambda r: (r["risk_per_entrant"], -r["aep_gwh"]))),
        ("balanced", sorted(without_reference,
                            key=lambda r: (knee_distance(r), r["evaluation_index"]))),
    )
    chosen, used = [], set()
    for label, ranked in rankings:
        row = next((r for r in ranked if r["evaluation_index"] not in used), None)
        if row is not None:
            used.add(row["evaluation_index"])
            chosen.append((label, row))
    return chosen, False


def optimise(args):
    required_verification(args)
    for name in ("benchmark", "smoke"):
        report = json.loads((args.output_dir / name / "report.json").read_text())
        if report["status"] != "PASS" or report["source_hashes"] != source_hashes():
            raise RuntimeError(f"{name} failed or inputs changed")
    components = model_components()
    birds = make_birds(components, TRAIN_BIRDS, TRAIN_BATCHES,
                       TRAIN_TRAJECTORY_SEED, TRAIN_HEIGHT_SEED)
    baseline = create_problem(components, birds).evaluate_raw_layout(components[8])
    measured_risk_ref = baseline["E_potential_strike"]
    if not math.isclose(measured_risk_ref, RISK_REFERENCE,
                        rel_tol=1e-10, abs_tol=1e-15):
        raise RuntimeError(
            f"Training reference risk changed: {measured_risk_ref:.16e} "
            f"!= {RISK_REFERENCE:.16e}"
        )
    risk_ref = RISK_REFERENCE
    problem = create_problem(components, birds, risk_ref, audit=True)
    initial, labels = initial_population(components, problem, POPULATION,
                                         INITIALISATION_SEED, random_reserved=5)
    stage = new_stage(args.output_dir / "optimisation")
    np.save(stage / "initial_population_xy_m.npy", initial)
    write_csv(stage / "initial_population.csv", [
        {"index": i, "source": label, **xy_columns(xy)}
        for i, (label, xy) in enumerate(zip(labels, initial))])
    write_csv(stage / "reference_training_batches.csv", baseline["batch_rows"])
    saved_sources(stage)
    import floris
    import pymoo
    settings = {
        "status": "running", "scenario": "final northern-gannet FHD", "model": "Model 10",
        "optimiser": "NSGA-II", "population": POPULATION, "generations": GENERATIONS,
        "intended_evaluations": POPULATION * GENERATIONS,
        "training_batches": TRAIN_BATCHES, "training_birds_per_batch": TRAIN_BIRDS,
        "training_trajectory_seeds": [TRAIN_TRAJECTORY_SEED + i for i in range(TRAIN_BATCHES)],
        "training_height_seeds": [TRAIN_HEIGHT_SEED + i for i in range(TRAIN_BATCHES)],
        "optimiser_seed": OPTIMISER_SEED, "initial_population_seed": INITIALISATION_SEED,
        "number_of_direction_sectors": N_DIRECTION_SECTORS,
        "confirmation_batches": CONFIRM_BATCHES,
        "confirmation_birds_per_batch": CONFIRM_BIRDS,
        "confirmation_trajectory_seeds": [CONFIRM_TRAJECTORY_SEED + i for i in range(CONFIRM_BATCHES)],
        "confirmation_height_seeds": [CONFIRM_HEIGHT_SEED + i for i in range(CONFIRM_BATCHES)],
        "confirmation_used_during_training": False,
        "working_minimum_meaningful_improvement_fraction": MINIMUM_IMPROVEMENT,
        "reference_aep_gwh": baseline["aep_gwh"],
        "reference_training_risk_per_entrant": risk_ref,
        "reference_training_E_RT_CE": baseline["E_RT_CE"],
        "aep_ceiling_gwh": components[9], "aep_loss_reference_gwh": LOSS_REFERENCE_GWH,
        "FHD": components[4].settings(),
        "turbines": 12, "site_m": [6000., 6000.], "rotor_diameter_m": 242.24,
        "minimum_spacing_m": problem.minimum_spacing_m,
        "bird_direction_deg_toward": 225., "rotor_plane_heading_deg_toward": 225.,
        "behaviour": components[3].settings(242.24),
        "band": {
            "rotor_rpm": components[6].rotor_rpm,
            "collective_pitch_deg": components[6].collective_pitch_deg,
            "fixed_wind_speed_m_s": 12.2589068261207,
            "bird_speed_m_s": components[6].bird_speed_m_s,
            "bird_length_m": components[6].bird_length_m,
            "bird_wingspan_m": components[6].bird_wingspan_m,
            "flight_style": components[6].flight_style,
            "wind_heading_deg_toward": components[6].wind_heading_deg_toward,
        },
        "hypervolume_reference_normalised": HYPERVOLUME_REFERENCE,
        "initial_population_labels": labels,
        "selection_rule": "All feasible evaluated layouts and the external reference; "
            "distinct highest-AEP, lowest-risk and Euclidean knee of the training "
            "nondominated front; the knee uses (best AEP minus AEP)/(front AEP range) "
            "and (risk minus lowest risk)/(front risk range). If reference dominates "
            "all feasible candidates, choose only the nearest competitor with "
            "archived energy-loss and training reference-risk scales.",
        "source_hashes": components[0],
        "versions": {"floris": floris.__version__, "pymoo": pymoo.__version__,
                     "numpy": np.__version__, "python": sys.version.split()[0]},
        "interpretation": "Expected Band potential strikes per entrant; no annual flux or residual avoidance",
    }
    write_json(stage / "run_settings.json", settings)
    start = time.perf_counter()
    algorithm = NSGA2(pop_size=POPULATION, sampling=initial.reshape(POPULATION, 24),
                      eliminate_duplicates=True)
    result = minimize(problem, algorithm, termination=("n_gen", GENERATIONS),
                      seed=OPTIMISER_SEED,
                      callback=AuditedProgress(stage, problem), verbose=False)
    elapsed = time.perf_counter() - start
    if result.X is None:
        raise RuntimeError("No feasible final population")
    total = len(problem.evaluation_rows)
    if total != int(result.algorithm.evaluator.n_eval):
        raise RuntimeError("Recorded candidates do not match actual evaluator count")
    if total != POPULATION * GENERATIONS:
        raise RuntimeError(f"Expected 480 candidate evaluations, got {total}")
    front, feasible = training_nondominated(problem.evaluation_rows,
                                            baseline["aep_gwh"], risk_ref)
    selected, reference_dominates = select_candidates(front, feasible,
                                                       baseline["aep_gwh"], risk_ref)
    for xy in np.asarray(result.pop.get("X")).reshape(-1, 12, 2):
        check_layout(xy, problem.minimum_spacing_m)
    saved_xys = [components[8] if row["evaluation_index"] == -1 else row_xy(row)
                 for row in front]
    for xy in saved_xys:
        check_layout(xy, problem.minimum_spacing_m)
    np.save(stage / "pareto_layouts_xy_m.npy", np.asarray(saved_xys))
    write_csv(stage / "pareto_training.csv", [
        {"front_index": i, "evaluation_index": row["evaluation_index"],
         "aep_gwh": row["aep_gwh"], "risk_per_entrant": row["risk_per_entrant"],
         "is_external_reference": row["evaluation_index"] == -1,
         **({f"batch_{j:02d}_risk": baseline["batch_rows"][j]["E_potential_strike"]
             for j in range(TRAIN_BATCHES)} if row["evaluation_index"] == -1 else
            {f"batch_{j:02d}_risk": row[f"batch_{j:02d}_risk"]
             for j in range(TRAIN_BATCHES)}),
         **xy_columns(saved_xys[i])} for i, row in enumerate(front)])
    selected_xy = np.asarray([row_xy(row) for _, row in selected]).reshape(-1, 12, 2)
    np.save(stage / "selected_for_confirmation_xy_m.npy", selected_xy)
    write_csv(stage / "selected_for_confirmation.csv", [
        {"label": label, "evaluation_index": row["evaluation_index"],
         "aep_gwh": row["aep_gwh"], "risk_per_entrant": row["risk_per_entrant"],
         "AEP_minus_reference_gwh": row["aep_gwh"] - baseline["aep_gwh"],
         **xy_columns(row_xy(row))} for label, row in selected])
    settings.update({
        "status": "completed", "runtime_seconds": elapsed,
        "evaluations": total, "feasible_evaluations": len(feasible),
        "training_front_size_including_reference": len(front),
        "reference_dominates_every_feasible_candidate": reference_dominates,
        "selected_layout_count": len(selected),
        "selected_coordinates_sha256": digest(stage / "selected_for_confirmation_xy_m.npy"),
        "selected_table_sha256": digest(stage / "selected_for_confirmation.csv"),
        "final_hypervolume": json.loads((stage / "progress.jsonl").read_text().splitlines()[-1])
          ["normalised_hypervolume"],
    })
    write_json(stage / "run_settings.json", settings)
    print(f"Optimisation complete: {total} candidates, {len(front)} training-front layouts; "
          f"{len(selected)} frozen for independent confirmation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "benchmark", "smoke", "optimise"))
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "model10_fhd_robust_seeded_50k")
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
