"""Reproducible 12-turbine FLORIS AEP versus Model 10 Band-risk optimisation.

This runner only composes the frozen Model 8/9/Band modules. Bird risk is
expected potential strikes per entering bird, with no annual flux or mortality
conversion. Its comparison to AEP is a two-objective NSGA-II search.

Example:
 PYTHONPATH=verification/_runtime_deps:verification/_runtime_source:upload:. \
 OPENBLAS_NUM_THREADS=1 python -u model10_run_baseline_nsga2.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "upload"), str(ROOT / "verification/_runtime_source")]

from config import Scenario  # noqa: E402
from floris_aep import FLORISAEPModel  # noqa: E402
from model8_meso_response import BehaviourScenario  # noqa: E402
from model9_rotor_transit import (  # noqa: E402
    DiscreteFlightHeightDistribution, Model9Population, RotorGeometry,
    RotorTransitModel,
)
from model10_band_strike import BandScenario, ProjectedBladeStations  # noqa: E402
from model10_aep_collision_problem import Model10AEPCollisionProblem  # noqa: E402
from run_model8_validation import make_fixed_layouts  # noqa: E402


def verify_file(path: Path, expected_sha256: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Source checksum changed: {path}")
    return digest


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_layout(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    points = np.array([[float(row["x_m"]), float(row["y_m"])] for row in data])
    if points.shape != (12, 2):
        raise ValueError("Expected exactly 12 turbine coordinates")
    return points


def populations(
    *, fhd: DiscreteFlightHeightDistribution, behaviour: BehaviourScenario,
    count: int, batches: int, seed: int, flight_seed: int,
) -> tuple[Model9Population, ...]:
    return tuple(Model9Population.generate_parallel(
        flight_height_distribution=fhd, flight_height_seed=flight_seed + i,
        site_width_m=6000, site_height_m=6000, heading_deg_toward=225,
        number_of_trajectories=count, maximum_response_events=12,
        longitudinal_margin_m=behaviour.response_radius_m + behaviour.return_length_m,
        trajectory_seed=seed + i,
    ) for i in range(batches))


class Progress(Callback):
    def __init__(self, destination: Path) -> None:
        super().__init__()
        self.destination = destination
        self.start = time.perf_counter()

    def notify(self, algorithm) -> None:
        f = np.asarray(algorithm.pop.get("F"), dtype=float)
        cv = np.asarray(algorithm.pop.get("CV"), dtype=float)
        feasible = f[np.ravel(cv) <= 1e-9]
        record = {
            "generation": int(algorithm.n_gen),
            "n_evaluations": int(algorithm.evaluator.n_eval),
            "feasible_count": len(feasible),
            "minimum_f0": float(np.min(feasible[:, 0])) if len(feasible) else None,
            "minimum_f1": float(np.min(feasible[:, 1])) if len(feasible) else None,
            "elapsed_s": round(time.perf_counter() - self.start, 3),
        }
        with self.destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def select_anchors(rows: list[dict]) -> dict[str, int]:
    aep = np.asarray([row["aep_gwh"] for row in rows])
    loss = np.asarray([row["aep_loss_gwh"] for row in rows])
    strikes = np.asarray([row["E_potential_strike"] for row in rows])

    def scale(v):
        span = np.ptp(v)
        return (v - np.min(v)) / span if span else np.zeros_like(v)

    return {
        "energy_endpoint": int(np.argmax(aep)),
        "balanced": int(np.argmin(np.hypot(scale(loss), scale(strikes)))),
        "collision_endpoint": int(np.argmin(strikes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop", type=int, default=24)
    parser.add_argument("--gen", type=int, default=20)
    parser.add_argument("--objective-batches", type=int, default=2)
    parser.add_argument("--objective-birds", type=int, default=1024)
    parser.add_argument("--fresh-batches", type=int, default=3)
    parser.add_argument("--fresh-birds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification/model10_baseline_nsga2_12t")
    args = parser.parse_args()
    if min(args.pop - 4, args.gen, args.objective_batches, args.objective_birds,
           args.fresh_batches, args.fresh_birds) <= 0:
        raise ValueError("Population must exceed four; other counts must be positive")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    import floris
    import pymoo

    config_file = ROOT / "verification/library_inputs/thesis/gch_iea15mw(2).yaml"
    source_zip = ROOT / "upload/IEA15MW_Band_input_sources(1).zip"
    fhd_file = ROOT / "upload/northern_gannet_Johnston2014_FHD(3).csv"
    energy_file = ROOT / "verification/runtime_inputs/energy_baseline.csv"
    hashes = {
        "model8": verify_file(ROOT / "upload/model8_meso_response(2).py", "2d9985f6c165a83fac5ed1493cefa1a65e8082a39d17e80e3776be12257e823d"),
        "model9": verify_file(ROOT / "upload/model9_rotor_transit.py", "3c821d5f62de8b292bc00190ae71ae1379be64ad4cd11c67dc631186b5650f09"),
        "band": verify_file(ROOT / "model10_band_strike.py", "2c095e33563323fb60a8e7d70cb3ab7b5a9672adbf0df98da32d006fa7ed96e0"),
        "floris_config": verify_file(config_file, "f6b788df9e38f006dfb6c68ddfb80ab090dad9cd555982a8d5c5e4b0c2f2d364"),
        "gannet_FHD": verify_file(fhd_file, "adc9ea903db9c8e34bfe5b88c382381b737c2e676d79af437f66d198208a0f2c"),
    }
    turbine = Path(floris.__file__).parent / "turbine_library/iea_15MW.yaml"
    hashes["floris_turbine"] = verify_file(turbine, "93d2bcfcb619b1819efe8149ce9e8231656e51a29f2ae5c1079a46e339f02feb")
    hashes["source_zip"] = hashlib.sha256(source_zip.read_bytes()).hexdigest()
    hashes["objective"] = hashlib.sha256((ROOT / "model10_aep_collision_problem.py").read_bytes()).hexdigest()
    energy = load_layout(energy_file)
    hashes["energy_layout"] = hashlib.sha256(energy_file.read_bytes()).hexdigest()

    scenario = Scenario()
    aep_model = FLORISAEPModel(scenario, config_file)
    rotor_diameter = float(aep_model.floris_rotor_diameter_m)
    if rotor_diameter != 242.24 or float(aep_model.floris_hub_height_m) != 150.0:
        raise AssertionError("The archived turbine geometry has changed")
    scenario.turbine.rotor_diameter_m = rotor_diameter
    behaviour = BehaviourScenario.from_rotor_diameters(
        rotor_diameter_m=rotor_diameter, response_probability=0.6,
        response_radius_D=2.0, encounter_radius_D=1.0,
        deflection_angle_deg=45.0, return_length_D=8.0, step_size_D=0.2,
    )
    fhd = DiscreteFlightHeightDistribution.from_csv(fhd_file)
    rotor = RotorTransitModel(
        behaviour, RotorGeometry(rotor_diameter, 150.0),
        rotor_plane_heading_deg_toward=225.0,
        flight_height_distribution=fhd,
    )
    # Existing wind_from_deg=225° is meteorological (from SW). In the bird
    # Cartesian frame it moves toward NE, heading 45°. This is fixed for all
    # crossings and never rotates Model 9's physical plane.
    assert scenario.site.wind_from_deg == 225.0
    band = BandScenario(
        rotor_rpm=7.517553314698912,
        collective_pitch_deg=6.760740271637748,
        bird_speed_m_s=14.9,
        bird_length_m=0.94, bird_wingspan_m=1.72,
        flight_style="gliding", wind_heading_deg_toward=45.0,
    )
    blade = ProjectedBladeStations.from_source_zip(source_zip)
    objective_pops = populations(
        fhd=fhd, behaviour=behaviour, count=args.objective_birds,
        batches=args.objective_batches, seed=args.seed,
        flight_seed=args.seed + 10000,
    )
    aep_ceiling = float(aep_model.ideal_aep(12))
    if not math.isclose(aep_ceiling, 854.4956585617186, abs_tol=1e-7):
        raise AssertionError("Wake-free AEP ceiling differs from the archived Model 9 baseline")
    # Fixed normalisation: energy layout evaluated on the SAME objective birds.
    provisional = Model10AEPCollisionProblem(
        scenario=scenario, aep_model=aep_model, rotor_model=rotor,
        populations=objective_pops, blade=blade, band_scenario=band,
        aep_ceiling_gwh=aep_ceiling, aep_loss_reference_gwh=137.96914764866568,
        strike_reference=1.0, record_evaluations=False,
    )
    baseline = provisional.evaluate_raw_layout(energy)
    if baseline["E_potential_strike"] <= 0:
        raise RuntimeError("Training collision reference is zero; do not optimise an undefined risk scale")
    if baseline["aep_gwh"] != 847.555657223372:
        raise AssertionError("Archived 12-turbine AEP has changed")
    problem = Model10AEPCollisionProblem(
        scenario=scenario, aep_model=aep_model, rotor_model=rotor,
        populations=objective_pops, blade=blade, band_scenario=band,
        aep_ceiling_gwh=aep_ceiling, aep_loss_reference_gwh=137.96914764866568,
        strike_reference=baseline["E_potential_strike"],
    )
    # Reuse the existing, tested feasible-seed sampler from the Model 8 runner.
    source = ROOT / "upload/run_model8_multiobjective(1).py"
    spec = importlib.util.spec_from_file_location("existing_model8_runner", source)
    assert spec is not None and spec.loader is not None
    original_runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original_runner
    spec.loader.exec_module(original_runner)
    fixed = make_fixed_layouts(
        site_width_m=6000.0, site_height_m=6000.0,
        bird_heading_deg_toward=225.0, minimum_spacing_m=4 * rotor_diameter,
    )
    sampling = original_runner.FeasibleSeededSampling(
        scenario=scenario, rotor_diameter_m=rotor_diameter,
        exact_seed_layouts={"energy": energy, **fixed}, seed=args.seed,
    )
    initial = sampling._do(problem, args.pop)
    write_csv(output / "initial_population.csv", [
        {"seed_source": sampling.last_labels[i], **{
            f"{axis}_{turbine}_m": float(initial[i, 2 * turbine + index])
            for turbine in range(12) for index, axis in enumerate(("x", "y"))
        }} for i in range(len(initial))
    ])
    settings = {
        "status": "running", "model": "Model10 fixed-plane Band conditional potential strikes",
        "objectives": ["maximise FLORIS AEP", "minimise expected potential strikes per entering bird"],
        "seed": args.seed, "population": args.pop, "generations": args.gen,
        "objective_batches": args.objective_batches,
        "objective_birds_per_batch": args.objective_birds,
        "fresh_batches": args.fresh_batches,
        "fresh_birds_per_batch": args.fresh_birds,
        "objective_trajectory_seeds": [x.trajectory_seed for x in objective_pops],
        "objective_height_seeds": [x.flight_height_seed for x in objective_pops],
        "fresh_trajectory_seeds": [30142 + i for i in range(args.fresh_batches)],
        "fresh_height_seeds": [40142 + i for i in range(args.fresh_batches)],
        "baseline_aep_gwh": baseline["aep_gwh"],
        "baseline_E_RT_CE": baseline["E_RT_CE"],
        "baseline_E_potential_strike": baseline["E_potential_strike"],
        "strike_reference_definition": "12-turbine energy layout, meso-response on, same optimisation bird batches",
        "aep_ceiling_gwh": aep_ceiling, "aep_loss_reference_gwh": 137.96914764866568,
        "turbine_count": 12, "site_size_m": [6000, 6000],
        "minimum_spacing_D": 4.0, "bird_heading_deg_toward_math": 225.0,
        "site_wind_from_deg_meteorological": 225.0,
        "fixed_wind_heading_deg_toward_math": 45.0,
        "rotor_plane_heading_deg_toward_math": 225.0,
        "approved_band": {
            "IEA_workbook_tab": "Rotor Performance", "workbook_row": 34,
            "wind_speed_m_s": 12.2589068261207,
            "rotor_rpm": band.rotor_rpm,
            "collective_pitch_deg": band.collective_pitch_deg,
            "bird_speed_m_s": band.bird_speed_m_s,
            "bird_length_m": band.bird_length_m,
            "bird_wingspan_m": band.bird_wingspan_m,
            "flight_style": band.flight_style,
            "incidence": "v_normal = 14.9 * abs(cos(actual segment heading - fixed plane normal))",
            "projected_blade_span_m": [float(blade.radius_m[0]), float(blade.radius_m[-1])],
            "nominal_disc_radius_m": 121.12,
        },
        "behaviour": behaviour.settings(rotor_diameter),
        "FHD": fhd.settings(),
        "versions": {"floris": floris.__version__, "pymoo": pymoo.__version__,
                     "numpy": np.__version__, "python": sys.version.split()[0]},
        "hashes": hashes,
        "interpretation": "potential strikes per entering bird, not annual collisions or mortality; no residual avoidance",
    }
    (output / "run_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Baseline 12-turbine AEP {baseline['aep_gwh']:.9f} GWh, E_RT_CE {baseline['E_RT_CE']:.10f}, potential strikes per entrant {baseline['E_potential_strike']:.10f}", flush=True)

    algorithm = NSGA2(pop_size=args.pop, sampling=sampling, eliminate_duplicates=True)
    start = time.perf_counter()
    result = minimize(
        problem, algorithm, termination=("n_gen", args.gen),
        seed=args.seed, callback=Progress(output / "progress.jsonl"), verbose=False,
    )
    if result.X is None:
        raise RuntimeError("No feasible NSGA-II layout found")
    X = np.atleast_2d(np.asarray(result.X, dtype=float))
    F = np.atleast_2d(np.asarray(result.F, dtype=float))
    np.save(output / "pareto_layouts_xy_m.npy", X.reshape(-1, 12, 2))
    rows = []
    for i, (vector, objectives) in enumerate(zip(X, F)):
        raw = problem.evaluate_raw_layout(vector.reshape(12, 2))
        rows.append({
            "solution_index": i, "aep_gwh": raw["aep_gwh"],
            "aep_loss_gwh": raw["aep_loss_gwh"],
            "N_started": raw["N_started"], "E_RT_CE": raw["E_RT_CE"],
            "E_potential_strike": raw["E_potential_strike"],
            "accepted_bird_turbine_pairs": raw["accepted_bird_turbine_pairs"],
            "normalised_aep_loss": float(objectives[0]),
            "normalised_potential_strikes": float(objectives[1]),
            "maximum_spacing_constraint": float(np.max(problem.spacing_constraints(vector))),
        })
    write_csv(output / "pareto_training.csv", rows)
    write_csv(output / "all_evaluated_candidates.csv", problem.evaluation_rows)

    anchors = select_anchors(rows)
    anchor_dir = output / "anchor_layouts"
    anchor_dir.mkdir(exist_ok=True)
    fresh = populations(
        fhd=fhd, behaviour=behaviour, count=args.fresh_birds,
        batches=args.fresh_batches, seed=30142, flight_seed=40142,
    )
    fresh_problem = Model10AEPCollisionProblem(
        scenario=scenario, aep_model=aep_model, rotor_model=rotor,
        populations=fresh, blade=blade, band_scenario=band,
        aep_ceiling_gwh=aep_ceiling, aep_loss_reference_gwh=137.96914764866568,
        strike_reference=baseline["E_potential_strike"], record_evaluations=False,
    )
    fresh_rows = []
    batch_rows = []
    for label, index in anchors.items():
        vector = X[index].reshape(12, 2)
        write_csv(anchor_dir / f"{label}.csv", [
            {"turbine_id": j, "x_m": float(x), "y_m": float(y)}
            for j, (x, y) in enumerate(vector)
        ])
        value = fresh_problem.evaluate_raw_layout(vector)
        batch = np.array([b["E_potential_strike"] for b in value["batch_rows"]])
        fresh_rows.append({
            "label": label, "solution_index": index,
            "aep_gwh": value["aep_gwh"], "training_E_potential_strike": rows[index]["E_potential_strike"],
            "fresh_E_potential_strike": value["E_potential_strike"],
            "fresh_strike_batch_sd": float(np.std(batch, ddof=1)) if len(batch)>1 else 0.0,
            "fresh_strike_batch_se": float(np.std(batch, ddof=1)/np.sqrt(len(batch))) if len(batch)>1 else 0.0,
            "fresh_E_RT_CE": value["E_RT_CE"],
            "fresh_N_started": value["N_started"],
        })
        batch_rows.extend({"label":label, "batch_index":i, **b} for i,b in enumerate(value["batch_rows"]))
        print(f"Fresh {label}: AEP {value['aep_gwh']:.4f} GWh; E_strike {value['E_potential_strike']:.9g}", flush=True)
    write_csv(output / "anchor_fresh_validation.csv", fresh_rows)
    write_csv(output / "anchor_fresh_batches.csv", batch_rows)
    settings.update({
        "status": "completed", "runtime_seconds": round(time.perf_counter()-start, 3),
        "number_of_pareto_layouts": len(rows), "anchors": anchors,
        "validation_scope": "Independent common-random-number batches for the three selected training anchors",
    })
    (output / "run_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Completed {args.gen} generations, {len(rows)} Pareto layouts, {len(fresh_rows)} fresh anchor checks", flush=True)


if __name__ == "__main__":
    main()
