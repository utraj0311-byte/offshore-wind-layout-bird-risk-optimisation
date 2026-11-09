"""Read-only Metric 4a pre-flight checks; does NOT start NSGA-II.

Use in Windows Anaconda Prompt, activated floris-env:
    python metric4a_preflight.py audit
    python metric4a_preflight.py bird
    python metric4a_preflight.py aep
    python metric4a_preflight.py benchmark

Each stage writes its report only into ./preflight_reports/.
Original ZIP archives, saved layouts and project files are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "model8_code"
OUT = ROOT / "preflight_reports"
for directory in (ROOT, SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

OPT_ZIP = ROOT / "archives" / "original_optimisation.zip"
CONF_ZIP = ROOT / "archives" / "original_confirmation.zip"
YAML = ROOT / "floris_config" / "gch_iea15mw.yaml"
SEED_CSV = ROOT / "model10_energy_seed_layout.csv"
LOCKED_CEILING = 854.4956585617186
LOCKED_AEP_LOSS_REF = 137.96914764866568
LOCKED_ENCOUNTER_REF = 0.6854
BATCHES = 5
PER_BATCH = 2000
OPT_SEED = 42
CONF_SEED = 20042


class StopCheck(Exception):
    pass


def sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def load_zip_file(path, suffix):
    with zipfile.ZipFile(path) as archive:
        files = [name for name in archive.namelist() if name.endswith("/" + suffix)]
        if len(files) != 1:
            raise StopCheck(f"Cannot uniquely find {suffix} in {path.name}")
        return archive.read(files[0])


def original():
    settings = json.loads(load_zip_file(OPT_ZIP, "run_settings.json"))
    table = pd.read_csv(io.BytesIO(load_zip_file(OPT_ZIP, "pareto_points.csv")))
    vectors = np.load(io.BytesIO(load_zip_file(OPT_ZIP, "pareto_decision_vectors.npy")), allow_pickle=False)
    initial = pd.read_csv(io.BytesIO(load_zip_file(OPT_ZIP, "initial_population.csv")))
    cs = json.loads(load_zip_file(CONF_ZIP, "confirmation_settings.json"))
    return settings, table, vectors, initial, cs


def report(stage, data):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{stage}_report.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"\nReport saved: {path}")


def check(label, observed, expected, *, atol=0, rtol=0):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        ok = bool(np.isclose(observed, expected, rtol=rtol, atol=atol))
    else:
        ok = observed == expected
    print(f"{'PASS' if ok else 'FAIL'} | {label}: observed={observed!r}; expected={expected!r}")
    if not ok:
        raise StopCheck(f"Pre-flight failed: {label}. No optimisation may be started.")


def endpoints(table, vectors):
    high_row = table.loc[table["aep_gwh"].idxmax()]
    low_row = table.loc[table["E_enc"].idxmin()]
    def collect(row):
        index = int(row["solution_index"])
        return index, vectors[index].reshape(-1, 2), row
    return {"highest-AEP": collect(high_row), "lowest-risk": collect(low_row)}


def historical_site(settings):
    """Original optimisation runner sets site dimensions to 6000 m explicitly.

    Historical run_settings.json omitted those two keys; the independent
    confirmation_settings.json retained both, and is checked here.
    """
    archived = json.loads(load_zip_file(CONF_ZIP, "confirmation_settings.json"))
    width, height = float(archived["site_width_m"]), float(archived["site_height_m"])
    if (width, height) != (6000.0, 6000.0):
        raise StopCheck("Original confirmation site dimensions conflict with the 6000-m runner")
    return width, height


def bundle_hashes():
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and "preflight_reports" not in p.parts and "__pycache__" not in p.parts
    }


def corridor_values(settings):
    from model8_meso_response import heading_unit_vectors, cross_unit_vectors
    width = float(historical_site(settings)[0])
    height = float(historical_site(settings)[1])
    angle = float(settings["bird_heading_deg_toward"])
    corners = np.array([[0,0],[width,0],[0,height],[width,height]], dtype=float)
    cross = cross_unit_vectors(angle)
    along = heading_unit_vectors(angle)
    return {
        "min_cross_m": float(min(corners @ cross)),
        "max_cross_m": float(max(corners @ cross)),
        "width_cross_m": float(np.ptp(corners @ cross)),
        "longitudinal_margin_m": (
            settings["problem"]["behaviour"]["response_radius_m"]
            + settings["problem"]["behaviour"]["return_length_m"]
        ),
        "entry_generation": "TrajectoryPopulation.generate_parallel; site cross projection, no encounter margin",
        "along_site_span_m": float(np.ptp(corners @ along)),
    }


def audit(args):
    settings, pareto, X, initial, confirmation = original()
    print("\n--- ARCHIVE / CODE AUDIT ---")
    # Verify that the supplied AEP implementation uses a configurable sector count.
    aep_source = (ROOT / "floris_aep.py").read_text(encoding="utf-8")
    config_source = (ROOT / "config.py").read_text(encoding="utf-8")
    runner_source = (SRC / "run_model8_multiobjective.py").read_text(encoding="utf-8")
    confirmation_source = (SRC / "run_model8_confirmation.py").read_text(encoding="utf-8")
    check("wind sector default declared in config", "n_sectors: int = 12" in config_source, True)
    check("FLORIS sector grid uses wind.n_sectors", "self.wind.n_sectors" in aep_source, True)
    check("original runner does not override wind sector count", "scenario.wind.n_sectors =" in runner_source, False)
    check("optimisation uses shared corridor generator", "TrajectoryPopulation.generate_parallel(" in runner_source, True)
    check("confirmation uses shared corridor generator", "TrajectoryPopulation.generate_parallel(" in confirmation_source, True)
    check("original direction sectors", settings["aep_model"]["number_of_direction_sectors"], 12)
    check("original objective population", settings["population"], 80)
    check("original generations", settings["generations"], 100)
    check("original seed", settings["seed"], 42)
    check("training batch count", settings["objective_batches"], 5)
    check("training trajectories per batch", settings["objective_trajectories_per_batch"], 2000)
    check("training seeds", settings["objective_random_number_seeds"], list(range(42, 47)))
    check("independent confirmation seeds", confirmation["confirmation_random_number_seeds"], list(range(20042, 20062)))
    check("confirmation batches", confirmation["confirmation_batches"], 20)
    check("confirmation trajectories/batch", confirmation["trajectories_per_batch"], 5000)
    check("reference Q_4a", settings["problem"]["encounter_reference"], LOCKED_ENCOUNTER_REF)
    check("energy-loss reference", settings["problem"]["aep_loss_reference_gwh"], LOCKED_AEP_LOSS_REF, atol=1e-10)
    check("wake-free reference", settings["problem"]["aep_ceiling_gwh"], LOCKED_CEILING, atol=1e-10)
    check("response probability", settings["problem"]["behaviour"]["response_probability"], 0.6)
    check("bird heading", settings["bird_heading_deg_toward"], 225.0)
    check("rotor diameter", settings["problem"]["rotor_diameter_m"], 242.24)
    check("minimum spacing", settings["problem"]["minimum_spacing_m"], 968.96)
    check("saved population count", len(initial), 80)
    check("saved Pareto vector shape", list(X.shape), [19, 24])
    check("saved Pareto table count", len(pareto), 19)
    check("saved initial exact families", int(initial["seed_source"].str.startswith("exact_").sum()), 4)
    check("saved initial jitter families", int(initial["seed_source"].str.startswith("jitter_").sum()), 40)
    check("saved initial random families", int((initial["seed_source"] == "random_feasible").sum()), 36)
    check("confirmation site width", confirmation["site_width_m"], historical_site(settings)[0])
    check("confirmation site height", confirmation["site_height_m"], historical_site(settings)[1])
    check("confirmation bird heading", confirmation["bird_heading_deg_toward"], settings["bird_heading_deg_toward"])
    check("confirmation behaviour", confirmation["behaviour"], settings["problem"]["behaviour"])
    configs = {"original_pymoo_version": settings.get("pymoo_version", "NOT RECORDED"),
               "laptop_pymoo_version": version("pymoo"),
               "laptop_floris_version": version("floris"),
               "laptop_numpy_version": version("numpy"),
               "laptop_python_version": platform.python_version(),
               "original_floris_yaml_sha256": "NOT RECORDED"}
    print("\nPackage versions:", json.dumps(configs, indent=2))
    check("FLORIS version target", version("floris"), "4.6.5")

    # Resolve historical YAML path; a filename alone does not prove file equality.
    project_root = args.project_root or ROOT.parent
    local_yaml = project_root / "floris_config" / "gch_iea15mw.yaml"
    if local_yaml.exists():
        print("Original-project YAML:", local_yaml)
        check("project YAML identical to bundled uploaded YAML", sha(local_yaml), sha(YAML))
        configs["project_yaml_equal_to_uploaded"] = True
    else:
        print("UNVERIFIED | Historical YAML not present at", local_yaml)
        configs["project_yaml_equal_to_uploaded"] = "UNVERIFIED"

    if args.reference_yaml is not None:
        if not args.reference_yaml.is_file():
            raise StopCheck(f"Explicit reference YAML not found: {args.reference_yaml}")
        check("explicit historical reference YAML matches bundled YAML", sha(args.reference_yaml), sha(YAML))
        configs["explicit_historical_yaml_equal_to_uploaded"] = True

    # Reconstruct exact 80-layout starting population using the source implementation.
    from config import Scenario
    from run_model8_multiobjective import FeasibleSeededSampling
    from run_model8_validation import make_fixed_layouts
    scenario = Scenario()
    scenario.site.n_turbines = 12
    scenario.site.width_m = 6000.0
    scenario.site.height_m = 6000.0
    scenario.site.min_spacing_D = 4.0
    scenario.site.migration_axis_deg = 225.0
    energy_csv = pd.read_csv(SEED_CSV).sort_values("turbine_id")
    energy = energy_csv[["x_m", "y_m"]].to_numpy(dtype=float)
    fixed = make_fixed_layouts(site_width_m=6000., site_height_m=6000.,
                               bird_heading_deg_toward=225., minimum_spacing_m=968.96)
    sampler = FeasibleSeededSampling(scenario=scenario, rotor_diameter_m=242.24,
                                    exact_seed_layouts={"energy": energy, **fixed}, seed=42)
    generated = sampler._do(None, 80)
    expected = initial.drop(columns=["seed_source"]).to_numpy(dtype=float)
    max_error = float(np.max(np.abs(generated - expected)))
    check("reconstructed initial coordinates maximum absolute error (m)", max_error, 0., atol=1e-9)
    check("reconstructed seed labels", sampler.last_labels, initial["seed_source"].to_list())

    # Same bird-entry generator used by original optimisation and confirmation.
    from run_model8_multiobjective import generate_population_batches
    from run_model8_confirmation import generate_confirmation_batches
    # Both wrappers delegate to TrajectoryPopulation.generate_parallel.
    corridor = corridor_values(settings)
    print("\nCorridor:", json.dumps(corridor, indent=2))
    print("Original AEP objective: (AEP_ceiling - AEP) / 137.96914764866568")
    print("Original bird objective: Q_4a / 0.6854")
    print("70+ conditions derived from sector count; sectors in config.WindResource and floris_aep._build_wind_conditions.")
    
    # Proposed settings compared with historical settings. Only base input n_sectors changes.
    proposed = json.loads(json.dumps(settings))
    proposed["aep_model"]["number_of_direction_sectors"] = 72
    proposed["aep_model"]["number_of_combined_conditions"] = 504
    proposed["n_sectors"] = 72
    proposed["pymoo_version"] = version("pymoo")
    proposed["provenance_note"] = "Proposed only; no optimisation has been launched"
    OUT.mkdir(exist_ok=True)
    (OUT / "proposed_run_settings.json").write_text(json.dumps(proposed, indent=2), encoding="utf-8")
    results = {"status": "PASS (historical version/YAML provenance may remain unverified)",
               "software": configs, "initial_population_max_error_m": max_error,
               "corridor": corridor, "bundled_file_sha256": bundle_hashes(),
               "historical_sector_count": 12, "proposed_sector_count": 72,
               "derived_conditions_original": 84, "derived_conditions_new": 504}
    report("audit", results)
    print("PASS | Source and archived initial population agree. No optimisation started.")


def create_birds(settings):
    from config import Scenario
    from model8_meso_response import TrajectoryPopulation
    scenario = Scenario()
    scenario.site.width_m = float(historical_site(settings)[0])
    scenario.site.height_m = float(historical_site(settings)[1])
    scenario.site.n_turbines = int(settings["problem"]["number_of_turbines"])
    behaviour = settings["problem"]["behaviour"]
    # Exact same TrajectoryPopulation.generate_parallel call and seeds as the
    # original generate_population_batches wrapper (see original source).
    populations = tuple(
        TrajectoryPopulation.generate_parallel(
            site_width_m=scenario.site.width_m,
            site_height_m=scenario.site.height_m,
            heading_deg_toward=float(settings["bird_heading_deg_toward"]),
            number_of_trajectories=int(settings["objective_trajectories_per_batch"]),
            maximum_response_events=scenario.site.n_turbines,
            longitudinal_margin_m=float(behaviour["response_radius_m"] + behaviour["return_length_m"]),
            seed=int(settings["seed"]) + i,
        ) for i in range(int(settings["objective_batches"]))
    )
    return populations


def create_behaviour(settings, probability=None):
    from model8_meso_response import BehaviourScenario
    b = settings["problem"]["behaviour"]
    return BehaviourScenario(
        response_probability=b["response_probability"] if probability is None else probability,
        response_radius_m=b["response_radius_m"],
        encounter_radius_m=b["encounter_radius_m"],
        deflection_angle_deg=b["deflection_angle_deg"],
        return_length_m=b["return_length_m"],
        step_size_m=b["step_size_m"],
        max_steps_factor=b["max_steps_factor"],
    )


def bird(args):
    print("\n--- BIRD TRAINING REPRODUCTION ---")
    settings, table, X, _, _ = original()
    from model8_meso_response import MesoResponseModel
    from model8_batching import summarise_meso_batches
    populations = create_birds(settings)
    results = {}
    for label, (index, xy, row) in endpoints(table, X).items():
        observed = summarise_meso_batches(MesoResponseModel(create_behaviour(settings)), xy, populations)
        check(label + " N_enc_farm", int(observed["N_enc_farm"]), int(row["N_enc_farm"]))
        check(label + " Q_4a", observed["E_enc"], float(row["E_enc"]), atol=1e-15)
        check(label + " P(>=1)", observed["P_ge_1"], float(row["P_ge_1"]), atol=1e-15)
        check(label + " completed", observed["N_completed"], int(row["N_completed"]))
        results[label] = {"solution_index": index, "Q_4a": observed["E_enc"],
                          "P_ge_1": observed["P_ge_1"], "N_enc_farm": observed["N_enc_farm"]}

    energy_xy = pd.read_csv(SEED_CSV).sort_values("turbine_id")[["x_m", "y_m"]].to_numpy(dtype=float)
    no_response = summarise_meso_batches(MesoResponseModel(create_behaviour(settings, probability=0.)),
                                          energy_xy, populations)
    check("no-response objective scaling", no_response["E_enc"], LOCKED_ENCOUNTER_REF, atol=1e-15)
    results["no_response_reference"] = no_response["E_enc"]
    report("bird", results)
    print("PASS | Exact count-level reproduction of saved training scores.")


def create_aep_model(settings, sectors):
    from config import Scenario
    from floris_aep import FLORISAEPModel
    scenario = Scenario()
    scenario.site.width_m = float(historical_site(settings)[0])
    scenario.site.height_m = float(historical_site(settings)[1])
    scenario.site.n_turbines = int(settings["problem"]["number_of_turbines"])
    scenario.site.min_spacing_D = float(settings["problem"]["minimum_spacing_D"])
    scenario.site.wind_from_deg = float(settings["aep_model"]["dominant_wind_from_deg"])
    scenario.wind.n_sectors = sectors
    aep = FLORISAEPModel(scenario, config_path=YAML,
                          turbulence_intensity=float(settings["aep_model"]["turbulence_intensity"]),
                          hours_per_year=float(settings["aep_model"]["hours_per_year"]))
    scenario.turbine.rotor_diameter_m = aep.floris_rotor_diameter_m
    check(f"{sectors} sectors; turbine", aep.turbine_type, settings["aep_model"]["turbine_type"])
    check(f"{sectors} sectors; diameter", aep.floris_rotor_diameter_m, 242.24)
    check(f"{sectors} sectors; number of conditions", aep.number_of_conditions, sectors * 7)
    return aep


def aep(args):
    print("\n--- LOCAL FLORIS REPRODUCTION ---")
    settings, table, vectors, _, _ = original()
    from model8_meso_response import MesoResponseModel
    from model8_batching import summarise_meso_batches
    model12 = create_aep_model(settings, 12)
    results = {}
    for label, (index, xy, row) in endpoints(table, vectors).items():
        measured = float(model12.evaluate(xy))
        error = abs(measured - float(row["aep_gwh"]))
        print(f"{label}: saved={float(row['aep_gwh']):.12f}, local={measured:.12f}, error={error:.9f} GWh/year")
        check(f"{label} 12-sector AEP tolerance <=0.001 GWh", error <= .001, True)
        results[label] = {"solution_index": index, "archived_12_sector_aep_gwh": float(row["aep_gwh"]),
                          "reproduced_12_sector_aep_gwh": measured, "abs_error_gwh": error}
    check("wake-free 12-sector ceiling", model12.ideal_aep(12), LOCKED_CEILING, atol=1e-6)

    model72 = create_aep_model(settings, 72)
    check("wake-free 72-sector ceiling", model72.ideal_aep(12), LOCKED_CEILING, atol=1e-6)
    populations = create_birds(settings)
    for label, (index, xy, row) in endpoints(table, vectors).items():
        result72 = float(model72.evaluate(xy))
        resultQ = summarise_meso_batches(MesoResponseModel(create_behaviour(settings)), xy, populations)
        check(f"{label} Q_4a unchanged at 72 sectors", resultQ["E_enc"], float(row["E_enc"]), atol=1e-15)
        results[label]["aep_72_sectors_gwh"] = result72
        results[label]["Q_4a_72_sectors"] = resultQ["E_enc"]
        print(f"{label}: 72-sector AEP = {result72:.12f} GWh/year; Q_4a = {resultQ['E_enc']:.7f}")
    report("aep", results)
    print("PASS | Archived 12-sector results reproduce; 72-sector wind resource changes only energy.")


def benchmark(args):
    print("\n--- LOCAL RUNTIME BENCHMARK (NO OPTIMISATION) ---")
    settings, table, vectors, _, _ = original()
    from model8_aep_problem import Model8AEPEncounterProblem
    from model8_meso_response import MesoResponseModel
    from run_model8_confirmation import generate_confirmation_batches
    populations = create_birds(settings)
    model = create_aep_model(settings, 72)
    problem = Model8AEPEncounterProblem(
        scenario=model.s, aep_model=model, meso_model=MesoResponseModel(create_behaviour(settings)),
        trajectory_populations=populations, rotor_diameter_m=242.24,
        aep_ceiling_gwh=LOCKED_CEILING,
        aep_loss_reference_gwh=LOCKED_AEP_LOSS_REF,
        encounter_reference=LOCKED_ENCOUNTER_REF,
        record_evaluations=False,
    )
    high = endpoints(table, vectors)["highest-AEP"][1]
    t0, cpu0 = time.perf_counter(), time.process_time()
    values = problem.evaluate(high.reshape(1, -1), return_values_of=["F", "G"])
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    total8k = wall * 8000 / 3600
    print(f"One objective evaluation (including 10,000 birds): {wall:.3f} s wall / {cpu:.3f} s CPU")
    print(f"8,000 evaluations × observed time: {total8k:.2f} hours (naive baseline; overhead and I/O excluded)")
    print(f"Logical CPU cores visible to Python: {os.cpu_count()} (this is NOT number of workers used)")
    print("Worker processes in provided NSGA-II code: 1; NumPy/BLAS may use extra native threads.")
    print("CPU-time/wall-time ratio:", round(cpu / max(wall,1e-12),2))
    print("Relevant thread settings:", {key:os.environ.get(key,"not set") for key in
                                       ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})
    print("Do not assume 4a + 4b parallel jobs are efficient until you observe CPU/RAM saturation.")
    # Benchmark one confirmation batch for each endpoint and each response setting.
    beh = settings["problem"]["behaviour"]
    batches = generate_confirmation_batches(
        site_width_m=historical_site(settings)[0], site_height_m=historical_site(settings)[1],
        heading_deg_toward=settings["bird_heading_deg_toward"],
        number_of_trajectories=5000, batch_count=1,
        maximum_response_events=12,
        longitudinal_margin_m=beh["response_radius_m"] + beh["return_length_m"],
        first_seed=CONF_SEED,
    )
    sample = batches[0]
    elapsed_conf = 0.
    for label, (_idx, xy, _row) in endpoints(table, vectors).items():
        for p in (0.6, 0.0):
            t0 = time.perf_counter()
            observed = MesoResponseModel(create_behaviour(settings, probability=p)).simulate(xy, sample)
            elapsed = time.perf_counter() - t0
            elapsed_conf += elapsed
            print(f"Confirmation benchmark {label} p={p:.1f}: {elapsed:.3f}s/5,000 birds; Q={observed.encounter_intensity:.6f}")
    estimated_conf = elapsed_conf * 20 / 60.
    print(f"20-batch × 2-layout × 2-response confirmation projected: {estimated_conf:.2f} min (plus overhead).")
    results = {"single_72_sector_evaluation_wall_seconds": wall, "cpu_seconds": cpu,
               "projected_8000_eval_hours": total8k,
               "logical_cpu_cores": os.cpu_count(), "parallel_python_workers": 1,
               "projected_two_end_20_batch_two_p_confirmation_minutes": estimated_conf,
               "old_12sector_8000_evaluations_seconds": 22454.7920434,
               "notes": "Only timing estimates: layout dependence, checkpoints, output, 1-degree checks, and laptop scheduling add overhead"}
    report("benchmark", results)
    print("BENCHMARK COMPLETE; no NSGA-II started.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("audit", "bird", "aep", "benchmark"),
                        help="Run one safe, read-only verification stage")
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Optional location of original bird_layout_opt folder, to compare its YAML")
    parser.add_argument("--reference-yaml", type=Path, default=None,
                        help="Optional historical gch_iea15mw(2).yaml to compare via SHA-256")
    args = parser.parse_args()
    stages = {"audit": audit, "bird": bird, "aep": aep, "benchmark": benchmark}
    try:
        stages[args.stage](args)
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        (OUT / f"{args.stage}_ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nSTOP —", type(exc).__name__, str(exc))
        print("No optimisation has been launched. See preflight_reports for the error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
