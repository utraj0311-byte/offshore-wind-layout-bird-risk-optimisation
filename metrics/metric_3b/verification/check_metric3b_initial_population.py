"""Read-only reproducibility check for Metric 3b's original 160 starting layouts.

Run from bird_layout_opt project root using the existing floris-env.
Does not run FLORIS, NSGA-II, or write files.
"""
from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
CODE_DIR = ROOT / "shielding_model_code"
ARCHIVE_DIR = ROOT / "results" / "shielding_v5_final" / "heading225_W6000_seed48_pop160_gen250"

for path in (CODE_DIR / "run_shielding_multiobjective.py", ARCHIVE_DIR / "run_settings.json", ARCHIVE_DIR / "initial_population.csv"):
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")

sys.path.insert(0, str(CODE_DIR))
from config import Scenario  # noqa: E402
from run_shielding_multiobjective import (  # noqa: E402
    FeasibleSeededSampling,
    load_layout_csv,
    load_prior_seed_csv,
    make_fixed_layouts,
)

settings = json.loads((ARCHIVE_DIR / "run_settings.json").read_text(encoding="utf-8"))
scenario = Scenario()
scenario.site.n_turbines = int(settings["problem"]["number_of_turbines"])
scenario.site.width_m = float(settings["site_width_m"])
scenario.site.height_m = float(settings["site_height_m"])
scenario.site.min_spacing_D = float(settings["problem"]["minimum_spacing_D"])
scenario.site.migration_axis_deg = float(settings["bird_heading_deg_toward"])
scenario.birds.migration_width_m = float(settings["corridor_width_m"])
scenario.turbine.rotor_diameter_m = float(settings["aep_model"]["floris_rotor_diameter_m"])

rotor_diameter_m = scenario.turbine.rotor_diameter_m
fixed_layouts = make_fixed_layouts(
    site_width_m=scenario.site.width_m,
    site_height_m=scenario.site.height_m,
    bird_heading_deg_toward=scenario.site.migration_axis_deg,
    minimum_spacing_m=scenario.site.min_spacing_D * rotor_diameter_m,
)
energy_seed_path = Path(settings["energy_seed_layout"])
if not energy_seed_path.is_absolute():
    energy_seed_path = ROOT / energy_seed_path
energy_seed = load_layout_csv(energy_seed_path, scenario.site.n_turbines)

prior_seed_path = Path(settings["initialisation"]["prior_seed_csv"])
if not prior_seed_path.is_absolute():
    prior_seed_path = ROOT / prior_seed_path
prior_layouts, prior_labels = load_prior_seed_csv(prior_seed_path, scenario.site.n_turbines)

sampler = FeasibleSeededSampling(
    scenario=scenario,
    rotor_diameter_m=rotor_diameter_m,
    exact_seed_layouts={"energy": energy_seed, **fixed_layouts},
    prior_seed_layouts=prior_layouts,
    prior_seed_labels=prior_labels,
    projection_preserving_seed_names=set(fixed_layouts),
    seed=int(settings["seed"]),
    jitter_scale_D=float(settings["initialisation"]["jitter_scale_D"]),
)

population = sampler._do(None, int(settings["population"]))
archived = pd.read_csv(ARCHIVE_DIR / "initial_population.csv")
coordinate_columns = [
    axis
    for turbine_idx in range(scenario.site.n_turbines)
    for axis in (f"x_{turbine_idx}_m", f"y_{turbine_idx}_m")
]
missing_columns = set(coordinate_columns + ["seed_source"]) - set(archived.columns)
if missing_columns:
    raise ValueError(f"Archive missing columns: {sorted(missing_columns)}")
archived_population = archived[coordinate_columns].to_numpy(dtype=float)
archived_labels = archived["seed_source"].astype(str).tolist()

if population.shape != archived_population.shape:
    raise AssertionError(f"Population shape differs: {population.shape} versus {archived_population.shape}")
maximum_error_m = float(np.max(np.abs(population - archived_population)))
labels_identical = sampler.last_labels == archived_labels
print(f"Reconstructed population: {population.shape[0]} layouts")
print(f"Number of prior-seed input layouts: {len(prior_layouts)}")
print(f"Included prior Pareto layouts: {sampler.last_prior_seed_count}")
print(f"Maximum absolute coordinate difference: {maximum_error_m:.12g} m")
print(f"All source labels identical: {labels_identical}")
counts = Counter(
    "exact" if label.startswith("exact_") else
    "prior" if label.startswith("prior_") else
    "jitter" if label.startswith("jitter_") else
    "random" if label == "random_feasible" else
    "other"
    for label in sampler.last_labels
)
print("Population composition:", dict(counts))
if maximum_error_m > 1e-9 or not labels_identical:
    raise AssertionError("FAIL: original initial population was NOT reproduced; stop pre-flight.")
print("PASS: all 160 initial layouts and source labels reproduced.")
print("No files written. No AEP evaluation or optimisation performed.")
