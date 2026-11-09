#!/usr/bin/env python3
"""Read-only replay discrepancy audit for Metric 3a.

Run from C:\\Users\\Deepak\\Desktop\\bird_layout_opt:
    python audit_metric3a_replay_mismatch.py

No GA, FLORIS, or optimisation runs; does not change any files.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TAG = "angle225p0_down8p0D_cross1p0D_seed42_pop100_gen{}"
ORIGINAL = ROOT / "results" / "directional_absorbing_risk_only" / TAG.format(120)
NEW = ROOT / "results" / "directional_absorbing_risk_only" / TAG.format(300)

ARCHIVED_SHA256 = {
    "config.py": "df56f80c33f8912eb44ef6380451d762ba71aabcd0aa002186bf506618fb074e",
    "directional_absorbing_risk.py": "3afae0b6cfc21adbd3ef81957a17fd261d1e1362a90508849bdd8522729235cf",
    "risk_only_problem.py": "169078731a621d93805974e90b03d12e23b476c6f9b2e1622096fd9e4b79df53",
    "run_directional_absorbing_risk_only.py": "8f8f469e3e89a74bd25925c3333ebe7ceef97ae9674b8dbcbd3059e9b7c51d60",
    "run_validation.py": "e86b0cea38e8900aa82db092ffd5011ab5f95c76934cb2c830786e76c51cd981",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def load_run(path: Path, expected_gen: int):
    if not path.exists():
        raise FileNotFoundError(f"Missing saved result folder: {path}")
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    conv = pd.read_csv(path / "convergence.csv")
    if not {"generation", "best_risk"}.issubset(conv.columns):
        raise ValueError(f"Missing convergence columns: {path}")
    generations = conv["generation"].to_numpy(dtype=int)
    values = conv["best_risk"].to_numpy(dtype=float)
    if len(generations) != expected_gen or not np.array_equal(generations, np.arange(1, expected_gen+1)):
        raise RuntimeError(f"Unexpected generation numbering at {path}: first/last/count={generations[0]}/{generations[-1]}/{len(generations)}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite risks: {path}")
    return summary, conv


def main():
    print("METRIC 3a — ORIGINAL VS 300-GENERATION SEED-42 AUDIT")
    print("Project root:", ROOT)
    print("Read-only: no files modified, no optimisation run.\n")
    old_settings, old_conv = load_run(ORIGINAL, 120)
    new_settings, new_conv = load_run(NEW, 300)

    print("1. RUN-SETTING COMPARISON")
    fields = ["seed", "population", "generations", "bird_travel_deg", "downstream_scale_D", "cross_scale_D", "maximum_shielding", "minimum_spacing_required_m"]
    for field in fields:
        a, b = old_settings.get(field, "NOT RECORDED"), new_settings.get(field, "NOT RECORDED")
        status = "EXPECTED" if field == "generations" else ("MATCH" if a==b else "DIFFERENT")
        print(f"  {field}: original={a!s}; new={b!s} [{status}]")

    print("\n2. EXACT HISTORICAL SOURCE-HASH COMPARISON")
    different = []
    for filename, historical_hash in ARCHIVED_SHA256.items():
        path=ROOT / filename
        if not path.exists():
            print(f"  {filename}: MISSING")
            different.append(filename)
        else:
            current_hash=sha256(path)
            match=current_hash==historical_hash
            print(f"  {filename}: {'MATCH' if match else 'DIFFERENT'} {current_hash}")
            if not match: different.append(filename)
    print("  Historical snapshots: these SHA-256 values were taken from the saved original Metric 3a verification archive.")

    print("\n3. CURRENT PYTHON ENVIRONMENT (historical original versions unrecorded)")
    print("  Python", platform.python_version())
    for pkg in ("pymoo","numpy","scipy"):
        print(" ",pkg,version(pkg))

    print("\n4. CONVERGENCE COMPARISON; SAME GENERATION LABELS")
    old = old_conv["best_risk"].to_numpy(dtype=float)
    new = new_conv["best_risk"].to_numpy(dtype=float)[:120]
    delta = np.abs(new-old)
    first = np.flatnonzero(delta>1e-9)
    print(f"  Generation 1: original={old[0]:.12f}, new={new[0]:.12f}, delta={delta[0]:.3e}")
    for gen in [2,10,30,60,120]:
        i=gen-1
        print(f"  Generation {gen}: original={old[i]:.12f}, new={new[i]:.12f}, delta={delta[i]:.3e}")
    print(f"  Maximum absolute difference (generations 1–120): {np.max(delta):.12f}")
    if len(first):
        i=int(first[0]); print(f"  FIRST DIVERGENCE: generation {i+1}; original={old[i]:.12f}; new={new[i]:.12f}")
    else:
        print("  EXACT REPLAY PASS: same Q history through generation 120")

    print("\n5. DETERMINISTIC INITIAL SAMPLING (NO GA)")
    if different:
        print("  SKIPPED: source files do not all match historical SHA-256; resolve code differences first.")
    else:
        try:
            from config import Scenario
            from directional_absorbing_risk import DirectionalAbsorbingRiskModel
            from run_validation import make_initial_population
            scenario=Scenario()
            initial_a = np.asarray(make_initial_population(scenario,100,42,"constant"),dtype=float)
            initial_b = np.asarray(make_initial_population(scenario,100,42,"constant"),dtype=float)
            matching = np.array_equal(initial_a,initial_b)
            model=DirectionalAbsorbingRiskModel(scenario=scenario,bird_travel_deg=225.,downstream_scale_D=8.,cross_scale_D=1.,maximum_shielding=.8)
            initial_q=np.asarray([model.collisions(x.reshape(12,2)) for x in initial_a],dtype=float)
            print(f"  Two independently regenerated batches of 100: {'IDENTICAL' if matching else 'NOT IDENTICAL'}")
            print(f"  Initial population SHA-256: {hashlib.sha256(initial_a.tobytes()).hexdigest()}")
            print(f"  Best objective among raw sampled layouts: {np.min(initial_q):.12f}")
            print("  Note: GA feasibility selection may differ from the raw-sample minimum.")
        except Exception as exc:
            print(f"  Sampling test couldn't complete: {type(exc).__name__}: {exc}")

    print("\nINTERPRETATION")
    if different:
        print("  Code mismatch or missing file found. Check changed files before attributing divergence to software versions.")
    elif len(first)==0:
        print("  Reproduces historical 120-generation Q history exactly.")
    elif first[0]==0:
        print("  Difference starts in initial population / first-generation population handling or objective evaluation.")
    else:
        print("  First generation agrees; divergence begins during later genetic-algorithm evolution.")
        print("  Operators, stochastic state, dependency versions, and execution conditions remain candidates; cause is not established.")
    print("  Never call the 300-generation run a verified continuation unless the first 120 generations match.")


if __name__ == "__main__":
    main()
