#!/usr/bin/env python3
"""Reproducible, read-only checks/sensitivity/figures for Metric 3a.

This script DOES NOT run a GA and does not alter existing optimisation outputs.
Use the existing run_directional_absorbing_risk_only.py to run GA seeds 42/43/44.

From project root:
    python metric3a_followup_analysis.py --check
    python metric3a_followup_analysis.py --sweep
    python metric3a_followup_analysis.py --compare
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TAG = "angle225p0_down8p0D_cross1p0D_seed{seed}_pop100_gen{gen}"
ORIGINAL_U = RESULTS / "directional_absorbing_risk_only" / TAG.format(seed=42, gen=120)
ORIGINAL_S = RESULTS / "directional_absorbing_seeded" / TAG.format(seed=42, gen=120)
OUTPUT = RESULTS / "metric3a_extended_study"
EXPECT = {"unseeded": 7.688414314122593, "seeded": 5.460555118557852,
          "aligned": 5.525407012391223, "cross_flow": 9.560725671350504}
ETAS = [0.01, 0.1, 0.3, 0.8]
SEEDS = [42, 43, 44]


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_xyz(p):
    if not p.is_file():
        raise FileNotFoundError(f"Required original data missing: {p}")
    d = pd.read_csv(p)
    if len(d) != 12 or not {"x_m", "y_m"}.issubset(d.columns):
        raise RuntimeError(f"Wrong turbine layout CSV format: {p}")
    a = d[["x_m", "y_m"]].to_numpy(dtype=float)
    if not np.isfinite(a).all():
        raise RuntimeError(f"Non-finite coordinates in {p}")
    return a


def inputs():
    sys.path.insert(0, str(HERE))
    from config import Scenario
    from directional_absorbing_risk import DirectionalAbsorbingRiskModel
    scenario = Scenario()
    if (scenario.site.n_turbines, scenario.site.width_m, scenario.site.height_m) != (12, 6000., 6000.):
        raise RuntimeError("Original Metric 3a site settings no longer match (12 turbines, 6000x6000m)")
    if abs(scenario.turbine.rotor_diameter_m-240) > 1e-9 or abs(scenario.site.min_spacing_D-4.) > 1e-9:
        raise RuntimeError("Original 3a geometry expected D=240m and spacing=4D=960m. Do not replace with 3b turbine settings.")
    model = lambda eta: DirectionalAbsorbingRiskModel(scenario=scenario, bird_travel_deg=225.,
                downstream_scale_D=8., cross_scale_D=1., maximum_shielding=float(eta))
    reference = load_xyz(ORIGINAL_S / "reference_layout.csv")
    layouts = {
        "aligned_reference": reference,
        "original_without_start": load_xyz(ORIGINAL_U / "best_layout.csv"),
        "original_with_start": load_xyz(ORIGINAL_S / "best_layout.csv"),
    }
    # Exact cross-flow reference geometry from the original runner's
    # create_reference_layout(aligned=False), not an approximate transformation.
    along, cross = model(0.8).direction_vectors()
    centre = np.array([3000., 3000.])
    layouts["cross_flow_reference"] = np.array([
        centre + secondary * along + primary * cross
        for secondary in [-650., 650.]
        for primary in [-2500., -1500., -500., 500., 1500., 2500.]
    ])
    return scenario, model, layouts


def check():
    scenario, m, layouts = inputs()
    observed = {
        "unseeded": m(.8).collisions(layouts["original_without_start"]),
        "seeded": m(.8).collisions(layouts["original_with_start"]),
        "aligned": m(.8).collisions(layouts["aligned_reference"]),
        "cross_flow": m(.8).collisions(layouts["cross_flow_reference"]),
    }
    for key, result in observed.items():
        err = abs(result - EXPECT[key])
        print(f"{key:>12}: recomputed {result:.12f}; archived {EXPECT[key]:.12f}; error {err:.2e}")
        if err > 1e-8:
            raise RuntimeError(f"Metric 3a reference check failed for {key}. Stopping.")
    for label, folder, key in [("original without start", ORIGINAL_U, "unseeded"),
                                ("original with start", ORIGINAL_S, "seeded")]:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if any([summary["seed"] != 42, summary["population"] != 100,
                summary["generations"] != 120, abs(summary["maximum_shielding"]-.8)>1e-12,
                abs(summary["bird_travel_deg"]-225.)>1e-12,
                abs(summary["downstream_scale_D"]-8.)>1e-12,
                abs(summary["cross_scale_D"]-1.)>1e-12]):
            raise RuntimeError(f"Archived setting mismatch in {folder}")
        conv = pd.read_csv(folder / "convergence.csv")
        if len(conv) != 120 or int(conv["generation"].iloc[-1]) != 120:
            raise RuntimeError(f"Unexpected original convergence length: {folder}")
        if abs(conv["best_risk"].iloc[-1]-EXPECT[key]) > 1e-8:
            raise RuntimeError(f"Unexpected original convergence endpoint: {folder}")
        print(f"Confirmed {label}: 120 generations, seed 42, population 100")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [HERE / n for n in ["config.py", "directional_absorbing_risk.py",
                              "run_directional_absorbing_risk_only.py", "run_validation.py",
                              "risk_only_problem.py"]]
    hashes = {p.name: sha256(p) if p.exists() else "MISSING" for p in paths}
    check_info = {"original_results": {k: str(v) for k, v in
                   [("without_start", ORIGINAL_U), ("with_start", ORIGINAL_S)]},
                  "metric_3a_parameters": {"D_m": 240., "min_spacing_m": 960.,
                                            "bird_toward_deg": 225., "downstream_scale_D": 8.,
                                            "cross_scale_D": 1., "eta": .8,
                                            "population": 100, "seed42_original_generations": 120},
                  "reference_checks": observed, "code_hashes": hashes}
    (OUTPUT / "preflight.json").write_text(json.dumps(check_info, indent=2), encoding="utf-8")
    print(f"PASS: preflight verified; metadata saved to {OUTPUT / 'preflight.json'}")


def sweep():
    check()
    _, m, layouts = inputs()
    rows=[]
    for eta in ETAS:
        values = {name: float(m(eta).collisions(xy)) for name,xy in layouts.items()}
        sorted_items = sorted(values.items(), key=lambda kv: kv[1])
        for name, q in values.items():
            rows.append({"eta": eta, "layout": name, "Q_3a": q,
                         "reduction_vs_unshielded_percent": 100*(12.-q)/12.,
                         "rank": 1+sum(other < q - 1e-10 for other in values.values())})
        print(f"eta={eta:.2f}: " + " < ".join(f"{k} ({v:.6f})" for k,v in sorted_items))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    d=pd.DataFrame(rows)
    d.to_csv(OUTPUT / "fixed_layout_eta_sweep.csv", index=False)
    rank_wide = d.pivot(index="eta", columns="layout", values="rank")
    rank_wide.to_csv(OUTPUT / "fixed_layout_rankings.csv")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8,4.8))
    for name, group in d.groupby("layout", sort=False):
        ax.plot(group["eta"], group["Q_3a"], "o-", label=name.replace("_", " "))
    ax.set_xlabel("Maximum shielding strength, $\\eta$")
    ax.set_ylabel("$Q_{3a}$ (a.u.)")
    ax.set_xticks(ETAS)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"fixed_layout_eta_sensitivity.{ext}", dpi=220)
    plt.close(fig)
    print(f"Sensitivity CSV and PDF: {OUTPUT}")


def lane_counts(xy, threshold=120.0):
    centred=np.asarray(xy, dtype=float)-3000.
    projected=np.sort((centred[:,0]-centred[:,1])/np.sqrt(2.))
    groups=[[projected[0]]]
    for x in projected[1:]:
        if x-groups[-1][-1] <= threshold:
            groups[-1].append(x)
        else:
            groups.append([x])
    return sorted((len(x) for x in groups), reverse=True)


def compare():
    check()
    _, m, base_layouts=inputs()
    original = pd.read_csv(ORIGINAL_U / "convergence.csv")
    aligned = EXPECT["aligned"]
    rows=[]
    histories={}
    layouts={}
    replay_verified=None
    for seed in SEEDS:
        folder=RESULTS / "directional_absorbing_risk_only" / TAG.format(seed=seed, gen=300)
        conv_file=folder / "convergence.csv"
        if not conv_file.exists():
            raise FileNotFoundError(f"Missing 300-generation run for seed {seed}: {conv_file}")
        conv=pd.read_csv(conv_file)
        if int(conv["generation"].iloc[-1]) != 300 or len(conv) != 300:
            raise RuntimeError(f"Expected exactly 300 generations for seed {seed}: {len(conv)} rows")
        if not conv["best_risk"].is_monotonic_decreasing:
            # Best of current GA population is not guaranteed monotonic for every implementation.
            print(f"Note: seed {seed} has non-monotonic population-best curve; checking values individually.")
        xy=load_xyz(folder / "best_layout.csv")
        final_q=float(m(.8).collisions(xy))
        if abs(final_q-float(conv["best_risk"].iloc[-1])) > 1e-8:
            raise RuntimeError(f"Final layout does not match convergence endpoint for seed {seed}")
        q120=float(conv.loc[conv["generation"]==120,"best_risk"].iloc[0])
        if seed == 42:
            differences=np.abs(conv["best_risk"].to_numpy()[:120] - original["best_risk"].to_numpy())
            replay_verified=bool(np.max(differences)<1e-8)
            print(f"Seed 42, generations 1–120: {'EXACT REPLAY PASS' if replay_verified else 'REPLAY MISMATCH'}; max |delta Q|={np.max(differences):.3e}")
            if not replay_verified:
                print("Do not label seed 42 as a verified extension of the original trajectory.")
        rows.append({"seed":seed,"Q_3a_at_generation_120":q120,
                     "Q_3a_at_generation_300":final_q,
                     "improvement_120_to_300":q120-final_q,
                     "beats_aligned_reference_at_300":bool(final_q<aligned),
                     "lane_group_sizes_D_over_2":str(lane_counts(xy)),
                     "source_result_folder":str(folder)})
        histories[seed]=conv
        layouts[seed]=xy
        print(f"Seed {seed}: Q120={q120:.9f}; Q300={final_q:.9f}; lane groups={lane_counts(xy)}")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "extended_run_comparison.csv",index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.0,4.8))
    for seed, conv in histories.items():
        ax.plot(conv["generation"],conv["best_risk"],label=f"Without prescribed start: seed {seed}")
    ax.plot(original["generation"], original["best_risk"],
            ls="--",lw=1.25,alpha=.8,label="Original seed 42 (120 gen)")
    existing_s=pd.read_csv(ORIGINAL_S/"convergence.csv")
    ax.plot(existing_s["generation"], existing_s["best_risk"],ls=":",
            label="With aligned starting layout (120 gen)")
    ax.axhline(aligned, ls="--",lw=1.,c="black", label="Aligned reference (5.525)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("$Q_{3a}$ (a.u.)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=7,loc="best")
    fig.tight_layout()
    for ext in ["png","pdf"]: fig.savefig(OUTPUT/f"extended_convergence.{ext}",dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15.2,4.1),sharex=True,sharey=True)
    along,_=m(.8).direction_vectors()
    for ax,(name,xy) in zip(axes,[(f"Seed {seed}",layouts[seed]) for seed in SEEDS] + [("Aligned reference",base_layouts["aligned_reference"])]):
        xy0=xy-3000.
        ax.scatter(xy0[:,0],xy0[:,1],s=28,zorder=4)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-3000.,-3000.),6000.,6000.,fill=False,ls="--",edgecolor="0.4"))
        ax.annotate("",xy=(2000+700*along[0],2400+700*along[1]),xytext=(2000,2400),arrowprops=dict(arrowstyle="->",lw=1.4))
        ax.set_xlim(-3350,3350);ax.set_ylim(-3350,3350);ax.set_aspect("equal")
        ax.set_title(name,fontsize=10);ax.set_xlabel("x position (m)",fontsize=9)
        ax.tick_params(labelsize=8);ax.grid(alpha=.15)
    axes[0].set_ylabel("y position (m)",fontsize=9)
    fig.tight_layout()
    for ext in ["png","pdf"]: fig.savefig(OUTPUT/f"extended_layouts.{ext}",dpi=220)
    plt.close(fig)
    print(f"PASS: three runs verified; report and figures saved in {OUTPUT}")


def main():
    ap=argparse.ArgumentParser(description="Metric 3a follow-up checks/sweep/summary (no optimisation)")
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check",action="store_true")
    group.add_argument("--sweep",action="store_true")
    group.add_argument("--compare",action="store_true")
    args=ap.parse_args()
    if args.check: check()
    elif args.sweep: sweep()
    else: compare()

if __name__=="__main__":
    main()
