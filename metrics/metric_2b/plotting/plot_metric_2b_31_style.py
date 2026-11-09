"""Create thesis-ready Metric 2b Pareto and two endpoint-layout figures.

READS EXISTING 360-sector results; NEVER starts an optimisation and NEVER edits
original saved results. Creates 3 PDFs + 3 PNGs in a thesis_figures folder.

From the project root in Windows Anaconda Prompt:
    python plot_metric_2b_31_style.py

Optional arguments:
    --results-dir "C:\\...\\metric_2b_360sectors_seed42_pop80_gen100"
    --references-file "C:\\...\\final_reference_comparison.csv"
    --output-dir "C:\\...\\some_separate_output_folder"
    --overwrite-figures  # regenerate ONLY the six thesis_figures files

The Pareto references use the NEW 360-sector AEP-only run and the original
risk-only layout measured at 1 degree, NOT the historical 12-sector AEP-only
reference. All numerical values are read from the saved CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


METRIC_FOLDER = "metric_2b_360sectors_seed42_pop80_gen100"
AEP_FOLDER = "metric_aep_only_360sectors_seed42_pop160_gen250"

# Same color family as the completed Metric 2a plotting script.
COLORS = {
    "ordinary": "#4C79A4",  # Turbines
    "pareto": "#718397",    # Pareto solutions
    "energy": "#4C79A4",    # Highest AEP
    "risk": "#E89210",      # Lowest risk
    "aep_ref": "#8364A2",   # Separate AEP-only search
    "risk_ref": "#947049",  # Separate risk-only search
    "grid": "#D8DFE5",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def read_endpoint(path: Path, n_turbines: int, width: float, height: float) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing endpoint coordinates: {path}")
    df = pd.read_csv(path)
    required = {"turbine_id", "x_m", "y_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path.name} missing columns: {sorted(required - set(df.columns))}")
    df = df.sort_values("turbine_id")
    ids = df["turbine_id"].to_numpy(int)
    if not np.array_equal(ids, np.arange(n_turbines)):
        raise ValueError(f"{path.name}: expected unique turbine IDs 0..{n_turbines-1}")
    xy = df[["x_m", "y_m"]].to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError(f"{path.name}: non-finite coordinates")
    if np.any(xy[:, 0] < -1e-8) or np.any(xy[:, 0] > width + 1e-8):
        raise ValueError(f"{path.name}: x-coordinate outside site")
    if np.any(xy[:, 1] < -1e-8) or np.any(xy[:, 1] > height + 1e-8):
        raise ValueError(f"{path.name}: y-coordinate outside site")
    return xy


def pair_distances(xy: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.linalg.norm(xy[i] - xy[j])
        for i in range(len(xy)) for j in range(i + 1, len(xy))
    ], dtype=float)


def gaussian_Q(xy: np.ndarray, sigma: float) -> float:
    d = pair_distances(xy)
    return float(np.exp(-(d ** 2) / (2 * sigma ** 2)).sum())


def read_references(ref_path: Path, rechecked_path: Path):
    """Read the actual 360-sector AEP-only search and risk-only reference."""
    if not ref_path.is_file():
        raise FileNotFoundError(
            "Cannot find the NEW AEP-only reference table:\n"
            f"  {ref_path}\n"
            "Its source is the completed 360-sector AEP-only results folder.\n"
            "Pass --references-file PATH if that folder is elsewhere."
        )
    ref = pd.read_csv(ref_path)
    needed = {"reference", "aep_1deg_gwh", "risk_Q"}
    if not needed.issubset(ref.columns):
        raise ValueError(f"Reference CSV missing: {sorted(needed - set(ref.columns))}")
    lower = ref["reference"].astype(str).str.lower()
    energy_rows = ref[lower.str.contains("aep-only", regex=False) &
                      ~lower.str.contains("historical", regex=False)]
    risk_rows = ref[lower.str.contains("risk-only", regex=False)]
    if len(energy_rows) != 1 or len(risk_rows) != 1:
        raise ValueError(
            "Expected one NEW AEP-only reference and one risk-only reference "
            "in final_reference_comparison.csv"
        )
    ae = energy_rows.iloc[0]
    ri = risk_rows.iloc[0]
    aep_ref = {"aep_gwh": float(ae["aep_1deg_gwh"]), "Q": float(ae["risk_Q"])}
    risk_ref = {"aep_gwh": float(ri["aep_1deg_gwh"]), "Q": float(ri["risk_Q"])}
    if not all(np.isfinite(list(aep_ref.values()) + list(risk_ref.values()))):
        raise ValueError("Non-finite single-objective references")

    if not rechecked_path.is_file():
        raise FileNotFoundError(f"Missing 2b reference check: {rechecked_path}")
    ref2 = pd.read_csv(rechecked_path)
    risk_only = ref2[ref2["reference"].astype(str).str.lower().eq("risk_only")]
    if len(risk_only) != 1:
        raise ValueError("single_objective_references_rechecked.csv lacks risk_only row")
    r = risk_only.iloc[0]
    if not np.isclose(risk_ref["aep_gwh"], float(r["aep_1deg_gwh"]), atol=1e-6):
        raise ValueError("Risk-only AEP inconsistent between reference files")
    if not np.isclose(risk_ref["Q"], float(r["risk_Q"]), atol=1e-8):
        raise ValueError("Risk-only Q inconsistent between reference files")
    return aep_ref, risk_ref


def load_and_validate(results_dir: Path, ref_path: Path):
    with (results_dir / "run_settings.json").open(encoding="utf-8") as f:
        settings = json.load(f)
    if settings.get("metric") != "2b" or settings.get("run_status") != "complete":
        raise ValueError("This is not a completed Metric 2b optimisation")
    if int(settings["wind"]["n_sectors"]) != 360:
        raise ValueError("Metric 2b results must use 360 wind sectors")
    sigma = float(settings["metric_formulation"]["sigma_m"])
    width = float(settings["site"]["width_m"])
    height = float(settings["site"]["height_m"])
    n = int(settings["site"]["n_turbines"])
    min_spacing = float(settings["site"]["minimum_spacing_m"])
    if (sigma, width, height, n, min_spacing) != (1920.0, 6000.0, 6000.0, 12, 960.0):
        raise ValueError("2b Gaussian length, site or turbine constraints unexpected")

    points = pd.read_csv(results_dir / "pareto_points.csv")
    needed = {"layout_id", "aep_gwh", "risk_Q", "spacing_violations", "boundary_violations"}
    if not needed.issubset(points.columns):
        raise ValueError(f"Pareto CSV missing: {sorted(needed - set(points.columns))}")
    if len(points) != 20 or points["layout_id"].nunique() != 20:
        raise ValueError("Expected exactly 20 distinct nondominated layouts")
    if int(points["spacing_violations"].sum()) or int(points["boundary_violations"].sum()):
        raise ValueError("Saved Pareto solutions contain spacing/boundary violations")

    anchors = pd.read_csv(results_dir / "anchor_summary.csv").set_index("layout")
    for key in ("energy_optimal", "collision_optimal"):
        if key not in anchors.index:
            raise ValueError(f"Missing Pareto anchor {key}")
    if not np.isclose(points["aep_gwh"].max(), float(anchors.loc["energy_optimal", "aep_gwh"]), atol=1e-6):
        raise ValueError("Highest-AEP anchor does not match Pareto front")
    if not np.isclose(points["risk_Q"].min(), float(anchors.loc["collision_optimal", "risk_Q"]), atol=1e-8):
        raise ValueError("Lowest-risk anchor does not match Pareto front")

    coords = {}
    for key in ("energy_optimal", "collision_optimal"):
        xy = read_endpoint(results_dir / f"layout_{key}.csv", n, width, height)
        distances = pair_distances(xy)
        if distances.min() < min_spacing - 1e-5:
            raise ValueError(f"{key}: turbine pair too close")
        q = gaussian_Q(xy, sigma)
        row = anchors.loc[key]
        if not np.isclose(q, float(row["risk_Q"]), atol=1e-8):
            raise ValueError(f"{key}: stored Gaussian Q does not match layout")
        if not np.isclose(float(distances.min()), float(row["minimum_pair_distance_m"]), atol=1e-5):
            raise ValueError(f"{key}: minimum pair distance inconsistent with archive")
        if not np.isclose(float(distances.mean()), float(row["mean_pair_distance_m"]), atol=1e-5):
            raise ValueError(f"{key}: mean pair distance inconsistent with archive")
        coords[key] = xy

    aep_ref, risk_ref = read_references(
        ref_path, results_dir / "single_objective_references_rechecked.csv"
    )
    return points, anchors, coords, aep_ref, risk_ref, width, height


def save(fig, stem: Path) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + extension), dpi=260,
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_pareto(points, anchors, aep_ref, risk_ref, out_dir):
    """Section 3.1 marker convention + separate reference symbols; no legend."""
    fig, ax = plt.subplots(figsize=(3.85, 2.95))
    ax.scatter(points["aep_gwh"], points["risk_Q"],
               s=17, color=COLORS["pareto"], edgecolor="#4D6171",
               linewidth=0.3, alpha=0.85, zorder=3)
    ae = anchors.loc["energy_optimal"]
    ri = anchors.loc["collision_optimal"]
    # Pareto endpoints identical to Section 3.1:
    # blue filled circle and orange open diamond.
    ax.scatter([ae["aep_gwh"]], [ae["risk_Q"]], s=60,
               marker="o", facecolor=COLORS["energy"],
               edgecolor="black", linewidth=0.7, zorder=5)
    ax.scatter([ri["aep_gwh"]], [ri["risk_Q"]], s=65,
               marker="D", facecolor="none", edgecolor=COLORS["risk"],
               linewidth=1.5, zorder=6)
    # The separate reference searches have their own distinct shapes.
    ax.scatter([aep_ref["aep_gwh"]], [aep_ref["Q"]], s=65,
               marker="^", facecolor=COLORS["aep_ref"],
               edgecolor="black", linewidth=0.65, zorder=6)
    ax.scatter([risk_ref["aep_gwh"]], [risk_ref["Q"]], s=58,
               marker="s", facecolor=COLORS["risk_ref"],
               edgecolor="black", linewidth=0.65, zorder=6)
    ax.set_xlabel("AEP (GWh/year)")
    ax.set_ylabel(r"$Q_{2b}$ (a.u.)")
    ax.grid(True, color=COLORS["grid"], linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)

    all_x = np.r_[points["aep_gwh"].to_numpy(float), aep_ref["aep_gwh"], risk_ref["aep_gwh"]]
    all_y = np.r_[points["risk_Q"].to_numpy(float), aep_ref["Q"], risk_ref["Q"]]
    xpad = max(0.18, 0.055 * float(np.ptp(all_x)))
    ypad = max(0.11, 0.075 * float(np.ptp(all_y)))
    ax.set_xlim(float(all_x.min() - xpad), float(all_x.max() + xpad))
    ax.set_ylim(float(all_y.min() - ypad), float(all_y.max() + ypad))
    fig.tight_layout(pad=0.45)
    save(fig, out_dir / "metric_2b_pareto_final_refs")

def plot_endpoint(xy, width, height, output):
    fig, ax = plt.subplots(figsize=(3.18, 3.15))
    ax.add_patch(Rectangle((0, 0), width, height, fill=False,
                           edgecolor="#596674", linewidth=0.8, zorder=1))
    ax.scatter(xy[:, 0], xy[:, 1], s=28, marker="o",
               facecolor=COLORS["ordinary"], edgecolor="black",
               linewidth=0.45, zorder=5)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-170, width + 170)
    ax.set_ylim(-170, height + 170)
    ax.xaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.grid(True, color=COLORS["grid"], linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    save(fig, output)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="The original Metric 2b results folder")
    parser.add_argument("--references-file", type=Path, default=None,
                        help="Path to the NEW AEP-only final_reference_comparison.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save new figures (default: Metric 2b/thesis_figures)")
    parser.add_argument("--overwrite-figures", action="store_true",
                        help="Regenerate ONLY these six plot files (never original plots/data)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if not (root / "jensen_1deg_approved" / "results").is_dir():
        root = Path.cwd()
    master_results = root / "jensen_1deg_approved" / "results"
    results_dir = args.results_dir or (master_results / METRIC_FOLDER)
    refs_file = args.references_file or (
        master_results / AEP_FOLDER / "final_reference_comparison.csv"
    )
    if not results_dir.is_dir():
        parser.error(f"Metric 2b results folder not found: {results_dir}")

    out_dir = args.output_dir or results_dir / "thesis_figures_final"
    if out_dir.resolve() == results_dir.resolve():
        parser.error("Refusing to create thesis figures in the original results root")
    stems = ("metric_2b_pareto_final_refs", "metric_2b_layout_energy_optimal",
             "metric_2b_layout_collision_optimal")
    paths = [out_dir / f"{stem}.{ext}" for stem in stems for ext in ("pdf", "png")]
    existing = [p for p in paths if p.exists()]
    if existing and not args.overwrite_figures:
        parser.error("Thesis figures already exist; no files changed:\n  "
                     + "\n  ".join(map(str, existing))
                     + "\nUse --overwrite-figures ONLY if you wish to regenerate these six plots.")

    points, anchors, coords, aep_ref, risk_ref, width, height = load_and_validate(
        results_dir, refs_file
    )
    configure_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_pareto(points, anchors, aep_ref, risk_ref, out_dir)
    plot_endpoint(coords["energy_optimal"], width, height,
                  out_dir / "metric_2b_layout_energy_optimal")
    plot_endpoint(coords["collision_optimal"], width, height,
                  out_dir / "metric_2b_layout_collision_optimal")

    print("\nMETRIC 2b THESIS PLOTS COMPLETE — NO OPTIMISATION RUN")
    print(f"Read-only Metric 2b results: {results_dir}")
    print(f"Reference CSV: {refs_file}")
    print(f"Figures saved in: {out_dir}")
    print("Pareto y-axis: Q_2b (a.u.)")
    print(f"Pareto solutions: {len(points)}")
    print(f"Highest-AEP endpoint: {float(anchors.loc['energy_optimal', 'aep_gwh']):.6f} GWh/year")
    print(f"Lowest-risk endpoint: Q = {float(anchors.loc['collision_optimal', 'risk_Q']):.6f}")
    print(f"AEP-only reference: {aep_ref['aep_gwh']:.6f} GWh/year; Q={aep_ref['Q']:.6f}")
    print(f"Risk-only reference: {risk_ref['aep_gwh']:.6f} GWh/year; Q={risk_ref['Q']:.6f}")
    print("Created 3 PDFs + 3 PNGs; original optimiser outputs untouched.")


if __name__ == "__main__":
    main()
