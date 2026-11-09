"""Generate publication-style Metric 2a figures from EXISTING optimisation results.

Does not perform any optimisation or change original result files.

Default Windows usage (from C:\\Users\\Deepak\\Desktop\\bird_layout_opt):
    python plot_metric_2a_31_style.py

Optional:
    python plot_metric_2a_31_style.py --overwrite-figures
    python plot_metric_2a_31_style.py --results-dir "C:\\...\\metric_2a_360sectors_seed42_pop60_gen60"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Works in Anaconda Prompt, without an interactive window.
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


RUN_FOLDER = "metric_2a_360sectors_seed42_pop60_gen60"
PALETTE = {
    "ordinary": "#4C79A4",
    "pareto": "#718397",
    "energy": "#4C79A4",
    "risk": "#E89210",
    "reach": "#B04D45",
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


def read_endpoint(path: Path, n_turbines: int, target_index: int) -> np.ndarray:
    df = pd.read_csv(path)
    required = {"turbine_id", "x_m", "y_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path.name}: missing columns {sorted(required - set(df.columns))}")
    df = df.sort_values("turbine_id")
    ids = df["turbine_id"].astype(int).to_numpy()
    if len(ids) != n_turbines or len(set(ids)) != n_turbines:
        raise ValueError(f"{path.name}: expected {n_turbines} uniquely indexed turbines")
    if target_index not in ids:
        raise ValueError(f"{path.name}: missing designated turbine {target_index}")
    coords = df[["x_m", "y_m"]].to_numpy(float)
    if not np.isfinite(coords).all():
        raise ValueError(f"{path.name}: non-finite coordinates")
    # The saved optimiser keeps turbine IDs 0, 1, ..., 11. Reject anything else.
    if not np.array_equal(ids, np.arange(n_turbines)):
        raise ValueError(f"{path.name}: unexpected turbine numbering {ids.tolist()}")
    return coords


def validate_results(results_dir: Path):
    settings_path = results_dir / "run_settings.json"
    with settings_path.open(encoding="utf-8") as handle:
        settings = json.load(handle)
    if settings.get("metric") != "2a":
        raise ValueError("run_settings.json does not describe Metric 2a")
    if settings.get("run_status") != "complete":
        raise ValueError("Metric 2a run is not marked complete")
    if settings["wind"]["n_sectors"] != 360:
        raise ValueError("Expected the final 360-sector Metric 2a run")
    if settings["metric_formulation"]["designated_turbine_rule"] != "permanently fixed index 0":
        raise ValueError("Unexpected designated-turbine rule")
    sigma = float(settings["metric_formulation"]["sigma_m"])
    if not np.isclose(sigma, 960.0):
        raise ValueError(f"Unexpected sigma: {sigma} m; expected 960 m")
    width = float(settings["site"]["width_m"])
    height = float(settings["site"]["height_m"])
    n_turbines = int(settings["site"]["n_turbines"])
    if width != 6000.0 or height != 6000.0 or n_turbines != 12:
        raise ValueError("Unexpected site/turbine settings")
    df = pd.read_csv(results_dir / "pareto_points.csv")
    if len(df) != 19:
        raise ValueError(f"Expected 19 nondominated layouts, got {len(df)}")
    for col in ("layout_id", "aep_gwh", "risk_Q", "spacing_violations", "boundary_violations"):
        if col not in df:
            raise ValueError(f"pareto_points.csv missing {col}")
    if df["spacing_violations"].sum() or df["boundary_violations"].sum():
        raise ValueError("Invalid layout found in archived Pareto points")
    energy = read_endpoint(results_dir / "layout_energy_optimal.csv", n_turbines, 0)
    risk = read_endpoint(results_dir / "layout_collision_optimal.csv", n_turbines, 0)
    anchors = pd.read_csv(results_dir / "anchor_summary.csv")
    a = anchors.set_index("layout")
    for label in ("energy_optimal", "collision_optimal"):
        if label not in a.index:
            raise ValueError(f"anchor_summary.csv is missing {label}")
    if not np.isclose(df["aep_gwh"].max(), a.loc["energy_optimal", "aep_gwh"], atol=1e-7):
        raise ValueError("Archived highest-AEP anchor is inconsistent with Pareto front")
    if not np.isclose(df["risk_Q"].min(), a.loc["collision_optimal", "risk_Q"], atol=1e-8):
        raise ValueError("Archived lowest-risk anchor is inconsistent with Pareto front")
    return df, energy, risk, anchors, sigma, width, height


def save_pdf_and_png(fig: plt.Figure, path_without_extension: Path) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(
            path_without_extension.with_suffix("." + extension),
            dpi=260,
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)


def plot_pareto(points: pd.DataFrame, anchors: pd.DataFrame, output_dir: Path):
    """Same marker convention as locked Section 3.1, without an in-plot legend."""
    fig, ax = plt.subplots(figsize=(3.85, 2.95))
    ax.scatter(
        points["aep_gwh"], points["risk_Q"],
        s=17, color=PALETTE["pareto"], edgecolor="#4D6171", linewidth=0.3,
        alpha=0.85, zorder=2,
    )
    anchors = anchors.set_index("layout")
    high = anchors.loc["energy_optimal"]
    low = anchors.loc["collision_optimal"]
    ax.scatter([high["aep_gwh"]], [high["risk_Q"]],
               s=60, marker="o", facecolor=PALETTE["energy"],
               edgecolor="black", linewidth=0.7, zorder=5)
    ax.scatter([low["aep_gwh"]], [low["risk_Q"]],
               s=65, marker="D", facecolor="none",
               edgecolor=PALETTE["risk"], linewidth=1.5, zorder=6)
    ax.set_xlabel("AEP (GWh/year)")
    ax.set_ylabel(r"$Q_{2a}$ (a.u.)")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    xx = points["aep_gwh"].to_numpy(float)
    yy = points["risk_Q"].to_numpy(float)
    xpad = max(0.15, (xx.max() - xx.min()) * 0.045)
    ypad = max(0.0010, (yy.max() - yy.min()) * 0.05)
    ax.set_xlim(float(xx.min() - xpad), float(xx.max() + xpad))
    ax.set_ylim(-ypad * 0.6, float(yy.max() + ypad))
    # Marker meanings are defined in the LaTeX caption, not in a legend.
    fig.tight_layout(pad=0.45)
    save_pdf_and_png(fig, output_dir / "metric_2a_pareto")

def plot_endpoint(xy: np.ndarray, sigma_m: float, width_m: float, height_m: float, output: Path):
    """One PDF per endpoint, sized for a 3-inch-wide LaTeX minipage."""
    fig, ax = plt.subplots(figsize=(3.18, 3.15))
    center = xy[0]
    other = xy[1:]
    reach_m = 3.0 * sigma_m
    ax.add_patch(Rectangle((0, 0), width_m, height_m, fill=False,
                           edgecolor="#596674", linewidth=0.8, zorder=1))
    ax.add_patch(Circle(
        tuple(center), reach_m, fill=False, linestyle=(0, (5, 3)),
        edgecolor=PALETTE["reach"], linewidth=1.3, zorder=3,
    ))
    ax.scatter(other[:, 0], other[:, 1], s=28, marker="o",
               facecolor=PALETTE["ordinary"], edgecolor="black",
               linewidth=0.45, zorder=5)
    ax.scatter(center[0], center[1], s=90, marker="*",
               facecolor="#E89210", edgecolor="black",
               linewidth=0.6, zorder=6)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-170, width_m + 170)
    ax.set_ylim(-170, height_m + 170)
    # Avoid colliding tick labels at the final ~3-inch printed size.
    ax.xaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.grid(True, color=PALETTE["grid"], linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    save_pdf_and_png(fig, output)
    nearest = np.linalg.norm(other - center[None, :], axis=1).min()
    return float(nearest)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Original Metric 2a results directory (defaults to Jensen master results).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="New figure directory; defaults to <results-dir>/thesis_figures.",
    )
    parser.add_argument(
        "--overwrite-figures", action="store_true",
        help="Allow replacing only the 3 generated figure pairs in thesis_figures_final.",
    )
    args = parser.parse_args()
    # Place script in bird_layout_opt project root; working folder otherwise works as fallback.
    root = Path(__file__).resolve().parent
    if not (root / "jensen_1deg_approved" / "results").is_dir():
        root = Path.cwd()
    results_dir = args.results_dir or (root / "jensen_1deg_approved" / "results" / RUN_FOLDER)
    if not results_dir.is_dir():
        parser.error(f"Cannot find results directory: {results_dir}")
    out_dir = args.output_dir or (results_dir / "thesis_figures_final")
    outputs = [
        out_dir / f"{stem}.{ext}"
        for stem in [
            "metric_2a_pareto",
            "metric_2a_layout_energy_optimal_3sigma",
            "metric_2a_layout_collision_optimal_3sigma",
        ]
        for ext in ("pdf", "png")
    ]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite_figures:
        parser.error(
            "These thesis figures already exist; refusing to overwrite: "
            + ", ".join(str(p) for p in existing)
            + " (use --overwrite-figures to replace only these plots)"
        )
    points, energy_xy, risk_xy, anchors, sigma, width, height = validate_results(results_dir)
    configure_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_pareto(points, anchors, out_dir)
    d_energy = plot_endpoint(
        energy_xy, sigma, width, height,
        out_dir / "metric_2a_layout_energy_optimal_3sigma",
    )
    d_risk = plot_endpoint(
        risk_xy, sigma, width, height,
        out_dir / "metric_2a_layout_collision_optimal_3sigma",
    )
    print("\nMETRIC 2a THESIS PLOTS COMPLETE — NO OPTIMISATION RUN")
    print(f"Read-only results: {results_dir}")
    print(f"Saved new figures: {out_dir}")
    print("Pareto y-axis: Q_2a (a.u.)")
    print(f"3 sigma radius: {3*sigma:.1f} m")
    print(f"Highest-AEP nearest neighbour: {d_energy:.2f} m")
    print(f"Lowest-risk nearest neighbour: {d_risk:.2f} m")
    print("3 PDF files and 3 PNG files created. Copy PDFs to your thesis figures/ folder.")


if __name__ == "__main__":
    main()
