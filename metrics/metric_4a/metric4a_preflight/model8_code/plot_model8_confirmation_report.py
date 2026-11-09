"""Create a report-ready Model 8 confirmation figure from saved CSV results.

This script is plotting-only. It does not run NSGA-II, FLORIS, or any bird
trajectories. By default, place this file in ``bird_layout_opt/model8_code``
and run it from the ``bird_layout_opt`` project root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULT_SUBDIR = Path(
    "results/model8_confirmation/"
    "p0p6_heading225_CB20xN5000_seed20042_C95_source_seed42"
)


def project_root() -> Path:
    """Find the project root when the script is in model8_code or the root."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "model8_code":
        return script_dir.parent
    return Path.cwd().resolve()


def parse_args() -> argparse.Namespace:
    root = project_root()
    default_csv = root / DEFAULT_RESULT_SUBDIR / "confirmation_layout_summary.csv"
    parser = argparse.ArgumentParser(
        description=(
            "Replot the saved Model 8 independent-confirmation results without "
            "rerunning trajectories, FLORIS, or NSGA-II."
        )
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=default_csv,
        help=f"Saved confirmation summary (default: {default_csv})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder (default: the same folder as input_csv).",
    )
    parser.add_argument(
        "--basename",
        default="model8_confirmation_report",
        help="Output filename without extension.",
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    """Read booleans reliably whether pandas loaded bools or text."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def confidence_multiplier(csv_path: Path, data: pd.DataFrame) -> tuple[float, float]:
    """Recover the exact confidence level and Student-t multiplier used."""
    settings_path = csv_path.parent / "confirmation_settings.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        multiplier = float(settings["student_t_critical_value"])
        confidence = float(settings.get("confidence_level", 0.95))
        return multiplier, confidence

    batch_counts = pd.to_numeric(
        data["confirmation_batch_count"], errors="raise"
    ).unique()
    if len(batch_counts) != 1:
        raise ValueError("All rows must have the same confirmation batch count.")

    try:
        from scipy.stats import t
    except ImportError as exc:
        raise FileNotFoundError(
            "confirmation_settings.json is missing and SciPy is unavailable, "
            "so the confidence multiplier cannot be reconstructed."
        ) from exc

    batch_count = int(batch_counts[0])
    confidence = 0.95
    multiplier = float(t.ppf(0.5 + confidence / 2.0, df=batch_count - 1))
    return multiplier, confidence


def check_columns(data: pd.DataFrame) -> None:
    required = {
        "solution_index",
        "aep_gwh",
        "confirmation_batch_count",
        "confirmation_E_enc",
        "confirmation_E_enc_batch_se",
        "confirmation_P_ge_1",
        "confirmation_P_ge_1_batch_se",
        "confirmation_exact_nondominated_E_enc",
        "confirmation_robustly_nondominated_E_enc",
        "confirmation_exact_nondominated_P_ge_1",
        "confirmation_robustly_nondominated_P_ge_1",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))


def padded_limits(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values - errors))
    high = float(np.max(values + errors))
    span = high - low
    padding = 0.035 * span if span > 0.0 else max(abs(low) * 0.02, 0.01)
    return low - padding, high + padding


def make_plot(data: pd.DataFrame, multiplier: float, confidence: float):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.35), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.77, wspace=0.12)

    panels = [
        {
            "metric": "confirmation_E_enc",
            "se": "confirmation_E_enc_batch_se",
            "exact": "confirmation_exact_nondominated_E_enc",
            "robust": "confirmation_robustly_nondominated_E_enc",
            "scale": 1.0,
            "title": "(a) Encounter intensity",
            "xlabel": r"Mean encounters per trajectory, $E_{\mathrm{enc}}$",
        },
        {
            "metric": "confirmation_P_ge_1",
            "se": "confirmation_P_ge_1_batch_se",
            "exact": "confirmation_exact_nondominated_P_ge_1",
            "robust": "confirmation_robustly_nondominated_P_ge_1",
            "scale": 100.0,
            "title": "(b) Routes with at least one encounter",
            "xlabel": r"Routes with $\geq 1$ encounter, $P_{\geq 1}$ (%)",
        },
    ]

    legend_handles = None
    legend_labels = None
    for axis, panel in zip(axes, panels):
        values = pd.to_numeric(data[panel["metric"]], errors="raise").to_numpy()
        standard_errors = pd.to_numeric(data[panel["se"]], errors="raise").to_numpy()
        values = values * panel["scale"]
        errors = multiplier * standard_errors * panel["scale"]
        aep = pd.to_numeric(data["aep_gwh"], errors="raise").to_numpy()
        exact = as_bool(data[panel["exact"]]).to_numpy()
        robust = as_bool(data[panel["robust"]]).to_numpy()

        all_handle = axis.errorbar(
            values,
            aep,
            xerr=errors,
            fmt="o",
            markersize=4.8,
            markerfacecolor="#9C9C9C",
            markeredgecolor="white",
            markeredgewidth=0.35,
            color="#9C9C9C",
            ecolor="#C5C5C5",
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            alpha=0.88,
            label=f"All {len(data)} finalists (mean and {confidence:.0%} CI)",
            zorder=1,
        )
        exact_handle = axis.scatter(
            values[exact],
            aep[exact],
            s=42,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.45,
            label="Mean-only Pareto front",
            zorder=3,
        )
        robust_handle = axis.scatter(
            values[robust],
            aep[robust],
            s=135,
            marker="*",
            color="#D55E00",
            edgecolor="#9B3200",
            linewidth=0.45,
            label="Uncertainty-aware front",
            zorder=4,
        )

        solution_12 = data["solution_index"].astype(int).eq(12).to_numpy()
        if np.any(solution_12):
            index = int(np.flatnonzero(solution_12)[0])
            axis.annotate(
                "Solution 12",
                xy=(values[index], aep[index]),
                xytext=(8, -15),
                textcoords="offset points",
                fontsize=9,
                fontweight="semibold",
                color="#8A2E00",
                ha="left",
                va="top",
                zorder=5,
            )

        axis.set_title(panel["title"], loc="left", fontweight="semibold", pad=8)
        axis.set_xlabel(panel["xlabel"], labelpad=8)
        axis.set_xlim(*padded_limits(values, errors))
        axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.65)
        axis.set_axisbelow(True)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)

        if legend_handles is None:
            legend_handles = [all_handle, exact_handle, robust_handle]
            legend_labels = [
                f"All {len(data)} finalists (mean and {confidence:.0%} CI)",
                "Mean-only Pareto front",
                "Uncertainty-aware front",
            ]

    axes[0].set_ylabel("FLORIS AEP (GWh/year)", labelpad=8)
    axes[1].tick_params(labelleft=False)

    fig.suptitle(
        f"Model 8: independent confirmation of {len(data)} finalist layouts",
        fontsize=15,
        fontweight="semibold",
        y=0.975,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.65,
    )
    fig.text(
        0.5,
        0.035,
        "Preferred direction is towards the top-left: higher energy and fewer encounters. "
        "Horizontal bars show Monte Carlo sampling uncertainty.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4D4D4D",
    )
    return fig


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_csv.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv).sort_values("solution_index").reset_index(drop=True)
    check_columns(data)
    multiplier, confidence = confidence_multiplier(input_csv, data)
    fig = make_plot(data, multiplier, confidence)

    png_path = output_dir / f"{args.basename}.png"
    pdf_path = output_dir / f"{args.basename}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Plot-only export complete. No trajectories were simulated.")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
