"""Metric 4a - revised thesis figures, read-only plotter.

Intended location on Windows:
  C:\\Users\\Deepak\\Desktop\\bird_layout_opt\\metric4a_preflight\\metric4a_plot72_corridor_adjusted.py

Run in the floris-env Anaconda Prompt:
  python metric4a_plot72_corridor_adjusted.py

Reads the completed optimisation and independent-evaluation result folders.
Does NOT run FLORIS, simulate birds, or change the original results.

Important: independent-evaluation Q_4a and its 95% CI are multiplied by
W_independent / W_optimisation so that the two samples can be displayed
on a common, original-corridor-width scale. This is a width-normalised
comparison, NOT an independent simulation within the original corridor.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parent.parent
OPT = ROOT / "metric_4a_72sectors"
CONF = ROOT / "metric_4a_72sectors_confirmation"
OUT = ROOT / "metric_4a_72sectors_figures_corridor_adjusted"

NEEDED = [
    OPT / "pareto_points.csv",
    OPT / "highest-AEP_layout.csv",
    OPT / "lowest-risk_layout.csv",
    OPT / "hypervolume.csv",
    CONF / "confirmation_batches.csv",
    CONF / "confirmation_endpoint_summary.csv",
    CONF / "confirmation_paired_contrasts.csv",
    CONF / "confirmation_settings.json",
]
for path in NEEDED:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
if OUT.exists():
    raise FileExistsError(
        f"The output directory already exists: {OUT}\n"
        "To protect existing figures, move/rename that directory before running again."
    )

FRONT = pd.read_csv(OPT / "pareto_points.csv")
BATCHES = pd.read_csv(CONF / "confirmation_batches.csv")
SUMMARY = pd.read_csv(CONF / "confirmation_endpoint_summary.csv")
CONTRASTS = pd.read_csv(CONF / "confirmation_paired_contrasts.csv")
HV = pd.read_csv(OPT / "hypervolume.csv")
SETTINGS = json.loads((CONF / "confirmation_settings.json").read_text(encoding="utf-8"))

W_OPT = float(SETTINGS["original_width_m"])
W_IND = float(SETTINGS["common_width_m"])
SCALE = W_IND / W_OPT
assert np.isclose(W_OPT, 8485.28137423857, atol=1e-5)
assert np.isclose(W_IND, 8787.57961404495, atol=1e-5)
assert np.isclose(SCALE, 1.0356261892179748, rtol=0, atol=1e-10)
assert len(FRONT) == 38
assert len(HV) == 100 and int(HV.generation.iloc[-1]) == 100
assert sorted(BATCHES.seed.unique()) == list(range(20042, 20062))
assert len(BATCHES) == 80
assert (BATCHES.N_failed == 0).all()

COL = {
    "front": "#326CAB",
    "energy": "#E58A1F",
    "risk": "#C6443B",
    "gray": "#748295",
    "outline": "#1F2937",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.labelsize": 11.2, "legend.fontsize": 8.5,
    "xtick.labelsize": 9.2, "ytick.labelsize": 9.2,
    "axes.edgecolor": "#CBD3DB", "axes.linewidth": 0.8,
    "xtick.color": "#334155", "ytick.color": "#334155",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})


def endpoint(name: str):
    selected = SUMMARY[
        (SUMMARY.layout == name) & (SUMMARY.response_probability == 0.6)
    ]
    assert len(selected) == 1, name
    row = selected.iloc[0]
    sample = BATCHES[
        (BATCHES.layout == name) & (BATCHES.response_probability == 0.6)
    ].sort_values("seed")["Q_4a"].to_numpy(dtype=float)
    assert len(sample) == 20
    unscaled_mean = float(sample.mean())
    ci_halfwidth = float(t.ppf(0.975, 19) * sample.std(ddof=1) / math.sqrt(20))
    assert np.isclose(unscaled_mean, row.confirmed_Q_4a, rtol=0, atol=1e-12)
    return {
        "row": row,
        "sample": sample,
        "raw_mean": unscaled_mean,
        "raw_ci_halfwidth": ci_halfwidth,
        "adjusted_mean": unscaled_mean * SCALE,
        "adjusted_ci_halfwidth": ci_halfwidth * SCALE,
        "optimisation_mean": float(row.training_Q_4a),  # archived field name
        "aep": float(row.aep_5deg_gwh),
    }


data = {name: endpoint(name) for name in ("highest-AEP", "lowest-risk")}
assert np.isclose(data["highest-AEP"]["aep"], 831.4654021246256, atol=1e-7)
assert np.isclose(data["lowest-risk"]["aep"], 810.9579417748998, atol=1e-7)
assert np.isclose(data["highest-AEP"]["optimisation_mean"], 0.2369, atol=1e-12)
assert np.isclose(data["lowest-risk"]["optimisation_mean"], 0.1058, atol=1e-12)

OUT.mkdir(parents=True, exist_ok=False)


def save_pair(fig, filename):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{filename}.{ext}", dpi=240,
                    bbox_inches="tight", pad_inches=0.24)
    plt.close(fig)


# FIGURE 1. Consistent with the other thesis fronts: bird objective horizontal,
# AEP vertical. The existing optimisation-sample coordinates are NOT scaled.
fig, ax = plt.subplots(figsize=(8.1, 5.7))
ax.scatter(
    FRONT.E_enc, FRONT.aep_gwh, s=42, color=COL["front"], alpha=0.78,
    edgecolor="white", linewidth=0.55, zorder=2,
    label="Pareto layouts (optimisation sample)",
)

for name, marker, color, size in [
    ("highest-AEP", "o", COL["energy"], 115),
    ("lowest-risk", "*", COL["risk"], 195),
]:
    v = data[name]
    ax.scatter(
        [v["optimisation_mean"]], [v["aep"]], s=size,
        color=color, marker=marker, edgecolor="white",
        linewidth=1.1, zorder=5,
        label=f"{name}: optimisation sample",
    )
    ax.errorbar(
        v["adjusted_mean"], v["aep"], xerr=v["adjusted_ci_halfwidth"],
        fmt="D", color=color, markerfacecolor="white",
        markeredgewidth=1.7, markersize=8.2,
        capsize=4, elinewidth=1.7, zorder=6,
        label=f"{name}: independent evaluation (95% CI)",
    )

ax.set_xlabel(r"$Q_{4\mathrm{a}}$ (encounters per entering bird)")
ax.set_ylabel("AEP (GWh/year)")
ax.grid(alpha=0.18, linewidth=0.7)
ax.set_axisbelow(True)
ax.margins(x=0.12, y=0.13)
ax.legend(
    loc="lower right", frameon=True, facecolor="white",
    edgecolor="#E1E6EC", fontsize=8.0, labelspacing=0.65,
)
fig.tight_layout(pad=1.5)
save_pair(fig, "metric4a_front_corridor_adjusted")

# FIGURE 2. 12 turbine centres, encounter zones (1D radius), aligned pairs.
D = 242.24
THETA = np.deg2rad(225.0)
flight = np.array([np.cos(THETA), np.sin(THETA)])


def aligned_pairs(xy: np.ndarray):
    result = []
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            vec = xy[j] - xy[i]
            if np.linalg.norm(vec) <= 8 * D + 1e-9:
                angle = abs(np.arctan2(
                    flight[0] * vec[1] - flight[1] * vec[0],
                    np.dot(flight, vec),
                ))
                angle = min(angle, abs(np.pi - angle))
                if angle <= np.deg2rad(5) + 1e-12:
                    result.append((i, j))
    return result


fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.8), sharex=True, sharey=True)
for ax, name, panel in zip(axes, ("highest-AEP", "lowest-risk"), ("a", "b")):
    xy = pd.read_csv(OPT / f"{name}_layout.csv")[["x_m", "y_m"]].to_numpy(float)
    assert xy.shape == (12, 2)
    ax.add_patch(Rectangle(
        (0, 0), 6000, 6000, facecolor="#F8FAFC", edgecolor="#607186",
        linewidth=1.2, zorder=0,
    ))
    for x, y in xy:
        ax.add_patch(Circle(
            (x, y), radius=D, facecolor=COL["front"], alpha=0.055,
            edgecolor=COL["front"], linewidth=0.9, zorder=1,
        ))
    pairs = aligned_pairs(xy)
    for i, j in pairs:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1],
                linestyle="--", color="#8795A5", linewidth=1.3, zorder=2)
    ax.scatter(
        xy[:, 0], xy[:, 1], s=64, color=COL["energy"] if panel == "a" else COL["risk"],
        edgecolor="white", linewidth=0.9, zorder=4,
    )
    ax.annotate(
        "", xy=(5300, 5240), xytext=(5840, 5780),
        arrowprops=dict(arrowstyle="-|>", linewidth=2.0, color=COL["outline"]),
    )
    ax.text(5870, 6010, "Bird flight 225°", ha="right", va="bottom",
            fontsize=9.0, color=COL["outline"])
    ax.set_xlim(-450, 6450)
    ax.set_ylim(-450, 6450)
    ax.set_xticks([0, 2000, 4000, 6000])
    ax.set_yticks([0, 2000, 4000, 6000])
    ax.set_xlabel("x position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.14)
    ax.text(0.5, -0.18, f"({panel}) {name}", transform=ax.transAxes,
            ha="center", va="top", fontsize=11, weight="bold", color="#273549")
    expected_pairs = 0 if name == "highest-AEP" else 7
    assert len(pairs) == expected_pairs, (name, len(pairs), expected_pairs)
    print(f"PASS | {name}: {len(pairs)} aligned pairs")
axes[0].set_ylabel("y position (m)")
# Explicit second-panel y label as requested, while preserving shared scale.
axes[1].set_ylabel("y position (m)")
fig.subplots_adjust(left=0.075, right=0.98, top=0.965, bottom=0.23, wspace=0.2)
save_pair(fig, "metric4a_endpoint_layouts")

# FIGURE 3. Convergence, not altered by the corridor rescaling.
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(HV.generation, HV.normalised_hypervolume, color=COL["front"], linewidth=2.1)
ax.scatter([100], [HV.normalised_hypervolume.iloc[-1]],
           color=COL["energy"], s=70, zorder=3)
ax.set_xlabel("Generation")
ax.set_ylabel("Normalised hypervolume")
ax.set_xlim(1, 102)
ax.grid(alpha=0.2)
fig.tight_layout(pad=1.5)
save_pair(fig, "metric4a_hypervolume")


# Machine-readable comparison. The input archives are never changed.
comparison = []
for name, v in data.items():
    old = v["optimisation_mean"]
    new = v["adjusted_mean"]
    comparison.append({
        "layout": name,
        "aep_5deg_gwh": v["aep"],
        "optimisation_sample_Q_4a": old,
        "independently_evaluated_Q_4a_raw_widened_corridor": v["raw_mean"],
        "independently_evaluated_Q_4a_original_corridor_width_adjusted": new,
        "adjusted_independent_Q_4a_ci95_lower": new - v["adjusted_ci_halfwidth"],
        "adjusted_independent_Q_4a_ci95_upper": new + v["adjusted_ci_halfwidth"],
        "adjusted_minus_optimisation_Q_4a_absolute": new - old,
        "adjusted_minus_optimisation_Q_4a_percent": 100 * (new / old - 1),
        "optimisation_corridor_width_m": W_OPT,
        "independent_corridor_width_m": W_IND,
        "normalisation_factor": SCALE,
    })
pd.DataFrame(comparison).to_csv(OUT / "metric4a_corridor_adjusted_comparison.csv", index=False,
                                float_format="%.12g")


def paired_contrast(p, metric):
    match = CONTRASTS[
        np.isclose(pd.to_numeric(CONTRASTS.response_probability, errors="coerce"), p)
        & (CONTRASTS.metric == metric)
    ]
    assert len(match) == 1, (p, metric)
    return match.iloc[0]


baseline_q = paired_contrast(0.6, "Q_4a")
zero_q = paired_contrast(0.0, "Q_4a")
zero_p = paired_contrast(0.0, "P_ge_1")
expected_no_response = 12 * (2 * D) / W_IND
assert abs(expected_no_response - 0.66159) < 5e-5
assert float(zero_q.ci95_low) < 0 < float(zero_q.ci95_high)

cap_front = (
    "Figure 1. Metric 4a energy-encounter Pareto front for the 72-sector (5°) "
    "optimisation. Blue markers show 38 non-dominated layouts scored with the "
    "optimisation sample (10,000 birds per layout; original corridor width "
    f"{W_OPT:,.2f} m). Filled endpoint markers show the optimisation-sample "
    r"$Q_{4\mathrm{a}}$; open diamonds and horizontal 95% confidence intervals show "
    "the independently evaluated endpoint scores (20 paired batches of 5,000 "
    "birds per layout). Independent evaluation used a common widened corridor "
    f"({W_IND:,.2f} m) to fully cover both layouts' encounter zones. For comparison "
    "on the original-corridor-width scale, independent values and confidence "
    f"intervals were multiplied by {W_IND:,.2f}/{W_OPT:,.2f} = {SCALE:.6f}. "
    "This is a width-normalised estimate, not a new independent simulation "
    "on the narrower corridor."
)
cap_layout = (
    "Figure 2. Metric 4a endpoint layouts: (a) highest-AEP and (b) lowest-risk "
    "at 5° wind-direction resolution. Filled markers denote turbine centres; "
    "pale circles show the 1D horizontal encounter radius; dashed lines indicate "
    "turbine pairs within 8D whose connecting line is within ±5° of the "
    "45°–225° bird-flight axis. Arrows indicate bird movement towards 225°. "
    "There are zero aligned pairs in (a) and seven in (b)."
)
cap_hv = (
    "Figure 3. Normalised hypervolume across the 100 generations of the "
    "72-sector Metric 4a NSGA-II optimisation (population 80, seed 42). "
    "Hypervolume describes coverage and extent of the stored objective-space "
    "solution set, not the ecological accuracy of the bird model."
)
(OUT / "metric4a_figure_captions.txt").write_text(
    "\n\n".join((cap_front, cap_layout, cap_hv)) + "\n", encoding="utf-8"
)

res = [
    "Metric 4a — results text (thesis terminology)",
    "",
    "The 72-sector optimisation retained 38 non-dominated layouts. The highest-AEP "
    "layout produced 831.4654 GWh/year with an optimisation-sample Q_4a of "
    "0.2369 encounters per entering bird; the lowest-risk layout produced "
    "810.9579 GWh/year with an optimisation-sample Q_4a of 0.1058. "
    "Their AEP difference was 20.5075 GWh/year (2.47% of highest-AEP).",
    "",
    f"To compare the independent evaluation (common corridor {W_IND:,.2f} m) "
    f"with the optimisation sample (corridor {W_OPT:,.2f} m), the independently "
    f"evaluated Q_4a scores and their intervals were multiplied by "
    f"{SCALE:.8f}. The adjusted independent values were "
    f"{data['highest-AEP']['adjusted_mean']:.6f} for highest-AEP and "
    f"{data['lowest-risk']['adjusted_mean']:.6f} for lowest-risk, "
    "respectively 2.78% and 1.85% above the optimisation-sample values. "
    "This adjustment is a corridor-width normalisation, not a repeat simulation "
    "within the original corridor.",
    "",
    "At the baseline response probability p = 0.6, the unadjusted independent "
    "evaluation on the shared widened corridor yielded Q_4a = 0.23510 "
    "(highest-AEP) and 0.10405 (lowest-risk). The paired difference "
    "(lowest-risk minus highest-AEP) was -0.13105 encounters per entering bird "
    "(95% confidence interval [-0.13410, -0.12800]); the lowest-risk endpoint "
    "therefore had approximately 55.7% fewer encounters for 2.47% lower AEP. "
    "After width normalisation, the paired difference was -0.13572 "
    "(95% confidence interval [-0.13888, -0.13256]); the percentage "
    "reduction is unchanged by a common multiplicative adjustment.",
    "",
    "The no-response control (p = 0) passed. With straight-flight trajectories "
    "and all encounter zones covered by the same corridor, the expected mean "
    "encounter count is layout-independent: it is set by the sum of encounter "
    "zone widths divided by corridor width, 12 × 2 × 242.24 / "
    f"{W_IND:.2f} ≈ {expected_no_response:.5f}. The independently evaluated "
    "Q_4a values were 0.66159 (highest-AEP) and 0.66172 (lowest-risk). "
    "Their paired difference was +0.00013, with a 95% confidence interval "
    "[-0.000117, +0.000377], including zero. This is consistent with "
    "layout-independence of the mean encounter count at p = 0.",
    "",
    "However, P(≥1) was 0.56535 (highest-AEP) versus 0.22331 (lowest-risk) "
    "at p = 0. The difference is geometric: aligned turbine lanes concentrate "
    "encounters along fewer incoming paths, so a smaller fraction of birds "
    "encounter at least one turbine while those birds can encounter multiple "
    "turbines. Consequently, unchanged mean encounter count does not imply "
    "unchanged probability of any encounter.",
    "",
    "These are horizontal encounter-based findings for two endpoint layouts "
    "under a prescribed bird-flight and response scenario, not rotor-transit "
    "or collision-mortality predictions. The confidence intervals reflect Monte "
    "Carlo sampling variability at fixed model parameters, not uncertainty in "
    "bird behaviour or optimiser coverage.",
]
(OUT / "metric4a_results_text.txt").write_text("\n".join(res) + "\n", encoding="utf-8")

print("PASS | width normalisation factor:", f"{SCALE:.9f}")
for name in ("highest-AEP", "lowest-risk"):
    v = data[name]
    change = 100 * (v["adjusted_mean"] / v["optimisation_mean"] - 1)
    print("PASS |", name, "original Q:", f"{v['optimisation_mean']:.6f}",
          "raw independent Q:", f"{v['raw_mean']:.6f}",
          "adjusted independent Q:", f"{v['adjusted_mean']:.6f}",
          "change from optimisation:", f"{change:+.2f}%")
print("PASS | p=0 control: Q_4a layout-independent within paired CI")
print("PASS | original archives remain unchanged; figures and text saved:", OUT)
