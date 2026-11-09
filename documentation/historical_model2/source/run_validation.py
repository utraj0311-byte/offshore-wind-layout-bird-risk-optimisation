import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from config import Scenario
from wake import AEPModel
from hypothetical_risk import HypotheticalCollisionModel
from validation_problem import ValidationLayoutProblem


# -------------------------------------------------
# Basic layout checks
# -------------------------------------------------

def pairwise_distances(xy):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)

    distances = []

    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            distances.append(np.linalg.norm(xy[i] - xy[j]))

    return np.array(distances)


def min_spacing(xy):
    distances = pairwise_distances(xy)

    if len(distances) == 0:
        return np.nan

    return float(np.min(distances))


def count_spacing_violations(xy, min_spacing_m):
    distances = pairwise_distances(xy)
    return int(np.sum(distances < min_spacing_m - 1e-6))


def count_boundary_violations(xy, width, height):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)

    outside_x = np.logical_or(xy[:, 0] < 0.0, xy[:, 0] > width)
    outside_y = np.logical_or(xy[:, 1] < 0.0, xy[:, 1] > height)

    return int(np.sum(np.logical_or(outside_x, outside_y)))


def is_feasible(xy, scenario):
    width = scenario.site.width_m
    height = scenario.site.height_m
    min_spacing_m = scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m

    if count_boundary_violations(xy, width, height) > 0:
        return False

    if count_spacing_violations(xy, min_spacing_m) > 0:
        return False

    return True


# -------------------------------------------------
# Initial population
# -------------------------------------------------

def make_one_grid_layout(scenario, rng):
    """
    Creates one grid-like layout with small random jitter.
    """
    n_turbines = scenario.site.n_turbines
    width = scenario.site.width_m
    height = scenario.site.height_m
    min_spacing_m = scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m

    rows = int(np.ceil(np.sqrt(n_turbines)))
    cols = int(np.ceil(n_turbines / rows))

    margin = min_spacing_m / 2.0

    xs = np.linspace(margin, width - margin, cols)
    ys = np.linspace(margin, height - margin, rows)

    base_points = []

    for y in ys:
        for x in xs:
            base_points.append([x, y])

    base_points = np.array(base_points[:n_turbines], dtype=float)

    for _ in range(100):
        jitter = rng.normal(
            loc=0.0,
            scale=0.10 * min_spacing_m,
            size=base_points.shape,
        )

        xy = base_points + jitter
        xy[:, 0] = np.clip(xy[:, 0], 0.0, width)
        xy[:, 1] = np.clip(xy[:, 1], 0.0, height)

        if is_feasible(xy, scenario):
            return xy

    return base_points


def make_random_feasible_layout(scenario, rng, x_range=None, y_range=None, max_tries=5000):
    """
    Creates a random feasible layout by placing turbines one by one.

    x_range and y_range can restrict placement to a safe region.
    """
    n_turbines = scenario.site.n_turbines
    width = scenario.site.width_m
    height = scenario.site.height_m
    min_spacing_m = scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m

    if x_range is None:
        x_range = (0.0, width)

    if y_range is None:
        y_range = (0.0, height)

    points = []

    for _ in range(max_tries):
        if len(points) == n_turbines:
            xy = np.array(points, dtype=float)
            if is_feasible(xy, scenario):
                return xy

        candidate = np.array(
            [
                rng.uniform(x_range[0], x_range[1]),
                rng.uniform(y_range[0], y_range[1]),
            ]
        )

        if len(points) == 0:
            points.append(candidate)
        else:
            existing = np.array(points)
            distances = np.linalg.norm(existing - candidate, axis=1)

            if np.all(distances >= min_spacing_m):
                points.append(candidate)

    # If random placement fails, fall back to grid
    return make_one_grid_layout(scenario, rng)


def make_west_biased_layout(scenario, rng):
    """
    Creates a layout mostly in the west/safe half.
    Used for half-site risk validation.

    West side is x < width/2.
    We leave a small buffer from the exact boundary.
    """
    width = scenario.site.width_m
    height = scenario.site.height_m

    return make_random_feasible_layout(
        scenario,
        rng,
        x_range=(0.0, width / 2.0 - 50.0),
        y_range=(0.0, height),
    )


def make_corridor_avoiding_layout(scenario, rng):
    """
    Creates a layout avoiding the central horizontal corridor.
    Used for central corridor validation.
    """
    n_turbines = scenario.site.n_turbines
    width = scenario.site.width_m
    height = scenario.site.height_m
    min_spacing_m = scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m

    corridor_width_m = 1200.0
    y_low_max = height / 2.0 - corridor_width_m / 2.0
    y_high_min = height / 2.0 + corridor_width_m / 2.0

    points = []
    max_tries = 8000

    for _ in range(max_tries):
        if len(points) == n_turbines:
            xy = np.array(points, dtype=float)
            if is_feasible(xy, scenario):
                return xy

        # Choose lower or upper safe region
        if rng.random() < 0.5:
            y = rng.uniform(0.0, y_low_max)
        else:
            y = rng.uniform(y_high_min, height)

        x = rng.uniform(0.0, width)

        candidate = np.array([x, y])

        if len(points) == 0:
            points.append(candidate)
        else:
            existing = np.array(points)
            distances = np.linalg.norm(existing - candidate, axis=1)

            if np.all(distances >= min_spacing_m):
                points.append(candidate)

    return make_one_grid_layout(scenario, rng)


def make_hotspot_avoiding_layout(scenario, rng):
    """
    Creates a layout avoiding the centre hotspot.
    """
    n_turbines = scenario.site.n_turbines
    width = scenario.site.width_m
    height = scenario.site.height_m
    min_spacing_m = scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m

    cx = width / 2.0
    cy = height / 2.0
    hotspot_radius_m = 900.0

    points = []
    max_tries = 8000

    for _ in range(max_tries):
        if len(points) == n_turbines:
            xy = np.array(points, dtype=float)
            if is_feasible(xy, scenario):
                return xy

        candidate = np.array(
            [
                rng.uniform(0.0, width),
                rng.uniform(0.0, height),
            ]
        )

        dist_to_hotspot = np.linalg.norm(candidate - np.array([cx, cy]))

        if dist_to_hotspot <= hotspot_radius_m:
            continue

        if len(points) == 0:
            points.append(candidate)
        else:
            existing = np.array(points)
            distances = np.linalg.norm(existing - candidate, axis=1)

            if np.all(distances >= min_spacing_m):
                points.append(candidate)

    return make_one_grid_layout(scenario, rng)


def make_initial_population(scenario, n_samples, seed, model_name):
    """
    Creates a diverse initial population.

    This is important for validation because we want NSGA-II to see:
    - energy-like layouts,
    - random feasible layouts,
    - collision-good layouts for known artificial risk fields.
    """
    rng = np.random.default_rng(seed)
    n_turbines = scenario.site.n_turbines

    X = np.zeros((n_samples, 2 * n_turbines))

    for k in range(n_samples):
        r = rng.random()

        if model_name == "half_site_east_risk":
            if r < 0.40:
                xy = make_west_biased_layout(scenario, rng)
            elif r < 0.70:
                xy = make_random_feasible_layout(scenario, rng)
            else:
                xy = make_one_grid_layout(scenario, rng)

        elif model_name == "central_corridor":
            if r < 0.40:
                xy = make_corridor_avoiding_layout(scenario, rng)
            elif r < 0.70:
                xy = make_random_feasible_layout(scenario, rng)
            else:
                xy = make_one_grid_layout(scenario, rng)

        elif model_name == "point_hotspot":
            if r < 0.40:
                xy = make_hotspot_avoiding_layout(scenario, rng)
            elif r < 0.70:
                xy = make_random_feasible_layout(scenario, rng)
            else:
                xy = make_one_grid_layout(scenario, rng)

        else:
            if r < 0.50:
                xy = make_random_feasible_layout(scenario, rng)
            else:
                xy = make_one_grid_layout(scenario, rng)

        X[k, :] = xy.reshape(-1)

    return X


class GridSampling(Sampling):
    def __init__(self, scenario, seed, model_name):
        super().__init__()
        self.scenario = scenario
        self.seed = seed
        self.model_name = model_name

    def _do(self, problem, n_samples, **kwargs):
        return make_initial_population(
            self.scenario,
            n_samples,
            self.seed,
            self.model_name,
        )


# -------------------------------------------------
# Hypervolume callback
# -------------------------------------------------

class HVCallback(Callback):
    """
    Tracks hypervolume during optimisation.

    Since the objectives are normalised, the reference point can be fixed.
    """

    def __init__(self, ref_point=np.array([1.2, 1.2])):
        super().__init__()

        self.ref_point = ref_point
        self.hv_indicator = HV(ref_point=ref_point)
        self.data["hv"] = []

    def notify(self, algorithm):
        if algorithm.opt is None:
            self.data["hv"].append(np.nan)
            return

        F = algorithm.opt.get("F")

        if F is None or len(F) == 0:
            self.data["hv"].append(np.nan)
            return

        hv = self.hv_indicator(F)
        self.data["hv"].append(float(hv))


# -------------------------------------------------
# Results processing
# -------------------------------------------------

def raw_objectives_dataframe(X, F_norm, scenario, aep_model, risk_model):
    rows = []

    min_spacing_m = scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m

    for idx, xvec in enumerate(X):
        xy = xvec.reshape(scenario.site.n_turbines, 2)

        aep = aep_model.evaluate(xy)
        risk = risk_model.collisions(xy)
        diagnostics = risk_model.diagnostics(xy)

        rows.append(
            {
                "layout_id": idx,
                "aep_gwh": aep,
                "risk": risk,
                "f1_normalised_minimise": F_norm[idx, 0],
                "f2_normalised_risk": F_norm[idx, 1],
                "min_spacing_m": min_spacing(xy),
                "spacing_violations": count_spacing_violations(xy, min_spacing_m),
                "boundary_violations": count_boundary_violations(
                    xy,
                    scenario.site.width_m,
                    scenario.site.height_m,
                ),
                "high_risk_count": diagnostics["high_risk_count"],
                "high_risk_fraction": diagnostics["high_risk_fraction"],
            }
        )

    return pd.DataFrame(rows)


def find_anchor_indices(df):
    """
    Finds three important layouts:
    - energy-optimal
    - collision-optimal
    - knee
    """
    energy_idx = int(df["aep_gwh"].idxmax())
    collision_idx = int(df["risk"].idxmin())

    aep = df["aep_gwh"].to_numpy()
    risk = df["risk"].to_numpy()

    aep_range = np.max(aep) - np.min(aep)
    risk_range = np.max(risk) - np.min(risk)

    if aep_range < 1e-9:
        aep_n = np.ones_like(aep)
    else:
        aep_n = (aep - np.min(aep)) / aep_range

    if risk_range < 1e-9:
        risk_n = np.ones_like(risk)
    else:
        # Higher is better after this transformation
        risk_n = 1.0 - (risk - np.min(risk)) / risk_range

    # Ideal point = high AEP and low risk = (1, 1)
    distance_to_ideal = np.sqrt((1.0 - aep_n) ** 2 + (1.0 - risk_n) ** 2)
    knee_idx = int(np.argmin(distance_to_ideal))

    return {
        "energy_optimal": energy_idx,
        "collision_optimal": collision_idx,
        "knee": knee_idx,
    }


def save_anchor_layouts(X, df, anchors, scenario, output_dir):
    rows = []

    for name, idx in anchors.items():
        xy = X[idx].reshape(scenario.site.n_turbines, 2)

        layout_df = pd.DataFrame(
            {
                "turbine": np.arange(1, scenario.site.n_turbines + 1),
                "x_m": xy[:, 0],
                "y_m": xy[:, 1],
            }
        )

        layout_df.to_csv(output_dir / f"layout_{name}.csv", index=False)

        rows.append(
            {
                "anchor": name,
                "layout_id": idx,
                "aep_gwh": df.loc[idx, "aep_gwh"],
                "risk": df.loc[idx, "risk"],
                "min_spacing_m": df.loc[idx, "min_spacing_m"],
                "spacing_violations": df.loc[idx, "spacing_violations"],
                "boundary_violations": df.loc[idx, "boundary_violations"],
                "high_risk_count": df.loc[idx, "high_risk_count"],
                "high_risk_fraction": df.loc[idx, "high_risk_fraction"],
            }
        )

    anchor_df = pd.DataFrame(rows)
    anchor_df.to_csv(output_dir / "anchor_summary.csv", index=False)

    return anchor_df


# -------------------------------------------------
# Plotting
# -------------------------------------------------

def plot_pareto(df, anchors, output_dir):
    plt.figure(figsize=(7, 5))

    plt.scatter(df["aep_gwh"], df["risk"])

    for name, idx in anchors.items():
        plt.scatter(df.loc[idx, "aep_gwh"], df.loc[idx, "risk"], s=120)

        plt.annotate(
            name,
            (df.loc[idx, "aep_gwh"], df.loc[idx, "risk"]),
            textcoords="offset points",
            xytext=(8, 8),
        )

    plt.xlabel("AEP (GWh/year)")
    plt.ylabel("Artificial collision risk")
    plt.title("Pareto frontier")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / "pareto_frontier.png", dpi=200)
    plt.close()


def plot_hypervolume(hv_values, output_dir):
    plt.figure(figsize=(7, 4))

    plt.plot(np.arange(1, len(hv_values) + 1), hv_values)

    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume trajectory")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / "hypervolume_trajectory.png", dpi=200)
    plt.close()


def plot_anchor_layouts(X, anchors, scenario, risk_model, output_dir):
    for name, idx in anchors.items():
        xy = X[idx].reshape(scenario.site.n_turbines, 2)

        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        risk_model.draw_background(ax)

        plt.scatter(xy[:, 0], xy[:, 1], s=80)

        for i, (x, y) in enumerate(xy):
            plt.annotate(
                str(i + 1),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
            )

        plt.xlim(0, scenario.site.width_m)
        plt.ylim(0, scenario.site.height_m)

        plt.xlabel("x position (m)")
        plt.ylabel("y position (m)")
        plt.title(f"{name} layout")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_dir / f"layout_{name}.png", dpi=200)
        plt.close()


# -------------------------------------------------
# Main script
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="half_site_east_risk",
        choices=[
            "constant",
            "half_site_east_risk",
            "central_corridor",
            "point_hotspot",
        ],
        help="Artificial risk model to test.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--gen", type=int, default=100)
    parser.add_argument("--risk_mult", type=float, default=1.0)

    args = parser.parse_args()

    scenario = Scenario()

    aep_model = AEPModel(scenario)

    risk_model = HypotheticalCollisionModel(
    scenario,
    model_name=args.model,
    risk_high=1000.0,
    risk_low=0.0,
    corridor_width_m=3000.0,
    hotspot_radius_m=2000.0,
    risk_multiplier=args.risk_mult,
)

    # Objective normalisation.
    # AEP maximum is approximated by wake-free ideal AEP.
    ideal_aep = aep_model.ideal_aep(scenario.site.n_turbines)

    aep_max = ideal_aep
    aep_min = 0.5 * ideal_aep

    risk_min = 0.0
    risk_max = risk_model.risk_scale()

    problem = ValidationLayoutProblem(
        scenario=scenario,
        aep_model=aep_model,
        risk_model=risk_model,
        aep_min=aep_min,
        aep_max=aep_max,
        risk_min=risk_min,
        risk_max=risk_max,
    )

    sampling = GridSampling(scenario, seed=args.seed, model_name=args.model)

    algorithm = NSGA2(
        pop_size=args.pop,
        sampling=sampling,
        eliminate_duplicates=True,
    )

    callback = HVCallback(ref_point=np.array([1.2, 1.2]))

    print("")
    print("Running optimiser validation")
    print("-----------------------------")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Population size: {args.pop}")
    print(f"Generations: {args.gen}")
    print("")

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", args.gen),
        seed=args.seed,
        callback=callback,
        verbose=True,
    )

    if result.X is None:
        print("")
        print("No feasible Pareto solutions were found.")
        print("Try increasing population/generations or check constraints.")
        print("")
        return

    X = np.asarray(result.X)
    F = np.asarray(result.F)

    if X.ndim == 1:
        X = X.reshape(1, -1)
        F = F.reshape(1, -1)

    output_dir = (
        Path("results")
        / f"validation_{args.model}_mult{args.risk_mult}_seed{args.seed}_pop{args.pop}_gen{args.gen}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    df = raw_objectives_dataframe(
        X,
        F,
        scenario,
        aep_model,
        risk_model,
    )

    df.to_csv(output_dir / "pareto_points.csv", index=False)

    anchors = find_anchor_indices(df)

    anchor_df = save_anchor_layouts(
        X,
        df,
        anchors,
        scenario,
        output_dir,
    )

    hv_df = pd.DataFrame(
        {
            "generation": np.arange(1, len(callback.data["hv"]) + 1),
            "hypervolume": callback.data["hv"],
        }
    )

    hv_df.to_csv(output_dir / "hypervolume.csv", index=False)

    plot_pareto(df, anchors, output_dir)
    plot_hypervolume(callback.data["hv"], output_dir)
    plot_anchor_layouts(X, anchors, scenario, risk_model, output_dir)

    print("")
    print("Validation run complete.")
    print(f"Saved outputs to: {output_dir}")
    print("")

    print("Anchor layouts:")
    print(anchor_df.to_string(index=False))
    print("")

    print("Quick checks:")
    print(f"Number of Pareto layouts saved: {len(df)}")
    print(f"Minimum spacing in Pareto set: {df['min_spacing_m'].min():.2f} m")
    print(f"Total spacing violations in Pareto set: {df['spacing_violations'].sum()}")
    print(f"Total boundary violations in Pareto set: {df['boundary_violations'].sum()}")
    print("")


if __name__ == "__main__":
    main()