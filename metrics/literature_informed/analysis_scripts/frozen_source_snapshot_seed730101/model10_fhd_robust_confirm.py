"""Independent 10 x 50,000 paired confirmation for frozen Model 10 layouts.

Run only after the separate optimiser has completed and selected layouts from
training data. This stage never changes or reselects candidate coordinates.
It checkpoints a full batch after all selected layouts have been evaluated;
rerunning after an interruption safely resumes at the next complete batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

import model10_fhd_robust_seeded_50k as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=experiment.ROOT / "model10_fhd_robust_seeded_50k")
    args = parser.parse_args()
    training = args.output_dir / "optimisation"
    settings = json.loads((training / "run_settings.json").read_text(encoding="utf-8"))
    if settings["status"] != "completed":
        raise RuntimeError("Training optimisation has not completed")
    if settings["source_hashes"] != experiment.source_hashes():
        raise RuntimeError("Verified model inputs or exact scripts changed after selection")
    chosen_npy = training / "selected_for_confirmation_xy_m.npy"
    chosen_csv = training / "selected_for_confirmation.csv"
    if (experiment.digest(chosen_npy) != settings["selected_coordinates_sha256"]
            or experiment.digest(chosen_csv) != settings["selected_table_sha256"]):
        raise RuntimeError("Frozen selection has been altered")
    choices = list(csv.DictReader(chosen_csv.open(newline="", encoding="utf-8")))
    positions = np.load(chosen_npy, allow_pickle=False)
    if positions.shape != (len(choices), 12, 2) or len(choices) > 3:
        raise ValueError("Invalid frozen candidate selection")
    if not choices:
        raise RuntimeError("No nonreference training candidate was frozen")
    if (settings["confirmation_batches"] != experiment.CONFIRM_BATCHES
            or settings["confirmation_birds_per_batch"] != experiment.CONFIRM_BIRDS
            or settings["working_minimum_meaningful_improvement_fraction"]
            != experiment.MINIMUM_IMPROVEMENT):
        raise RuntimeError("Predeclared confirmation design has changed")
    trajectory_seeds = settings["confirmation_trajectory_seeds"]
    height_seeds = settings["confirmation_height_seeds"]
    assert trajectory_seeds == [experiment.CONFIRM_TRAJECTORY_SEED + i
                                for i in range(experiment.CONFIRM_BATCHES)]
    assert height_seeds == [experiment.CONFIRM_HEIGHT_SEED + i
                            for i in range(experiment.CONFIRM_BATCHES)]
    assert set(trajectory_seeds + height_seeds).isdisjoint(
        settings["training_trajectory_seeds"] + settings["training_height_seeds"])
    components = experiment.model_components()
    reference = components[8]
    layouts = {"reference": reference}
    for record, xy in zip(choices, positions):
        if not np.array_equal(experiment.row_xy(record), xy):
            raise RuntimeError("Saved candidate table does not match frozen coordinates")
        experiment.check_layout(xy, 968.96)
        layouts[record["label"]] = xy
    if len(layouts) != len(choices) + 1:
        raise ValueError("Duplicate candidate labels")
    stage = args.output_dir / "confirmation"
    plan_path = stage / "plan.json"
    rows_path = stage / "fresh_batches.csv"
    plan = {
        "scenario": "final northern-gannet FHD", "status": "planned",
        "paired_bird_trajectories_within_batch": True,
        "batch_count": experiment.CONFIRM_BATCHES,
        "birds_per_batch": experiment.CONFIRM_BIRDS,
        "trajectory_seeds": trajectory_seeds, "height_seeds": height_seeds,
        "candidate_labels": list(layouts),
        "training_settings_sha256_at_confirmation_start": experiment.digest(training / "run_settings.json"),
        "selected_coordinates_sha256": settings["selected_coordinates_sha256"],
        "selected_table_sha256": settings["selected_table_sha256"],
        "source_hashes": settings["source_hashes"],
        "paired_CI": "two-sided Student-t across ten independent batch risk differences; df=9",
        "working_minimum_meaningful_improvement_fraction": experiment.MINIMUM_IMPROVEMENT,
    }
    if stage.exists():
        if not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("Existing confirmation stage has a different frozen plan")
        if (stage / "report.json").exists():
            raise FileExistsError("Confirmation report is already complete")
        existing = list(csv.DictReader(rows_path.open(newline="", encoding="utf-8"))) if rows_path.exists() else []
    else:
        stage.mkdir(parents=True)
        experiment.write_json(plan_path, plan)
        existing = []
    labels = list(layouts)
    for i in range(experiment.CONFIRM_BATCHES):
        completed = [r for r in existing if int(r["batch_index"]) == i]
        if completed:
            if len(completed) != len(labels) or [r["layout"] for r in completed] != labels:
                raise RuntimeError("Partial or reordered confirmation batch in checkpoint")
            if any(int(r["trajectory_seed"]) != trajectory_seeds[i]
                   or int(r["flight_height_seed"]) != height_seeds[i] for r in completed):
                raise RuntimeError("Confirmation checkpoint has the wrong bird seeds")
            continue
        birds = experiment.make_birds(components, experiment.CONFIRM_BIRDS,
                                      1, trajectory_seeds[i], height_seeds[i])
        problem = experiment.create_problem(
            components, birds, settings["reference_training_risk_per_entrant"])
        added = []
        for label, xy in layouts.items():
            raw = problem.evaluate_raw_layout(xy)
            record = raw["batch_rows"][0]
            assert int(record["N_started"]) == experiment.CONFIRM_BIRDS
            added.append({
                "batch_index": i, "layout": label,
                "trajectory_seed": trajectory_seeds[i],
                "flight_height_seed": height_seeds[i],
                "N_started": record["N_started"],
                "AEP_GWh": raw["aep_gwh"],
                "risk_per_entrant": record["E_potential_strike"],
                "conditional_rotor_transits_per_entrant": record["E_RT_CE"],
            })
        existing.extend(added)
        experiment.write_csv(rows_path, existing)
        print(f"Completed independent confirmation batch {i + 1}/10 "
              f"(seed {trajectory_seeds[i]})", flush=True)

    ref_batches = [float(r["risk_per_entrant"]) for r in existing
                   if r["layout"] == "reference"]
    if len(ref_batches) != experiment.CONFIRM_BATCHES:
        raise RuntimeError("Missing reference batch")
    fresh_ref = float(np.mean(ref_batches))
    if fresh_ref <= 0:
        raise RuntimeError("Fresh reference risk is zero; relative difference undefined")
    comparison_rows = []
    paired_rows = []
    ref_aep = float(settings["reference_aep_gwh"])
    for selected in choices:
        label = selected["label"]
        candidate = [r for r in existing if r["layout"] == label]
        if len(candidate) != experiment.CONFIRM_BATCHES:
            raise RuntimeError(f"Incomplete confirmation for {label}")
        risk_batches = np.asarray([float(r["risk_per_entrant"]) for r in candidate])
        difference = risk_batches - np.asarray(ref_batches)
        mean = float(np.mean(difference))
        candidate_mean = float(np.mean(risk_batches))
        if not math.isclose(candidate_mean - fresh_ref, mean, rel_tol=0, abs_tol=1e-16):
            raise AssertionError("Paired difference does not match candidate-reference mean")
        stderr = float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
        halfwidth = float(student_t.ppf(.975, len(difference) - 1) * stderr)
        low, high = mean - halfwidth, mean + halfwidth
        for i, value in enumerate(difference):
            paired_rows.append({
                "layout": label, "batch_index": i,
                "trajectory_seed": trajectory_seeds[i],
                "candidate_risk": risk_batches[i], "reference_risk": ref_batches[i],
                "candidate_minus_reference_risk": float(value),
            })
        aep = float(selected["aep_gwh"])
        if not all(math.isclose(float(r["AEP_GWh"]), aep, abs_tol=1e-7)
                   for r in candidate):
            raise RuntimeError(f"Deterministic AEP changed for {label}")
        if low <= 0 <= high:
            outcome = "inconclusive_risk_difference"
        elif candidate_mean <= (1 - experiment.MINIMUM_IMPROVEMENT) * fresh_ref and high < 0:
            outcome = "confirmed_at_predeclared_2pct_threshold"
        else:
            outcome = "not_confirmed_at_2pct_threshold"
        comparison_rows.append({
            "layout": label, "fresh_risk_per_entrant": candidate_mean,
            "fresh_reference_risk_per_entrant": fresh_ref,
            "training_risk_per_entrant": float(selected["risk_per_entrant"]),
            "paired_mean_candidate_minus_reference": mean,
            "paired_95pct_t_lower": low, "paired_95pct_t_upper": high,
            "relative_candidate_minus_reference_pct": 100 * mean / fresh_ref,
            "AEP_GWh": aep,
            "AEP_minus_reference_GWh": aep - ref_aep,
            "meets_2pct_mean_reduction": candidate_mean <= .98 * fresh_ref,
            "paired_95pct_interval_entirely_below_zero": high < 0,
            "outcome": outcome,
        })
    experiment.write_csv(stage / "paired_differences.csv", paired_rows)
    experiment.write_csv(stage / "fresh_comparisons.csv", comparison_rows)
    experiment.write_json(stage / "report.json", {
        "status": "completed", "fresh_reference_risk_per_entrant": fresh_ref,
        "fresh_reference_AEP_GWh": ref_aep,
        "pairs": comparison_rows,
        "interpretation": "Expected potential strikes per entrant; independent horizontal "
            "trajectory batches with gannet height probabilities integrated analytically; "
            "paired intervals are per-comparison 95% intervals, not joint familywise intervals. "
            "No statement of convergence, annual fatalities or behaviour in real birds.",
        "plan": plan,
    })
    print("Frozen-layout independent paired confirmation complete", flush=True)


if __name__ == "__main__":
    main()
