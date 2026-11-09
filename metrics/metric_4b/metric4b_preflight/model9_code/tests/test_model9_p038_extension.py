from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model9_p038_extension import (
    combine_p032_p038_batches,
    extension_decision,
    load_prior_confirmation_archive,
    paired_difference_frame,
    validate_prior_confirmation,
)


MODULE_DIR = Path(__file__).resolve().parents[1]


def paired_frame(values: dict[str, list[float]]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    batch_count = len(next(iter(values.values())))
    for candidate_id, metrics in values.items():
        for batch_index, metric in enumerate(metrics):
            started = 1000
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "batch_index": batch_index,
                    "trajectory_seed": 300 + batch_index,
                    "flight_height_seed": 400 + batch_index,
                    "N_started": started,
                    "N_completed": started,
                    "N_failed": 0,
                    "N_enc_farm": 100,
                    "E_enc": 0.1,
                    "P_ge_1": 0.08,
                    "N_RT_farm": int(round(metric * started)),
                    "E_RT": metric,
                    "P_RT": metric * 0.8,
                    "N_RT_CE_farm": metric * started,
                    "E_RT_CE": metric,
                    "P_RT_CE": metric * 0.8,
                    "P_height_band": 0.05,
                    "mean_route_extension_m": 75.0,
                }
            )
    return pd.DataFrame(rows)


class PriorArchiveTests(unittest.TestCase):
    def test_bundled_completed_confirmation_is_exact_and_reusable(self) -> None:
        manifest = json.loads(
            (MODULE_DIR / "data" / "model9_p038_extension_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        prior = manifest["prior_confirmation"]
        archive = MODULE_DIR / str(prior["archive_path"])
        batches, settings, decision = load_prior_confirmation_archive(
            archive,
            expected_sha256=str(prior["archive_sha256"]),
        )
        p032 = validate_prior_confirmation(
            batches,
            settings,
            decision,
            manifest,
        )
        self.assertEqual(len(p032), 15)
        self.assertEqual(int(p032["N_started"].sum()), 1_500_000)
        self.assertEqual(int(p032["N_failed"].sum()), 0)


class ExtensionDecisionTests(unittest.TestCase):
    @property
    def prior_decision(self) -> dict[str, object]:
        return {
            "primary_status": "risk_difference_inconclusive",
            "primary_tradeoff_gate_passed": False,
        }

    def test_clear_p038_reduction_confirms_wide_tradeoff(self) -> None:
        p032 = paired_frame(
            {"pareto_032": [0.0200, 0.0201, 0.0199, 0.0202, 0.0198]}
        )
        p038 = paired_frame(
            {"pareto_038": [0.0100, 0.0101, 0.0099, 0.0102, 0.0098]}
        )
        combined = combine_p032_p038_batches(p032, p038)
        decision = extension_decision(
            combined,
            aep_gwh={"pareto_032": 100.0, "pareto_038": 97.5},
            prior_decision=self.prior_decision,
        )
        self.assertTrue(decision["extension_tradeoff_gate_passed"])
        self.assertEqual(
            decision["overall_model9_status"],
            "wide_tradeoff_confirmed_but_high_energy_p037_not_confirmed",
        )

    def test_sign_changing_p038_difference_remains_inconclusive(self) -> None:
        p032 = paired_frame({"pareto_032": [0.020] * 5})
        p038 = paired_frame(
            {"pareto_038": [0.019, 0.021, 0.019, 0.021, 0.020]}
        )
        combined = combine_p032_p038_batches(p032, p038)
        decision = extension_decision(
            combined,
            aep_gwh={"pareto_032": 100.0, "pareto_038": 97.5},
            prior_decision=self.prior_decision,
        )
        self.assertFalse(decision["extension_tradeoff_gate_passed"])
        self.assertEqual(
            decision["extension_status"],
            "pareto_038_difference_inconclusive",
        )

    def test_paired_output_uses_p038_minus_p032(self) -> None:
        p032 = paired_frame({"pareto_032": [0.020, 0.021, 0.022]})
        p038 = paired_frame({"pareto_038": [0.010, 0.011, 0.012]})
        combined = combine_p032_p038_batches(p032, p038)
        paired = paired_difference_frame(combined)
        self.assertEqual(len(paired), 3)
        self.assertTrue(
            np.allclose(paired["pareto_038_minus_pareto_032"], -0.01)
        )


if __name__ == "__main__":
    unittest.main()
