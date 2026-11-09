from __future__ import annotations

import unittest

import pandas as pd

from run_model9_validation import (
    estimator_precision_comparison,
    paired_layout_differences,
    validation_diagnostic,
)


class ValidationUncertaintyTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        values = {
            "a": ([0.20, 0.21, 0.19], [0.010, 0.011, 0.009]),
            "b": ([0.30, 0.31, 0.29], [0.020, 0.021, 0.019]),
            "c": ([0.40, 0.41, 0.39], [0.030, 0.031, 0.029]),
        }
        for layout, (encounters, transits) in values.items():
            for batch_index, (encounter, transit) in enumerate(
                zip(encounters, transits)
            ):
                rows.append(
                    {
                        "layout": layout,
                        "batch_index": batch_index,
                        "E_enc": encounter,
                        "E_RT": transit,
                    }
                )
        self.batch_results = pd.DataFrame(rows)
        self.summary = (
            self.batch_results.groupby("layout", as_index=False)[["E_enc", "E_RT"]]
            .mean()
            .assign(
                RT_to_encounter_ratio=lambda x: x["E_RT"] / x["E_enc"],
                N_RT_farm=lambda x: (x["E_RT"] * 3000).round().astype(int),
            )
        )

    def test_paired_differences_use_matching_batch_indices(self) -> None:
        paired = paired_layout_differences(self.batch_results)
        row = paired.loc[
            (paired["layout_first"] == "a")
            & (paired["layout_second"] == "b")
            & (paired["metric"] == "E_RT")
        ].iloc[0]
        self.assertEqual(row["batch_count"], 3)
        self.assertAlmostEqual(row["mean_first_minus_second"], -0.01)
        self.assertTrue(bool(row["ci95_excludes_zero"]))

    def test_diagnostic_reports_resolved_synthetic_ranking(self) -> None:
        paired = paired_layout_differences(self.batch_results)
        diagnostic = validation_diagnostic(
            self.summary,
            self.batch_results,
            paired,
        )
        self.assertTrue(
            diagnostic["rotor_transit_ranking_identical_across_batches"]
        )
        self.assertTrue(
            diagnostic["all_pairwise_E_RT_95pct_intervals_exclude_zero"]
        )

    def test_precision_comparison_quantifies_variance_reduction(self) -> None:
        batch_results = pd.DataFrame(
            {
                "layout": ["a"] * 4,
                "batch_index": [0, 1, 2, 3],
                "E_RT": [0.01, 0.03, 0.01, 0.03],
                "E_RT_CE": [0.019, 0.021, 0.019, 0.021],
            }
        )
        comparison = estimator_precision_comparison(batch_results).iloc[0]
        self.assertAlmostEqual(comparison["mean_sampled_minus_conditional"], 0.0)
        self.assertAlmostEqual(comparison["sd_reduction_factor"], 10.0)
        self.assertAlmostEqual(comparison["variance_reduction_factor"], 100.0)
        self.assertFalse(bool(comparison["difference_ci95_excludes_zero"]))

    def test_diagnostic_keeps_sampled_and_conditional_ratios_distinct(self) -> None:
        batch_results = self.batch_results.copy()
        batch_results["E_RT_CE"] = batch_results["E_RT"] * 0.5
        summary = (
            batch_results.groupby("layout", as_index=False)[
                ["E_enc", "E_RT", "E_RT_CE"]
            ]
            .mean()
            .assign(
                RT_to_encounter_ratio=lambda x: x["E_RT"] / x["E_enc"],
                RT_CE_to_encounter_ratio=lambda x: (
                    x["E_RT_CE"] / x["E_enc"]
                ),
                N_RT_farm=lambda x: (x["E_RT"] * 3000).round().astype(int),
                N_RT_CE_farm=lambda x: x["E_RT_CE"] * 3000,
            )
        )
        paired = paired_layout_differences(batch_results)
        precision = estimator_precision_comparison(batch_results)
        diagnostic = validation_diagnostic(
            summary,
            batch_results,
            paired,
            precision,
        )

        self.assertEqual(diagnostic["optimisation_estimator"], "E_RT_CE")
        self.assertAlmostEqual(
            diagnostic["minimum_RT_CE_to_encounter_ratio"],
            0.5 * diagnostic["minimum_RT_to_encounter_ratio"],
        )


if __name__ == "__main__":
    unittest.main()
