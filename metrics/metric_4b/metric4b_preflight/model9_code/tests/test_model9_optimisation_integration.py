from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np

from config import Scenario, Turbine
from model9_optimisation_utils import (
    load_initial_population,
    minimum_pairwise_spacing,
)


ROOT = Path(__file__).resolve().parents[1]


if importlib.util.find_spec("floris") is None:
    floris_stub = ModuleType("floris")
    floris_stub.__file__ = str(ROOT / "_test_floris" / "__init__.py")
    floris_stub.FlorisModel = object  # type: ignore[attr-defined]
    sys.modules["floris"] = floris_stub

from floris_aep import FLORISAEPModel


class FakeFlorisModel:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.number_of_turbines = 0
        self.number_of_conditions = 0
        self.hours_per_year = 8760.0

    def set(self, **kwargs: object) -> None:
        self.number_of_turbines = len(np.asarray(kwargs["layout_x"]))
        self.number_of_conditions = len(np.asarray(kwargs["wind_speeds"]))

    def run(self) -> None:
        pass

    def get_turbine_powers(self) -> np.ndarray:
        return np.full(
            (self.number_of_conditions, self.number_of_turbines),
            1.0e6,
            dtype=float,
        )

    def get_farm_AEP(self, *, freq: np.ndarray, hours_per_year: float) -> float:
        self.hours_per_year = hours_per_year
        return float(1.0e6 * self.number_of_turbines * hours_per_year * np.sum(freq))


class ConfigurationTests(unittest.TestCase):
    def test_corrected_geometry_and_programmatic_wind_resource(self) -> None:
        scenario = Scenario()
        self.assertEqual(scenario.turbine.rotor_diameter_m, 242.24)
        self.assertEqual(scenario.turbine.hub_height_m, 150.0)
        self.assertEqual(scenario.wind.n_sectors, 12)
        np.testing.assert_array_equal(
            scenario.wind.speed_bins,
            np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
        )
        frequencies = scenario.wind.sector_frequencies(
            scenario.site.wind_from_deg
        )
        self.assertAlmostEqual(float(frequencies.sum()), 1.0, places=14)
        self.assertEqual(int(np.argmax(frequencies)), 7)  # 210-degree sector


class InitialPopulationTests(unittest.TestCase):
    def load(self, population_size: int) -> tuple[np.ndarray, list[str], list[int]]:
        scenario = Scenario()
        return load_initial_population(
            ROOT / "data" / "model8_initial_population.csv",
            population_size=population_size,
            seed=42,
            number_of_turbines=scenario.site.n_turbines,
            site_width_m=scenario.site.width_m,
            site_height_m=scenario.site.height_m,
            minimum_spacing_m=(
                scenario.site.min_spacing_D * scenario.turbine.rotor_diameter_m
            ),
        )

    def test_smoke_population_retains_all_four_exact_anchors(self) -> None:
        vectors, labels, rows = self.load(8)
        self.assertEqual(vectors.shape, (8, 24))
        self.assertEqual(rows[:4], [0, 1, 2, 3])
        self.assertEqual(
            labels[:4],
            ["exact_energy", "exact_aligned", "exact_staggered", "exact_wider_gap"],
        )

    def test_population_subset_is_deterministic_and_feasible(self) -> None:
        first, _, rows_first = self.load(24)
        second, _, rows_second = self.load(24)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(rows_first, rows_second)
        minimum = min(
            minimum_pairwise_spacing(vector.reshape(12, 2)) for vector in first
        )
        self.assertGreaterEqual(minimum + 1.0e-9, 4.0 * 242.24)

    def test_full_population_preserves_model8_row_order(self) -> None:
        vectors, _, rows = self.load(80)
        self.assertEqual(vectors.shape, (80, 24))
        self.assertEqual(rows, list(range(80)))


class FlorisCouplingTests(unittest.TestCase):
    @property
    def config_path(self) -> Path:
        return ROOT / "floris_config" / "gch_iea15mw.yaml"

    @property
    def turbine_path(self) -> Path:
        return ROOT / "floris_config" / "iea_15MW_REFERENCE.yaml"

    def construct(self, scenario: Scenario) -> FLORISAEPModel:
        with patch("floris_aep.FlorisModel", FakeFlorisModel):
            return FLORISAEPModel(
                scenario,
                self.config_path,
                turbine_reference_path=self.turbine_path,
                installed_turbine_path=self.turbine_path,
            )

    def test_wrapper_builds_exactly_84_normalised_conditions(self) -> None:
        model = self.construct(Scenario())
        self.assertEqual(model.number_of_conditions, 84)
        self.assertEqual(len(np.unique(model.wind_directions_deg)), 12)
        self.assertEqual(len(np.unique(model.wind_speeds_ms)), 7)
        self.assertAlmostEqual(float(model.condition_frequencies.sum()), 1.0, 14)

    def test_wrapper_rejects_obsolete_240_m_scenario(self) -> None:
        scenario = Scenario(turbine=Turbine(rotor_diameter_m=240.0))
        with self.assertRaisesRegex(ValueError, "rotor diameter"):
            self.construct(scenario)

    def test_constant_power_stub_integrates_expected_aep(self) -> None:
        model = self.construct(Scenario())
        aep = model.evaluate(np.array([[0.0, 0.0], [1000.0, 1000.0]]))
        self.assertAlmostEqual(aep, 17.52, places=12)


if __name__ == "__main__":
    unittest.main()
