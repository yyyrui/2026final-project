"""Check ordinary Naive semantics and the author-source compatibility boundary."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sktime_sales_adapter_tests", ROOT / "sktime-m4/scripts/sales_naive.py"
)
sales_naive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sales_naive)


class SktimeSalesTests(unittest.TestCase):
    def test_real_sktime_forecasts_last_value_for_each_series_and_horizon(self):
        # Means and the same weekday last week differ from the final observation.
        history = np.array([
            [900., 800., 4., 6., 8., 10., 12., 14., 2.5],
            [700., 600., 30., 25., 20., 15., 10., 5., 0.],
        ])
        completed = []
        forecast = sales_naive.forecast_sales(
            history, context_length=7, horizon=7,
            progress=lambda current, total: completed.append((current, total)),
        )
        np.testing.assert_array_equal(forecast, [[2.5] * 7, [0.] * 7])
        self.assertEqual(completed, [(2, 2)])
        np.testing.assert_array_equal(history[:, -1], [2.5, 0.])

    def test_rejects_nonfinite_context(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    sales_naive.forecast_sales(
                        np.array([[1., invalid, 3.]]), context_length=3, horizon=7
                    )

    def test_rejects_missing_history_context(self):
        with self.assertRaisesRegex(ValueError, "enough context"):
            sales_naive.forecast_sales(np.array([[1., 2.]]), context_length=3)

    def test_rejects_changed_or_missing_author_naive_definition(self):
        definitions = (
            '"Naive": NaiveForecaster(strategy="seasonal_last", sp=7)',
            '"Naive": NaiveForecaster(strategy="mean")',
            '"sNaive": NaiveForecaster(strategy="last")',
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "construct.py"
            for definition in definitions:
                with self.subTest(definition=definition):
                    source.write_text(
                        "def construct_M4_forecasters(sp, fh):\n"
                        f"    return {{{definition}}}\n",
                        encoding="utf-8",
                    )
                    with patch.object(sales_naive, "SOURCE_PATH", source):
                        with self.assertRaises(ValueError):
                            sales_naive.source_spec()


if __name__ == "__main__":
    unittest.main()
