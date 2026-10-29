import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from comparison.data import load_sales
from run_comparison import metrics


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rows = [dict(city_id=0, store_id=s, product_id=5, dt=date, sale_amount=s + day)
                for s in range(3) for day, date in enumerate(pd.date_range('2024-01-01', periods=14))]
        self.train = pd.DataFrame(rows)
        self.evaluation = self.train.groupby('store_id').tail(7).copy()
        self.train = self.train.groupby('store_id').head(7).copy()
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.train.to_parquet(self.root / 'train.parquet')
        self.evaluation.to_parquet(self.root / 'eval.parquet')

    def load(self):
        return load_sales(self.root / 'train.parquet', self.root / 'eval.parquet', 2)

    def test_alignment_is_independent_of_row_order(self):
        first = self.load()
        self.evaluation = self.evaluation.sample(frac=1, random_state=5)
        self.train = self.train.sample(frac=1, random_state=6)
        self.write()
        second = self.load()
        np.testing.assert_array_equal(first.train, second.train)
        np.testing.assert_array_equal(first.actual, second.actual)
        np.testing.assert_array_equal(first.actual[:, 0], first.train[:, -1] + 1)

    def test_reject_duplicate(self):
        self.train = pd.concat([self.train, self.train.iloc[:1]])
        self.write()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load()

    def test_reject_missing_day(self):
        self.train = self.train.iloc[1:]
        self.write()
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.load()

    def test_reject_overlap(self):
        self.evaluation['dt'] -= pd.Timedelta(days=1)
        self.write()
        with self.assertRaisesRegex(ValueError, 'next 7 days'):
            self.load()

    def test_reject_key_mismatch(self):
        self.evaluation.loc[self.evaluation.store_id == 0, 'store_id'] = 9
        self.write()
        with self.assertRaisesRegex(ValueError, 'keys differ'):
            self.load()

    def test_metrics_and_zero_denominator(self):
        score = metrics(np.array([[1., 3.]]), np.array([[[2., 2.], [1., 1.]]]))
        self.assertEqual(score['MAE'], 1.5)
        self.assertAlmostEqual(score['RMSE'], np.sqrt(2.5))
        self.assertEqual(score['WAPE'], .75)
        self.assertIsNone(metrics(np.zeros((1, 2)), np.ones((1, 2, 3)))['WAPE'])
        with self.assertRaises(ValueError):
            metrics(np.ones((1, 2)), np.full((1, 2, 3), np.nan))


if __name__ == '__main__':
    unittest.main()
