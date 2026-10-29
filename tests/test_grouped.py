import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from comparison.full import run_grouped, score_sums, scores, verify_signature, run_lock


class GroupedTests(unittest.TestCase):
    def test_aggregate_uses_observations_not_group_means(self):
        parts = [score_sums([1, 2], [2, 4]), score_sums([10], [4])]
        total = {k: sum(x[k] for x in parts) for k in parts[0]}
        result = scores(total)
        self.assertEqual(result['MAE'], 3)
        self.assertAlmostEqual(result['RMSE'], np.sqrt(41/3))
        self.assertAlmostEqual(result['WAPE'], 9/13)

    def test_signature_rejects_changed_data_or_settings(self):
        with self.assertRaisesRegex(ValueError, 'Resume rejected'):
            verify_signature({'epochs': 2}, {'epochs': 3})

    def test_lock_blocks_second_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(Path(tmp)):
                with self.assertRaisesRegex(RuntimeError, 'Another process'):
                    with run_lock(Path(tmp)):
                        pass

    def test_interrupt_resume_and_complete_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(city_id=0, store_id=0, product_id=s, dt=d, sale_amount=float(s+t))
                    for s in range(3) for t,d in enumerate(pd.date_range('2024-01-01',periods=28))]
            frame = pd.DataFrame(rows)
            train, evaluation = root/'train.parquet', root/'eval.parquet'
            frame.groupby('product_id').head(21).to_parquet(train)
            frame.groupby('product_id').tail(7).to_parquet(evaluation)
            args = Namespace(threads=1,output_dir=root/'output', resume=False, train_path=train,
                             eval_path=evaluation, all_series=True,num_series=4,seed=42,horizon=7,
                             context_length=2,group_size=2,data_only=False)
            calls = []
            def tactis(values, args, folder):
                calls.append(len(values))
                return np.repeat(values[:, -1:, None], 7, axis=1), []
            def chronos(values, args, pipeline=None):
                return np.repeat(values[:, -1:, None], 7, axis=1)
            with patch('comparison.full.train_tactis', side_effect=tactis), \
                 patch('comparison.full.load_chronos', return_value=object()), \
                 patch('comparison.full.predict_chronos', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    run_grouped(args)
            self.assertFalse((args.output_dir/'SUCCESS').exists())
            self.assertFalse((args.output_dir/'metrics.csv').exists())
            args.resume = True
            with patch('comparison.full.train_tactis', side_effect=tactis), \
                 patch('comparison.full.load_chronos', return_value=object()), \
                 patch('comparison.full.predict_chronos', side_effect=chronos):
                run_grouped(args)
            self.assertEqual(calls, [2, 1])  # finished model in first group was not retrained
            self.assertTrue((args.output_dir/'SUCCESS').exists())
            result = pd.read_csv(args.output_dir/'tactis2_predictions.csv')
            self.assertEqual(len(result), 21)
            self.assertEqual(len(result.drop_duplicates(['city_id','store_id','product_id','dt'])),21)
            score = pd.read_csv(args.output_dir/'metrics.csv')
            self.assertTrue(score.observations.eq(21).all())
            self.assertTrue(score.MAE.eq(4).all())
            args.group_size = 3
            with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                run_grouped(args)


if __name__ == '__main__':
    unittest.main()
