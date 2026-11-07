"""Numerical equivalence and protocol checks for the shared three-model runner."""
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from comparison import runner as experiment
from comparison.data import load_sales
from unittest.mock import patch


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        experiment.seed_all(1102)

    def test_self_attention_output_and_gradients(self):
        from types import MethodType
        original = torch.nn.TransformerEncoderLayer(16, 2, 24, dropout=0.0).double()
        bounded = deepcopy(original)
        bounded.query_block = 3
        bounded._sa_block = MethodType(experiment.bounded_self_attention, bounded)
        x = torch.randn(11, 2, 16, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        expected, actual = original(x), bounded(y)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
        weights = torch.randn_like(expected)
        (expected * weights).sum().backward()
        (actual * weights).sum().backward()
        torch.testing.assert_close(y.grad, x.grad, atol=1e-9, rtol=1e-8)
        for (name, a), (_, b) in zip(original.named_parameters(), bounded.named_parameters()):
            torch.testing.assert_close(b.grad, a.grad, atol=1e-9, rtol=1e-8, msg=name)

    def test_copula_loss_and_all_gradients(self):
        from tactis.model.decoder import AttentionalCopula
        original = AttentionalCopula(input_dim=8, attention_heads=2, attention_layers=2,
            attention_dim=4, mlp_layers=2, mlp_dim=12, resolution=10, dropout=0.0).double()
        bounded = deepcopy(original)
        bounded.query_block = 3
        values = [torch.randn(2, 12, 8, dtype=torch.float64), torch.rand(2, 12, dtype=torch.float64),
                  torch.randn(2, 8, 8, dtype=torch.float64), torch.rand(2, 8, dtype=torch.float64)]
        inputs = [x.clone().requires_grad_() for x in values]
        copies = [x.clone().requires_grad_() for x in values]
        expected = original.loss(*inputs, num_series=4, num_timesteps=2)
        actual = experiment.bounded_copula_loss(bounded, *copies, num_series=4, num_timesteps=2)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
        expected.sum().backward()
        actual.sum().backward()
        for a, b in zip(inputs, copies):
            torch.testing.assert_close(b.grad, a.grad, atol=1e-9, rtol=1e-8)
        for (name, a), (_, b) in zip(original.named_parameters(), bounded.named_parameters()):
            torch.testing.assert_close(b.grad, a.grad, atol=1e-9, rtol=1e-8, msg=name)

    @staticmethod
    def fixture(folder):
        rows = []
        for series in range(5):
            for day, date in enumerate(pd.date_range("2024-01-01", periods=18)):
                rows.append(dict(city_id=1, store_id=1, product_id=series, dt=date,
                                 sale_amount=float(1 + series + day % 4)))
        frame = pd.DataFrame(rows)
        frame[frame.dt < "2024-01-17"].sample(frac=1, random_state=7).to_parquet(folder / "train.parquet", index=False)
        frame[frame.dt >= "2024-01-17"].to_parquet(folder / "eval.parquet", index=False)

    def test_sampling_matches_existing_and_excludes_eval_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            paths = folder / "train.parquet", folder / "eval.parquet"
            panel = load_sales(*paths, num_series=3, seed=1102, horizon=2)
            expected_ids = np.sort(np.random.default_rng(1102).choice(5, 3, replace=False))
            np.testing.assert_array_equal(panel.keys.product_id.to_numpy(), expected_ids)
            expected_train = np.array([[1 + series + day % 4 for day in range(16)] for series in expected_ids])
            np.testing.assert_array_equal(panel.train, expected_train)
            evaluation = pd.read_parquet(paths[1])
            evaluation.sale_amount += 100000
            evaluation.to_parquet(paths[1], index=False)
            changed = load_sales(*paths, num_series=3, seed=1102, horizon=2)
            pd.testing.assert_frame_equal(panel.keys, changed.keys)
            np.testing.assert_array_equal(panel.train, changed.train)
            # Missing observations must fail rather than quietly fill or drop IDs.
            evaluation.iloc[1:].to_parquet(paths[1], index=False)
            with self.assertRaises(ValueError):
                load_sales(*paths, num_series=5, seed=1102, horizon=2)

    def test_joint_training_and_csv_alignment(self):
        from sales_naive import forecast_sales
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            panel = load_sales(folder / "train.parquet", folder / "eval.parquet", 3, 1102, 2)
            args = experiment.arguments(["--context", "4", "--horizon", "2",
                "--epochs", "1", "--samples", "2", "--attention-block", "3", "--device", "cpu"])
            args.work_dir = folder
            forecast = experiment.train_tactis(panel, args)
            naive = forecast_sales(panel.train, context_length=4, horizon=2)
            frame = experiment.prediction_frame(panel, {"tactis2": forecast, "sktime_m4_naive": naive})
            self.assertEqual(len(frame), 6)
            self.assertFalse(frame.duplicated(experiment.KEYS + ["dt"]).any())
            np.testing.assert_array_equal(frame.actual.to_numpy(), panel.actual.reshape(-1))
            self.assertTrue(np.isfinite(forecast).all())
            np.testing.assert_array_equal(naive, np.repeat(panel.train[:, -1:], 2, axis=1))
            state = torch.load(folder / "tactis2.pt", weights_only=False)
            self.assertIsNone(state["config"]["bagging_size"])
            self.assertEqual(state["config"]["num_series"], 3)
            self.assertEqual(experiment.score(np.zeros((1, 2)), np.ones((1, 2)))["WAPE"], None)

    def test_all_series_selection_is_seed_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            paths = folder / "train.parquet", folder / "eval.parquet"
            first = load_sales(*paths, num_series=None, seed=1, horizon=2)
            second = load_sales(*paths, num_series=None, seed=99, horizon=2)
            self.assertEqual(len(first.keys), 5)
            self.assertEqual(first.total_series, 5)
            np.testing.assert_array_equal(first.keys.product_id, np.arange(5))
            pd.testing.assert_frame_equal(first.keys, second.keys)
            np.testing.assert_array_equal(first.train, second.train)

    def test_entrypoint_selection_and_output_paths(self):
        sample = experiment.arguments([])
        full = experiment.arguments([], mode="all")
        self.assertEqual(sample.num_series, 5000)
        self.assertIsNone(full.num_series)
        self.assertNotEqual(sample.work_dir, full.work_dir)
        self.assertEqual(sample.train_path, experiment.ROOT / "dataset/train.parquet")
        self.assertEqual(full.output, full.work_dir / "predictions.csv")
        custom = experiment.arguments(["--work-dir", "outputs/custom"], mode="all")
        self.assertEqual(custom.output, experiment.ROOT / "outputs/custom/predictions.csv")

    def test_lfs_pointer_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            pointer = Path(temporary) / "train.parquet"
            pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                load_sales(pointer, pointer, num_series=None)

    def test_full_entrypoint_three_models_from_another_directory(self):
        model = experiment.ROOT / "chronos/chronos-forecasting/test/dummy-chronos2-model"
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            artifacts = folder / "artifacts"
            command = [sys.executable, str(experiment.ROOT / "test_all.py"),
                "--train-path", str(folder / "train.parquet"),
                "--eval-path", str(folder / "eval.parquet"), "--work-dir", str(artifacts),
                "--chronos2-model", str(model), "--offline", "--context", "4", "--horizon", "2",
                "--epochs", "1", "--samples", "2", "--attention-block", "3", "--device", "cpu"]
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((artifacts / "manifest.json").read_text())
            self.assertEqual(manifest["selected_series"], 5)
            self.assertEqual(manifest["total_series"], 5)
            self.assertIsNone(manifest["arguments"]["num_series"])
            self.assertIsNone(manifest["tactis_config"]["bagging_size"])
            frame = pd.read_csv(artifacts / "predictions.csv")
            self.assertEqual(len(frame), 10)
            self.assertFalse(frame.isna().any().any())
            table = pd.read_csv(artifacts / "metrics.csv")
            self.assertEqual(set(table.model), {"tactis2", "chronos2", "sktime_m4_naive"})
            self.assertTrue(table.observations.eq(10).all())
            # Existing experiments must not be overwritten.
            repeat = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("Existing results are protected", repeat.stderr)

    @patch.object(experiment, "SAMPLE_SERIES", 3)
    def test_three_model_cli_with_local_chronos2_fixture(self):
        import json
        # Upstream's bundled dummy Chronos-2 weights exercise real loading and
        # inference offline. They are NOT the pretrained benchmark model.
        model = experiment.PROJECT / "chronos/chronos-forecasting/test/dummy-chronos2-model"
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.fixture(folder)
            output, artifacts = folder / "forecast.csv", folder / "artifacts"
            experiment.main(["--train-path", str(folder / "train.parquet"),
                "--eval-path", str(folder / "eval.parquet"), "--output", str(output),
                "--work-dir", str(artifacts), "--chronos2-model", str(model), "--offline",
                "--context", "4", "--horizon", "2", "--epochs", "1",
                "--samples", "2", "--chronos2-batch-size", "2", "--attention-block", "3",
                "--device", "cpu"])
            frame = pd.read_csv(output)
            self.assertEqual(frame.columns.tolist(), experiment.KEYS +
                ["dt", "actual", "tactis2", "chronos2", "sktime_m4_naive"])
            self.assertEqual(len(frame), 6)
            self.assertFalse(frame.isna().any().any())
            table = pd.read_csv(artifacts / "metrics.csv")
            self.assertEqual(set(table.model), {"tactis2", "chronos2", "sktime_m4_naive"})
            self.assertTrue(table.MAE.is_monotonic_increasing)
            self.assertTrue(table.observations.eq(6).all())
            self.assertEqual(json.loads((artifacts / "status.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
