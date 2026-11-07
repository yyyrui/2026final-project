"""Migration checks for the actual Chronos-2 API and quantile semantics."""
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from comparison import chronos2
from comparison.runner import arguments


class Chronos2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = chronos2.ROOT / "chronos/chronos-forecasting/test/dummy-chronos2-model"

    def test_both_entrypoints_select_pinned_small_model(self):
        for mode in ("sample", "all"):
            args = arguments([], mode=mode)
            self.assertEqual(args.chronos2_model, "autogluon/chronos-2-small")
            self.assertEqual(args.chronos2_revision, chronos2.MODEL_SPEC["model_revision"])
            self.assertIn("chronos2", str(args.work_dir))
        custom = arguments(["--chronos2-model", "amazon/chronos-2"])
        self.assertIsNone(custom.chronos2_revision)

    def test_rejects_original_chronos_checkpoint_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text(json.dumps({
                "model_type": "t5", "architectures": ["T5ForConditionalGeneration"],
                "chronos_pipeline_class": "ChronosPipeline", "chronos_config": {}}))
            args = arguments(["--chronos2-model", str(folder), "--offline"])
            with self.assertRaisesRegex(ValueError, "Expected a Chronos-2 checkpoint"):
                chronos2.validate_model(args)

    def test_cached_hub_checkpoint_never_attempts_network_offline(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            snapshot = folder / "models--test-org--chronos2-fixture/snapshots" / revision
            shutil.copytree(self.model, snapshot)
            args = arguments(["--chronos2-model", "test-org/chronos2-fixture",
                              "--chronos2-revision", revision, "--offline", "--device", "cpu",
                              "--context", "4", "--horizon", "2"])
            args.work_dir = folder
            panel = SimpleNamespace(keys=[0], train=np.array([[1, 2, 3, 4]], dtype=np.float32))
            with patch.object(chronos2, "loading_options", return_value=dict(
                    revision=revision, cache_dir=str(folder), local_files_only=True)), \
                    patch("requests.sessions.Session.request", side_effect=AssertionError("Network attempted")):
                result = chronos2.predict_chronos2(panel, args)
            self.assertEqual(result.shape, (1, 2))
            self.assertTrue(np.isfinite(result).all())
            provenance = json.loads((folder / "chronos2_model.json").read_text())
            self.assertEqual(provenance["resolved_revision"], revision)

    def test_native_quantiles_median_alignment_and_batch_independence(self):
        from chronos import Chronos2Pipeline

        panel = SimpleNamespace(keys=list(range(3)), train=np.array([
            [1, 2, 3, 5, 8, 13], [10, 9, 8, 6, 4, 3], [0, 1, 0, 2, 0, 4],
        ], dtype=np.float32))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            args = arguments(["--chronos2-model", str(self.model), "--offline",
                              "--context", "4", "--horizon", "3", "--device", "cpu",
                              "--chronos2-batch-size", "2", "--samples", "1"])
            args.work_dir = folder
            result = chronos2.predict_chronos2(panel, args)
            saved = np.load(folder / "chronos2_quantiles.npz")
            levels = saved["quantile_levels"]
            q50 = list(levels).index(0.5)
            self.assertEqual(saved["quantiles"].shape, (3, 3, len(levels)))
            np.testing.assert_array_equal(result, saved["quantiles"][:, :, q50])
            self.assertFalse((folder / "chronos_samples.npz").exists())
            self.assertFalse((folder / "chronos2_samples.npz").exists())
            pipeline = Chronos2Pipeline.from_pretrained(str(self.model), device_map="cpu", local_files_only=True)
            reference = pipeline.predict(torch.from_numpy(panel.train[:, -4:]).unsqueeze(1),
                                         prediction_length=3, context_length=4, cross_learning=False)
            expected = torch.cat(reference, dim=0)[:, q50, :].numpy()
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
            # --samples controls TACTiS2 only; batching must not mix series.
            args.samples = 999
            args.chronos2_batch_size = 1
            repeated = chronos2.predict_chronos2(panel, args)
            np.testing.assert_allclose(result, repeated, rtol=1e-5, atol=1e-5)
            provenance = json.loads((folder / "chronos2_model.json").read_text())
            self.assertEqual(provenance["pipeline_class"], "Chronos2Pipeline")
            self.assertEqual(provenance["point_quantile"], 0.5)
            self.assertFalse(provenance["cross_learning"])
            self.assertIn("model.safetensors", provenance["local_files_sha256"])


if __name__ == "__main__":
    unittest.main()
