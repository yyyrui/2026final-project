import json
from pathlib import Path
import unittest

import torch

from joint_comparison.resources import estimate,blockers
from joint_comparison.runner import build_model,set_stage_parameters,windows

ROOT=Path(__file__).resolve().parents[1]


class JointTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/tactis2-paper-traffic.json').read_text())

    def test_dense_allocation_and_3090_rejection(self):
        report=estimate(50000,self.config)
        self.assertEqual(report['validation_one_dense_tensor_bytes'],3920000000000)
        self.assertTrue(blockers(report,dict(free_bytes=24*2**30)))
        self.assertEqual(report['joint_autoregressive_steps_per_sample_call'],350000)

    def test_sample_batching_does_not_change_joint_validation(self):
        first=estimate(50000,self.config)
        self.config['sampling']['samples_per_call']=10
        second=estimate(50000,self.config)
        self.assertEqual(first['validation_one_dense_tensor_bytes'],second['validation_one_dense_tensor_bytes'])
        self.assertEqual(first['sampling_predicted_kv_cache_bytes']*10,second['sampling_predicted_kv_cache_bytes'])

    def test_stage_two_freezes_marginals_and_retains_global_embeddings(self):
        model=build_model(8,self.config,'cpu',stage=2)
        set_stage_parameters(model,2)
        self.assertEqual(model.flow_series_encoder.num_embeddings,8)
        self.assertEqual(model.copula_series_encoder.num_embeddings,8)
        self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if 'copula' not in n))
        self.assertTrue(any(p.requires_grad for n,p in model.named_parameters() if 'copula' in n))

    def test_window_future_is_strictly_after_history(self):
        data=torch.arange(30).repeat(3,1)
        ht,hist,pt,future=windows(data,[10,12],7,7,'cpu')
        self.assertEqual(tuple(hist.shape),(2,3,7))
        self.assertTrue(torch.equal(hist[:,:,-1]+1,future[:,:,0]))


if __name__=='__main__':unittest.main()
