#!/usr/bin/env python3
"""One TACTiS2 for all selected series. Defaults to preflight, not training."""
import argparse
from copy import deepcopy
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')


def positive(value):
    value=int(value)
    if value<1: raise argparse.ArgumentTypeError('must be positive')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/tactis2-paper-traffic.json')
    p.add_argument('--train-path',type=Path,default=ROOT/'dataset/train.parquet')
    p.add_argument('--eval-path',type=Path,default=ROOT/'dataset/eval.parquet')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--threads',type=positive,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--execute',action='store_true',help='Execute after successful resource preflight')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--smoke-series',type=positive,help='Explicit small correctness test, NOT full-data evaluation')
    p.add_argument('--chronos-model',default='amazon/chronos-t5-tiny')
    p.add_argument('--chronos-revision',default='29d808298f1a62493e7b9a5e08529d0d930fa189')
    p.add_argument('--chronos-batch-size',type=positive,default=8)
    args=p.parse_args()
    for key in ('config','train_path','eval_path','output_dir'):
        v=getattr(args,key)
        if v is not None and not v.is_absolute(): setattr(args,key,ROOT/v)
    if args.offline: os.environ['HF_HUB_OFFLINE']='1'
    if args.resume and (not args.execute or not args.output_dir): p.error('--resume requires --execute and --output-dir')
    import torch
    from comparison.data import load_sales
    from comparison.full import atomic_json,digest,run_lock
    from joint_comparison.resources import estimate,hardware,blockers
    from joint_comparison.runner import run
    torch.set_num_threads(args.threads)
    config=json.loads(args.config.read_text())
    if args.smoke_series:
        if args.smoke_series>64: p.error('smoke-series must be <=64')
        config=deepcopy(config)
        config['training'].update(max_epochs_per_stage=2,steps_per_epoch=2,batch_size=1,patience=2)
        config['sampling'].update(num_samples=4,samples_per_call=2)
    panel=load_sales(args.train_path,args.eval_path,args.smoke_series,args.seed,config['horizon'])
    scope='SMOKE_ONLY' if args.smoke_series else 'ALL_SERIES_JOINT'
    output=args.output_dir or ROOT/'outputs'/datetime.now().strftime('joint-%Y%m%d-%H%M%S-%f')
    if args.resume:
        if not (output/'manifest.json').exists(): p.error('Resume manifest missing')
    else: output.mkdir(parents=True,exist_ok=False)
    with run_lock(output):
        paths=[ROOT/'run_joint_comparison.py',*sorted((ROOT/'joint_comparison').glob('*.py')),
               *sorted((ROOT/'TACTiS2/TACTiS/tactis/model').glob('*.py')),
               ROOT/'comparison/data.py',ROOT/'comparison/full.py',ROOT/'comparison/models.py',
               *sorted((ROOT/'chronos/chronos-forecasting/src').rglob('*.py'))]
        identity=dict(config=config,scope=scope,seed=args.seed,series=len(panel.keys),threads=args.threads,
                      chronos=[args.chronos_model,args.chronos_revision,args.chronos_batch_size],
                      data={key:digest(getattr(args,key)) for key in ('train_path','eval_path')},
                      source={str(path.relative_to(ROOT)):digest(path) for path in paths},
                      versions={key:importlib.metadata.version(key) for key in
                                ('torch','numpy','pandas','pyarrow','transformers','accelerate','einops')})
        if args.resume and json.loads((output/'manifest.json').read_text())['identity'] != identity:
            raise ValueError('Resume data, code, configuration or package mismatch')
        report=estimate(len(panel.keys),config)
        capacity=hardware(args.device)
        reasons=blockers(report,capacity)
        if args.device=='cuda' and not torch.cuda.is_available(): reasons.append('CUDA not available on this machine')
        report.update(hardware=capacity,blockers=reasons,scope=scope,
                      status='blocked' if reasons else 'preflight_passed_not_a_peak_memory_guarantee')
        atomic_json(output/'resource_report.json',report)
        if not args.resume:
            atomic_json(output/'manifest.json',dict(identity=identity,train_dates=[str(panel.train_dates[0]),str(panel.train_dates[-1])],
                        eval_dates=[str(panel.eval_dates[0]),str(panel.eval_dates[-1])],
                        note='One shared model; random bagging only in training; full unbagged validation and joint sampling.'))
            panel.keys.to_csv(output/'series.csv',index=False)
        print(json.dumps(report,indent=2),flush=True)
        if not args.execute:
            print(f'Preflight only: {output}',flush=True)
            return
        if reasons:
            atomic_json(output/'status.json',dict(status='blocked_before_training',reasons=reasons))
            raise SystemExit('Resource preflight blocked training. No new model comparison was run.')
        try:
            run(panel,config,args,output,identity)
        except BaseException as exc:
            atomic_json(output/'status.json',dict(status='interrupted_or_failed',error=str(exc)))
            raise


if __name__=='__main__': main()
