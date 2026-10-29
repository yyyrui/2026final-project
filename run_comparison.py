#!/usr/bin/env python3
"""Small, reproducible real-sales comparison; paths are relative to this file."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

import numpy as np
import pandas as pd
import torch
from comparison.data import load_sales
from comparison.models import train_tactis, predict_chronos


def metrics(actual, samples):
    if samples.shape[:2] != actual.shape or samples.ndim != 3 or not np.isfinite(samples).all():
        raise ValueError('Forecast shape mismatch or nonfinite predictions')
    # Same sample median and raw (unclipped) forecasts for both models.
    prediction = np.median(samples, axis=-1)
    error = prediction.astype(float) - actual
    denominator = float(np.abs(actual).sum())
    return dict(MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.square(error).mean())),
                WAPE=float(np.abs(error).sum() / denominator) if denominator else None)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-path', type=Path, default=ROOT / 'dataset/train.parquet')
    parser.add_argument('--eval-path', type=Path, default=ROOT / 'dataset/eval.parquet')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--num-series', type=positive, default=4)
    parser.add_argument('--context-length', type=positive, default=28)
    parser.add_argument('--horizon', type=positive, default=7)
    parser.add_argument('--epochs', type=positive, default=2, help='epochs per TACTiS stage')
    parser.add_argument('--steps-per-epoch', type=positive, default=5)
    parser.add_argument('--num-samples', type=positive, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=positive, default=2)
    parser.add_argument('--chronos-model', default='amazon/chronos-t5-tiny', help='Original Chronos checkpoint')
    parser.add_argument('--chronos-batch-size', type=positive, default=8)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    if args.context_length < 2 or args.learning_rate <= 0:
        parser.error('context-length must be >= 2 and learning-rate must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    if args.num_series > 64:
        parser.error('This small joint-model runner supports at most 64 series; larger runs need grouping and memory planning')
    for field in ('train_path', 'eval_path', 'output_dir'):
        path = getattr(args, field)
        if path is not None and not path.is_absolute():
            setattr(args, field, ROOT / path)
    output = args.output_dir or ROOT / 'outputs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    panel = load_sales(args.train_path, args.eval_path, args.num_series, args.seed, args.horizon)
    panel.keys.to_csv(output / 'series.csv', index=False)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    total_series=panel.total_series, selected_series=len(panel.keys),
                    train_dates=[str(panel.train_dates[0].date()), str(panel.train_dates[-1].date())],
                    validation_dates=[str(panel.train_dates[-args.horizon].date()), str(panel.train_dates[-1].date())],
                    eval_dates=[str(panel.eval_dates[0].date()), str(panel.eval_dates[-1].date())],
                    point_forecast='sample median, no clipping',
                    note='Small smoke test, not a tuned benchmark. Both models receive the same final context window.',
                    versions={'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if args.data_only:
        print(f'Data validation passed: {output}', flush=True)
        return
    results = []
    for name in ('tactis2', 'chronos'):
        print(f'Running {name} ...', flush=True)
        started = time.perf_counter()
        if name == 'tactis2':
            samples, log = train_tactis(panel.train, args, output)
            (output / 'training_log.json').write_text(json.dumps(log, indent=2))
        else:
            samples = predict_chronos(panel.train, args)
        score = metrics(panel.actual, samples)
        score.update(model=name, seconds=time.perf_counter() - started)
        np.savez_compressed(output / f'{name}_samples.npz', samples=samples)
        predictions = panel.keys.loc[panel.keys.index.repeat(args.horizon)].reset_index(drop=True)
        predictions['dt'] = np.tile(panel.eval_dates, len(panel.keys))
        predictions['actual'] = panel.actual.ravel()
        predictions['prediction'] = np.median(samples, axis=-1).ravel()
        predictions.to_csv(output / f'{name}_predictions.csv', index=False)
        results.append(score)
        pd.DataFrame(results).to_csv(output / 'metrics.csv', index=False)
        print(score, flush=True)
    (output / 'SUCCESS').write_text('Both models completed and forecasts passed shape/finite checks.\n')
    print(f'Comparison complete: {output}', flush=True)


if __name__ == '__main__':
    main()
