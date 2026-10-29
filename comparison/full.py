"""Bounded-memory, resumable grouped forecasting with atomic result files."""
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from .data import KEYS, load_sales
from .models import ROOT, load_chronos, predict_chronos, train_tactis


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


@contextmanager
def run_lock(output):
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is using {output}') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def signature(args):
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('output_dir', 'resume', 'data_only')}
    sources = [ROOT / 'run_comparison.py', *sorted((ROOT / 'comparison').glob('*.py')),
               *sorted((ROOT / 'TACTiS2/TACTiS/tactis/model').glob('*.py')),
               *sorted((ROOT / 'chronos/chronos-forecasting/src').rglob('*.py'))]
    return dict(arguments=settings, data={str(p): digest(p) for p in (args.train_path, args.eval_path)},
                source={str(p.relative_to(ROOT)): digest(p) for p in sources},
                packages={p: importlib.metadata.version(p) for p in
                          ('torch', 'numpy', 'pandas', 'pyarrow', 'transformers', 'accelerate', 'einops')})


def verify_signature(saved, current):
    if saved != current:
        raise ValueError('Resume rejected: data, code, packages or parameters changed. Use a new output directory.')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def score_sums(actual, prediction):
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if actual.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid prediction shape/values')
    error = prediction - actual
    return dict(count=int(actual.size), absolute=float(np.abs(error).sum()),
                squared=float(np.square(error).sum()), denominator=float(np.abs(actual).sum()))


def scores(total):
    return dict(MAE=total['absolute'] / total['count'], RMSE=np.sqrt(total['squared'] / total['count']),
                WAPE=total['absolute'] / total['denominator'] if total['denominator'] else None,
                observations=total['count'])


def result_frame(panel, start, stop, samples):
    expected = (stop - start, len(panel.eval_dates))
    if samples.ndim != 3 or samples.shape[:2] != expected or not np.isfinite(samples).all():
        raise ValueError('Forecast shape mismatch or nonfinite samples')
    frame = panel.keys.iloc[start:stop].loc[lambda x: x.index.repeat(len(panel.eval_dates))].reset_index(drop=True)
    frame['dt'] = np.tile(panel.eval_dates, stop-start)
    frame['actual'] = panel.actual[start:stop].ravel()
    frame['prediction'] = np.median(samples, axis=-1).ravel()
    return frame


def read_result(path, panel, start, stop):
    frame = pd.read_parquet(path)
    expected = panel.keys.iloc[start:stop].loc[lambda x: x.index.repeat(len(panel.eval_dates))].reset_index(drop=True)
    if not frame[KEYS].equals(expected):
        raise ValueError(f'Wrong series IDs in {path}')
    if not np.array_equal(frame.dt.to_numpy(), np.tile(panel.eval_dates, stop-start)):
        raise ValueError(f'Wrong dates in {path}')
    if not np.array_equal(frame.actual.to_numpy(), panel.actual[start:stop].ravel()):
        raise ValueError(f'Wrong actuals in {path}')
    return frame, score_sums(frame.actual, frame.prediction)


def run_grouped(args):
    torch.set_num_threads(args.threads)
    output = args.output_dir or ROOT / 'outputs' / datetime.now().strftime('grouped-%Y%m%d-%H%M%S-%f')
    if args.resume:
        if not (output / 'manifest.json').exists():
            raise ValueError('Cannot resume: manifest.json is missing')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        _run(args, output)


def _run(args, output):
    identity = signature(args)
    manifest_path = output / 'manifest.json'
    if args.resume:
        verify_signature(json.loads(manifest_path.read_text())['signature'], identity)
    panel = load_sales(args.train_path, args.eval_path, None if args.all_series else args.num_series,
                       args.seed, args.horizon)
    if panel.train.shape[1] - args.horizon < args.context_length + args.horizon:
        raise ValueError('Insufficient training/validation history')
    groups = list(range(0, len(panel.keys), args.group_size))
    if not args.resume:
        panel.keys.assign(group=np.arange(len(panel.keys)) // args.group_size).to_csv(output / 'series.csv', index=False)
        atomic_json(manifest_path, dict(signature=identity, series=len(panel.keys), groups=len(groups),
                    expected_rows_per_model=len(panel.keys)*args.horizon,
                    grouping='Sort city_id/store_id/product_id, consecutive fixed-size groups; groups may cross stores.',
                    model='Independent small TACTiS2 per group; pretrained original Chronos, shared checkpoint.',
                    train_dates=[str(panel.train_dates[0]), str(panel.train_dates[-1])],
                    validation_dates=[str(panel.train_dates[-args.horizon]), str(panel.train_dates[-1])],
                    eval_dates=[str(panel.eval_dates[0]), str(panel.eval_dates[-1])],
                    point_forecast='Sample median, no clipping. Same final context for both models.',
                    limitation='Full series coverage does not imply converged training. Random training windows; validation held out.'))
    if args.data_only:
        print(f'Data validation passed: {len(panel.keys)} series, {len(groups)} groups', flush=True)
        return
    completed = 0
    totals = {name: dict(count=0, absolute=0., squared=0., denominator=0.) for name in ('tactis2', 'chronos')}
    pipeline = None
    started = time.perf_counter()
    newly_completed = 0
    print(f'Output: {output}\n{len(panel.keys)} series / {len(groups)} groups', flush=True)
    try:
        for group, start in enumerate(groups):
            stop = min(start + args.group_size, len(panel.keys))
            folder = output / 'groups' / f'{group:05d}'
            folder.mkdir(parents=True, exist_ok=True)
            for model_index, name in enumerate(totals):
                path = folder / f'{name}.parquet'
                if not path.exists():
                    group_args = Namespace(**vars(args))
                    group_args.seed = args.seed + group * 2 + model_index
                    seed_all(group_args.seed)
                    print(f'Group {group+1}/{len(groups)} {name}, series {start}:{stop}', flush=True)
                    if name == 'tactis2':
                        samples, log = train_tactis(panel.train[start:stop], group_args, folder)
                        atomic_json(folder / 'training_log.json', log)
                    else:
                        if pipeline is None:
                            pipeline = load_chronos(args)
                        seed_all(group_args.seed)  # model loading must not change sampling RNG
                        samples = predict_chronos(panel.train[start:stop], group_args, pipeline=pipeline)
                    frame = result_frame(panel, start, stop, samples)
                    temporary = path.with_suffix('.parquet.tmp')
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)  # only a validated complete forecast is resumable
                    del samples
                    newly_completed += 1
                _, subtotal = read_result(path, panel, start, stop)
                for key, value in subtotal.items():
                    totals[name][key] += value
            completed += 1
            elapsed = time.perf_counter() - started
            remaining = (len(groups)-completed)*2
            atomic_json(output / 'progress.json', dict(status='running', completed_groups=completed,
                        total_groups=len(groups), completed_series=stop, total_series=len(panel.keys),
                        elapsed_this_session_seconds=elapsed,
                        estimated_remaining_seconds=elapsed/newly_completed*remaining if newly_completed else None,
                        updated_utc=datetime.now(timezone.utc).isoformat()))
            partial = pd.DataFrame([dict(model=name, **scores(total)) for name,total in totals.items()])
            partial.to_csv(output / 'metrics_partial.csv.tmp', index=False)
            (output / 'metrics_partial.csv.tmp').replace(output / 'metrics_partial.csv')
        # Recheck every persisted partition and export only after full coverage.
        rows = []
        for name in totals:
            path = output / f'{name}_predictions.csv'
            temporary = path.with_suffix('.csv.tmp')
            total = dict(count=0, absolute=0., squared=0., denominator=0.)
            with temporary.open('w') as stream:
                for group, start in enumerate(groups):
                    frame, subtotal = read_result(output/'groups'/f'{group:05d}'/f'{name}.parquet',
                                                  panel, start, min(start+args.group_size,len(panel.keys)))
                    frame.to_csv(stream, index=False, header=(group == 0))
                    for key, value in subtotal.items():
                        total[key] += value
            if total['count'] != len(panel.keys)*args.horizon:
                raise ValueError('Incomplete forecast coverage')
            temporary.replace(path)
            rows.append(dict(model=name, **scores(total)))
        pd.DataFrame(rows).to_csv(output / 'metrics.csv.tmp', index=False)
        (output / 'metrics.csv.tmp').replace(output / 'metrics.csv')
        atomic_json(output / 'progress.json', dict(status='complete', completed_groups=len(groups),
                    total_groups=len(groups), completed_series=len(panel.keys), total_series=len(panel.keys)))
        (output / 'SUCCESS').write_text('Both models cover every selected series and evaluation day.\n')
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        print(f'Complete: {output}', flush=True)
    except BaseException as exc:
        atomic_json(output / 'progress.json', dict(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                    completed_groups=completed, total_groups=len(groups), error=str(exc)))
        raise
