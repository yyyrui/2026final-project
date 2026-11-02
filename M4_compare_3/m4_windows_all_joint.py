# -*- coding: utf-8 -*-
"""Windows / Python 3.11. Strict ALL 100,000-series TACTiS-2 experiment.
Run `python m4_windows_all_joint.py --help`. No model scores are bundled.
See README_m4_windows.md for methodological and hardware limitations.
"""
# %% 1. ALL installation and downloads (run this cell first)
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from copy import deepcopy

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
REPO = 'yyyrui/2026final-project'
COMMIT = '70db6cf5891af48b3032f939dc8f6634d1693bee'
CHRONOS_ID = 'amazon/chronos-t5-tiny'
CHRONOS_REV = '29d808298f1a62493e7b9a5e08529d0d930fa189'
# frequency: (official count, horizon, M4 MASE seasonal lag)
SPECS = {'Yearly': (23000, 6, 1), 'Quarterly': (24000, 8, 4),
         'Monthly': (48000, 18, 12), 'Weekly': (359, 13, 1),
         'Daily': (4227, 14, 1), 'Hourly': (414, 48, 24)}
METHODS = ('Naive', 'sNaive', 'Naive2', 'SES', 'Holt', 'Damped', 'Theta', 'Com')
PACKAGES = ['numpy==1.26.4', 'pandas==2.2.3', 'scipy==1.13.1',
            'sktime==0.35.0', 'statsmodels==0.14.4', 'einops==0.8.1',
            'chronos-forecasting==1.5.2', 'transformers==4.48.3',
            'accelerate==1.1.1', 'huggingface-hub==0.28.1',
            'pyarrow==18.1.0', 'psutil==6.1.1']


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'check'], default='all')
    p.add_argument('--work-dir', type=Path, default=ROOT / 'm4_work')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'm4_results')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--torch-index', choices=['cu121', 'cpu'], default='cu121')
    p.add_argument('--skip-install', action='store_true')
    p.add_argument('--offline', action='store_true', help='skip all setup; require prior complete setup')
    p.add_argument('--context', type=int, default=6)
    p.add_argument('--train-horizon', type=int, default=6)
    p.add_argument('--epochs', type=int, default=20, help='fixed epochs PER stage; no Test tuning')
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    # Ignore Jupyter kernel flags when running # %% cells; CLI keeps strict argument parsing.
    a = p.parse_args([] if 'ipykernel' in sys.modules else None)
    if min(a.context, a.train_horizon, a.epochs, a.samples) < 1 or a.context < 2:
        p.error('context >= 2; other numeric settings must be positive')
    a.work_dir = a.work_dir.resolve(); a.output_dir = a.output_dir.resolve()
    return a


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def download(url, path):
    """Atomic download; completed files reused, partial files never accepted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    temp = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            print('Download:', url, flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'M4-research-data-preparation'})
            with urllib.request.urlopen(req, timeout=120) as r, temp.open('wb') as f:
                shutil.copyfileobj(r, f)
            os.replace(temp, path)
            return
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)


def setup(a):
    os.environ['HF_HOME'] = str(a.work_dir / 'cache' / 'huggingface')
    if a.offline:
        return
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    if not a.skip_install:
        if sys.version_info[:2] != (3, 11):
            raise SystemExit('Use a fresh Python 3.11 environment on Windows; then rerun.')
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.5.1',
                        '--index-url', 'https://download.pytorch.org/whl/' + a.torch_index], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', *PACKAGES], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    base = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/TACTiS2/TACTiS/'
    files = ['tactis/__init__.py', 'tactis/model/__init__.py', 'LICENSE', 'NOTICE']
    files += ['tactis/model/' + f + '.py' for f in ['tactis', 'decoder', 'encoder', 'flow', 'marginal', 'utils']]
    for f in files:
        download(base + f, a.work_dir / 'vendor' / f)
    m4 = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/'
    download(m4 + 'M4-info.csv', a.work_dir / 'raw' / 'M4-info.csv')
    for freq in SPECS:
        for split in ['train', 'test']:
            name = f'{freq}-{split}.csv'
            download(m4 + split.capitalize() + '/' + name, a.work_dir / 'raw' / name)
    from huggingface_hub import snapshot_download
    snapshot_download(CHRONOS_ID, revision=CHRONOS_REV, local_dir=str(a.work_dir / 'chronos'),
                      allow_patterns=['config.json', 'generation_config.json', 'model.safetensors'])
    paths = [p for folder in ['raw', 'vendor', 'chronos']
             for p in (a.work_dir / folder).rglob('*') if p.is_file() and '.cache' not in p.parts]
    write_json(a.work_dir / 'download_manifest.json', {
        'repo_commit': COMMIT, 'chronos_revision': CHRONOS_REV,
        'files': {str(p.relative_to(a.work_dir)): digest(p) for p in paths}})


if __name__ == '__main__':
    ARGS = arguments()
    setup(ARGS)  # All network activity ends here. Subsequent cells operate offline.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# %% 2. Prepare ALL M4 observations, ID mappings, and model adapters
import numpy as np
import pandas as pd


def read_rows(path):
    """Remove only CSV trailing padding; internal missing values are errors."""
    result = {}
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            key, values = row[0], row[1:]
            while values and values[-1].strip().lower() in ('', 'nan'):
                values.pop()
            if key in result:
                raise ValueError(f'Duplicate ID: {key}')
            y = np.asarray(values, dtype=np.float64)
            if not len(y) or not np.isfinite(y).all():
                raise ValueError(f'Empty/nonfinite/internal missing data: {key}')
            result[key] = y
    return result


def load_all(raw, strict=True):
    records = []
    for freq, (count, horizon, sp) in SPECS.items():
        train = read_rows(raw / f'{freq}-train.csv')
        test = read_rows(raw / f'{freq}-test.csv')
        if train.keys() != test.keys():
            raise ValueError(f'{freq}: Train/Test IDs differ')
        if strict and len(train) != count:
            raise ValueError(f'{freq}: {len(train)} series, expected {count}')
        for key in sorted(train, key=lambda k: int(k[1:])):
            if not key.startswith(freq[0]) or len(test[key]) != horizon:
                raise ValueError(f'Invalid frequency/horizon for {key}')
            records.append(dict(id=key, frequency=freq, horizon=horizon, sp=sp,
                                train=train[key], test=test[key]))
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate IDs across frequencies')
    if strict and len(records) != 100000:
        raise ValueError('All 100,000 series are mandatory')
    if strict:
        info = pd.read_csv(raw / 'M4-info.csv')
        if info.M4id.duplicated().any() or set(info.M4id) != {r['id'] for r in records}:
            raise ValueError('M4-info ID coverage mismatch')
    return records


def history_matrix(records, context, horizon=0, offset=0):
    """All series at SAME relative offset from their own Train endpoint.
    Never pad, interpolate, merge frequencies, or read official Test values.
    """
    width = context + horizon
    if min(len(r['train']) for r in records) < width + offset:
        raise ValueError('Context/horizon/offset exceeds shortest Train history')
    return np.stack([r['train'][len(r['train']) - offset - width:len(r['train']) - offset]
                     for r in records]).astype(np.float32)


def prepare(records, a):
    folder = a.work_dir / 'prepared'; folder.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    # One complete table; list columns preserve each series' full, ragged history.
    table = pa.table({
        'unique_id': [r['id'] for r in records],
        'frequency': [r['frequency'] for r in records],
        'horizon': [r['horizon'] for r in records], 'seasonal_period': [r['sp'] for r in records],
        'train_values': pa.array([r['train'] for r in records], type=pa.list_(pa.float64())),
        'test_values': pa.array([r['test'] for r in records], type=pa.list_(pa.float64()))})
    pq.write_table(table, folder / 'm4_all_series.parquet', compression='zstd')
    pd.DataFrame([{k: r[k] for k in ['id', 'frequency', 'horizon', 'sp']} |
                  {'row': i, 'train_length': len(r['train'])} for i, r in enumerate(records)]
                 ).to_csv(folder / 'series_index.csv', index=False)
    history = history_matrix(records, a.context)
    np.save(folder / 'tactis_history.npy', history[None])  # [1,100000,C]
    np.save(folder / 'chronos_history.npy', history)       # [100000,C], ONE predict call
    target = np.full((len(records), 48), np.nan, dtype=np.float64)
    for i, r in enumerate(records):
        target[i, :r['horizon']] = r['test']
    np.savez(folder / 'official_targets.npz', actual=target, score_mask=np.isfinite(target))
    # Both arrays below have ALL series; no group IDs or subsets exist.
    train = history_matrix(records, a.context, a.train_horizon)
    np.savez(folder / 'tactis_latest_training_window.npz',
             hist=train[None, :, :a.context], future=train[None, :, a.context:])
    report = {'series': len(records), 'official_target_observations': sum(r['horizon'] for r in records),
              'train_observations_preserved': sum(len(r['train']) for r in records),
              'min_train_length': min(len(r['train']) for r in records),
              'max_train_length': max(len(r['train']) for r in records),
              'nonpositive_train_series': sum(bool(np.any(r['train'] <= 0)) for r in records),
              'context': a.context, 'train_horizon': a.train_horizon,
              'common_training_windows': min(len(r['train']) for r in records) - a.context - a.train_horizon + 1,
              'alignment': 'relative to each series endpoint, NOT a shared calendar',
              'full_raw_history_preserved': True,
              'training_scope': 'all series in every step; only common end-aligned windows fit original dense model',
              'benchmark_context': 'TACTiS2/Chronos: last C; sktime: full Train for seasonal estimation; NOT context-matched',
              'counts': {f: sum(r['frequency'] == f for r in records) for f in SPECS}}
    write_json(folder / 'data_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report

# %% 3. Strict joint TACTiS-2, full-coverage baselines, and resource check

def model_config(n):
    enc = dict(attention_layers=2, attention_heads=4, attention_dim=8,
               attention_feedforward_dim=32, dropout=0.0)
    return dict(num_series=n, bagging_size=None, flow_series_embedding_dim=8,
                copula_series_embedding_dim=8, flow_input_encoder_layers=2,
                copula_input_encoder_layers=2, data_normalization='standardization',
                loss_normalization='both', positional_encoding={'dropout': 0.0},
                flow_temporal_encoder=enc.copy(), copula_temporal_encoder=enc.copy(),
                copula_decoder=dict(min_u=0.05, max_u=0.95,
                    attentional_copula=dict(attention_heads=4, attention_layers=2, attention_dim=8,
                        mlp_layers=2, mlp_dim=32, resolution=100, dropout=0.0,
                        attention_mlp_class='_simple_linear_projection'),
                    dsf_marginal=dict(mlp_layers=2, mlp_dim=32, flow_layers=2, flow_hid_dim=16)),
                skip_copula=True, experiment_mode='forecasting')


def resource_report(n, a, available=None):
    # Verified against the pinned decoder's explicit einsum product allocation.
    h, c, heads, size = a.train_horizon, a.context, 4, 4
    decoder = heads * (n*h) * (n*(c+h)) * size
    temporal = (c+h) * heads * n*n * size
    # Multiple dense products, masks, softmax outputs and gradients coexist.
    budget = 6 * max(decoder, temporal)
    if available is None:
        import torch, psutil
        available = torch.cuda.mem_get_info()[0] if a.device == 'cuda' and torch.cuda.is_available() else (
            psutil.virtual_memory().available if a.device == 'cpu' else 0)
    return {'series': n, 'decoder_single_product_bytes': decoder,
            'temporal_attention_dense_equivalent_bytes': temporal,
            'conservative_required_bytes': budget, 'available_device_bytes': int(available),
            'blocked': available * 0.8 < budget,
            'note': 'Estimate, not a measured peak. Single device; no pooling of GPU RAM. No grouping fallback.'}


def create_tactis(a, n):
    import torch
    sys.path.insert(0, str(a.work_dir / 'vendor'))
    from tactis.model.tactis import TACTiS
    torch.manual_seed(a.seed)
    return TACTiS(**deepcopy(model_config(n))).to(a.device)


def tensors(values, a):
    import torch
    data = torch.from_numpy(values[None]).to(a.device)
    ht = torch.arange(a.context, device=a.device)[None, None]
    pt = torch.arange(a.context, a.context+a.train_horizon, device=a.device)[None, None]
    return ht, data[:, :, :a.context], pt, data[:, :, a.context:]


def train_joint(records, a, folder):
    import torch
    model = create_tactis(a, len(records))
    windows = min(len(r['train']) for r in records) - a.context - a.train_horizon + 1
    if windows < 1:
        raise ValueError('Not enough observations for a common joint training window')
    rng = np.random.default_rng(a.seed)
    log = []
    for stage in (1, 2):
        if stage == 2:
            model.initialize_stage2(); model.to(a.device)
        for name, p in model.named_parameters():
            p.requires_grad_(stage == 1 or 'copula' in name)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=1e-3 if stage == 1 else 5e-4)
        for epoch in range(a.epochs):
            model.train(); losses = []
            # Exhaust ALL common offsets each epoch. Every update includes all N series.
            for offset in rng.permutation(windows):
                values = history_matrix(records, a.context, a.train_horizon, int(offset))
                opt.zero_grad(set_to_none=True)
                marginal, copula = model.loss(*tensors(values, a))
                loss = (-marginal if stage == 1 else copula).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss; no skipped series or Naive substitution')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                opt.step(); losses.append(float(loss.detach().cpu()))
            record = dict(stage=stage, epoch=epoch+1, loss=float(np.mean(losses)),
                          windows=windows, series_per_update=len(records))
            log.append(record); print(record, flush=True)
            write_json(folder / 'training_log.json', log)
        torch.save({'model': model.state_dict(), 'stage': stage, 'config': model_config(len(records))},
                   folder / f'tactis_stage{stage}.pt')
    model.eval()
    return model


def predict_joint(model, records, a, folder):
    import torch
    n = len(records); max_h = max(r['horizon'] for r in records)
    result = np.lib.format.open_memmap(folder / 'tactis_samples.npy', mode='w+',
                                      dtype=np.float32, shape=(n, max_h, a.samples))
    torch.manual_seed(a.seed+100)
    # Each trajectory maintains ALL N series. Only Monte Carlo trajectories are serial.
    for draw in range(a.samples):
        history = history_matrix(records, a.context)
        offset = 0
        while offset < max_h:
            # Always sample the trained horizon, even on the final block; crop only for output.
            values = np.concatenate([history, np.zeros((n, a.train_horizon), dtype=np.float32)], axis=1)
            ht, hv, pt, _ = tensors(values, a)
            with torch.no_grad():
                sampled = model.sample(1, ht, hv, pt)[0, :, -a.train_horizon:, 0].cpu().numpy()
            if sampled.shape != (n, a.train_horizon) or not np.isfinite(sampled).all():
                raise RuntimeError('Invalid joint samples')
            used = min(a.train_horizon, max_h-offset)
            result[:, offset:offset+used, draw] = sampled[:, :used]
            history = np.concatenate([history, sampled], axis=1)[:, -a.context:].copy()
            offset += used
        result.flush(); print(f'Joint trajectory {draw+1}/{a.samples}: all {n} series', flush=True)
    return np.median(result, axis=2)


def predict_chronos(records, a):
    os.environ['HF_HOME'] = str(a.work_dir / 'cache' / 'huggingface')
    import torch
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(str(a.work_dir / 'chronos'),
        device_map=a.device, torch_dtype=torch.float32, local_files_only=True)
    torch.manual_seed(a.seed+200)
    context = torch.from_numpy(history_matrix(records, a.context))
    with torch.no_grad():
        # Exactly ONE call containing all 100,000 series. No external batch loop.
        samples = pipeline.predict(context, prediction_length=48, num_samples=a.samples)
    if samples.shape != (len(records), a.samples, 48):
        raise RuntimeError(f'Unexpected Chronos shape: {samples.shape}')
    pred = np.median(samples.float().cpu().numpy(), axis=1)
    if not np.isfinite(pred).all():
        raise RuntimeError('Nonfinite Chronos output')
    return pred


def seasonality_test(y, sp):
    from statsmodels.tsa.stattools import acf
    y = np.asarray(y, dtype=float).reshape(-1)
    if sp == 1 or len(y) < 3*sp or np.std(y) == 0:
        return False
    ac = acf(y, nlags=sp, fft=False)
    limit = 1.645 / np.sqrt(len(y)) * np.sqrt(np.cumsum(np.r_[1, 2*ac[1:]**2]))
    return bool(abs(ac[sp]) > limit[sp-1])


def sktime_one(values, horizon, sp):
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.transformations.series.detrend import ConditionalDeseasonalizer
    # Seasonal fitting uses original Train history. No Naive fallback on failures.
    seasonal_model = 'multiplicative' if np.all(np.asarray(values) > 0) else 'additive'
    def seasonal(f):
        if sp == 1:
            return f
        return TransformedTargetForecaster([
            ('deseason', ConditionalDeseasonalizer(sp=sp, model=seasonal_model,
                                                  seasonality_test=seasonality_test)), ('model', f)])
    methods = {
        'Naive': NaiveForecaster(strategy='last'),
        'sNaive': NaiveForecaster(strategy='last', sp=sp),
        'Naive2': seasonal(NaiveForecaster(strategy='last')),
        'SES': seasonal(ExponentialSmoothing()),
        'Holt': seasonal(ExponentialSmoothing(trend='add', damped_trend=False)),
        'Damped': seasonal(ExponentialSmoothing(trend='add', damped_trend=True)),
        'Theta': seasonal(ThetaForecaster(deseasonalize=False)),
    }
    y = pd.Series(np.asarray(values, dtype=float)); fh = np.arange(1, horizon+1)
    preds, errors = {}, {}
    for name, f in methods.items():
        try:
            if name == 'sNaive' and len(y) < sp:
                raise ValueError(f'context {len(y)} < seasonal period {sp}')
            pred = f.fit(y, fh=fh).predict().to_numpy(dtype=float)
            if pred.shape != (horizon,) or not np.isfinite(pred).all():
                raise ValueError('Nonfinite or wrong-shape forecast')
            preds[name] = pred
        except Exception as exc:
            errors[name] = f'{type(exc).__name__}: {exc}'
    if all(name in preds for name in ['SES', 'Holt', 'Damped']):
        preds['Com'] = np.mean([preds[name] for name in ['SES', 'Holt', 'Damped']], axis=0)
    else:
        errors['Com'] = 'One or more component methods failed'
    return preds, errors


def point_metrics(train, actual, pred, sp):
    error = np.abs(actual-pred)
    denom = np.abs(actual)+np.abs(pred)
    smape = 200 * np.mean(np.divide(error, denom, out=np.zeros_like(error), where=denom != 0))
    scale = np.mean(np.abs(train[sp:]-train[:-sp]))
    return {'sMAPE': float(smape), 'MASE': float(np.mean(error)/scale) if scale > 0 else None,
            'MAE': float(np.mean(error))}


def evaluate(records, predictions, folder):
    rows = []
    expected = sum(r['horizon'] for r in records)
    for name, pred in predictions.items():
        with (folder / f'predictions_{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['unique_id', 'frequency', 'step', 'actual', 'prediction'])
            for i, r in enumerate(records):
                values = pred[i, :r['horizon']]
                valid = np.isfinite(values).all()
                metrics = point_metrics(r['train'], r['test'], values, r['sp']) if valid else {}
                rows.append(dict(model=name, id=r['id'], frequency=r['frequency'], valid=bool(valid), **metrics))
                for j, (actual, forecast) in enumerate(zip(r['test'], values), 1):
                    writer.writerow([r['id'], r['frequency'], j, actual, forecast if np.isfinite(forecast) else ''])
    frame = pd.DataFrame(rows); frame.to_csv(folder / 'per_series_metrics.csv', index=False)
    summary = []
    for name in predictions:
        sub = frame[frame.model == name]
        valid = int(sub.valid.sum()); mase_valid = int(sub.MASE.notna().sum())
        complete = valid == len(records)
        summary.append(dict(model=name, expected_series=len(records), valid_series=valid,
                            mase_defined_series=mase_valid, complete=complete,
                            expected_observations=expected,
                            sMAPE=float(sub.sMAPE.mean()) if complete else None,
                            MASE=float(sub.MASE.mean()) if complete and mase_valid == len(records) else None))
    write_json(folder / 'metrics.json', summary)
    pd.DataFrame(summary).to_csv(folder / 'metrics.csv', index=False)
    return summary

# %% 4. Execute: preparation -> resource check -> training -> benchmarks -> evaluation

def main(a):
    if a.output_dir.exists():
        raise SystemExit('Output directory already exists. Choose a new --output-dir; no mixed runs/overwrites.')
    a.output_dir.mkdir(parents=True)
    status = a.output_dir / 'status.json'
    write_json(status, {'status': 'preparing', 'results_generated': False})
    try:
        manifest = a.work_dir / 'download_manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError('Complete chunk 1 before offline execution')
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        for name, sha in saved['files'].items():
            path = a.work_dir / name
            if not path.is_file() or digest(path) != sha:
                raise ValueError(f'Missing or modified downloaded asset: {name}')
        records = load_all(a.work_dir / 'raw')
        report = prepare(records, a)
        write_json(a.output_dir / 'manifest.json', {
            'repo_commit': COMMIT, 'chronos': [CHRONOS_ID, CHRONOS_REV],
            'settings': {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
            'data': report, 'model': model_config(len(records)),
            'script_sha256': digest(Path(__file__)) if '__file__' in globals() else None,
            'download_manifest_sha256': digest(manifest),
            'packages': {x.split('==')[0]: importlib.metadata.version(x.split('==')[0])
                         for x in PACKAGES + ['torch']},
            'training': 'fixed epochs, no validation or test-based selection; not paper reproduction',
            'sktime_scope': list(METHODS), 'OWA': 'not computed; official Naive2 reference not bundled',
            'chronos_caveat': 'M4 may overlap pretraining; not a guaranteed zero-shot comparison'})
        if a.mode == 'prepare':
            write_json(status, {'status': 'data_prepared_only', 'results_generated': False}); return
        resources = resource_report(len(records), a)
        write_json(a.output_dir / 'resource_report.json', resources)
        print(json.dumps(resources, indent=2), flush=True)
        if resources['blocked']:
            write_json(status, {'status': 'blocked_before_training', 'results_generated': False,
                                'reason': 'Insufficient single-device memory for mandatory full joint model'})
            raise SystemExit('Full joint memory requirement exceeds available RAM/VRAM. No grouping fallback.')
        if a.mode == 'check':
            write_json(status, {'status': 'preflight_passed_not_a_peak_memory_guarantee', 'results_generated': False}); return
        import torch
        model = train_joint(records, a, a.output_dir)
        predictions = {'TACTiS2': predict_joint(model, records, a, a.output_dir)}
        del model
        if a.device == 'cuda': torch.cuda.empty_cache()
        predictions['Chronos'] = predict_chronos(records, a)
        if a.device == 'cuda': torch.cuda.empty_cache()
        for method in METHODS:
            predictions['sktime_'+method] = np.full((len(records), 48), np.nan)
        with (a.output_dir / 'sktime_failures.jsonl').open('w', encoding='utf-8') as f:
            for i, r in enumerate(records):
                # Local classical estimators necessarily fit one series at a time, not a joint model.
                # Use full history for seasonal estimators: otherwise common C=6 makes sp=12/24 impossible.
                pred, errors = sktime_one(r['train'], r['horizon'], r['sp'])
                for method, values in pred.items(): predictions['sktime_'+method][i, :r['horizon']] = values
                if errors:
                    f.write(json.dumps({'id': r['id'], 'errors': errors}, ensure_ascii=False)+'\n'); f.flush()
                if i % 1000 == 0: print(f'sktime coverage: {i}/{len(records)}', flush=True)
        summary = evaluate(records, predictions, a.output_dir)
        complete = all(row['complete'] for row in summary)
        write_json(status, {'status': 'complete' if complete else 'incomplete_with_failures',
                            'results_generated': True, 'complete': complete})
        if complete: (a.output_dir / 'SUCCESS').write_text('All models cover all official targets.\n')
    except Exception as exc:
        write_json(status, {'status': 'failed', 'results_generated': False, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main(ARGS)
