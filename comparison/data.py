"""Strict daily panels shared by both forecasting models."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['city_id', 'store_id', 'product_id']


@dataclass
class SalesPanel:
    keys: pd.DataFrame
    train_dates: pd.DatetimeIndex
    eval_dates: pd.DatetimeIndex
    train: np.ndarray  # [series, day]
    actual: np.ndarray
    total_series: int


def read_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=KEYS + ['dt', 'sale_amount'])
    if frame.isna().any().any():
        raise ValueError(f'{path}: null keys, dates or targets')
    frame['dt'] = pd.to_datetime(frame['dt'], errors='raise')
    if not frame['dt'].eq(frame['dt'].dt.normalize()).all():
        raise ValueError(f'{path}: expected daily dates')
    if frame.duplicated(KEYS + ['dt']).any():
        raise ValueError(f'{path}: duplicate series/date rows; aggregation must be explicit')
    if not np.isfinite(frame['sale_amount'].to_numpy(dtype=float)).all():
        raise ValueError(f'{path}: nonfinite targets')
    return frame


def load_sales(train_path: Path, eval_path: Path, num_series=4, seed=42, horizon=7) -> SalesPanel:
    if num_series < 1 or horizon < 1:
        raise ValueError('num_series and horizon must be positive')
    train, evaluation = read_daily(train_path), read_daily(eval_path)
    keys = train[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    eval_keys = evaluation[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    if not keys.equals(eval_keys):
        raise ValueError('Train/eval series keys differ')
    train_dates = pd.DatetimeIndex(sorted(train.dt.unique()))
    eval_dates = pd.DatetimeIndex(sorted(evaluation.dt.unique()))
    if not train_dates.equals(pd.date_range(train_dates[0], train_dates[-1], freq='D')):
        raise ValueError('Training dates are not consecutive daily dates')
    expected = pd.date_range(train_dates[-1] + pd.Timedelta(days=1), periods=horizon)
    if not eval_dates.equals(expected):
        raise ValueError(f'Eval must be the next {horizon} days after train')
    for name, frame, dates in [('train', train, train_dates), ('eval', evaluation, eval_dates)]:
        if not frame.groupby(KEYS).size().eq(len(dates)).all():
            raise ValueError(f'{name}: missing daily observations; no implicit zero filling')
    if num_series > len(keys):
        raise ValueError(f'Requested {num_series} series but only {len(keys)} exist')
    indices = np.sort(np.random.default_rng(seed).choice(len(keys), num_series, replace=False))
    selected = keys.iloc[indices].reset_index(drop=True)
    index = pd.MultiIndex.from_frame(selected)

    def panel(frame, dates):
        subset = frame.merge(selected, on=KEYS, how='inner', validate='many_to_one')
        return subset.pivot(index=KEYS, columns='dt', values='sale_amount').reindex(
            index=index, columns=dates).to_numpy(dtype=np.float32)

    return SalesPanel(selected, train_dates, eval_dates, panel(train, train_dates),
                      panel(evaluation, eval_dates), len(keys))
