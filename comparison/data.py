"""Strict daily panels shared by all three forecasting models."""
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


def load_sales(train_path, eval_path, num_series=5000, seed=1102, horizon=7):
    """Scan Parquet in bounded batches; num_series=None keeps every series.

    Sampling uses sorted TRAIN IDs only, without inspecting evaluation targets.
    No individual observations are sampled and no missing dates are imputed.
    """
    import pyarrow.parquet as pq

    def all_keys(path):
        found = set()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=KEYS):
            frame = batch.to_pandas()
            if frame.isna().any().any():
                raise ValueError(f"{path}: null series IDs")
            found.update(frame.drop_duplicates().itertuples(index=False, name=None))
        if not found:
            raise ValueError(f"{path}: empty data")
        return pd.DataFrame(sorted(found), columns=KEYS)

    for path in (train_path, eval_path):
        with Path(path).open("rb") as stream:
            if stream.read(4) != b"PAR1":
                raise ValueError(f"{path}: not a Parquet file; run git lfs pull to download the dataset")
    if (num_series is not None and num_series < 1) or horizon < 1:
        raise ValueError("num_series and horizon must be positive")
    keys = all_keys(train_path)
    if num_series is not None and num_series > len(keys):
        raise ValueError(f"Requested {num_series} series, only {len(keys)} available")
    if num_series is None:
        selected = keys.copy()  # Full mode never samples or silently caps the panel.
    else:
        indices = np.sort(np.random.default_rng(seed).choice(len(keys), num_series, replace=False))
        selected = keys.iloc[indices].reset_index(drop=True)
    if not keys.equals(all_keys(eval_path)):
        raise ValueError("Train/eval series keys differ")

    def selected_rows(path):
        pieces = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=KEYS + ["dt", "sale_amount"]):
            part = batch.to_pandas().merge(selected, on=KEYS, how="inner", validate="many_to_one")
            if len(part):
                pieces.append(part)
        if not pieces:
            raise ValueError(f"{path}: no selected observations")
        frame = pd.concat(pieces, ignore_index=True)
        if frame.isna().any().any() or frame.duplicated(KEYS + ["dt"]).any():
            raise ValueError(f"{path}: null data or duplicate series/date observations")
        frame["dt"] = pd.to_datetime(frame["dt"], errors="raise")
        if not frame.dt.eq(frame.dt.dt.normalize()).all():
            raise ValueError(f"{path}: timestamps must be daily dates")
        # Check again after parsing in case different strings represented one day.
        if frame.duplicated(KEYS + ["dt"]).any():
            raise ValueError(f"{path}: duplicate parsed series/date observations")
        if not np.isfinite(frame.sale_amount.to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: nonfinite sales values")
        return frame

    train, evaluation = selected_rows(train_path), selected_rows(eval_path)
    train_dates = pd.DatetimeIndex(sorted(train.dt.unique()))
    eval_dates = pd.DatetimeIndex(sorted(evaluation.dt.unique()))
    if not train_dates.equals(pd.date_range(train_dates[0], train_dates[-1], freq="D")):
        raise ValueError("Training dates are not consecutive")
    if not eval_dates.equals(pd.date_range(train_dates[-1] + pd.Timedelta(days=1), periods=horizon)):
        raise ValueError(f"Evaluation must cover the next {horizon} days after training")
    index = pd.MultiIndex.from_frame(selected)

    def matrix(frame, dates):
        if not frame.groupby(KEYS).size().reindex(index, fill_value=0).eq(len(dates)).all():
            raise ValueError("missing daily observations for selected series; no implicit filling")
        values = frame.pivot(index=KEYS, columns="dt", values="sale_amount").reindex(
            index=index, columns=dates).to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Panel contains missing/nonfinite values after float32 conversion")
        return values

    return SalesPanel(selected, train_dates, eval_dates, matrix(train, train_dates),
                      matrix(evaluation, eval_dates), len(keys))
