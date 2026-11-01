"""Sales adapter for the author's ordinary Naive baseline.

The author's scripts/construct.py selects NaiveForecaster(strategy="last").
This adapter uses that same forecasting rule through the modern official sktime
API, explicitly setting sp=1. It does not import the obsolete dependencies of
the 2020 M4 runner, and does not claim to reproduce that runner's environment.
Every sales series is actually fitted and predicted by sktime; the last-value
formula below is an independent output check, not the forecasting implementation.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


MODEL_SPEC = {"strategy": "last", "sp": 1}
SOURCE_PATH = Path(__file__).resolve().with_name("construct.py")
SOURCE_REPOSITORY = "https://github.com/mloning/sktime-m4"
SOURCE_PAPER = "https://arxiv.org/abs/2005.08067"


def source_spec() -> dict:
    """Verify the downloaded author's model definition without importing it."""
    source = SOURCE_PATH.read_bytes()
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    matches = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "construct_M4_forecasters":
            continue
        for item in ast.walk(node):
            if not isinstance(item, ast.Dict):
                continue
            for key, value in zip(item.keys, item.values):
                if isinstance(key, ast.Constant) and key.value == "Naive":
                    matches.append(value)
    if len(matches) != 1:
        raise ValueError("Cannot uniquely identify the author's ordinary Naive model.")
    call = matches[0]
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
            or call.func.id != "NaiveForecaster" or call.args
            or len(call.keywords) != 1 or call.keywords[0].arg != "strategy"
            or not isinstance(call.keywords[0].value, ast.Constant)
            or call.keywords[0].value.value != "last"):
        raise ValueError("The author's Naive definition changed; review the sales adapter.")
    return {
        "repository": SOURCE_REPOSITORY,
        "paper": SOURCE_PAPER,
        "source_file": "sktime-m4/scripts/construct.py",
        "source_line": call.lineno,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "original_model": 'NaiveForecaster(strategy="last")',
        "adapted_model": "sktime.forecasting.naive.NaiveForecaster",
        "parameters": MODEL_SPEC.copy(),
        "port_note": "Same ordinary last-value rule through modern sktime; not the original 0.4.0 environment.",
    }


def forecast_sales(
    values: np.ndarray,
    context_length: int = 28,
    horizon: int = 7,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Return [series, horizon] forecasts from actual sktime fit/predict calls.

    The caller provides training history only. No evaluation observations or
    TACTiS2 predictions are accepted by this function. Progress is reported every
    1,000 series and at completion. Naive fits no optimized parameters; fit
    records the history whose final observation determines every future value.
    """
    from sktime.forecasting.naive import NaiveForecaster

    source_spec()
    if (not isinstance(context_length, int) or isinstance(context_length, bool)
            or context_length < 1 or not isinstance(horizon, int)
            or isinstance(horizon, bool) or horizon < 1):
        raise ValueError("context_length and horizon must be positive integers.")
    data = np.asarray(values)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < context_length:
        raise ValueError("Expected nonempty [series, days] history with enough context.")
    history = data[:, -context_length:].astype(np.float64, copy=False)
    if not np.isfinite(history).all():
        raise ValueError("Naive context contains nonfinite values.")
    total = history.shape[0]
    predictions = np.empty((total, horizon), dtype=np.float64)
    training_index = pd.RangeIndex(context_length)
    expected_index = pd.RangeIndex(context_length, context_length + horizon)
    relative_horizon = np.arange(1, horizon + 1)
    for index in range(total):
        model = NaiveForecaster(**MODEL_SPEC)
        model.fit(pd.Series(history[index], index=training_index), fh=relative_horizon)
        forecast = model.predict()
        if not isinstance(forecast, pd.Series) or not forecast.index.equals(expected_index):
            raise ValueError(f"Unexpected Naive forecast dates for series {index}.")
        result = forecast.to_numpy(dtype=np.float64)
        if result.shape != (horizon,) or not np.isfinite(result).all():
            raise ValueError(f"Invalid Naive forecast values for series {index}.")
        if not np.array_equal(result, np.full(horizon, history[index, -1])):
            raise ValueError(f"Naive last-value invariant failed for series {index}.")
        predictions[index] = result
        completed = index + 1
        if progress is not None and (completed % 1000 == 0 or completed == total):
            progress(completed, total)
    return predictions
