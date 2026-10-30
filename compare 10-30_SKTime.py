#!/usr/bin/env python3
"""Run the author's ordinary Naive on the sales panel and compare saved TACTiS2.

From the project virtual environment:
    python "compare 10-30_SKTime.py"

Paths are relative to this file, not the terminal's current directory. Sales
paths live in sktime-m4/scripts/sales_config.py. The historical M4 runner expects
M4 CSVs and obsolete packages; this entry uses its verified Naive definition
with the installed official sktime implementation. TACTiS2 is NOT retrained.
An existing output directory or report is never overwritten. For another run,
pass both --output-dir and --report-path with new names.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SALES_SCRIPTS = ROOT / "sktime-m4" / "scripts"
sys.path.insert(0, str(SALES_SCRIPTS))
import sales_config
from sales_naive import forecast_sales, source_spec
from comparison.data import KEYS, load_sales


def fingerprint(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def display_path(path):
    return Path(os.path.relpath(path, ROOT)).as_posix()


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=sales_config.TRAIN_PATH)
    parser.add_argument("--eval-path", type=Path, default=sales_config.EVAL_PATH)
    parser.add_argument("--tactis-results", type=Path, default=sales_config.TACTIS_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=sales_config.OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=sales_config.REPORT_PATH)
    parser.add_argument("--jobs", type=positive, default=2, help="CPU workers for independent Naive calls")
    args = parser.parse_args()
    for name in ("train_path", "eval_path", "tactis_results", "output_dir", "report_path"):
        path = getattr(args, name)
        setattr(args, name, (path if path.is_absolute() else ROOT / path).resolve())
    return args


def check_reference(args):
    """Fail before forecasting if this is not the same data/preprocessing run."""
    reference = args.tactis_results
    for name in ("SUCCESS", "manifest.json", "metrics.csv", "series.csv"):
        if not (reference / name).is_file():
            raise ValueError(f"Missing completed TACTiS2 reference: {reference / name}")
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    signature = manifest["signature"]
    settings = signature["arguments"]
    if not settings.get("all_series") or manifest["series"] != 50000 or settings["horizon"] != 7:
        raise ValueError("Expected the completed 50,000-series, seven-day TACTiS2 experiment.")
    hashes = {}
    for name in ("train_path", "eval_path"):
        # Original paths may point to another computer; use them only as manifest keys.
        hashes[name] = fingerprint(getattr(args, name))
        if hashes[name] != signature["data"][settings[name]]:
            raise ValueError(f"{name} differs from the data used by TACTiS2.")
    if fingerprint(ROOT / "comparison/data.py") != signature["source"]["comparison/data.py"]:
        raise ValueError("Shared preprocessing changed since the TACTiS2 experiment; review before comparing.")
    return manifest, hashes


def load_reference_predictions(panel, reference, manifest):
    """Recompute TACTiS2 metrics from exact Parquet values, not rounded CSV."""
    from comparison.full import read_result, scores

    settings = manifest["signature"]["arguments"]
    group_size = settings["group_size"]
    if group_size < 1:
        raise ValueError("Invalid reference group size")
    keys = pd.read_csv(reference / "series.csv")
    if not keys[KEYS].equals(panel.keys):
        raise ValueError("TACTiS2 series IDs/order differ from the shared sales panel.")
    if not np.array_equal(keys["group"], np.arange(len(panel.keys)) // group_size):
        raise ValueError("TACTiS2 group assignment mismatch.")
    frames, source_hashes = [], {}
    total = dict(count=0, absolute=0., squared=0., denominator=0.)
    for group, start in enumerate(range(0, len(panel.keys), group_size)):
        path = reference / "groups" / f"{group:05d}" / "tactis2.parquet"
        frame, subtotal = read_result(path, panel, start, min(start + group_size, len(panel.keys)))
        frames.append(frame)
        source_hashes[path.relative_to(reference).as_posix()] = fingerprint(path)
        for key, value in subtotal.items():
            total[key] += value
    if len(frames) != manifest["groups"] or total["count"] != manifest["expected_rows_per_model"]:
        raise ValueError("Incomplete TACTiS2 group/forecast coverage.")
    metrics = scores(total)
    previous = pd.read_csv(reference / "metrics.csv")
    previous = previous.loc[previous.model.eq("tactis2")]
    if len(previous) != 1 or int(previous.iloc[0].observations) != total["count"]:
        raise ValueError("Invalid TACTiS2 reference metrics.")
    for key in ("MAE", "RMSE", "WAPE"):
        if not np.isclose(metrics[key], previous.iloc[0][key], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Recomputed TACTiS2 {key} does not match the saved result.")
    return pd.concat(frames, ignore_index=True), metrics, source_hashes


def naive_chunk(values, context, horizon):
    # Only silence this known API-transition warning; keep actual data/model errors.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The default of config ``remember_data`` will change",
            category=FutureWarning, module=r"sktime\.forecasting\.base\._base",
        )
        return forecast_sales(values, context_length=context, horizon=horizon)


def run_naive(values, context, horizon, jobs, output):
    """Every series goes through sktime; chunks only schedule independent calls."""
    from joblib import Parallel, delayed, parallel_config

    total = len(values)
    started = time.perf_counter()
    starts = list(range(0, total, 1000))
    history = values[:, -context:].copy()
    predictions = np.empty((total, horizon), dtype=np.float64)
    with parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=1):
        chunks = Parallel(return_as="generator")(
            delayed(naive_chunk)(history[start:start + 1000], context, horizon) for start in starts
        )
        for start, chunk in zip(starts, chunks):
            stop = start + len(chunk)
            predictions[start:stop] = chunk
            elapsed = time.perf_counter() - started
            print(f"Naive {stop:,}/{total:,} series; {elapsed:.1f} seconds", flush=True)
            write_json(output / "progress.json", dict(status="running_naive", completed_series=stop,
                        total_series=total, elapsed_seconds=elapsed))
    return predictions, time.perf_counter() - started


def make_report(panel, reference, metrics, wins, run_info, args):
    settings = reference["signature"]["arguments"]
    rows = metrics.set_index("model")
    tactis, naive = rows.loc["tactis2_grouped"], rows.loc["naive_sktime"]
    labels = {"tactis2_grouped": "TACTiS2（既有分組小型模型）", "naive_sktime": "Naïve（sktime，本次執行）"}
    table = ["| 模型 | MAE ↓ | RMSE ↓ | WAPE ↓ | 預測筆數 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in rows.iterrows():
        table.append(f"| {labels[name]} | {row.MAE:.12f} | {row.RMSE:.12f} | {row.WAPE * 100:.10f}% | {int(row.observations):,} |")
    comparisons = []
    for key in ("MAE", "RMSE", "WAPE"):
        a, b = float(tactis[key]), float(naive[key])
        if np.isclose(a, b, rtol=0, atol=1e-12):
            comparisons.append(f"- {key}：兩者相同（容差 1e-12）。")
        else:
            winner = "TACTiS2" if a < b else "Naïve"
            change = (b - a) / b * 100 if b else None
            detail = (f"；TACTiS2 相對 Naïve 的誤差{'降低' if change > 0 else '增加'} {abs(change):.4f}%"
                      if change is not None else "；Naïve 誤差為零，不計相對百分比")
            comparisons.append(f"- {key}：**{winner} 較低**{detail}。")
    date = lambda value: str(value.date())
    output = display_path(args.output_dir)
    return f"""# compare 10-30_SKTime：TACTiS2 與 Naïve

執行時間：{run_info['created_at']}（Asia/Taipei）。

本次對原銷售資料的 **50,000 條序列全部執行 sktime Naïve**，每條預測 7 天，
共 350,000 筆。TACTiS2 使用先前已完成的結果，逐筆核對後重新計算指標，**本次沒有重新訓練 TACTiS2**。

## 實際比較結果

{chr(10).join(table)}

三個指標都是越低越好。Markdown 顯示四捨五入數值；完整浮點數保存在 [{output}/metrics.csv]({output}/metrics.csv)。

{chr(10).join(comparisons)}

以每條序列七天的 MAE 比較：TACTiS2 較低 {wins['tactis2']:,} 條，
Naïve 較低 {wins['naive']:,} 條，相同 {wins['ties']:,} 條（容差 1e-12）。
整體指標由全部 350,000 筆誤差統計，不是把各組 RMSE 或 WAPE 直接平均。

## 使用哪份資料

- 原始訓練檔：`{display_path(args.train_path)}`，{date(panel.train_dates[0])}～{date(panel.train_dates[-1])}，90 天。
- 原始評估檔：`{display_path(args.eval_path)}`，{date(panel.eval_dates[0])}～{date(panel.eval_dates[-1])}，7 天。
- 序列鍵：`city_id / store_id / product_id`；預測目標：`sale_amount`。
- 兩檔 SHA256 均與先前 TACTiS2 manifest 相同；沒有另一份被改寫銷售數值的訓練檔。
- 共用 `comparison/data.py`，先檢查缺值、重複、日期連續性，再重整成「序列 × 日」並轉成 float32。
- 兩者最終輸入均取最後 {settings['context_length']} 天：{date(panel.train_dates[-settings['context_length']])}～{date(panel.train_dates[-1])}。
  普通 Naïve 實際只依賴最後一天；未使用 eval 真值來訓練或挑選參數。
- TACTiS2 的標準化是在模型內進行，輸出已還原銷售單位；本比較不再額外標準化。

## Naïve 來源與路徑調整

- 作者配套程式：[mloning/sktime-m4](https://github.com/mloning/sktime-m4)。
- 論文：Markus Löning、Franz J. Király（2020），
  [Forecasting with sktime: Designing sktime's New Forecasting API and Applying It to Replicate and Extend the M4 Study](https://arxiv.org/abs/2005.08067)。
- 原始 `sktime-m4/scripts/construct.py` 的普通 Naïve 定義為 `NaiveForecaster(strategy="last")`。
- 本次透過 `scripts/sales_naive.py`，使用官方 sktime **{run_info['packages']['sktime']}** 的
  `NaiveForecaster(strategy="last", sp=1)`；每條序列確實呼叫 `fit()` 和 `predict()`。
  `fit()` 只記錄歷史，不做梯度訓練；未來七天都預測為 {date(panel.train_dates[-1])} 的數值。
- 原始 M4 runner 要求 M4 metadata／CSV、sktime 0.4.0 與舊 API，因此不能只替換路徑就讀 parquet。
  本次沿用作者模型規則並移植資料介面，**沒有宣稱重現 2020 年套件環境或 M4 論文成績**。
- 銷售實驗的輸入、既有 TACTiS2 結果、輸出路徑統一由
  [`sktime-m4/scripts/sales_config.py`](sktime-m4/scripts/sales_config.py) 設定，依專案位置解析，沒有寫死本機帳號。
  銷售入口不使用原本僅供 M4 的 `scripts/config.py`。

## TACTiS2 的實際設定與限制

- 預測來源：`{display_path(args.tactis_results)}/groups/*/tactis2.parquet`。
- 依城市／店鋪／商品排序，每 {settings['group_size']} 條建立一個獨立小型 TACTiS2，
  共 {reference['groups']:,} 組，最後一組 16 條；分組可能跨店鋪。
- 每個訓練階段 {settings['epochs']} epochs，每 epoch {settings['steps_per_epoch']} steps；
  使用 {settings['num_samples']} 個預測樣本的中位數，不裁切負值。
- 2024-03-28～06-18 抽取訓練視窗；06-19～06-25 作驗證。
- 這是短訓練、小型分組模型的結果，不代表原論文完整設定，也不代表已充分收斂。
- 本報告只評估點預測，不能用這些指標直接證明 copula 學好了跨商品相依關係。
- 只有一個七天測試區間，不能把本次排名推廣到所有資料或所有時間。
- 原 Chronos 比較結果保留於 [FULL_COMPARISON.md](FULL_COMPARISON.md)，本次未重跑或移除。

## 驗證與輸出

- 全部原始資料雜湊與前次實驗一致，共用資料載入程式也與前次相同。
- {reference['groups']:,} 個 TACTiS2 預測分組皆核對 ID、日期、真值、有限數值與完整覆蓋。
- TACTiS2 重新計算的 MAE／RMSE／WAPE 與既有 metrics.csv 一致。
- Naïve 的全部輸出皆與「最後觀測值延續七天」獨立核對相同。
- 新輸出的兩模型預測與同一個真值欄並列，每個「序列／日期」恰好一筆。
- 本次 Naïve 執行耗時 {run_info['naive_seconds']:.3f} 秒（含工作程序啟動，{args.jobs} 個 CPU workers）。
  TACTiS2 是重用結果，因此本報告不作訓練速度比較。

| 檔案 | 用途 |
| --- | --- |
| [{output}/metrics.csv]({output}/metrics.csv) | 兩模型整體指標；WAPE 為比例，WAPE_percent 為百分比 |
| [{output}/predictions.parquet]({output}/predictions.parquet) | 350,000 筆真值與兩模型預測，保留浮點精度 |
| [{output}/predictions.csv]({output}/predictions.csv) | 可直接開啟的相同預測表 |
| [{output}/per_day_metrics.csv]({output}/per_day_metrics.csv) | 每天各模型的指標 |
| [{output}/manifest.json]({output}/manifest.json) | 來源、設定、套件版本、資料與預測檔案雜湊 |
| [{output}/verification.json]({output}/verification.json) | 本次驗證摘要 |

## 重新執行

先啟用專案 `.venv`，確認已安裝 `requirements-comparison.txt`，執行：

```bash
python "compare 10-30_SKTime.py"
```

既有輸出與報告不會被覆蓋。再次執行時改用新的輸出名稱，例如：

```bash
python "compare 10-30_SKTime.py" --output-dir outputs/compare-10-30_SKTime-rerun --report-path "compare 10-30_SKTime-rerun.md"
```

這支入口需要本機已有 TACTiS2 的完整結果；若從 GitHub 換到另一台電腦，
`outputs/` 目前被忽略，必須另行帶過去。也要保留 `sktime-m4/scripts/sales_config.py`、
`sales_naive.py` 與原作者 `construct.py`；只複製入口檔不足以執行。
"""


def main():
    args = arguments()
    if args.output_dir.exists() or args.report_path.exists():
        raise SystemExit("輸出或報告已存在；請用 --output-dir 與 --report-path 指定新名稱。")
    started = time.perf_counter()
    created_at = datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")
    reference, hashes = check_reference(args)
    settings = reference["signature"]["arguments"]
    author = source_spec()
    panel = load_sales(args.train_path, args.eval_path, num_series=None,
                       seed=settings["seed"], horizon=settings["horizon"])
    if len(panel.keys) != reference["series"] or len(panel.train_dates) != 90:
        raise ValueError("Panel dimensions differ from the original experiment.")
    if [str(panel.eval_dates[0]), str(panel.eval_dates[-1])] != reference["eval_dates"]:
        raise ValueError("Evaluation dates differ from TACTiS2.")
    print("Data hashes match; all 50,000 series loaded. Verifying saved TACTiS2 forecasts...", flush=True)
    frame, tactis_metrics, prediction_hashes = load_reference_predictions(panel, args.tactis_results, reference)
    print(f"TACTiS2 reference verified: {tactis_metrics}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from comparison.full import run_lock, scores, score_sums

    with run_lock(args.output_dir):
        try:
            predictions, naive_seconds = run_naive(panel.train, settings["context_length"],
                                                   settings["horizon"], args.jobs, args.output_dir)
            expected = np.repeat(panel.train[:, -1:], settings["horizon"], axis=1)
            if not np.array_equal(predictions, expected):
                raise ValueError("Full-panel Naive forecast verification failed.")
            naive_metrics = scores(score_sums(panel.actual, predictions))
            frame = frame.rename(columns={"prediction": "tactis2_prediction"})
            frame["naive_prediction"] = predictions.ravel()
            if frame.duplicated(KEYS + ["dt"]).any() or len(frame) != 350000:
                raise ValueError("Prediction coverage/uniqueness failure.")
            metrics = pd.DataFrame([
                dict(model="tactis2_grouped", **tactis_metrics),
                dict(model="naive_sktime", **naive_metrics),
            ])
            metrics["WAPE_percent"] = metrics.WAPE * 100
            tactis_error = np.abs(frame.tactis2_prediction.to_numpy(dtype=np.float64).reshape(panel.actual.shape)
                                  - panel.actual.astype(np.float64)).mean(axis=1)
            naive_error = np.abs(predictions - panel.actual.astype(np.float64)).mean(axis=1)
            ties = np.isclose(tactis_error, naive_error, rtol=0, atol=1e-12)
            wins = dict(tactis2=int(((tactis_error < naive_error) & ~ties).sum()),
                        naive=int(((naive_error < tactis_error) & ~ties).sum()), ties=int(ties.sum()))
            day_rows = []
            for day, group in frame.groupby("dt", sort=True):
                for model, column in (("tactis2_grouped", "tactis2_prediction"), ("naive_sktime", "naive_prediction")):
                    day_rows.append(dict(dt=str(day.date()), model=model, **scores(score_sums(group.actual, group[column]))))
            metrics.to_csv(args.output_dir / "metrics.csv", index=False, float_format="%.17g")
            frame.to_parquet(args.output_dir / "predictions.parquet", index=False)
            frame.to_csv(args.output_dir / "predictions.csv", index=False, float_format="%.17g")
            pd.DataFrame(day_rows).to_csv(args.output_dir / "per_day_metrics.csv", index=False, float_format="%.17g")
            panel.keys.to_csv(args.output_dir / "series.csv", index=False)
            from sktime.forecasting.naive import NaiveForecaster
            source_paths = [Path(__file__), ROOT / "comparison/data.py", ROOT / "comparison/full.py",
                            SALES_SCRIPTS / "sales_config.py", SALES_SCRIPTS / "sales_naive.py", SALES_SCRIPTS / "construct.py"]
            run_info = dict(
                created_at=created_at, series=len(panel.keys), observations_per_model=len(frame),
                data_sha256=hashes, naive_source=author, naive_seconds=naive_seconds,
                total_seconds=time.perf_counter() - started, jobs=args.jobs,
                packages={name: importlib.metadata.version(name) for name in
                          ("sktime", "numpy", "pandas", "pyarrow", "scikit-learn", "joblib")},
                source_sha256={display_path(path): fingerprint(path) for path in source_paths},
                installed_naive_source_sha256=fingerprint(inspect.getfile(NaiveForecaster)),
                tactis_retrained=False, tactis_reference_directory=display_path(args.tactis_results),
                tactis_manifest=reference, tactis_prediction_sha256=prediction_hashes,
                output_sha256={name: fingerprint(args.output_dir / name) for name in
                               ("predictions.parquet", "metrics.csv", "per_day_metrics.csv")},
                series_mae_comparison=wins,
            )
            write_json(args.output_dir / "manifest.json", run_info)
            write_json(args.output_dir / "verification.json", dict(
                passed=True, series=50000, rows_per_model=350000, groups_verified=reference["groups"],
                checks=["raw data SHA256 match", "shared preprocessing SHA256 match", "series IDs and dates aligned",
                        "all TACTiS2 group actuals match eval", "TACTiS2 metrics recomputed and matched",
                        "real sktime fit/predict for all series", "all Naive values match last observation",
                        "350000 unique series/date rows", "finite predictions"],
            ))
            report = make_report(panel, reference, metrics, wins, run_info, args)
            args.report_path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation preserves a report created by another process/user.
            with args.report_path.open("x", encoding="utf-8") as stream:
                stream.write(report)
            write_json(args.output_dir / "progress.json", dict(status="complete", completed_series=50000,
                       total_series=50000, elapsed_seconds=time.perf_counter() - started))
            (args.output_dir / "SUCCESS").write_text("Naive executed; saved TACTiS2 forecasts verified. 50000 series / 350000 forecasts per model.\n", encoding="utf-8")
            print(metrics.to_string(index=False), flush=True)
            print(f"Report: {args.report_path}", flush=True)
        except BaseException as exc:
            write_json(args.output_dir / "status.json", dict(status="interrupted_or_failed", error=str(exc)))
            raise


if __name__ == "__main__":
    main()
