#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全部 50,000 條銷售序列：單一 TACTiS2 聯合訓練、驗證與預測，再比較 Chronos。

這次只提供程式，未執行此入口。Python 檔案的副檔名是 .py。

移到另一台電腦
---------------
請下載整個專案，保留本檔、comparison/、joint_comparison/、configs/、
TACTiS2/、chronos/、dataset/ 和 requirements-comparison.txt 的相對位置。
本檔是執行入口，不能只複製這一個檔案而省略模型原始碼。
資料使用 Git LFS；git clone 後，在專案目錄執行 git lfs pull 取得實際 parquet。
不要搬 Mac 的 .venv；在目標電腦以 Python 3.10 以上建立新的環境。
NVIDIA 電腦先依 https://pytorch.org/get-started/locally/ 安裝適用的 CUDA PyTorch，
再於該環境執行 python -m pip install -r requirements-comparison.txt。

正式執行（檔名有空白，請保留引號）
--------------------------------
    python "FULLTEST 1030.py" --device cuda
只檢查資料與記憶體、不訓練：
    python "FULLTEST 1030.py" --device cuda --check-only
中斷後續跑（從最近完成的 epoch 繼續，輸出目錄不可更換）：
    python "FULLTEST 1030.py" --device cuda --resume
沒有 NVIDIA CUDA 時可明確選 --device cpu，但不會因此降低記憶體需求。
原版程式目前沒有在此入口提供 MPS 或多 GPU 分散式執行。

實驗定義
---------
* 全部 50,000 條序列，共用一組 TACTiS2 權重；每次更新也使用全部序列。
  不做固定分組，不做訓練序列抽樣（bagging_size=None），沒有縮小測試模式。
* 使用原版 TACTiS2 核心與論文 Table 8 / Table 10 的 traffic 架構與學習率，
  設定來源為 configs/tactis2-paper-traffic.json。這是套用到銷售資料的新實驗，
  不是論文已訓練權重，也不宣稱重現論文分數。
* 論文訓練使用 bagging；本入口依「每次使用全部序列」要求停用它，
  這是明確的實驗差異。時間視窗仍為 7 天歷史預測 7 天，並非把 97 天
  一次用於訓練；eval 的 7 天只能用於最後評估，不能混入訓練。
* train 的前 41 天用於抽取訓練視窗，後 49 天作為 7 個驗證視窗。
  每階段 512 steps/epoch，最多 36 訓練小時、50 epochs 無改善即停止，
  並有 1,000 epochs 的額外上限；不保證收斂。時間視窗 batch_size=1。
* Chronos baseline 固定為原版 amazon/chronos-t5-tiny 的指定權重，不重新訓練。
  它以小批次處理序列，但只載入同一個預訓練模型；這與把 TACTiS2 拆成
  多個獨立模型不同。兩者使用同樣的最後 7 天觀測與 7 天測試日期。
* 每模型取 100 個樣本的中位數，計算 MAE、RMSE、WAPE（比例，乘 100 為 %）。

資源限制與結果
---------------
原版 dense attention 在 50,000 序列 / 7+7 天 / 4 heads / float32 下，
光一個張量就是 3,920,000,000,000 bytes，約 3.92 TB，尚未計入其他張量
與反向傳播。一般桌機或單張 3090 無法容納；此入口不會偷偷改成分組，
會在建立模型前回報記憶體不足並停止。它也不會自動合併多張 GPU 的記憶體。

預設輸出 outputs/fulltest-1030/：
  manifest.json / resource_report.json / status.json：設定、資源與執行狀態。
  training_log.json / last.pt / best_stage1.pt / best_stage2.pt / tactis.pt：訓練與權重。
  tactis2_predictions.csv / chronos_predictions.csv：每模型 350,000 筆預測。
  metrics.csv：兩個模型各一列的 MAE、RMSE、WAPE。
  SUCCESS：只有兩模型皆完成才會出現；資源檢查通過不代表完成實驗。

論文：https://arxiv.org/pdf/2310.01327，Appendix C.2、Table 8、Table 10。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_SERIES = 50_000
CONFIG_PATH = ROOT / "configs" / "tactis2-paper-traffic.json"


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("必須大於 0")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="全部 50,000 序列、一個 TACTiS2、不抽樣或分組；完成後比較 Chronos。"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--train-path", type=Path, default=ROOT / "dataset/train.parquet")
    parser.add_argument("--eval-path", type=Path, default=ROOT / "dataset/eval.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/fulltest-1030")
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="從相同輸出目錄的 checkpoint 繼續")
    parser.add_argument("--check-only", action="store_true", help="只檢查資料和資源，不訓練或預測")
    parser.add_argument("--offline", action="store_true", help="Chronos 已下載完成時才使用")
    args = parser.parse_args()
    if args.resume and args.check_only:
        parser.error("--resume 與 --check-only 請分開使用")
    for name in ("train_path", "eval_path", "output_dir"):
        path = getattr(args, name)
        setattr(args, name, path if path.is_absolute() else ROOT / path)
    args.chronos_model = "amazon/chronos-t5-tiny"
    args.chronos_revision = "29d808298f1a62493e7b9a5e08529d0d930fa189"
    args.chronos_batch_size = 8
    return args


def check_inputs(args):
    required = [CONFIG_PATH, args.train_path, args.eval_path,
                ROOT / "comparison/data.py", ROOT / "comparison/models.py",
                ROOT / "comparison/full.py", ROOT / "joint_comparison/runner.py",
                ROOT / "joint_comparison/resources.py",
                ROOT / "TACTiS2/TACTiS/tactis/model/tactis.py",
                ROOT / "chronos/chronos-forecasting/src/chronos/chronos.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("專案檔案不完整，請下載整個專案：\n" + "\n".join(missing))
    for path in (args.train_path, args.eval_path):
        with path.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise SystemExit(f"{path} 是 Git LFS 指標，請先在專案目錄執行 git lfs pull。")
    packages = ("torch", "numpy", "pandas", "pyarrow", "transformers", "accelerate", "einops", "psutil")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("目前 Python 缺少套件：" + ", ".join(missing) +
                         "\n請在此環境執行 python -m pip install -r requirements-comparison.txt。")
    return packages


def main():
    args = parse_args()
    packages = check_inputs(args)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from comparison.data import load_sales
    from comparison.full import atomic_json, digest, run_lock
    from joint_comparison.resources import blockers, estimate, hardware
    from joint_comparison.runner import run

    torch.set_num_threads(args.threads)
    config = deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    config["name"] = "fulltest_1030_traffic_transfer_without_bagging"
    config["training"]["bagging_size"] = None
    config["provenance"]["fulltest_override"] = (
        "All 50,000 series in every training step; no bagging or separate group models. "
        "This deliberately differs from paper training bagging."
    )
    panel = load_sales(args.train_path, args.eval_path, num_series=None,
                       seed=args.seed, horizon=config["horizon"])
    if len(panel.keys) != EXPECTED_SERIES or panel.total_series != EXPECTED_SERIES:
        raise SystemExit(f"此入口要求全部 {EXPECTED_SERIES:,} 條序列，讀到 {len(panel.keys):,} 條。")
    if panel.train.shape[1] != 90 or panel.actual.shape[1] != 7:
        raise SystemExit("此實驗要求 90 天 train 與接續 7 天 eval，請檢查資料檔。")

    sources = [Path(__file__).resolve(), CONFIG_PATH,
               *sorted((ROOT / "comparison").glob("*.py")),
               *sorted((ROOT / "joint_comparison").glob("*.py")),
               *sorted((ROOT / "TACTiS2/TACTiS/tactis").rglob("*.py")),
               *sorted((ROOT / "chronos/chronos-forecasting/src").rglob("*.py"))]
    identity = dict(
        config=config, scope="ALL_SERIES_JOINT_NO_BAGGING", series=EXPECTED_SERIES,
        seed=args.seed, threads=args.threads, device=args.device,
        chronos=[args.chronos_model, args.chronos_revision, args.chronos_batch_size],
        data={name: digest(getattr(args, name)) for name in ("train_path", "eval_path")},
        source={path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
        versions={name: importlib.metadata.version(name) for name in packages},
    )
    report = estimate(EXPECTED_SERIES, config)
    capacity = hardware(args.device)
    reasons = blockers(report, capacity)
    if args.device == "cuda" and not torch.cuda.is_available():
        reasons.append("CUDA is not available in this Python environment.")
    report.update(hardware=capacity, blockers=reasons, scope=identity["scope"],
                  status="blocked" if reasons else "preflight_passed_not_a_peak_memory_guarantee")
    print("TACTiS2: 50,000 條序列 / 一個模型 / 訓練、驗證、預測皆不分組。", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    # A read-only check does not reserve the normal output directory or leave a
    # misleading SUCCESS file. A later normal run can use the same command path.
    if args.check_only:
        if reasons:
            raise SystemExit("檢查完成：資源不足；沒有建立模型或執行訓練。")
        print("資料與資源初步檢查通過；沒有執行訓練或預測。", flush=True)
        return

    output = args.output_dir
    if args.resume:
        if not (output / "manifest.json").is_file():
            raise SystemExit("續跑目錄沒有 manifest.json；請選擇原本的輸出目錄。")
    elif output.exists():
        raise SystemExit("輸出目錄已存在；續跑請加 --resume，新實驗請指定新的 --output-dir。")
    else:
        output.mkdir(parents=True, exist_ok=False)

    with run_lock(output):
        manifest = output / "manifest.json"
        if args.resume:
            if json.loads(manifest.read_text(encoding="utf-8"))["identity"] != identity:
                raise SystemExit("資料、程式、套件或設定與原實驗不同，不能混用 checkpoint。")
        else:
            cutoff = len(panel.train_dates) - config["validation_windows"] * config["horizon"]
            atomic_json(manifest, dict(
                identity=identity, created_utc=datetime.now(timezone.utc).isoformat(),
                train_dates=[str(panel.train_dates[0]), str(panel.train_dates[cutoff - 1])],
                validation_dates=[str(panel.train_dates[cutoff]), str(panel.train_dates[-1])],
                eval_dates=[str(panel.eval_dates[0]), str(panel.eval_dates[-1])],
                expected_rows_per_model=EXPECTED_SERIES * config["horizon"],
                point_forecast="Median of 100 samples; no clipping; WAPE is a ratio.",
            ))
            panel.keys.to_csv(output / "series.csv", index=False)
        if (output / "SUCCESS").exists():
            print(f"此實驗已完成，結果：{output / 'metrics.csv'}", flush=True)
            return
        atomic_json(output / "resource_report.json", report)
        if reasons:
            atomic_json(output / "status.json", dict(status="blocked_before_training", reasons=reasons))
            raise SystemExit("資源不足，已停止；不會改成分組模型。詳見 resource_report.json。")
        try:
            run(panel, config, args, output, identity)
        except BaseException as exc:
            atomic_json(output / "status.json", dict(status="interrupted_or_failed", error=str(exc)))
            raise


if __name__ == "__main__":
    main()
