# -*- coding: utf-8 -*-
"""
TACTiS2（主要模型） vs. Chronos / sktime-m4（baseline）— 全部 50,000 條序列一次執行
====================================================================================

專案：https://github.com/yyyrui/2026final-project
資料：dataset/train.parquet（2024-03-28 ~ 06-25，90 天）、dataset/eval.parquet（06-26 ~ 07-02，7 天）
      50,000 條序列（city_id, store_id, product_id），目標 sale_amount（日銷量）

在 Colab 上的用法
-----------------
1. 執行階段 → 變更執行階段類型 → GPU（A100 / L4 最佳，T4 可跑但 TACTiS2 抽樣較慢）。
2. 把整份檔案貼進一個 cell 執行（或上傳後在 cell 執行 `%run tactis2_full_colab.py`）。
   - 以 cell 執行時會掛載 Google Drive，所有輸出與 checkpoint 存在 Drive，
     斷線後重新執行同一個 cell 會自動從中斷處接續。
3. 需要調整的設定都在下方「設定」區塊。

「不切小塊」的做法
-------------------
* TACTiS2：50,000 條序列共用「一個」模型（一組權重、50,000 個 series embedding），
  不再像舊版分成 1,563 個 32 條的獨立小模型。
  - 訓練：照論文做法，每個 optimizer step 隨機抽一組序列（bagging）。這只是隨機梯度的
    minibatch，資料集本身沒被切開，每條序列都會被抽到、都更新同一個模型。
    （論文 / 原始碼的 copula 是 dense attention，50,000 條同時算一次 loss 需要約 3.9 TB 記憶體，
    舊版 FULLTEST 1030.py 就是因此被擋下來；bagging 是 TACTiS2 原作者處理高維資料的方式。）
  - 驗證：stage 1（marginal flow）直接用全部 50,000 條序列算；stage 2（copula）用固定種子的
    多組 bag 算（dense copula 無法一次算 50,000 條）。
  - 預測：一次對全部 50,000 條序列 × 7 天 = 350,000 個變數做「聯合」抽樣，copula 看得到所有序列。
* Chronos：同一個預訓練模型對 50,000 條序列推論（GPU 批次只是運算上的 batch）。
* sktime-m4：M4 統計 benchmark，逐條序列 fit / predict（本來就是單變量方法），多核心平行。

時間標籤的判斷（程式會實際檢查並輸出 data_audit.json）
------------------------------------------------------
* dt（日期）：必要。檢查可解析、皆為整日、每條序列每天恰一筆、日期連續、eval 緊接在 train 之後。
  實際資料完全乾淨，不需補值或刪除。
* hours_sale（24 小時銷量）：每列 24 小時加總與 sale_amount 完全相等（誤差 < 1e-14），
  屬於重複資訊；且有大量凌晨 0–5 點的銷量。對日預測不必要 → 不使用。
* hours_stock_status / stock_hour6_22_cnt（逐小時缺貨標記）：夜間（非營業時段）約 40% 被標成缺貨，
  且約 77 萬個「缺貨小時」仍有銷量、2.5 萬天「24 小時全缺貨」卻有銷量 → 標籤互相矛盾，不使用。
* holiday_flag：2024-04-28（日）、2024-05-11（六）是中國官方調休「補班日」，資料卻標為假日 → 有誤。
  三個模型都只用目標值本身，所以不使用（只在 audit 中回報）。
  → 結論：唯一必要的時間標籤是 dt；模型輸入只有 (序列 ID, dt, sale_amount)。
"""

# %% ============================== 設定 ==============================
REPO_URL = "https://github.com/yyyrui/2026final-project.git"
REPO_DIR = "/content/2026final-project"
RUN_NAME = "full50k_tactis2_vs_baselines_v1"   # 換新實驗請改名；同名會接續舊進度
USE_DRIVE = True                                # 以 cell 執行時把輸出存到 Google Drive（可續跑）
SEED = 42

HORIZON = 7                 # 預測 06-26 ~ 07-02
VALIDATION_DAYS = 7         # train 最後 7 天（06-19 ~ 06-25）只做驗證 / 挑權重
CLIP_NEGATIVE = True        # 銷量不可能為負：所有模型的樣本 / 預測都截在 0
BASELINE_CONTEXT = None     # Chronos / sktime 使用的歷史長度；None = 全部 90 天

# ---- TACTiS2 ----
# 架構與學習率取自 repo 的 configs/tactis2-paper-traffic.json（論文 Table 8/10 traffic 設定）
TACTIS_CONTEXT = 7          # 與 repo config 相同；copula 聯合抽樣成本約與 (context+horizon) 成正比
TACTIS_STD_FLOOR = 0.1      # 標準化的最小標準差（原始碼為 1e-8，整週為 0 的序列會把未來值放大到 1e7 → 數值爆炸）
TACTIS_NUM_SAMPLES = 100    # 預測樣本數（論文 100）；T4 上可改 50 以節省時間
TACTIS_SAMPLES_PER_CALL = None   # None = 依 GPU 記憶體自動決定
TACTIS_STAGES = {
    1: dict(lr=2.5e-3, weight_decay=0.0, clip=1000.0,  bag=256, batch=32,
            steps_per_epoch=200, max_epochs=200, patience=10, max_hours=1.5),
    2: dict(lr=6.7e-4, weight_decay=0.0, clip=10000.0, bag=20,  batch=32,
            steps_per_epoch=200, max_epochs=200, patience=10, max_hours=1.5),
}
STAGE2_VAL_BAGS = 64        # stage 2 每次驗證呼叫的固定 bag 數
STAGE2_VAL_CALLS = 8        # 64 × 8 = 512 個 bag（約 10,240 條序列）
TACTIS_MODEL = {
    "flow_series_embedding_dim": 16,
    "copula_series_embedding_dim": 64,
    "flow_input_encoder_layers": 7,
    "copula_input_encoder_layers": 5,
    "input_encoding_normalization": True,
    "data_normalization": "standardization",
    "loss_normalization": "both",
    "positional_encoding": {"dropout": 0.0},
    "flow_temporal_encoder": {"attention_layers": 4, "attention_heads": 4, "attention_dim": 8,
                              "attention_feedforward_dim": 8, "dropout": 0.0},
    "copula_temporal_encoder": {"attention_layers": 4, "attention_heads": 4, "attention_dim": 8,
                                "attention_feedforward_dim": 8, "dropout": 0.0},
    "copula_decoder": {
        "min_u": 0.05, "max_u": 0.95,
        "attentional_copula": {"attention_heads": 4, "attention_layers": 2, "attention_dim": 48,
                               "mlp_layers": 6, "mlp_dim": 16, "resolution": 100,
                               "attention_mlp_class": "_simple_linear_projection",
                               "dropout": 0.0, "activation_function": "relu"},
        "dsf_marginal": {"mlp_layers": 6, "mlp_dim": 16, "flow_layers": 4, "flow_hid_dim": 16},
    },
    "skip_copula": True,
    "experiment_mode": "forecasting",
}

# ---- Chronos（原版 Chronos，與 repo 先前比較相同的權重）----
CHRONOS_MODEL = "amazon/chronos-t5-tiny"
CHRONOS_REVISION = "29d808298f1a62493e7b9a5e08529d0d930fa189"   # 換其他模型時設 None
CHRONOS_NUM_SAMPLES = 100
CHRONOS_BATCH = 256

# ---- sktime-m4（Löning & Király 2020 的 M4 benchmark，見 sktime-m4/scripts/construct.py）----
SKTIME_METHODS = ["Naive", "sNaive", "Naive2", "SES", "Holt", "Damped", "Theta", "Com"]
SKTIME_SP = 7               # 日資料的週季節
SKTIME_BLOCK = 250          # 平行運算時每個工作的序列數（不影響結果）

# %% ============================== 環境 ==============================
import importlib.util
import json
import os
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def install_packages():
    need = []
    if importlib.util.find_spec("sktime") is None:
        need.append("sktime==1.2.0")
    for module, spec in [("statsmodels", "statsmodels"), ("einops", "einops>=0.7,<1"),
                         ("transformers", "transformers>=4.41,<6"), ("accelerate", "accelerate>=1.1,<2"),
                         ("pyarrow", "pyarrow>=14")]:
        if importlib.util.find_spec(module) is None:
            need.append(spec)
    if need:
        quoted = " ".join(f'"{p}"' for p in need)
        if sh(f"{sys.executable} -m pip install -q {quoted}", check=False).returncode != 0:
            # sktime 1.2.0 與當下 Colab 套件衝突時，改裝相容的最新版
            sh(f"{sys.executable} -m pip install -q " + quoted.replace('"sktime==1.2.0"', '"sktime"'))


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(64).startswith(b"version https://git-lfs")


def fetch_repo():
    repo = Path(REPO_DIR)
    if not (repo / ".git").exists():
        sh("git lfs version >/dev/null 2>&1 || (apt-get -qq update && apt-get -qq install -y git-lfs)", check=False)
        sh("git lfs install --skip-repo", check=False)
        sh(f"git clone --depth 1 {REPO_URL} {repo}")
    for name in ("train.parquet", "eval.parquet"):
        path = repo / "dataset" / name
        if is_lfs_pointer(path):
            sh(f"cd {repo} && git lfs pull --include='dataset/*.parquet'", check=False)
        if is_lfs_pointer(path):
            import urllib.request
            url = f"https://media.githubusercontent.com/media/yyyrui/2026final-project/main/dataset/{name}"
            print("下載 LFS 檔案：", url, flush=True)
            urllib.request.urlretrieve(url, path)
        if is_lfs_pointer(path):
            raise SystemExit(f"{path} 仍是 Git LFS 指標檔，無法取得實際資料。")
    return repo


def output_dir():
    base = Path("/content/outputs")
    if USE_DRIVE:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            base = Path("/content/drive/MyDrive/2026final-project-outputs")
        except Exception as exc:  # 用 !python 執行時無法掛載 Drive
            print("未掛載 Google Drive，輸出存在 /content/outputs：", exc, flush=True)
    out = base / RUN_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def save_npy(path, array):
    tmp = Path(str(path) + ".tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


install_packages()
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

KEYS = ["city_id", "store_id", "product_id"]

# %% ============================== 資料與時間標籤檢查 ==============================
# 2024 中國官方放假 / 調休（只列資料期間 03-28 ~ 07-02）
CN_2024_HOLIDAYS = {"2024-04-04", "2024-04-05", "2024-04-06", "2024-05-01", "2024-05-02", "2024-05-03",
                    "2024-05-04", "2024-05-05", "2024-06-08", "2024-06-09", "2024-06-10"}
CN_2024_MAKEUP_WORKDAYS = {"2024-04-07", "2024-04-28", "2024-05-11"}


def read_daily(path):
    frame = pd.read_parquet(path, columns=KEYS + ["dt", "sale_amount"])
    frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce")
    report = {"rows": len(frame), "unparseable_dt": int(frame["dt"].isna().sum())}
    if report["unparseable_dt"]:
        raise ValueError(f"{path}: {report['unparseable_dt']} 筆 dt 無法解析")
    report["dt_with_time_of_day"] = int((frame["dt"] != frame["dt"].dt.normalize()).sum())
    frame["dt"] = frame["dt"].dt.normalize()     # 日資料只需要日期；時分秒（若有）不具意義
    if frame[KEYS + ["sale_amount"]].isna().any().any():
        raise ValueError(f"{path}: 序列 ID 或 sale_amount 有缺值")
    if not np.isfinite(frame["sale_amount"].to_numpy(float)).all():
        raise ValueError(f"{path}: sale_amount 有非有限值")
    report["negative_sales"] = int((frame["sale_amount"] < 0).sum())
    report["duplicate_series_day"] = int(frame.duplicated(KEYS + ["dt"]).sum())
    if report["duplicate_series_day"]:
        raise ValueError(f"{path}: 同一序列同一天有多筆，需先決定如何合併")
    return frame, report


def audit_hourly_labels(path, batch_rows=500_000):
    """逐批讀 24 小時欄位，只做統計，不進模型。"""
    pf = pq.ParquetFile(path)
    cols = ["dt", "sale_amount", "hours_sale", "hours_stock_status", "stock_hour6_22_cnt", "holiday_flag"]
    stats = dict(rows=0, bad_hour_length=0, hours_sum_mismatch=0, rows_with_sales_00_05=0,
                 sales_hours_total=0, sales_hours_marked_stockout=0, sales_on_24h_stockout_day=0,
                 stock_count_mismatch=0)
    stockout_by_hour = np.zeros(24)
    holiday = {}
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        hs_col, st_col = batch.column("hours_sale"), batch.column("hours_stock_status")
        ok = (np.asarray(hs_col.value_lengths()) == 24) & (np.asarray(st_col.value_lengths()) == 24)
        stats["bad_hour_length"] += int((~ok).sum())
        if not ok.all():
            continue  # 長度不是 24 的情況本資料不存在；若出現只計數
        hs = hs_col.flatten().to_numpy().reshape(-1, 24)
        st = st_col.flatten().to_numpy().reshape(-1, 24)
        daily = batch.column("sale_amount").to_numpy()
        stats["rows"] += len(daily)
        stats["hours_sum_mismatch"] += int((np.abs(hs.sum(1) - daily) > 1e-6).sum())
        stats["rows_with_sales_00_05"] += int((hs[:, :6].sum(1) > 0).sum())
        stats["sales_hours_total"] += int((hs > 0).sum())
        stats["sales_hours_marked_stockout"] += int(((hs > 0) & (st == 1)).sum())
        stats["sales_on_24h_stockout_day"] += int(((daily > 0) & (st.sum(1) == 24)).sum())
        stats["stock_count_mismatch"] += int((st[:, 6:22].sum(1) != batch.column("stock_hour6_22_cnt").to_numpy()).sum())
        stockout_by_hour += st.sum(0)
        days = pd.DataFrame({"dt": batch.column("dt").to_pandas(), "h": batch.column("holiday_flag").to_numpy()})
        for d, row in days.groupby("dt")["h"].agg(["min", "max"]).iterrows():
            lo, hi = holiday.get(d, (1, 0))
            holiday[d] = (min(lo, int(row["min"])), max(hi, int(row["max"])))
    stats["stockout_share_by_hour_percent"] = np.round(stockout_by_hour / max(stats["rows"], 1) * 100, 1).tolist()
    holiday_issues = []
    for d, (lo, hi) in sorted(holiday.items()):
        day = pd.Timestamp(d)
        expected = int(d in CN_2024_HOLIDAYS or (day.dayofweek >= 5 and d not in CN_2024_MAKEUP_WORKDAYS))
        if lo != hi:
            holiday_issues.append(dict(dt=d, issue="同一天 holiday_flag 不一致"))
        elif lo != expected:
            holiday_issues.append(dict(dt=d, weekday=day.day_name(), holiday_flag=int(lo),
                                       official_2024_calendar=expected))
    stats["holiday_flag_vs_official_calendar"] = holiday_issues
    return stats


def load_panel(repo, out):
    train, train_report = read_daily(repo / "dataset" / "train.parquet")
    evaluation, eval_report = read_daily(repo / "dataset" / "eval.parquet")
    keys = train[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    if not keys.equals(evaluation[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)):
        raise ValueError("train / eval 的序列集合不同")
    train_dates = pd.DatetimeIndex(sorted(train.dt.unique()))
    eval_dates = pd.DatetimeIndex(sorted(evaluation.dt.unique()))
    if not train_dates.equals(pd.date_range(train_dates[0], train_dates[-1], freq="D")):
        raise ValueError("train 日期不連續")
    if not eval_dates.equals(pd.date_range(train_dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")):
        raise ValueError(f"eval 必須是 train 之後連續 {HORIZON} 天")
    for name, frame, dates in [("train", train, train_dates), ("eval", evaluation, eval_dates)]:
        if not frame.groupby(KEYS).size().eq(len(dates)).all():
            raise ValueError(f"{name}: 有序列缺日；不自動補 0")
    index = pd.MultiIndex.from_frame(keys)

    def pivot(frame, dates):
        return frame.pivot(index=KEYS, columns="dt", values="sale_amount").reindex(
            index=index, columns=dates).to_numpy(np.float32)

    y_train, y_eval = pivot(train, train_dates), pivot(evaluation, eval_dates)
    audit_file = out / "data_audit.json"
    if not audit_file.exists():
        print("檢查 24 小時欄位與假日標籤（只做檢查，不進模型）…", flush=True)
        audit = dict(
            decision=("唯一必要的時間標籤是 dt（日）。hours_sale 與 sale_amount 完全重複；"
                      "hours_stock_status / stock_hour6_22_cnt 與實際銷量矛盾；holiday_flag 有調休日標錯。"
                      "三個模型只用 (序列 ID, dt, sale_amount)。"),
            train=dict(daily=train_report, dates=[str(train_dates[0].date()), str(train_dates[-1].date())],
                       hourly=audit_hourly_labels(repo / "dataset" / "train.parquet")),
            eval=dict(daily=eval_report, dates=[str(eval_dates[0].date()), str(eval_dates[-1].date())],
                      hourly=audit_hourly_labels(repo / "dataset" / "eval.parquet")),
            series=len(keys),
            zero_sales_share_train=float((y_train == 0).mean()),
        )
        write_json(audit_file, audit)
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    print(json.dumps({k: audit[k] for k in ("decision", "series")}, ensure_ascii=False, indent=1))
    for part in ("train", "eval"):
        h = audit[part]["hourly"]
        print(f"[{part}] dt 有時分秒: {audit[part]['daily']['dt_with_time_of_day']}, 重複: "
              f"{audit[part]['daily']['duplicate_series_day']}, 24h 加總不符: {h['hours_sum_mismatch']}, "
              f"缺貨小時仍有銷量: {h['sales_hours_marked_stockout']:,}/{h['sales_hours_total']:,}, "
              f"假日標籤疑誤: {h['holiday_flag_vs_official_calendar']}", flush=True)
    del train, evaluation
    return keys, train_dates, eval_dates, y_train, y_eval


# %% ============================== 指標 ==============================
QUANTILES = np.round(np.arange(0.1, 1.0, 0.1), 2)


def point_metrics(actual, pred):
    actual, pred = actual.astype(np.float64), pred.astype(np.float64)
    err = pred - actual
    denom = np.abs(actual).sum()
    return dict(MAE=np.abs(err).mean(), RMSE=np.sqrt(np.square(err).mean()),
                WAPE=np.abs(err).sum() / denom if denom else np.nan, observations=int(actual.size))


def weighted_quantile_loss(actual, quantile_preds):
    """GluonTS 的 mean_wQuantileLoss（CRPS 的近似），quantile_preds: [..., len(QUANTILES)]。"""
    actual = actual.astype(np.float64)[..., None]
    diff = actual - quantile_preds
    loss = np.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).sum(axis=tuple(range(diff.ndim - 1)))
    return float(np.mean(2 * loss / np.abs(actual).sum()))


def summarize_samples(samples):
    """samples: [series, horizon, samples] → (中位數點預測, 分位數)"""
    if CLIP_NEGATIVE:
        samples = np.clip(samples, 0, None)
    return np.median(samples, axis=-1), np.quantile(samples, QUANTILES, axis=-1).transpose(1, 2, 0)


# %% ============================== TACTiS2 ==============================
def import_tactis(repo):
    sys.path.insert(0, str(repo / "TACTiS2" / "TACTiS"))
    import tactis.model.tactis as tactis_module

    def standardization_init(self, hist_value):
        std, mean = torch.std_mean(hist_value, dim=2, unbiased=True, keepdim=True)
        self.std = std.clamp(min=TACTIS_STD_FLOOR)
        self.mean = mean

    # 只改最小標準差（數值穩定），其他完全使用 repo 內的原版 TACTiS2 程式碼
    tactis_module.NormalizationStandardization.__init__ = standardization_init
    return tactis_module.TACTiS


def build_tactis(TACTiS, num_series, stage, device):
    model = TACTiS(num_series=num_series, bagging_size=None, **deepcopy(TACTIS_MODEL))
    if stage == 2:
        model.initialize_stage2()
    return model.to(device)


def set_trainable(model, stage):
    for name, p in model.named_parameters():
        p.requires_grad_(stage == 1 or "copula" in name)


def make_windows(values, ends, device):
    C, H = TACTIS_CONTEXT, HORIZON
    hist = torch.stack([values[:, e - C:e] for e in ends])
    future = torch.stack([values[:, e:e + H] for e in ends])
    B = len(ends)
    ht = torch.arange(C, device=device)[None, None, :].expand(B, 1, C)
    pt = torch.arange(C, C + H, device=device)[None, None, :].expand(B, 1, H)
    return ht, hist, pt, future


def validate(model, values, stage, cutoff, device):
    model.eval()
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED + 999)
        if stage == 1:
            model.bagging_size = None   # 全部 50,000 條序列一起算 marginal NLL
            marginal, _ = model.loss(*make_windows(values, [cutoff], device))
            return float((-marginal).mean())
        model.bagging_size = TACTIS_STAGES[2]["bag"]
        losses = []
        for _ in range(STAGE2_VAL_CALLS):
            _, copula = model.loss(*make_windows(values, [cutoff] * STAGE2_VAL_BAGS, device))
            losses.append(float(copula.mean()))
        return float(np.mean(losses))


def rng_snapshot(rng, device):
    return dict(numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state() if device == "cuda" else None)


def train_tactis(TACTiS, y_train, out, device):
    final = out / "tactis2_final.pt"
    N, T = y_train.shape
    if final.exists():
        model = build_tactis(TACTiS, N, 2, device)
        model.load_state_dict(torch.load(final, map_location=device)["model"])
        print("已有訓練好的 TACTiS2，略過訓練。", flush=True)
        return model
    values = torch.from_numpy(y_train).to(device)
    cutoff = T - VALIDATION_DAYS                 # 驗證視窗：預測 train 的最後 7 天
    last_train_end = cutoff - HORIZON            # 訓練視窗的目標不碰驗證日
    if last_train_end < TACTIS_CONTEXT:
        raise ValueError("訓練天數不足")
    state_file, log_file = out / "tactis2_state.pt", out / "tactis2_training_log.json"
    state = torch.load(state_file, map_location=device, weights_only=False) if state_file.exists() else None
    start_stage = state["stage"] if state else 1
    model = build_tactis(TACTiS, N, start_stage, device)
    if state:
        model.load_state_dict(state["model"])
    logs = state["logs"] if state else []
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    for stage in (1, 2):
        if stage < start_stage:
            continue
        cfg = TACTIS_STAGES[stage]
        if stage == 2 and start_stage == 1:
            model.initialize_stage2()
            model.to(device)
        set_trainable(model, stage)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                     lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        epoch, best, patience, seconds = 0, float("inf"), 0, 0.0
        if state and state["stage"] == stage:
            optimizer.load_state_dict(state["optimizer"])
            epoch, best, patience, seconds = state["epoch"], state["best"], state["patience"], state["seconds"]
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"].cpu())
            if state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        best_file = out / f"tactis2_best_stage{stage}.pt"
        print(f"\n=== TACTiS2 stage {stage}：{'marginal flow' if stage == 1 else 'copula（flow 凍結）'}，"
              f"bag={cfg['bag']} × batch={cfg['batch']} ===", flush=True)
        while epoch < cfg["max_epochs"] and patience < cfg["patience"] and seconds < cfg["max_hours"] * 3600:
            model.train()
            model.bagging_size = cfg["bag"]
            started, losses, skipped = time.perf_counter(), [], 0
            for _ in range(cfg["steps_per_epoch"]):
                ends = rng.integers(TACTIS_CONTEXT, last_train_end + 1, size=cfg["batch"])
                optimizer.zero_grad(set_to_none=True)
                marginal, copula = model.loss(*make_windows(values, ends, device))
                loss = (-marginal if stage == 1 else copula).mean()
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                if not torch.isfinite(norm):
                    skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                losses.append(loss.item())
            val = validate(model, values, stage, cutoff, device)
            if device == "cuda":
                torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            epoch += 1
            if val < best or not best_file.exists():
                best, patience = min(val, best), 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "validation": val}, best_file)
            else:
                patience += 1
            logs.append(dict(stage=stage, epoch=epoch, train_loss=float(np.mean(losses)) if losses else None,
                             validation_loss=val, skipped_steps=skipped, minutes=round(seconds / 60, 2)))
            print(logs[-1], flush=True)
            write_json(log_file, logs)
            torch.save(dict(stage=stage, model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                            best=best, patience=patience, seconds=seconds, logs=logs,
                            **rng_snapshot(rng, device)), state_file)
        model.load_state_dict(torch.load(best_file, map_location=device)["model"])
        print(f"stage {stage} 完成，最佳驗證 loss = {best:.5f}", flush=True)
        state = None
        if stage == 1:
            # 進入 stage 2 前存檔，續跑時從 stage 2 開始
            model.initialize_stage2()
            model.to(device)
            set_trainable(model, 2)
            opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=TACTIS_STAGES[2]["lr"])
            torch.save(dict(stage=2, model=model.state_dict(), optimizer=opt2.state_dict(), epoch=0,
                            best=float("inf"), patience=0, seconds=0.0, logs=logs,
                            **rng_snapshot(rng, device)), state_file)
            start_stage = 2
    torch.save({"model": model.state_dict(), "config": TACTIS_MODEL, "context": TACTIS_CONTEXT}, final)
    return model


def auto_samples_per_call(num_series):
    if TACTIS_SAMPLES_PER_CALL:
        return TACTIS_SAMPLES_PER_CALL
    cop = TACTIS_MODEL["copula_decoder"]["attentional_copula"]
    heads, layers, dim = cop["attention_heads"], cop["attention_layers"], cop["attention_dim"]
    pred_vars, hist_vars = num_series * HORIZON, num_series * TACTIS_CONTEXT
    fixed = 2 * layers * heads * hist_vars * dim * 4 + 3 * 2**30        # 歷史 key/value + 編碼器等
    per_sample = 2 * layers * heads * pred_vars * dim * 4 + 24 * pred_vars * 16 * 4   # 抽樣 KV cache + marginal 反函數
    free = torch.cuda.mem_get_info()[0]
    return int(max(1, min(25, (free * 0.85 - fixed) // per_sample)))


def sample_tactis(model, y_train, out, device):
    """一次對全部 50,000 條序列 × 7 天聯合抽樣；多次呼叫只是分批產生獨立的樣本。"""
    folder = out / "tactis2_samples"
    folder.mkdir(exist_ok=True)
    N = y_train.shape[0]
    model.eval()
    model.bagging_size = None
    hist = torch.from_numpy(y_train[None, :, -TACTIS_CONTEXT:]).to(device)
    ht = torch.arange(TACTIS_CONTEXT, device=device)[None, None, :]
    pt = torch.arange(TACTIS_CONTEXT, TACTIS_CONTEXT + HORIZON, device=device)[None, None, :]
    per_call = auto_samples_per_call(N)
    print(f"\nTACTiS2 聯合抽樣：{N:,} 條序列 × {HORIZON} 天 = {N * HORIZON:,} 個變數，"
          f"共 {TACTIS_NUM_SAMPLES} 個樣本，每次 {per_call} 個", flush=True)
    parts = []
    for offset in range(0, TACTIS_NUM_SAMPLES, per_call):
        count = min(per_call, TACTIS_NUM_SAMPLES - offset)
        file = folder / f"{offset:04d}_{count:02d}.npy"
        if not file.exists():
            started = time.perf_counter()
            torch.manual_seed(SEED + 100_000 + offset)
            with torch.no_grad():
                draw = model.sample(count, ht, hist, pt)[0, :, -HORIZON:, :].float().cpu().numpy()
            if not np.isfinite(draw).all():
                raise ValueError("TACTiS2 樣本出現非有限值")
            save_npy(file, draw)
            minutes = (time.perf_counter() - started) / 60
            left = (TACTIS_NUM_SAMPLES - offset - count) / count * minutes
            print(f"  樣本 {offset + count}/{TACTIS_NUM_SAMPLES} 完成，本批 {minutes:.1f} 分，預估剩 {left:.0f} 分", flush=True)
        parts.append(np.load(file))
    return np.concatenate(parts, axis=-1)


# %% ============================== Chronos ==============================
def load_chronos_pipeline(repo, device):
    try:
        sys.path.insert(0, str(repo / "chronos" / "chronos-forecasting" / "src"))
        from chronos import ChronosPipeline
    except Exception as exc:
        print("repo 內 chronos 原始碼無法載入，改用 pip 套件：", exc, flush=True)
        sys.path = [p for p in sys.path if "chronos-forecasting" not in p]
        for name in [m for m in sys.modules if m == "chronos" or m.startswith("chronos.")]:
            del sys.modules[name]
        sh(f"{sys.executable} -m pip install -q chronos-forecasting")
        from chronos import ChronosPipeline
    return ChronosPipeline.from_pretrained(CHRONOS_MODEL, revision=CHRONOS_REVISION,
                                           device_map=device, torch_dtype=torch.float32)


def run_chronos(repo, y_train, out, device):
    file = out / "chronos_samples.npy"
    if file.exists():
        return np.load(file)
    pipeline = load_chronos_pipeline(repo, device)
    history = y_train if BASELINE_CONTEXT is None else y_train[:, -BASELINE_CONTEXT:]
    torch.manual_seed(SEED + 200_000)
    batches, started = [], time.perf_counter()
    print(f"\nChronos（{CHRONOS_MODEL}）推論 {len(history):,} 條序列，context={history.shape[1]} 天", flush=True)
    for start in range(0, len(history), CHRONOS_BATCH):
        context = torch.from_numpy(history[start:start + CHRONOS_BATCH].copy())
        with torch.no_grad():
            pred = pipeline.predict(context, prediction_length=HORIZON, num_samples=CHRONOS_NUM_SAMPLES)
        batches.append(pred.float().cpu().numpy().transpose(0, 2, 1))   # → [series, horizon, samples]
        if (start // CHRONOS_BATCH) % 40 == 0:
            print(f"  {start + len(context):,}/{len(history):,}（{time.perf_counter() - started:.0f}s）", flush=True)
    samples = np.concatenate(batches).astype(np.float32)
    save_npy(file, samples)
    del pipeline
    torch.cuda.empty_cache()
    return samples


# %% ============================== sktime-m4 ==============================
def seasonality_test_R(y, sp):
    """sktime-m4/scripts/utils.py 的 M4 季節性檢定（R 版翻譯，90% 信賴）。"""
    from statsmodels.tsa.stattools import acf
    y = np.asarray(y, dtype=float)
    if sp == 1 or len(y) < 3 * sp or np.std(y) == 0:
        return False
    coefs = acf(y, nlags=sp, fft=False)
    limits = 1.645 / np.sqrt(len(y)) * np.sqrt(np.cumsum(np.append(1, 2 * coefs[1:] ** 2)))
    return bool(np.abs(coefs[sp]) > limits[sp - 1])


def sktime_block(block, methods, sp, horizon):
    """construct.py 的 construct_M4_forecasters 改寫成目前 sktime API。"""
    import warnings
    warnings.filterwarnings("ignore")
    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.transformations.series.detrend import ConditionalDeseasonalizer

    def deseasonalise(forecaster, model):
        return TransformedTargetForecaster([
            ("deseasonalise", ConditionalDeseasonalizer(seasonality_test=seasonality_test_R, sp=sp, model=model)),
            ("forecast", forecaster)])

    def make(name, model):
        if name == "Naive":
            return NaiveForecaster(strategy="last")
        if name == "sNaive":
            return NaiveForecaster(strategy="last", sp=sp)        # 舊 API 的 strategy="seasonal_last"
        if name == "Naive2":
            return deseasonalise(NaiveForecaster(strategy="last"), model)
        if name == "SES":
            return deseasonalise(ExponentialSmoothing(), model)
        if name == "Holt":
            return deseasonalise(ExponentialSmoothing(trend="add", damped_trend=False), model)
        if name == "Damped":
            return deseasonalise(ExponentialSmoothing(trend="add", damped_trend=True), model)
        if name == "Theta":
            return deseasonalise(ThetaForecaster(deseasonalize=False), model)
        raise KeyError(name)

    fitted = [m for m in methods if m != "Com"]
    if "Com" in methods:   # Com = SES/Holt/Damped 等權平均（EnsembleForecaster 預設 mean）
        fitted += [m for m in ("SES", "Holt", "Damped") if m not in fitted]
    fh = np.arange(1, horizon + 1)
    preds = {m: np.empty((len(block), horizon)) for m in fitted}
    fails = {m: 0 for m in fitted}
    for i, values in enumerate(block):
        y = pd.Series(values.astype(float), index=pd.RangeIndex(len(values)))
        # M4 用乘法季節；有 0 銷量時乘法分解不成立，改用加法
        model = "multiplicative" if (values > 0).all() else "additive"
        for m in fitted:
            try:
                f = make(m, model)
                f.fit(y, fh=fh)
                p = f.predict().to_numpy(dtype=float)
                if p.shape != (horizon,) or not np.isfinite(p).all():
                    raise ValueError("invalid forecast")
            except Exception:
                fails[m] += 1
                p = np.full(horizon, values[-1], dtype=float)    # 失敗時退回 Naive，並計數
            preds[m][i] = p
    if "Com" in methods:
        preds["Com"] = (preds["SES"] + preds["Holt"] + preds["Damped"]) / 3
        fails["Com"] = 0
    return {m: preds[m] for m in methods}, {m: fails[m] for m in methods}


def sktime_job(start, block, folder):
    preds, fails = sktime_block(block, SKTIME_METHODS, SKTIME_SP, HORIZON)
    np.savez(Path(folder) / f"{start:06d}.npz", **preds, **{f"fail_{m}": v for m, v in fails.items()})
    return start


def run_sktime(y_train, out):
    file = out / "sktime_predictions.npz"
    if file.exists():
        data = np.load(file)
        return {m: data[m] for m in SKTIME_METHODS}
    from joblib import Parallel, delayed
    history = y_train if BASELINE_CONTEXT is None else y_train[:, -BASELINE_CONTEXT:]
    folder = out / "sktime_blocks"
    folder.mkdir(exist_ok=True)
    starts = list(range(0, len(history), SKTIME_BLOCK))
    todo = [s for s in starts if not (folder / f"{s:06d}.npz").exists()]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    jobs = os.cpu_count() or 2
    print(f"\nsktime-m4：{', '.join(SKTIME_METHODS)}，{len(history):,} 條序列，"
          f"剩 {len(todo)}/{len(starts)} 個區塊，{jobs} 核心", flush=True)

    started = time.perf_counter()
    for k in range(0, len(todo), jobs * 8):   # 分段送出，方便看進度與續跑
        Parallel(n_jobs=jobs, backend="loky")(
            delayed(sktime_job)(s, history[s:s + SKTIME_BLOCK].copy(), str(folder)) for s in todo[k:k + jobs * 8])
        done = min(k + jobs * 8, len(todo))
        rate = (time.perf_counter() - started) / done
        print(f"  {done}/{len(todo)} 區塊，預估剩 {rate * (len(todo) - done) / 60:.0f} 分", flush=True)
    preds = {m: [] for m in SKTIME_METHODS}
    fails = {m: 0 for m in SKTIME_METHODS}
    for s in starts:
        data = np.load(folder / f"{s:06d}.npz")
        for m in SKTIME_METHODS:
            preds[m].append(data[m])
            fails[m] += int(data[f"fail_{m}"])
    preds = {m: np.concatenate(v).astype(np.float32) for m, v in preds.items()}
    np.savez(file, **preds)
    write_json(out / "sktime_fit_failures.json", fails)
    print("sktime 失敗後退回 Naive 的序列數：", fails, flush=True)
    return preds


# %% ============================== 主程式 ==============================
def main():
    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU：TACTiS2 對 50,000 條序列的跨序列 attention 必須在 CUDA 上用 memory-efficient attention。")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    repo = fetch_repo()
    out = output_dir()
    print("輸出目錄：", out, "| GPU：", torch.cuda.get_device_name(), flush=True)
    keys, train_dates, eval_dates, y_train, y_eval = load_panel(repo, out)
    N = len(keys)
    write_json(out / "manifest.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), repo=REPO_URL, series=N,
        train_dates=[str(train_dates[0].date()), str(train_dates[-1].date())],
        validation_dates=[str(train_dates[-VALIDATION_DAYS].date()), str(train_dates[-1].date())],
        eval_dates=[str(eval_dates[0].date()), str(eval_dates[-1].date())],
        tactis=dict(context=TACTIS_CONTEXT, stages=TACTIS_STAGES, model=TACTIS_MODEL, std_floor=TACTIS_STD_FLOOR,
                    num_samples=TACTIS_NUM_SAMPLES, scope="ONE model, all 50,000 series; bagged SGD; full joint sampling"),
        chronos=dict(model=CHRONOS_MODEL, revision=CHRONOS_REVISION, num_samples=CHRONOS_NUM_SAMPLES),
        sktime=dict(methods=SKTIME_METHODS, sp=SKTIME_SP), baseline_context=BASELINE_CONTEXT or len(train_dates),
        clip_negative=CLIP_NEGATIVE, point_forecast="sample median",
        versions={p: __import__(p).__version__ for p in ("torch", "numpy", "pandas")}))

    # 1) TACTiS2
    TACTiS = import_tactis(repo)
    model = train_tactis(TACTiS, y_train, out, device)
    tactis_samples = sample_tactis(model, y_train, out, device)
    del model
    torch.cuda.empty_cache()
    # 2) Chronos
    chronos_samples = run_chronos(repo, y_train, out, device)
    # 3) sktime-m4
    sktime_preds = run_sktime(y_train, out)

    # 4) 評估
    rows, point = [], {}
    for name, samples in [("TACTiS2", tactis_samples), ("Chronos", chronos_samples)]:
        median, quantiles = summarize_samples(samples)
        point[name] = median
        rows.append(dict(model=name, role="main" if name == "TACTiS2" else "baseline",
                         **point_metrics(y_eval, median), wQL=weighted_quantile_loss(y_eval, quantiles)))
    for name, pred in sktime_preds.items():
        pred = np.clip(pred, 0, None) if CLIP_NEGATIVE else pred
        point[f"sktime_{name}"] = pred
        rows.append(dict(model=f"sktime_{name}", role="baseline", **point_metrics(y_eval, pred), wQL=np.nan))
    metrics = pd.DataFrame(rows)
    metrics["WAPE_percent"] = metrics["WAPE"] * 100
    metrics.to_csv(out / "metrics.csv", index=False)

    per_day = []
    for name, pred in point.items():
        for d, day in enumerate(eval_dates):
            per_day.append(dict(model=name, dt=str(day.date()), **point_metrics(y_eval[:, d], pred[:, d])))
    pd.DataFrame(per_day).to_csv(out / "per_day_metrics.csv", index=False)

    frame = keys.loc[keys.index.repeat(HORIZON)].reset_index(drop=True)
    frame["dt"] = np.tile(eval_dates, N)
    frame["actual"] = y_eval.ravel()
    for name, pred in point.items():
        frame[name] = pred.ravel().astype(np.float32)
    frame.to_parquet(out / "predictions.parquet", index=False)

    print("\n================ 結果（越低越好；50,000 條序列 × 7 天 = 350,000 筆）================")
    print(metrics.sort_values("MAE").to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    best = metrics[metrics.role == "baseline"].sort_values("MAE").iloc[0]
    tac = metrics[metrics.model == "TACTiS2"].iloc[0]
    print(f"\nTACTiS2 MAE 相對最佳 baseline（{best.model}）："
          f"{(tac.MAE / best.MAE - 1) * 100:+.2f}%（負值代表 TACTiS2 較好）")

    try:
        import matplotlib.pyplot as plt
        picks = np.random.default_rng(SEED).choice(N, 6, replace=False)
        tq = np.quantile(np.clip(tactis_samples, 0, None) if CLIP_NEGATIVE else tactis_samples, [0.1, 0.9], axis=-1)
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        for ax, i in zip(axes.ravel(), picks):
            ax.plot(train_dates[-28:], y_train[i, -28:], color="black", label="train")
            ax.plot(eval_dates, y_eval[i], color="black", ls="--", label="actual")
            ax.fill_between(eval_dates, tq[0, i], tq[1, i], alpha=0.25, label="TACTiS2 10–90%")
            ax.plot(eval_dates, point["TACTiS2"][i], label="TACTiS2")
            ax.plot(eval_dates, point["Chronos"][i], label="Chronos")
            if best.model != "Chronos":
                ax.plot(eval_dates, point[best.model][i], label=best.model)
            ax.set_title(" / ".join(f"{k}={keys.iloc[i][k]}" for k in KEYS), fontsize=9)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "examples.png", dpi=120)
        plt.show()
    except Exception as exc:
        print("繪圖略過：", exc)
    (out / "SUCCESS").write_text("TACTiS2, Chronos and sktime-m4 cover all series and evaluation days.\n")
    print("\n全部完成，輸出：", out)


if __name__ == "__main__":
    main()
