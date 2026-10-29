# 銷售資料模型比較

在專案最外層執行（不需要搬動模型）：

```bash
source .venv/bin/activate
python run_comparison.py
```

預設使用 CPU、固定 seed=42 抽取 4 組真實城市／店鋪／商品序列，預測 2024-06-26 至 2024-07-02。第一次會下載 `amazon/chronos-t5-tiny` 公開權重，快取在 `.cache/huggingface`。固定使用**原版 Chronos**，不是 Bolt 或 Chronos-2。

若重建環境，可用 `uv pip install --python .venv/bin/python -r requirements-comparison.txt`（或在含 pip 的虛擬環境使用 `python -m pip install -r requirements-comparison.txt`）。直接載入兩個現有原始碼目錄，不需安裝 GluonTS 或 PyTorchTS；原有 `TACTiS2/TACTiS/train.py` 保持原樣，此入口使用模型核心建立獨立訓練流程。

## 資料與訓練規則

- 只讀 `city_id`, `store_id`, `product_id`, `dt`, `sale_amount`。未加入天氣、促銷、庫存等特徵。
- 驗證整份資料的序列集合相同、每天一筆、日期連續、目標有限且無缺值。缺日不會自動補零，重複列不會自動加總。
- 抽樣只依訓練檔的序列 ID 與 seed，與 eval 銷售量無關；兩模型共用相同序列順序。
- TACTiS2 是小型設定的聯合模型；第一階段訓練 marginal flow，第二階段凍結 flow、訓練 copula。每階段預設 2 epochs，每 epoch 5 個視窗。
- 訓練視窗的目標只到 06/18；06/19–06/25 留作驗證，分別選取兩階段最佳權重。沒有用 eval 選模型，也沒有選完後重新訓練全部 90 天。
- 最終預測時，兩模型都使用訓練檔最後 28 天的真實觀測，預測相同 7 天。Chronos 只做預訓練推論，不微調。
- 每模型產生 20 個樣本，以中位數作點預測；不裁切負值。比較 MAE、RMSE 與 WAPE（比例，乘 100 為百分比）。全部實際值為零時 WAPE 留空。

## 輸出

每次執行建立新的 `outputs/<timestamp>/`，已有的指定輸出目錄會拒絕覆寫：

- `manifest.json`：參數、日期、版本與實驗說明。
- `series.csv`：抽中的序列 ID。
- `tactis.pt`、`training_log.json`：兩階段模型權重與訓練紀錄。重建 checkpoint 時需先用 config 建立模型，再呼叫 `initialize_stage2()` 後載入 state dict。
- `tactis2_predictions.csv`、`chronos_predictions.csv`：ID、日期、真值及點預測。
- `*_samples.npz`：`[series, horizon, sample]` 的原始預測樣本。
- `metrics.csv`：共同指標與各模型耗時（TACTiS 含訓練；Chronos 含載入及首次下載，因此不能直接作推論速度比較）。
- `SUCCESS`：兩模型皆成功完成才寫入；失敗時保留已完成產物供檢查。

```bash
# 僅驗證資料
python run_comparison.py --data-only
# 增加試驗規模
python run_comparison.py --num-series 16 --epochs 20 --steps-per-epoch 50 --num-samples 100
# CUDA 主機可指定 --device cuda；這次僅驗證 CPU，未支援 MPS
# 從其他工作目錄呼叫也可以，預設及相對參數路徑皆以入口目錄為基準
python /absolute/path/to/2026final-project/run_comparison.py
# 測試
python -m unittest discover -s tests -v
```

這是整合 smoke test，不能據此判定正式模型優劣。入口限制最多 64 組，避免直接將 50,000 組送入 joint copula；全量實驗需要另設分組、資源配置和完整訓練計畫。固定種子可重現抽樣，但不同 PyTorch／硬體環境的模型數值不保證完全一致。

## 本機已完成的試跑

`outputs/smoke-verified/` 已完成上述預設 4 組序列測試，且從 `/tmp` 呼叫絕對入口路徑成功，證實不依賴目前工作目錄。6 項單元測試全數通過。

| 模型 | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| TACTiS2 | 0.563548 | 0.790441 | 52.8444% |
| 原版 Chronos tiny | 0.475959 | 0.715994 | 44.6311% |

權重快取已下載。離線重跑可用 `HF_HUB_OFFLINE=1 python run_comparison.py`，避免網路檢查。試跑所用公開權重 snapshot 為 `29d808298f1a62493e7b9a5e08529d0d930fa189`。`requirements-comparison.lock.txt` 記錄本次虛擬環境的確切套件版本；跨平台安裝優先使用範圍相依檔。

## 全部 50,000 組序列

已新增 `--all-series` 的固定分組、斷點續跑與全量評估模式。
前述最多 64 條限制僅適用舊的單模型模式；分組模式限制的是每組最多 64 條。
完整操作和限制見 [FULL_COMPARISON.md](FULL_COMPARISON.md)。
