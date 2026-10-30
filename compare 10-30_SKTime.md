# compare 10-30_SKTime：TACTiS2 與 Naïve

執行時間：2026-10-30T23:40:22+08:00（Asia/Taipei）。

本次對原銷售資料的 **50,000 條序列全部執行 sktime Naïve**，每條預測 7 天，
共 350,000 筆。TACTiS2 使用先前已完成的結果，逐筆核對後重新計算指標，**本次沒有重新訓練 TACTiS2**。

## 實際比較結果

| 模型 | MAE ↓ | RMSE ↓ | WAPE ↓ | 預測筆數 |
| --- | ---: | ---: | ---: | ---: |
| TACTiS2（既有分組小型模型） | 0.518433445292 | 0.899392691688 | 43.4543543192% | 350,000 |
| Naïve（sktime，本次執行） | 0.511644324092 | 0.859646518118 | 42.8852998325% | 350,000 |

三個指標都是越低越好。Markdown 顯示四捨五入數值；完整浮點數保存在 [outputs/compare-10-30_SKTime/metrics.csv](outputs/compare-10-30_SKTime/metrics.csv)。

- MAE：**Naïve 較低**；TACTiS2 相對 Naïve 的誤差增加 1.3269%。
- RMSE：**Naïve 較低**；TACTiS2 相對 Naïve 的誤差增加 4.6235%。
- WAPE：**Naïve 較低**；TACTiS2 相對 Naïve 的誤差增加 1.3269%。

以每條序列七天的 MAE 比較：TACTiS2 較低 23,471 條，
Naïve 較低 26,529 條，相同 0 條（容差 1e-12）。
整體指標由全部 350,000 筆誤差統計，不是把各組 RMSE 或 WAPE 直接平均。

## 使用哪份資料

- 原始訓練檔：`dataset/train.parquet`，2024-03-28～2024-06-25，90 天。
- 原始評估檔：`dataset/eval.parquet`，2024-06-26～2024-07-02，7 天。
- 序列鍵：`city_id / store_id / product_id`；預測目標：`sale_amount`。
- 兩檔 SHA256 均與先前 TACTiS2 manifest 相同；沒有另一份被改寫銷售數值的訓練檔。
- 共用 `comparison/data.py`，先檢查缺值、重複、日期連續性，再重整成「序列 × 日」並轉成 float32。
- 兩者最終輸入均取最後 28 天：2024-05-29～2024-06-25。
  普通 Naïve 實際只依賴最後一天；未使用 eval 真值來訓練或挑選參數。
- TACTiS2 的標準化是在模型內進行，輸出已還原銷售單位；本比較不再額外標準化。

## Naïve 來源與路徑調整

- 作者配套程式：[mloning/sktime-m4](https://github.com/mloning/sktime-m4)。
- 論文：Markus Löning、Franz J. Király（2020），
  [Forecasting with sktime: Designing sktime's New Forecasting API and Applying It to Replicate and Extend the M4 Study](https://arxiv.org/abs/2005.08067)。
- 原始 `sktime-m4/scripts/construct.py` 的普通 Naïve 定義為 `NaiveForecaster(strategy="last")`。
- 本次透過 `scripts/sales_naive.py`，使用官方 sktime **1.2.0** 的
  `NaiveForecaster(strategy="last", sp=1)`；每條序列確實呼叫 `fit()` 和 `predict()`。
  `fit()` 只記錄歷史，不做梯度訓練；未來七天都預測為 2024-06-25 的數值。
- 原始 M4 runner 要求 M4 metadata／CSV、sktime 0.4.0 與舊 API，因此不能只替換路徑就讀 parquet。
  本次沿用作者模型規則並移植資料介面，**沒有宣稱重現 2020 年套件環境或 M4 論文成績**。
- 銷售實驗的輸入、既有 TACTiS2 結果、輸出路徑統一由
  [`sktime-m4/scripts/sales_config.py`](sktime-m4/scripts/sales_config.py) 設定，依專案位置解析，沒有寫死本機帳號。
  銷售入口不使用原本僅供 M4 的 `scripts/config.py`。

## TACTiS2 的實際設定與限制

- 預測來源：`outputs/full-comparison/groups/*/tactis2.parquet`。
- 依城市／店鋪／商品排序，每 32 條建立一個獨立小型 TACTiS2，
  共 1,563 組，最後一組 16 條；分組可能跨店鋪。
- 每個訓練階段 2 epochs，每 epoch 5 steps；
  使用 20 個預測樣本的中位數，不裁切負值。
- 2024-03-28～06-18 抽取訓練視窗；06-19～06-25 作驗證。
- 這是短訓練、小型分組模型的結果，不代表原論文完整設定，也不代表已充分收斂。
- 本報告只評估點預測，不能用這些指標直接證明 copula 學好了跨商品相依關係。
- 只有一個七天測試區間，不能把本次排名推廣到所有資料或所有時間。
- 原 Chronos 比較結果保留於 [FULL_COMPARISON.md](FULL_COMPARISON.md)，本次未重跑或移除。

## 驗證與輸出

- 全部原始資料雜湊與前次實驗一致，共用資料載入程式也與前次相同。
- 1,563 個 TACTiS2 預測分組皆核對 ID、日期、真值、有限數值與完整覆蓋。
- TACTiS2 重新計算的 MAE／RMSE／WAPE 與既有 metrics.csv 一致。
- Naïve 的全部輸出皆與「最後觀測值延續七天」獨立核對相同。
- 新輸出的兩模型預測與同一個真值欄並列，每個「序列／日期」恰好一筆。
- 本次 Naïve 執行耗時 75.959 秒（含工作程序啟動，2 個 CPU workers）。
  TACTiS2 是重用結果，因此本報告不作訓練速度比較。

| 檔案 | 用途 |
| --- | --- |
| [outputs/compare-10-30_SKTime/metrics.csv](outputs/compare-10-30_SKTime/metrics.csv) | 兩模型整體指標；WAPE 為比例，WAPE_percent 為百分比 |
| [outputs/compare-10-30_SKTime/predictions.parquet](outputs/compare-10-30_SKTime/predictions.parquet) | 350,000 筆真值與兩模型預測，保留浮點精度 |
| [outputs/compare-10-30_SKTime/predictions.csv](outputs/compare-10-30_SKTime/predictions.csv) | 可直接開啟的相同預測表 |
| [outputs/compare-10-30_SKTime/per_day_metrics.csv](outputs/compare-10-30_SKTime/per_day_metrics.csv) | 每天各模型的指標 |
| [outputs/compare-10-30_SKTime/manifest.json](outputs/compare-10-30_SKTime/manifest.json) | 來源、設定、套件版本、資料與預測檔案雜湊 |
| [outputs/compare-10-30_SKTime/verification.json](outputs/compare-10-30_SKTime/verification.json) | 本次驗證摘要 |

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
