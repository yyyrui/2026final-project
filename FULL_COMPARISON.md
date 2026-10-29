# 全資料比較

本模式處理 train.parquet / eval.parquet 的全部 50,000 條序列，最終每模型
預測 350,000 個 series/day。它不是單一 50,000 維 TACTiS2：依
city_id/store_id/product_id 排序，每 32 條建立一個獨立的小型 TACTiS2，
共 1,563 組；分組可能跨店鋪。每組各自訓練 flow 和 copula。
Chronos 是原版 amazon/chronos-t5-tiny，固定權重 revision
29d808298f1a62493e7b9a5e08529d0d930fa189，只載入一次。

## 開始執行

在專案根目錄：

```bash
.venv/bin/python -u run_comparison.py --all-series --offline --output-dir outputs/full-comparison
```

這個目錄已用於本次全量執行。正在執行時不要再啟動同目錄工作。
獨立新實驗請換一個尚不存在的 `--output-dir`。`--offline` 使用已下載的權重；
新電腦尚未下載權重時應省略此參數。Windows 尚不支援目前使用的 Unix 檔案鎖。

VS Code「執行與偵錯」新增了「全量比較（50,000 組，CPU）」與
「續跑全量比較（CPU）」設定，固定使用專案 .venv。編輯器右上角
Run Python File 不會自動加上全量參數；直接執行入口仍預設 4 條序列。

## 訓練設定及比較限制

- CPU，2 個 PyTorch threads，每組每階段 2 epochs，每 epoch 5 個隨機視窗。
- 每次預測各產生 20 個樣本，以中位數算 MAE / RMSE / WAPE，不裁切負數。
- 03/28–06/18 可用作訓練視窗；06/19–06/25 僅作驗證、挑選階段最佳權重。
- 兩模型都用截至 06/25 的最後 28 天觀測，預測 06/26–07/02。eval 不用於選模型。
- **全部序列都有參與，但極短訓練不代表收斂，也不代表每個可能的訓練視窗都被抽過。**
- 如要增加訓練，可在新輸出目錄加 `--epochs 20 --steps-per-epoch 50`。
  這會大幅增加時間，需要重新量測及檢查驗證曲線，不能混接舊輸出。
- 不同組大小與初始種子會影響結果；這不是原論文的完整模型設定。
- 此程式未加日曆、促銷、天氣等特徵。

## 輸出及中斷

- `manifest.json`：完整參數、資料 SHA-256、模型及入口原始碼 SHA-256、套件版本。
- `series.csv`：所有 ID 與固定 group 編號。
- `groups/<group>/tactis.pt`、`training_log.json`：每組模型與兩階段損失。
- `groups/<group>/{tactis2,chronos}.parquet`：完成的分組預測，含 ID/date/actual/prediction。
  原子寫入完成後才允許續跑跳過；不保留全量原始樣本，以減少儲存。
- `progress.json`：狀態、已完成批次／序列、執行中估計剩餘秒數。
- `metrics_partial.csv`：已完成的完整批次指標，**不是全量最終結果**。
- `metrics.csv`：只有所有批次成功後才輸出，每模型 observations 必須為 350000。
- `tactis2_predictions.csv` / `chronos_predictions.csv`：全量逐日預測，完成後合併產生。
- `SUCCESS`：全部完成標記；不存在代表還沒完整成功。

中斷後，同樣參數加 `--resume`：

```bash
.venv/bin/python -u run_comparison.py --all-series --offline --output-dir outputs/full-comparison --resume
```

已完成的「批次／模型」會跳過；正在訓練但尚未寫入預測的模型，會從該組重新訓練。
不是 epoch 級續訓。模型種子依固定 group 編號設定，Chronos 載入不會改變該組採樣種子。
資料、程式、套件或參數不同時拒絕續跑，避免混用不一致結果。
同一輸出目錄只允許一個寫入程序。

本次由助理啟動的執行日誌在 `outputs/full-comparison.log`；可在終端看最後幾行：

```bash
tail -n 10 outputs/full-comparison.log
cat outputs/full-comparison/progress.json
```

電腦休眠或程式關閉可能暫停／中斷運算；重新啟動時使用上述續跑命令。
所有結果仍在本機的 outputs/，沒有自動上傳 GitHub。

## 驗證

```bash
.venv/bin/python -m unittest discover -s tests -v
```

測試包含資料對齊、缺日／重複／日期重疊防護、全量模式、不同組大小的全域指標、
模擬中斷續跑、資料／參數不一致拒絕續跑與並行寫入鎖。
實際 33 條序列的端到端結果位於 outputs/grouped-verified/，覆蓋 32+1 的最後批次情況。

## 本次全量結果（已完成）

`outputs/full-comparison/` 的 1,563 個分組全部成功，兩模型各有 50,000 條序列、
350,000 筆預測。全量 CSV 已獨立核對唯一 ID/日期、七天覆蓋、兩模型真值一致、
有限數值及重算指標，結果記錄於 `verification.json`。10 項單元測試通過。

| 模型 | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| TACTiS2（分組小型模型） | 0.5184334453 | 0.8993926917 | 43.45435432% |
| 原版 Chronos tiny | 0.4653382462 | 0.8078348709 | 39.00399022% |

這是全資料覆蓋、短訓練設定下的結果，不能解讀成兩模型充分調參後的效能上限。

之後要重新跑全量且自動建立新輸出目錄：

```bash
.venv/bin/python run_comparison.py --all-series --offline
```

VS Code 的「全量比較（50,000 組，CPU）」也使用這個自動新目錄設定。
「續跑全量比較（CPU）」則指向本次 outputs/full-comparison；若要續跑其他實驗，
需修改該設定的 --output-dir，並保持原本實驗參數。
