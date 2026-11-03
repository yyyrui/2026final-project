# 5,000 條銷售序列比較

在 `1104` 資料夾執行 `1102.py`。程式預設使用旁邊的
`2026final-project/dataset/train.parquet` 與 `eval.parquet`，不使用 M4 資料集。

## 安裝與執行

建議 Python 3.11，於自己的虛擬環境安裝：

```powershell
python -m pip install -r requirements-1102.txt
python 1102.py --device cuda
```

GPU 版 PyTorch 請依 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/)
選擇適合機器的 CUDA wheel；上述 requirements 不會自行替你選擇 CUDA 版本。
省略 `--device cuda` 時會自動使用可用的 CUDA，否則用 CPU。
首次執行 Chronos 會下載固定版本的公開預訓練權重，之後可使用 `--offline`。
這支程式會實際訓練模型，不會自動安裝套件。

只檢查與保存抽樣，不訓練：

```powershell
python 1102.py --prepare-only --work-dir 1102_data_check
```

原始資料、既有訓練腳本與模型原始碼都不會被修改。若指定的工作目錄或最終
CSV 已存在，程式會停止以保護結果；另一次實驗可指定新的 `--work-dir` 與
`--output`。每次訓練從頭開始，沒有斷點續跑功能。

## 實驗規則

- 固定 `seed=1102`，從排序後的訓練序列 ID 中均勻抽出 **5,000 條不同序列**。
  每條由 `city_id, store_id, product_id` 識別。保存完整序列，不是抽 5,000 筆日資料。
  Parquet 以批次掃描，只保留抽中序列的完整觀測，降低讀取記憶體。
- 三個模型共用同一份 ID 順序、最後 28 天歷史，以及相同 7 天測試真值。
  原始資料預期為 90 天訓練、2024-06-26～2024-07-02 測試。
  日期不連續、缺值、重複觀測或 Train/Eval ID 不一致均會報錯，不補零。
- **TACTiS-2**：單一模型聯合處理全部 5,000 條；`bagging_size=None`。
  每一個 optimizer step 都包含全部序列，每個 epoch 遍歷全部合格訓練視窗。
  先訓練 marginal，再凍結 marginal 訓練 copula；每階段最多 30 epochs，
  連續 8 epochs 驗證未改善就停止。使用小型架構，不是論文 Traffic 設定的重現。
- 最後 7 個訓練日只用於驗證與選 checkpoint。訓練視窗目標不跨入驗證段；
  不使用 eval 選模型，也不在選好 checkpoint 後重訓。最終預測歷史可以包含
  已觀測的驗證週，三模型相同。
- 注意力以 query 分塊及 checkpoint 重算降低峰值記憶體；每個 query 仍看到
  全部合法 keys，所有分塊的梯度合併後才更新一次權重。這不會拆成不同模型。
  此處只改寫記憶體計算方式，原始模型檔案不變；數值與梯度等價性由測試檢查。
- **Chronos**：原版 `amazon/chronos-t5-tiny`，固定 revision，使用預訓練推論，
  不微調。預設每次推論 32 條獨立序列；這不會訓練多個模型。
- **sktime-m4**：沿用專案既有 `sales_naive.py` 的
  `NaiveForecaster(strategy="last", sp=1)`；每條序列實際呼叫 sktime fit/predict。
  `sktime-m4` 是方法集合，這次比較其中 Naive，沒有宣稱重現整套 M4 方法。
- 兩個機率模型各產生 20 個樣本，以中位數作點預測；Naive 是確定性預測。
  不裁切負值。主排名為 MAE，另報 RMSE、WAPE（比例，乘 100 才是百分比），
  均越低越好。實際值總和為零時 WAPE 留空。耗時包含模型載入與訓練，
  不能當作單純推論速度比較。

## 輸出

- **`1102.csv`**：全部三個模型成功才產生，預設 35,000 列，每條序列每個測試日一列。
  欄位為 `city_id, store_id, product_id, dt, actual, tactis2, chronos, sktime_m4_naive`。
- **`1102_artifacts/metrics.csv`**：三個模型的 MAE 排名、RMSE、WAPE、觀測數及總耗時。
- `selected_series.csv`、`selected_data.npz`：抽中 ID 與三個模型共用的資料。
- `manifest.json`：參數、版本、資料與原始碼 SHA-256、模型設定與比較規則。
- `best_stage1.pt`、`best_stage2.pt`、`tactis2.pt`、`training_log.json`：權重與訓練紀錄。
- `*_samples.npz`、`*_predictions.npy`：樣本與各模型預測。
- `status.json`：目前階段或失敗原因；失敗時不會把部分結果當作完整 `1102.csv`。

5,000 條聯合模型仍有很大的運算量。分塊降低記憶體，沒有消除二次方的計算成本；
尤其 copula 每個樣本需依序抽取 35,000 個變數，完整預測可能很久。
`--attention-block 64` 可進一步降低注意力峰值記憶體，會增加計算開銷。
不同硬體或套件版本的數值可能略有差異，固定 seed 保證同版本下的抽樣可重現。

## 驗證

```powershell
python -m unittest test_1102 -v
```

測試比較分塊前後的注意力輸出、copula loss、輸入與全部參數梯度，並檢查
抽樣與原本 loader 一致、不依賴測試真值、缺日拒絕、小型聯合模型兩階段訓練
與預測、CSV 對齊，以及用上游內附的本機 dummy Chronos 權重實際跑通三模型
入口與 CSV 輸出。dummy 權重只用於離線測試，正式預設仍為預訓練的
`amazon/chronos-t5-tiny`。小資料測試結果不代表完整 5,000 條的模型表現。

PyTorch 分塊內使用官方
[scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。
