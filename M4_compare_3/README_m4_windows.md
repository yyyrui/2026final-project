# Windows：完整 M4 與 TACTiS-2 / Chronos / sktime 基準

這份程式依照「100,000 條同時聯合建模，不能分批」製作。沒有序列子集、分組模型、序列 bagging 或自動降規。資料下載及套件安裝全部在第一個 `# %%` chunk；後續模型載入強制使用本地檔案。

**這是資料準備與實驗程式，不是已完成的 100,000 維模型實驗。一般 Windows 桌機無法執行這個原版 dense copula 規模，程式會在訓練前明確停止。**

## Windows 執行

先安裝 Python 3.11 64-bit（Python 本身不能由尚未啟動的 Python chunk 安裝）。把 `m4_windows_all_joint.py` 放在有足夠磁碟空間的資料夾。PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe .\m4_windows_all_joint.py
```

這個命令會依序安裝、下載、準備全部資料、檢查資源；通過後才訓練和評估。預設安裝 PyTorch 2.5.1 CUDA 12.1 wheel，需要相容的 NVIDIA 驅動；沒有 NVIDIA GPU 時可指定 `--device cpu --torch-index cpu`，但不能因此消除記憶體瓶頸。不依賴 Bash、Git LFS、Unix 檔案鎖或 Colab。

只產生完整資料（可在一般 Windows 電腦進行）：

```powershell
.\.venv\Scripts\python.exe .\m4_windows_all_joint.py --mode prepare --output-dir .\prepare_result
```

下載完成後，不再連網的資源檢查：

```powershell
.\.venv\Scripts\python.exe .\m4_windows_all_joint.py --offline --mode check --output-dir .\check_result
```

每次使用新的輸出目錄，避免不同實驗結果混用。若資料與套件已經準備好，重跑必須使用 `--offline`。第一個 chunk 安裝套件後，在已載入舊套件的 VS Code Interactive/Jupyter kernel 中應重啟 kernel 再執行；建議直接使用上述 `.py` 指令。程式提供 `# %%` 分段，不是 `.ipynb`。

## 資料格式

| 頻率 | 序列數 | 官方預測步數 | MASE lag |
|---|---:|---:|---:|
| Yearly | 23,000 | 6 | 1 |
| Quarterly | 24,000 | 8 | 4 |
| Monthly | 48,000 | 18 | 12 |
| Weekly | 359 | 13 | 1 |
| Daily | 4,227 | 14 | 1 |
| Hourly | 414 | 48 | 24 |
| 合計 | 100,000 | 每序列依官方設定 | — |

六種頻率是官方資料屬性，不會各自訓練六個 TACTiS2。所有序列合併為一個固定 ID 順序。Train 與 Test 保持分離；Test 只用於評分。僅刪除 CSV 尾端的空白欄位，保留真實的零；遇到內部缺值、重複 ID 或數量不足立即報錯。

`m4_work/prepared/` 的檔案：

- `m4_all_series.parquet`：100,000 列；`train_values`、`test_values` 為不等長 list，完整保留全部原始觀測。可以 `pd.read_parquet(...)` 讀取。
- `series_index.csv`：共同列順序、ID、頻率、官方 horizon、季節週期及 Train 長度。
- `tactis_history.npy`：`[1, 100000, context]`，全部序列同時輸入。
- `chronos_history.npy`：`[100000, context]`，全部序列一次呼叫 `predict`。
- `tactis_latest_training_window.npz`：完整聯合的 `hist` 與 `future`，都取自 Train。
- `official_targets.npz`：`actual[100000,48]` 與 `score_mask`；超出官方 horizon 的位置為 NaN，不作為觀測、訓練標籤或評分項目。
- `data_report.json`：實際序列／觀測數、最短歷史、共同訓練視窗數等。

已於本機實際轉換並檢查全部資料：24,002,047 筆 Train 觀測、1,277,717 筆 Test 目標、最短 Train 13 筆。`m4_prepared_data.zip` 提供上述 prepared 檔案；它不含套件或模型權重，也不等同完整離線 work-dir。Python 第一個 chunk 仍會準備正式執行需要的原始檔與權重。

## 嚴格聯合建模的限制與實驗定義

M4 是不同頻率、長度與領域的單變量資料集合，沒有天然共同日曆。本程式將各自 Train 的最後一筆定義成相對時間 0。**年資料的一步仍是一年、月資料的一步仍是一月；相對位置相同不代表同一日期或有真實共同衝擊。** 這是額外的混合頻率聯合實驗假設，不能宣稱重現 TACTiS-2 論文或標準 M4 多變量實驗。

原版模型需要密集、等長輸入。預設 `context=6`、`train_horizon=6`，在每條序列尾端對齊，使用所有序列都擁有的共同視窗。每次更新包括全部 100,000 條；每個 epoch 遍歷所有可用共同相對 offset。**完整歷史保存在資料檔，但長序列超過共同視窗範圍的較早歷史不參與 TACTiS2 訓練；「全部序列」不等於「每個原始觀測都進入 loss」。** 不填補不存在的觀測、不複製短序列、不假造日期。實測預設設定只有 2 個共同訓練視窗，這會嚴重限制訓練資訊量；程式可驗證執行流程，但不保證研究上的有效性。增加 context/horizon 可能超過最短 Train，會明確報錯。

TACTiS2 使用專案內固定 commit 的核心、兩階段訓練（先 marginal flow，再凍結 flow 訓練 copula），`bagging_size=None`；模型大小與預設 20 epochs／階段為此入口的實驗設定，不是論文設定或已驗證收斂值。固定 epochs，不使用官方 Test 挑選超參數或 checkpoint，也未提供驗證集調參或斷點續訓。

預測時每次對所有序列聯合產生 6 步，將生成值接回歷史，遞迴到 48 步；對較短 horizon 只評分前面官方要求的部分。20 條 Monte Carlo 軌跡逐條生成，每條軌跡始終含全部序列；沒有把序列拆成批次。這是分塊遞迴預測，不是直接訓練 48 步模型。

## 基準與評分

- Chronos：固定 `amazon/chronos-t5-tiny` 與權重 revision，不重新訓練；最後 6 個觀測作為 context；一次呼叫涵蓋全部 100,000 條並預測 48 步。原版 Chronos 每條序列獨立，批次維度不是跨序列聯合分布。
- sktime：Naive、sNaive、Naive2、SES、Holt、Damped、Theta、Com，採專案 sktime-m4 方法的現代 API 改寫。**這是 8 種明列的基準，不是完整 sktime-m4 所有方法的重現**；不包含 ARIMA、MLP、RNN、Theta-BC。各方法依定義逐序列擬合，涵蓋全部 ID，不建立分組聯合模型。
- sktime 使用完整 Train，以估計月／小時季節性；因此與 TACTiS2／Chronos 的 6-step context **不同**，結果不應解讀成僅由模型架構造成的公平差異。這個選擇避免僅有 6 筆歷史卻要求估計 12／24 期季節性。
- 使用官方 horizon、逐序列 sMAPE 與 MASE，再對所有序列平均，避免六種頻率平均時錯誤給相同權重。MASE 分母來自完整官方 Train；零分母記為未定義並揭露數量。MAE 另存於逐序列表，因尺度不同不作主要排名。
- 未計算 OWA；沒有以季節性 Naive 假裝官方 Naive2 分母。
- 若 Train 有零或負值，sktime 季節調整明確使用加法；全正值使用乘法。這是適應資料的差異，不假稱完全等同官方 M4 乘法方法。
- sktime 擬合失敗寫入 `sktime_failures.jsonl`，不暗中換成 Naive；任何模型缺少序列時不發出完整結果標記、不輸出該模型的全量平均分。
- Chronos 預訓練資料可能包含 M4；本比較不宣稱嚴格 zero-shot。

## 記憶體與結果狀態

預設 N=100000、C=6、H=6、4 heads、float32：copula 的單一 attention product 為
`4 × (100000×6) × (100000×12) × 4 bytes = 11,520,000,000,000 bytes`，約 **11.52 TB**。這還不是完整訓練記憶體；程式採 6 倍主要張量作保守預算，約 69.12 TB，實際峰值仍須量測。多張 GPU 的顯存不會自動合併，這個入口也沒有模型分散式實作。僅換成 Windows、增加 batch size 或換 CUDA wheel 都無法解決。

資源不足時會有 `resource_report.json`、`status.json`（`blocked_before_training`），不會有模型分數或 `SUCCESS`。通過檢查仍非峰值保證；完整執行後才會產生預測、訓練日誌及 metrics。Chronos 的單次 100,000 條推論也可能耗盡顯存，本程式同樣不會自動改成小批次。

本機的縮小合成 fixture 測試只驗證程式介面、資料對齊、loss/backprop、樣本形狀與評分；不代表已跑完 M4、不代表 Windows/CUDA 已實測，也不提供模型準確度證據。驗證結果見 `validation_report.json`。

## 來源

- 使用者專案（固定 commit）：https://github.com/yyyrui/2026final-project/tree/70db6cf5891af48b3032f939dc8f6634d1693bee
- TACTiS-2：https://github.com/ServiceNow/TACTiS （本入口下載專案已納入的 TACTiS2 核心，避免直接取 upstream main 的不同版本）
- 官方 M4：https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
- M4 頻率設定參考：https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/m4.py
- Chronos：https://github.com/amazon-science/chronos-forecasting
- Chronos 固定權重：https://huggingface.co/amazon/chronos-t5-tiny/tree/29d808298f1a62493e7b9a5e08529d0d930fa189
- sktime 方法原始定義：https://github.com/yyyrui/2026final-project/blob/70db6cf5891af48b3032f939dc8f6634d1693bee/sktime-m4/scripts/construct.py
