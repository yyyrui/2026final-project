# TACTiS2、Chronos 與 sktime M4 Naive 銷售預測比較

本專案只保留兩個實驗入口，使用 FreshRetailNet-50K 的日銷售資料，比較相同的三個模型：

| 入口 | 資料範圍 | TACTiS2 訓練方式 |
| --- | --- | --- |
| `test_5000.py` | seed=1102，固定抽取 5,000 條完整序列 | 單一聯合模型，每一步包含全部選取序列 |
| `test_all.py` | 自動讀取資料中的全部序列，目前為 50,000 條 | 單一聯合模型，不分群、不抽取序列子集 |

「5,000 條」指不同的 `(city_id, store_id, product_id)` 時間序列，不是 5,000 列日資料。
`test_all.py` 不提供縮小序列數或自動退回分群的選項。

## 目錄

```text
2026final-project/
├── TACTiS2/               # 上游模型原始碼及授權
├── chronos/               # 上游原始碼與離線測試用 dummy 權重
├── sktime-m4/
│   └── scripts/           # 原始方法定義與 sales_naive.py 適配器
├── dataset/
│   ├── train.parquet
│   └── eval.parquet
├── comparison/
│   ├── __init__.py
│   ├── data.py            # 資料檢查、抽樣或全量載入
│   └── runner.py          # 兩入口共用的三模型訓練、推論與評分
├── tests/                 # 程式正確性測試，不是正式實驗入口
├── test_5000.py
├── test_all.py
├── requirements.txt
├── README.md
├── .python-version
├── .gitignore
├── .gitattributes
└── .git/
```

`.venv/`、`.cache/`、`outputs/` 是本機環境、下載快取和執行產物，已排除在 Git 之外。
根目錄只放兩個實驗 `.py`；模型與共用程式仍需要各自的 Python 檔案。

## 安裝

使用 Python 3.10 或 3.11。`.python-version` 使用 3.10。

macOS / Linux：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

若有 NVIDIA GPU，先安裝適用的 CUDA PyTorch，再安裝 requirements。
入口的 `--device auto` 會在 CUDA 可用時使用 CUDA，否則使用 CPU；目前不支援 MPS 或多 GPU。
在 VS Code 執行 `Python: Select Interpreter`，選擇此專案 `.venv` 裡的 Python。

Chronos 正式執行使用固定的 `chronos-forecasting==1.5.2` 套件，避免本地較新的上游原始碼引入
Chronos-2 等額外依賴。`chronos/` 仍保留作為原始碼參考與測試資源，不需要額外 `pip install -e`。
首次正式推論會下載 `amazon/chronos-t5-tiny`，固定 revision
`29d808298f1a62493e7b9a5e08529d0d930fa189`；下載完成後才可使用 `--offline`。

資料由 Git LFS 管理。新電腦複製專案後，先執行：

```bash
git lfs install
git lfs pull
```

`train.parquet` 約 106 MB，`eval.parquet` 約 8.44 MB。若只有約 130 bytes，表示尚未下載實際資料。

## 執行

先檢查真實資料與保存選取結果，不訓練模型：

```bash
python test_5000.py --prepare-only --work-dir outputs/check5000
python test_all.py --prepare-only --work-dir outputs/check_all
```

正式實驗：

```bash
python test_5000.py --device cuda
python test_all.py --device cuda
```

沒有 NVIDIA GPU 時可省略 `--device cuda`，使用自動選擇的裝置。
預設輸出分別為 `outputs/sample5000/` 與 `outputs/all/`。
再次執行要換新的工作目錄，例如：

```bash
python test_5000.py --work-dir outputs/sample5000_run2
python test_all.py --work-dir outputs/all_run2
```

已有的工作目錄或最終預測檔不會覆寫；目前沒有中斷續訓功能。
使用 `--help` 查看其他參數。相對資料與輸出路徑都以專案根目錄為準，
所以從其他目錄使用入口的絕對路徑也可執行。

完整 50,000 條聯合訓練及抽樣的運算量很大。本程式以注意力 query 分塊及 checkpoint
重算降低記憶體峰值，每個 query 仍可讀取全部合法 keys，並非把序列拆成獨立模型。
這不會消除二次方計算成本，也不保證一般 CPU 或單張 GPU 能完成。
`--prepare-only` 通過只代表資料檢查成功，不代表完整訓練已完成或記憶體足夠。

## 比較規則

- 三個模型：TACTiS-2、原版 Chronos T5 tiny、sktime M4 **Naive**。
  M4 是方法集合，這裡只使用 `NaiveForecaster(strategy="last", sp=1)`，不額外比較其他 M4 方法。
- 只使用序列 ID、日期 `dt`、目標 `sale_amount`；不加入其他欄位。
- 訓練資料為 2024-03-28 至 2024-06-25，共 90 天；評估為 2024-06-26 至 2024-07-02，共 7 天。
- 同一入口的三個模型共用序列順序、最後 28 天歷史與相同的 7 天評估目標。
  資料缺日、重複、非有限值或 Train/Eval ID 不一致時停止，不自動補零。
- TACTiS2 沿用原 5,000 條實驗的小型架構，`bagging_size=None`。
  每個 epoch 遍歷全部合格訓練視窗，每次更新包含全部選取序列。
  先訓練 marginal，再凍結 marginal 訓練 copula；每階段最多 30 epochs，patience=8。
- train 的最後 7 天只作驗證與挑選 checkpoint；不使用 eval 挑模型，也不重新訓練驗證資料。
- Chronos 使用預訓練權重，不微調；推論批次不代表把 TACTiS2 分群。
- sktime Naive 逐條實際呼叫 fit/predict。`construct.py` 只作來源定義核對，不匯入其舊版依賴。
- TACTiS2、Chronos 預設產生 20 個樣本，取中位數；Naive 為確定性預測，不裁切負值。
- 依 MAE 排名，另提供 RMSE、WAPE。WAPE 是比例，乘 100 才是百分比；MAE、RMSE 不應標成百分比。
  耗時包含模型載入與訓練，不是單純推論速度。

新全量入口與 5,000 條入口共用小型架構，沒有沿用已移除的 `FULLTEST 1030.py` 的論文 Traffic
大架構，也沒有沿用 Colab 版的訓練 bagging 或額外 8 種 sktime 比較。
這是新的比較設定，舊流程的分數不能直接當作目前程式的結果。

## 輸出

每個工作目錄包含：

- `selected_series.csv`、`selected_data.npz`：選取的 ID 與完整訓練／測試矩陣。
- `manifest.json`：模式、資料及程式雜湊、版本、模型設定與比較規則。
- `best_stage1.pt`、`best_stage2.pt`、`tactis2.pt`、`training_log.json`：權重與訓練紀錄。
- `*_samples.npz`、`*_predictions.npy`：各模型的樣本或點預測。
- `metrics_partial.csv`：已完成模型的暫時分數。
- `metrics.csv`：三個模型全部成功後才產生的正式排名。
- `predictions.csv`：三模型皆成功後產生；5,000 條為 35,000 列，全量為 350,000 列。
  欄位：`city_id, store_id, product_id, dt, actual, tactis2, chronos, sktime_m4_naive`。
- `status.json`：`prepared`、`running`、`failed` 或 `complete`；只有 `complete` 代表整個實驗完成。

可用 `--output` 改變最終預測 CSV 的位置，其餘產物仍放在 `--work-dir`。

## 程式驗證

```bash
python -m unittest discover -s tests -v
```

測試包含資料對齊、完整序列抽樣／全量選取、LFS 指標提示、注意力數值與梯度等價性、
兩階段訓練、Naive 行為，以及從不同工作目錄執行全量入口的三模型端到端測試。
小型測試使用上游隨附的 dummy Chronos 權重，不能當作正式預訓練模型的實驗成績。

## GitHub 與 VS Code

此專案的遠端為 `https://github.com/yyyrui/2026final-project.git`，主分支為 `main`。
在 VS Code 開啟整個專案資料夾，在整合終端檢查：

```bash
git remote -v
git fetch origin
git status -sb
```

需要授權時依 VS Code 提示登入 GitHub。修改檔案後要 commit 並 push，GitHub 才會更新；
本機修改不會自動上傳。請保留 `.git/`、`.gitattributes` 與 `.gitignore`。
舊流程的已提交版本可從 Git 歷史找回；`outputs/` 和本機環境不會上傳。
