# Replicating and Extending the M4 Study with sktime


This is a companion repository for our paper: 

* [Markus Löning, Franz Király: “Forecasting with sktime: Designing sktime's New Forecasting API and Applying It to Replicate and Extend the M4 Study”, 2020; arXiv:2005.08067](https://arxiv.org/abs/2005.08067)

The repository contains the code for reproducing our results. Our code is heavily based on [sktime](https://github.com/alan-turing-institute/sktime).

* `Makefile` contains convenience commands for running the replication and
 extension from the command line,
* `scripts/` contains our code for replicating and extending the M4 study,
* `results/replicated/` contains our obtained predictive performance results,
* `requirements.txt` contains the list of required packages to run our scripts.

To reproduce our results, we recommend to download the [original repository of
 the M4 study](https://github.com/Mcompetitions/M4-methods), particularly we
  use 

* the [M4 data set](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset),  
* the individual [point forecast results](https://github.com/Mcompetitions/M4-methods/tree/master/Point%20Forecasts) for comparison,
* the [summary results file](https://github.com/Mcompetitions/M4-methods/blob/master/Evaluation%20and%20Ranks.xlsx) to cross-check results.

In addition, you may have to adjust paths for the input and output
 directories in `scripts/config.py`. 

## 本專案的銷售資料比較入口（2026-10-30）

銷售實驗使用專案根目錄的 `compare 10-30_SKTime.py`，以及
`scripts/sales_config.py` 裡依專案位置解析的輸入／輸出路徑：

- `dataset/train.parquet`、`dataset/eval.parquet`：與先前 TACTiS2 相同的原始資料。
- `outputs/full-comparison/`：先前已完成的 TACTiS2 分組結果；重新驗證後使用，不重訓。
- `outputs/compare-10-30_SKTime/`：本次 Naïve 預測與比較指標。
- `compare 10-30_SKTime.md`：自動生成的比較報告。

在根目錄的專案虛擬環境執行：

```bash
python "compare 10-30_SKTime.py"
```

`scripts/sales_naive.py` 沿用作者 `construct.py` 的普通
`NaiveForecaster(strategy="last")` 定義，以官方 sktime 1.2.0 的
`NaiveForecaster(strategy="last", sp=1)` 逐條預測，並核對輸出。
銷售入口使用根目錄 `requirements-comparison.txt`；本目錄的
`requirements.txt` 保留作者 sktime 0.4.0 環境，不應混裝。

原始 `replicate.py`／`config.py` 是 M4 專用流程，需要 M4 metadata 與 CSV；
銷售入口不會匯入這個帶有作者絕對路徑的 config，也不會改動作者的 M4 結果。
這是銷售資料介面移植與相同 Naïve 規則的實驗，不是 M4 原始環境的逐項重現。
