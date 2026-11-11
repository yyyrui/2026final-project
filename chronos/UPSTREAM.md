# Chronos-2 Small

本專案預設使用 `autogluon/chronos-2-small`，官方列出的最小 Chronos-2 checkpoint：
28M 參數（27,934,624），float32 權重約 112 MB。
`amazon/chronos-2` 與 `autogluon/chronos-2-synth` 都是 120M。
較少參數是本次降低運算需求的選擇依據，不是所有硬體上的速度保證。

- 官方模型列表：https://github.com/amazon-science/chronos-forecasting#available-models
- 模型卡：https://huggingface.co/autogluon/chronos-2-small
- 固定 revision：`ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a`
- 設定：https://huggingface.co/autogluon/chronos-2-small/blob/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a/config.json
- API：https://github.com/amazon-science/chronos-forecasting/blob/0aba28e9360b6355d5a919596c90b16c25939f02/src/chronos/chronos2/pipeline.py
- 查核日期：2026-11-07

`model.json` 記錄模型、來源、版本與套件來源雜湊。
`chronos2-small-config.json` 是上述固定模型版本的官方設定備份。
原生分位數為 `[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]`。
比較程式直接取 q=0.5，不使用舊版 Chronos 的 Monte Carlo 抽樣介面。

## 程式來源

`chronos-forecasting/` 已替換成官方 **2.3.2** release 的執行期原始碼。
來源為 PyPI sdist，SHA-256：
`910b0891310b74598a937bb9fe8447ea16393bcc2e1ad8d328c1f8099f909201`。
發布 commit：`0aba28e9360b6355d5a919596c90b16c25939f02`。

保留上游 `src/`、套件設定、README、LICENSE、NOTICE，以及
`test/dummy-chronos2-model/`。原本舊模型的 notebook、評估結果、訓練腳本與 dummy 權重已移除。
未修改上游執行期原始碼；其中的舊家族類別與共用工具是官方套件內部相依，
不代表比較入口仍會執行它們。兩入口只載入 `Chronos2Pipeline`，並拒絕其他家族的 checkpoint。

專案根目錄的 `requirements.txt` 會以 editable 模式安裝這份原始碼。
真實 Small 權重下載到被 Git 忽略的 `.cache/huggingface/`。
隨附的 dummy 權重僅用於離線程式測試，不可用來報告模型比較成績。
