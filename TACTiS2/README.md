# 11/4 前 找+傳資料集給老師還有跑benchmark
建議妳先從這篇ICLR 2024讀起

[https://arxiv.org/abs/2310.01327](https://arxiv.org/abs/2310.01327)

程式在[https://github.com/ServiceNow/tactis](https://github.com/ServiceNow/tactis)

這運用Empirical CDF & Copula觀念對高維度時間數列建構聯合機率分配，之後討論時再跟妳解釋Copula。

2.動手跑這個例子
[https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/hierarchical_forecasting.html?utm_source=chatgpt.com](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/hierarchical_forecasting.html?utm_source=chatgpt.com)
接著它的loss應該不是Poisson Mixture，妳可以用AI協助研究是否能接它的Poisson Mixture Loss, see
[https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html?utm_source=chatgpt.com#poisson-mixture-mesh-pmm](https://nixtlaverse.nixtla.io/neuralforecast/losses.pytorch.html?utm_source=chatgpt.com#poisson-mixture-mesh-pmm)

程式在[https://github.com/Nixtla/neuralforecast?utm_source=chatgpt.com](https://github.com/Nixtla/neuralforecast?utm_source=chatgpt.com)
|網址|是什麼？|你要拿它做什麼？|

|第一個：Hierarchical Forecast|完整的階層式預測教學|先照著跑，取得原版結果|
|第二個：Poisson Mixture Mesh|套件已提供的 loss 說明|研究怎麼放進第一個範例|
|第三個：GitHub|NeuralForecast 的原始程式碼|遇到不清楚的實作或相容性問題時查閱|
