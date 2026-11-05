實驗模型 
 
TACTiS2 
[2310.01327] TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series 
專為多變量概率時間序列預測與插值設計的深度學習模型。它能靈活處理大規模變量、複雜分佈、時間未對齊、不均勻採樣頻率以及缺失值補全問題 
Chronos 
[2403.07815] Chronos: Learning the Language of Time Series 
核心邏輯就是將連續的時間序列打包成高維 Token（Patch Embedding），丟進 Encoder-only Transformer 進行特徵交互與學習，最後再將輸出的未來 Token 轉回時間序列數值 
 
sktime-m4 
[2005.08067] Forecasting with sktime: Designing sktime's New Forecasting API and Applying It to Replicate and Extend the M4 Study 
以最後一個觀察到的歷史數值作為未來的預測值 
 
資料集 
FreshRetailNet-50K 
Dingdong-Inc/FreshRetailNet-50K · Datasets at Hugging Face 
 
包含約 20% 的有機缺貨數據 ( 20% 是「缺貨」的比例。商品賣光了，店家仍然知道當天實際賣了多少，但不知道「如果還有貨，原本能多賣多少」。所以銷量欄可以有數字，真正看不到的是那部分需求。) 
包含來自18個主要城市898家門市的5萬個店鋪產品90天時間序列，詳細的每小時銷售數據，涵蓋865種易腐商品，並附有細緻的缺貨事件註解。本資料集獨有的每小時庫存狀態紀錄 
 
 
 

 
原先想法(COMPARISON.md) 
拆成小組後執行 
1,563 個分組全部成功，一組32個，各有 50,000 條序列、 350,000 筆預測 
如果你看到的是 MAE、RMSE、WAPE 那張比較表，依照剛剛確認的分組程式，它是把所有已完成分組的預測合起來，算出每個模型的整體分數。 所以只顯示一張表，不會每組各列一張。 
模型 
MAE 
RMSE 
WAPE 
TACTiS2（分組小型模型） 
0.5184334453 
0.8993926917 
43.45435432% 
Chronos  
0.4653382462 
0.8078348709 
39.00399022% 
Naïve（sktime) 
0.5116443241 
0.8596465181 
42.88529983% 
 
 
MAE: Mean Absolute Error 
RMSE:Root Mean Square Error 
WAPE:Mean Absolute Percentage Error 
 
這些數值都是越小越好 
 
第二步想法(tactis2_full_colab.ipynb) 
不要分組，全部資料進去跑->資料量太大了要訓練很久就沒跑了 
 
第三步想法(test_1102.py) 
抽樣5,000條然後去跑 
 
模型 
MAE 
RMSE 
WAPE 
TACTiS2 
44.1664% 
74.4074% 
37.2215% 
Chronos  
46.0146% 
80.2076% 
38.7791% 
Naïve（sktime) 
51.2189% 
86.9092% 
43.1650% 
 
 
 
