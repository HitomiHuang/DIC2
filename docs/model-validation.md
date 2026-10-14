# 模型設定與驗證紀錄

[返回專案說明](../README.md)

以下保留專案原有的資料設定、計算方法與驗證紀錄。

## 資料集與預設設定

資料來源：[Avik Jain / 100-Days-Of-ML-Code 的 50_Startups.csv](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)。內建檔案：[dist/50_Startups.csv](../dist/50_Startups.csv)。

| 項目 | 設定 |
| --- | --- |
| 樣本 | 50 家公司，無缺值、無重複資料列 |
| 預測目標 | Profit |
| 原始輸入 | R&D Spend、Administration、Marketing Spend、State |
| 州別 | California、Florida、New York |
| 類別編碼 | One-hot，移除第一類；California 為基準 |
| 編碼後特徵 | 3 個支出欄位 + State_Florida + State_New York，共 5 個 |
| 訓練／測試 | 40／10 筆，test_size=0.2 |
| 隨機種子 | random_state=42 |
| 比較範圍 | k=1…5，共 25 組結果 |
| 最終迴歸模型 | LinearRegression，包含截距 |

數值欄位以**訓練集**中位數補值並標準化；類別欄位以訓練集眾數補值再編碼。測試集不參與訓練或特徵選取。

原始資料 SHA-256：

~~~text
c6075ca5a7551243b31193bad9228f0a1b9468cdde2475c6586c56dcfac5630d
~~~

## 五種方法如何計算

這裡比較的是**特徵選取方法**。Lasso／ElasticNet 用於排序，選好特徵後，五種方法均重新擬合 LinearRegression；不是直接比較五種不同的最終預測模型。

| 方法 | 選取規則 |
| --- | --- |
| Lasso | 對標準化目標 y 以 α=0.01 訓練，按係數絕對值排序，取前 k 個 |
| Forward SFS | 每一步加入使訓練集三折 CV 平均 MSE 最低的特徵 |
| RFE | 以 LinearRegression 反覆擬合，移除係數平方最小的特徵，直到剩 k 個 |
| ElasticNet | α=0.01、l1_ratio=0.5，按係數絕對值排序，取前 k 個 |
| Exhaustive Search | 枚舉大小為 k 的全部子集，選取訓練集三折 CV 平均 MSE 最低者 |

CV 使用 KFold(n_splits=3, shuffle=True, random_state=42)，自訂種子會同步套用；三個 fold 的 MSE 等權平均。係數採穩定排序，零係數同分時保留原欄位順序。

**前處理限制**：內層 CV 使用已在外層訓練集擬合的前處理結果，沒有在各 fold 重新擬合 imputer/scaler。本次 50 Startups 無缺值，OLS 對這種可逆線性縮放不變，獨立驗證一致；若用含缺值的自訂資料進行嚴格模型選擇，應改用各 fold 獨立擬合前處理的 Pipeline。

## 計算驗證

**驗證日期：2026-10-14。25 組特徵選取與指標全部通過。**

[verify_calculations.py](../verify_calculations.py) 的獨立驗證部分不呼叫原分析函式，也不使用 sklearn 的選取器、估計器或指標函式，而是另外以 NumPy 實作：

1. 重建資料切分與五個特徵，確認訓練／測試集合沒有交集。
2. 用 coordinate descent 核對 Lasso／ElasticNet 排序。
3. 以最小平方法逐步消除特徵，核對 RFE。
4. 逐一重算三折 CV，核對 Forward SFS 與完整子集搜尋。
5. 用 numpy.linalg.lstsq 重算預測，再依公式核對 RMSE、R²、MAE。
6. 檢查原始支出單位的係數與截距能重現每筆測試預測。
7. 另以原分析程式重跑，確認預先計算結果仍與目前程式一致。
8. 核對瀏覽器兩張比較圖的全部座標，以及瀏覽器重新運算匯出的 25 組結果。

| 驗證項目 | 結果 |
| --- | --- |
| 25 組特徵集合 | 全部一致 |
| NumPy 重算 RMSE／R²／MAE 最大絕對差 | 約 1.82 × 10⁻¹¹ |
| 原始單位係數重建預測最大絕對差 | 約 8.00 × 10⁻¹¹ |
| 瀏覽器重新計算最大指標絕對差 | 約 1.46 × 10⁻¹¹ |
| RMSE／R² 共 50 個圖上座標 | 全部符合數值與座標映射 |

上述差異均為浮點運算誤差。完整紀錄：[稽核 JSON](calculation-audit.json)、[驗證結果 CSV](verified-results.csv)、[瀏覽器重新計算結果](browser-recalculated-results.csv)、[圖表座標](browser-chart-coordinates.json)。

### 預設 RMSE

| 方法 | k=1 | k=2 | k=3 | k=4 | k=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lasso | 7,714.33 | 8,206.33 | 8,995.91 | 9,055.96 | 9,055.96 |
| Forward SFS | 7,714.33 | 9,152.27 | 8,995.91 | 9,015.11 | 9,055.96 |
| RFE | 7,714.33 | 8,206.33 | 8,995.91 | 9,055.96 | 9,055.96 |
| ElasticNet | 7,714.33 | 8,206.33 | 8,995.91 | 9,055.96 | 9,055.96 |
| Exhaustive Search | 7,714.33 | 9,152.27 | 8,995.91 | 9,316.54 | 9,055.96 |

預設切分下，五種方法在 k=1 都選取 R&D Spend，RMSE 約 **7,714.33**、R² 約 **0.926511**。畫面起始選取 k=2 是為方便比較方法，**不代表 k=2 最佳**。

### 指標公式與解讀

對測試集實際值 y、預測值 ŷ 與樣本數 n：

~~~text
RMSE = sqrt(sum((y - ŷ)²) / n)
R²   = 1 - sum((y - ŷ)²) / sum((y - mean(y))²)
MAE  = sum(abs(y - ŷ)) / n
~~~

- RMSE 越低越好，單位與 Profit 相同；R² 越高越好，可能為負數。
- 曲線重疊代表方法選到相同特徵或得到相同預測，並非漏畫曲線。
- 縱軸依目前顯示數值縮放，不一定從零開始；切換圖例可能改變軸範圍。
- 測試集只有 10 筆，結果受切分影響；不能單憑特徵變多、測試誤差上升就斷言過度擬合。
- 若據測試集分數挑選模型，正式評估應再使用獨立資料或巢狀 CV。
- 本站記錄實際切分、編碼與選取設定，不以參考網站的固定數值為正確答案，也不保證重現其 k=2 最佳結論。

## 重現驗證

本機環境：Python 3.14.3、NumPy 2.5.4、pandas 3.0.6、scikit-learn 1.9.1。瀏覽器使用 Pyodide 0.27.7 隨附套件，已另外逐筆比對。

建議先建立並啟用 Python 虛擬環境，再執行：

~~~bash
python -m pip install -r requirements.txt
python verify_calculations.py
python qa-engine.py
npm test
~~~

重新產生預設結果：

~~~bash
python prepare-defaults.py
python verify_calculations.py
~~~

qa-engine.py 檢查缺值、類別編碼、固定種子、原始單位係數與無效資料。qa-ui.mjs 使用 DOM 替身檢查圖表切換、圖例、特徵數、設定變更、worker 呼叫與匯出，**不能取代真正的瀏覽器測試**；本次另完成實際瀏覽器重新運算、匯出及桌面／手機版檢查。

