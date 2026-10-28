# Regression Learning Lab (迴歸分析互動學習實驗室)

一個基於 Streamlit 的互動式迴歸分析教學與實驗平台，協助使用者從資料探索、特徵選擇、模型擬合、評估指標、殘差診斷到模型比較與報告匯出，完成完整的迴歸分析工作流程。

---

## 專案結構 (Project Structure)

```text
regression-learning-lab/
│
├── app.py                     # 主入口程式 (Streamlit entry point)
├── requirements.txt           # 專案依賴套件清單
├── README.md                  # 專案說明文件
├── .gitignore                 # Git 忽略檔案設定
│
├── .streamlit/
│   └── config.toml            # Streamlit 主題與伺服器設定
│
├── pages/                     # 各工作流程頁面
│   ├── home.py                # 首頁：平台介紹與流程導引
│   ├── dataset.py             # 1. 資料集上傳與載入
│   ├── exploration.py         # 2. 探索性資料分析 (EDA)
│   ├── feature_selection.py   # 3. 特徵工程與選擇
│   ├── regression.py          # 4. 迴歸模型建立與擬合
│   ├── evaluation.py          # 5. 模型成效評估
│   ├── diagnostics.py         # 6. 殘差診斷與假設檢定
│   ├── comparison.py          # 7. 多模型對比與分析
│   └── report.py              # 8. 綜合報告生成與匯出
│
├── src/                       # 核心演算法與工具模組 (模組化程式碼)
├── data/                      # 範例資料集或快取資料
├── configs/                   # 設定檔 (如預設模型參數等)
├── tests/                     # 單元測試與功能測試
└── docs/                      # 相關說明文件與教材
```

---

## 快速開始 (Quick Start)

### 1. 啟用 Conda 環境
```bash
conda activate regression-lab
```

### 2. 安裝依賴套件
```bash
pip install -r requirements.txt
```

### 3. 啟動 Streamlit 應用程式
```bash
streamlit run app.py
```
