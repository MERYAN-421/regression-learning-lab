import streamlit as st

st.set_page_config(
    page_title="Regression Learning Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📈 Regression Learning Lab")
    st.markdown("### 迴歸分析互動學習實驗室")
    st.write(
        """
        歡迎來到 **Regression Learning Lab**！這是一個專為學習與實作迴歸分析設計的互動式平台。
        你可以從左側側邊欄選擇不同階段的工作流程：
        
        1. **Home**: 實驗室概覽與指南
        2. **Dataset**: 資料集載入與預覽
        3. **Exploration**: 探索性資料分析 (EDA) 與視覺化
        4. **Feature Selection**: 特徵選擇與工程
        5. **Regression**: 建立與訓練多元迴歸模型 (OLS, Ridge, Lasso, ElasticNet 等)
        6. **Evaluation**: 模型表現評估 (R², MSE, RMSE, MAE 等)
        7. **Diagnostics**: 迴歸診斷 (殘差常態性、同質變異性、共線性、離群值)
        8. **Comparison**: 跨模型成效比較
        9. **Report**: 匯出實驗分析報告
        """
    )

    st.info("👈 請點選左側側邊欄的頁面開始體驗完整流程！")

if __name__ == "__main__":
    main()
