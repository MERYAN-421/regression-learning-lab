import streamlit as st

st.set_page_config(page_title="Regression Models | Regression Lab", page_icon="⚙️", layout="wide")

st.title("⚙️ 迴歸模型配適 (Regression Modeling)")
st.markdown("---")

st.markdown(
    """
    ### 建立與訓練迴歸模型
    支援的模型演算法包含：
    1. **Ordinary Least Squares (OLS / 線性迴歸)** (支援 statsmodels 輸出完整檢定報表)
    2. **Ridge Regression (L2 正則化)**
    3. **Lasso Regression (L1 正則化)**
    4. **ElasticNet (彈性網絡)**
    5. **Polynomial Regression (多項式迴歸)**
    6. **Decision Tree & Random Forest Regressor (樹狀模型)**
    7. **Support Vector Regressor (SVR)**
    """
)

st.selectbox(
    "選擇演算法：",
    [
        "Linear Regression (OLS)",
        "Ridge Regression",
        "Lasso Regression",
        "ElasticNet",
        "Polynomial Regression",
        "Random Forest Regressor",
        "Support Vector Regressor (SVR)"
    ]
)
