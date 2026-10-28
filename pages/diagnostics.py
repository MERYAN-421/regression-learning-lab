import streamlit as st

st.set_page_config(page_title="Diagnostics | Regression Lab", page_icon="🩺", layout="wide")

st.title("🩺 殘差診斷與統計檢定 (Regression Diagnostics)")
st.markdown("---")

st.markdown(
    """
    ### Gauss-Markov 假設與統計診斷
    嚴謹的線性迴歸必須驗證殘差是否滿足基本假設：
    
    1. **殘差 vs 擬合值圖 (Residuals vs Fitted)**: 檢驗線性假定與同質變異性 (Homoscedasticity)。
    2. **常態 Q-Q 圖 (Normal Q-Q Plot)**: 檢驗殘差是否服從常態分佈 (Shapiro-Wilk 檢定)。
    3. **Scale-Location 圖**: 檢驗殘差變異數是否均勻 (Breusch-Pagan 檢定)。
    4. **Cook's 距離與影響點分析 (Cook's Distance & Leverage)**: 揪出對模型估計具關鍵影響的異常觀測值。
    5. **自相關性檢定**: Durbin-Watson 統計量檢定殘差是否存在自我相關。
    """
)
