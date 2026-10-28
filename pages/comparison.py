import streamlit as st

st.set_page_config(page_title="Comparison | Regression Lab", page_icon="⚖️", layout="wide")

st.title("⚖️ 模型比較與競賽 (Model Comparison)")
st.markdown("---")

st.markdown(
    """
    ### 多模型並列比較
    綜合比較不同模型在訓練集與驗證集上的表現：
    - **指標橫向對比表** (Train vs Test $R^2$, RMSE, MAE)
    - **效能雷達圖 (Radar Chart)** 與柱狀比較圖
    - **殘差分佈疊加圖**
    - **過擬合 (Overfitting) 與欠擬合 (Underfitting) 警示診斷**
    """
)

st.info("請先於 [Regression] 頁面訓練至少兩個模型以進行交叉評比。")
