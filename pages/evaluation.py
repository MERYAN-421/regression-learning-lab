import streamlit as st

st.set_page_config(page_title="Evaluation | Regression Lab", page_icon="📊", layout="wide")

st.title("📊 模型成效評估 (Model Evaluation)")
st.markdown("---")

st.markdown(
    """
    ### 迴歸評估指標與視覺化
    量測預測值 $\hat{y}$ 與真實值 $y$ 之間的差距：
    
    - **$R^2$ (判定係數)**: 解釋變異百分比
    - **Adjusted $R^2$ (調整後判定係數)**: 考慮特徵數量後的擬合度
    - **MSE (均方誤差)** & **RMSE (均方根誤差)**: 對大誤差敏感的距離量測
    - **MAE (平均絕對誤差)**: 穩健的平均誤差指標
    - **MAPE (平均絕對百分比誤差)**: 相對百分比誤差
    """
)

st.info("模型訓練完成後，評估指標將於此處即時計算並繪製實際值 vs 預測值對比圖。")
