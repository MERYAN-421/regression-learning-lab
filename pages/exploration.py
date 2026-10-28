import streamlit as st

st.set_page_config(page_title="Exploration | Regression Lab", page_icon="🔍", layout="wide")

st.title("🔍 探索性資料分析 (Exploration & EDA)")
st.markdown("---")

st.markdown(
    """
    ### 資料探索與特徵分佈
    在此階段你可以：
    - 檢視各數值欄位的描述性統計量 (Mean, Std, Min, Max, Quantiles)
    - 繪製單變量分佈直方圖 (Histogram & KDE)
    - 檢視相關係數矩陣熱力圖 (Correlation Heatmap)
    - 繪製特徵與目標變數之散佈圖 (Scatter Plots)
    """
)

if "dataset" not in st.session_state:
    st.warning("⚠️ 尚未載入資料集，請先至 [Dataset] 頁面載入或上傳資料。")
else:
    df = st.session_state["dataset"]
    st.subheader("描述性統計")
    st.dataframe(df.describe())
