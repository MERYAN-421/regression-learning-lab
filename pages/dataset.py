import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset | Regression Lab", page_icon="📁", layout="wide")

st.title("📁 資料集載入與管理 (Dataset)")
st.markdown("---")

st.markdown("### 選擇或上傳資料集")

data_source = st.radio(
    "資料來源：",
    ("內建範例資料集 (Sample Datasets)", "上傳自訂 CSV 檔案 (Upload CSV)"),
    horizontal=True
)

if data_source == "上傳自訂 CSV 檔案 (Upload CSV)":
    uploaded_file = st.file_uploader("選擇 CSV 檔案", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["dataset"] = df
        st.success("檔案載入成功！")
else:
    sample_name = st.selectbox(
        "選擇內建範例：",
        ["California Housing (加州房價)", "Diabetes (糖尿病指標)", "Tips (小費資料集)"]
    )
    st.info(f"已選擇範例資料集: {sample_name}")

if "dataset" in st.session_state:
    st.subheader("資料預覽")
    st.dataframe(st.session_state["dataset"].head())
