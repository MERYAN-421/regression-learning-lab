import streamlit as st

st.set_page_config(page_title="Feature Selection | Regression Lab", page_icon="🎯", layout="wide")

st.title("🎯 特徵選擇與工程 (Feature Selection)")
st.markdown("---")

st.markdown(
    """
    ### 特徵工程與多重共線性檢驗
    本模組提供特徵篩選與轉換功能：
    - **特徵標準化與編碼** (StandardScaler, MinMaxScaler, One-Hot Encoding)
    - **多重共線性檢測 (VIF)**: 計算 Variance Inflation Factor 排除高度相關變數
    - **特徵重要度與相關度篩選**: F-Regression, Mutual Information, Lasso 稀疏篩選
    - **自定義選擇預測變數 (X) 與目標變數 (y)**
    """
)

if "dataset" not in st.session_state:
    st.warning("⚠️ 尚未載入資料集，請先至 [Dataset] 頁面載入或上傳資料。")
