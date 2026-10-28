import streamlit as st

st.set_page_config(page_title="Report | Regression Lab", page_icon="📝", layout="wide")

st.title("📝 實驗報告與匯出 (Experiment Report)")
st.markdown("---")

st.markdown(
    """
    ### 實驗摘要與報告生成
    自動彙整本次迴歸分析之關鍵發現與結果：
    - 資料集摘要與特徵前處理紀錄
    - 最佳模型挑選與參數配置
    - 評估指標統計表與關鍵圖表
    - 殘差診斷結論與模型適用限制
    """
)

col1, col2 = st.columns(2)
with col1:
    st.button("📥 匯出 Markdown 格式摘要")
with col2:
    st.button("📥 匯出 HTML / PDF 格式報告")
