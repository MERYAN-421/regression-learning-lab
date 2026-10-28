import streamlit as st

st.title("📊 Data Exploration")
st.subheader("Exploratory Data Analysis (EDA) & Preprocessing")

st.info(
    """
    🚧 **Module Status: Under Development (Scheduled for Milestone 2)**
    
    This module will provide interactive tools to explore data distributions, detect missing values,
    examine correlation matrices, and perform essential statistical transformations.
    """
)

st.divider()

st.header("🎯 Module Objectives")
st.markdown(
    """
    - **Descriptive Statistics**: Compute location (mean, median), dispersion (standard deviation, IQR), and shape metrics (skewness, kurtosis).
    - **Missing Value Handling**: Visualize missingness patterns and test various imputation techniques (median, iterative, or indicator creation).
    - **Correlation Matrix**: Uncover pairwise linear relationships using interactive heatmaps to preemptively identify multicollinear feature pairs.
    - **Target Distribution**: Inspect target variable skewness and test log/Box-Cox transformations to satisfy normality assumptions.
    """
)

st.divider()

st.header("🛠️ Planned Exploration Tools")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 📈 Distribution & Anomaly Analysis")
        st.markdown(
            """
            - **Univariate Histograms & KDE**: Compare empirical distributions against Gaussian reference curves.
            - **Tukey Boxplots**: Identify potential statistical outliers ($1.5 \\times \\text{IQR}$ rule).
            - **Quantile Inspection**: Evaluate 1st, 5th, 95th, and 99th percentiles for heavy tails.
            """
        )

with col2:
    with st.container(border=True):
        st.markdown("#### 🔗 Bivariate & Multilateral Relationships")
        st.markdown(
            """
            - **Interactive Heatmap**: Pearson ($r$) and Spearman ($\\rho$) rank correlations with significance p-value filtering.
            - **Feature-Target Scatter Plots**: Interactive Plotly scatter diagrams with preliminary OLS trendlines.
            - **Pairplot Inspection**: Visual triage for multi-feature interaction terms.
            """
        )

st.divider()

st.header("⚙️ Planned Exploration Controls (Wireframe)")

c1, c2, c3 = st.columns(3)
with c1:
    st.selectbox("Select Numerical Feature (Preview)", ["-- Available in Milestone 2 --"], disabled=True)
with c2:
    st.selectbox("Transformation Technique (Preview)", ["None (Raw)", "Log Transform", "Box-Cox", "StandardScaler"], disabled=True)
with c3:
    st.selectbox("Correlation Metric (Preview)", ["Pearson", "Spearman"], disabled=True)

st.caption("Exploratory visualization components will be powered by Plotly and modularized under `src/exploration/`.")
