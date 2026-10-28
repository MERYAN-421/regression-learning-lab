import streamlit as st

st.title("🔍 Feature Selection")
st.subheader("Filter, Wrapper, and Embedded Dimensionality Reduction")

st.info(
    """
    🚧 **Module Status: Under Development (Scheduled for Milestone 2)**
    
    This module will provide algorithmic feature selection and multicollinearity filtering to reduce
    high-dimensional feature spaces (14–39 variables) down to the most statistically informative predictor subsets.
    """
)

st.divider()

st.header("🎯 Module Objectives")
st.markdown(
    """
    - **Multicollinearity Reduction**: Compute Variance Inflation Factors (VIF) to eliminate redundant regressors ($VIF > 5$ or $10$).
    - **Methodological Comparison**: Compare the selected subsets across Filter, Wrapper, and Embedded paradigms.
    - **Prevent Overfitting**: Observe how subset selection affects train vs. test out-of-sample generalization.
    """
)

st.divider()

st.header("🔬 Supported Feature Selection Paradigms")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### 1. Filter Methods")
        st.caption("Fast, model-agnostic statistical ranking")
        st.markdown(
            """
            - **Pearson Correlation ($r$)**: Linear correlation ranking with target variable.
            - **ANOVA F-statistic (`f_regression`)**: Linear hypothesis test ($F = \\frac{MS_{reg}}{MS_{res}}$).
            - **Mutual Information (`mutual_info_regression`)**: Non-parametric information theoretic metric capturing non-linear relationships.
            """
        )

with col2:
    with st.container(border=True):
        st.markdown("#### 2. Wrapper Methods")
        st.caption("Iterative model-driven greedy search")
        st.markdown(
            """
            - **Recursive Feature Elimination (RFE)**: Prunes the least significant weights recursively.
            - **Forward Sequential Feature Selection (SFS)**: Starts empty, greedily adds features maximizing cross-validated $R^2$.
            - **Backward SFS**: Starts full, greedily eliminates features minimizing score loss.
            """
        )

with col3:
    with st.container(border=True):
        st.markdown("#### 3. Embedded Methods")
        st.caption("Inherent sparsity regularization")
        st.markdown(
            """
            - **Lasso Regression (L1 Penalty)**: Drives non-essential regression coefficients exactly to zero.
            - **ElasticNet**: Combines L1 sparsity with L2 stability for collinear feature clusters.
            - **Regularization Path Visualizer**: Traces coefficient shrinkage as penalty parameter $\\alpha$ varies.
            """
        )

st.divider()

st.header("⚙️ Planned Selection Configuration (Wireframe)")

c1, c2, c3 = st.columns(3)
with c1:
    st.selectbox("Selection Algorithm (Preview)", [
        "Pearson Correlation Filter",
        "ANOVA F-regression",
        "Mutual Information",
        "Recursive Feature Elimination (RFE)",
        "Forward Sequential Selection (SFS)",
        "Backward Sequential Selection (SFS)",
        "Lasso L1 Sparsity Selection"
    ], disabled=True)
with c2:
    st.slider("Target Number of Features $k$ (Preview)", min_value=3, max_value=25, value=8, disabled=True)
with c3:
    st.number_input("Max VIF Threshold (Preview)", min_value=2.0, max_value=15.0, value=5.0, step=0.5, disabled=True)

st.caption("Feature selection algorithms will be encapsulated in `src/features/` and integrated in Milestone 2.")
