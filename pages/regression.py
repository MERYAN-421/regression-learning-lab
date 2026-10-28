import streamlit as st

st.title("📈 Regression Model")
st.subheader("Ordinary Least Squares (OLS) & Statistical Summary")

st.info(
    """
    🚧 **Module Status: Under Development (Scheduled for Milestone 3)**
    
    This module will train Multiple Linear Regression models using statsmodels and scikit-learn,
    generating full econometric summary tables, estimated coefficients, confidence intervals, and LaTeX formulas.
    """
)

st.divider()

st.header("🎯 Module Objectives")
st.markdown(
    """
    - **Parameter Estimation**: Solve the normal equations $\\hat{\\beta} = (X^T X)^{-1} X^T y$ via Ordinary Least Squares.
    - **Hypothesis Testing on Coefficients**: Evaluate individual two-tailed hypothesis tests ($H_0: \\beta_j = 0$ vs $H_1: \\beta_j \\neq 0$) using Student's $t$-distribution.
    - **Overall Model Significance**: Test joint significance across all explanatory variables via the omnibus $F$-test.
    - **Mathematical Transparency**: Formulate the fitted regression equation in interactive mathematical notation ($LaTeX$).
    """
)

st.divider()

st.header("📑 Planned OLS Summary Output Specification")

st.markdown(
    """
    In Milestone 3, fitting the model will render an econometric report containing:
    """
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 1. Global Model Fit Statistics")
        st.markdown(
            """
            - **$R^2$ & Adjusted $R^2$**: Proportion of variance explained, penalizing superfluous predictors ($p$).
            - **$F$-statistic & Prob($F$)**: Global omnibus test for linear association.
            - **Log-Likelihood, AIC, and BIC**: Information criteria for model selection and parsimony comparison.
            """
        )

with col2:
    with st.container(border=True):
        st.markdown("#### 2. Regressor Coefficients & Inferences")
        st.markdown(
            """
            - **Point Estimates ($\\hat{\\beta}_j$)**: Marginal effect of $X_j$ holding all other features constant.
            - **Standard Errors ($SE(\\hat{\\beta}_j)$)**: Precision of the estimated coefficient.
            - **$t$-statistic & $p$-value**: Statistical significance assessment against $\\alpha = 0.05$.
            - **95% Confidence Intervals**: Interval estimation for true population parameters $[\\beta_{lower}, \\beta_{upper}]$.
            """
        )

st.divider()

st.header("⚙️ Planned Regression Controls (Wireframe)")

c1, c2, c3 = st.columns(3)
with c1:
    st.selectbox("Regression Estimator (Preview)", [
        "Ordinary Least Squares (OLS)",
        "Ridge Regression (L2)",
        "Lasso Regression (L1)",
        "ElasticNet (L1 + L2)"
    ], disabled=True)
with c2:
    st.checkbox("Fit Intercept Term ($\\beta_0$)", value=True, disabled=True)
with c3:
    st.selectbox("Covariance Estimator (Preview)", ["Standard (Non-robust)", "HC0 (White's Heteroscedastic)", "HC3 (Jackknife Robust)"], disabled=True)

st.caption("Statistical model fitting will be managed by `src/models/` and statsmodels in Milestone 3.")
