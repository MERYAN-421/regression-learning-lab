import streamlit as st

st.title("🧪 Assumption Diagnostics")
st.subheader("Statistical Tests & Diagnostic Plots for the Five Gauss-Markov Assumptions")

st.info(
    """
    🚧 **Module Status: Under Development (Scheduled for Milestone 4)**
    
    This module will provide rigorous statistical hypothesis testing and diagnostic visualization suites
    to verify all five classical assumptions of Multiple Linear Regression.
    """
)

st.divider()

st.header("🎯 Module Objectives")
st.markdown(
    """
    Ordinary Least Squares (OLS) produces **Best Linear Unbiased Estimators (BLUE)** only when the Gauss-Markov assumptions hold.
    Violations lead to biased coefficient estimates, deflated standard errors, and invalid statistical inferences ($p$-values).
    This module equips students with formal hypothesis tests and diagnostic plots to detect and understand these violations.
    """
)

st.divider()

st.header("🔬 The Five Classical Regression Assumptions")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 1. Linearity of the Data-Generating Process")
        st.markdown(
            """
            - **Assumption**: $E[y|X] = X\\beta$ (The relationship between regressors and expected response is linear).
            - **Diagnostic Plot**: **Residuals vs. Fitted Values** with LOWESS non-parametric curve.
            - **Formal Statistical Test**: **Ramsey RESET Test** (powers of fitted values $\\hat{y}^2, \\hat{y}^3$).
            - **Remediation**: Polynomial terms, log transforms, or generalized additive models.
            """
        )
    
    with st.container(border=True):
        st.markdown("#### 2. Independence of Error Terms (No Autocorrelation)")
        st.markdown(
            """
            - **Assumption**: $Cov(\\epsilon_i, \\epsilon_j) = 0$ for all $i \\neq j$.
            - **Diagnostic Plot**: **Autocorrelation Plot (ACF)** & Residuals vs. Time/Order.
            - **Formal Statistical Test**: **Durbin-Watson Test** ($DW \\approx 2.0$), **Ljung-Box Test**.
            - **Remediation**: Lagged endogenous variables, Newey-West HAC standard errors.
            """
        )

    with col2:
        with st.container(border=True):
            st.markdown("#### 3. Homoscedasticity (Constant Error Variance)")
            st.markdown(
                """
                - **Assumption**: $Var(\\epsilon_i|X) = \\sigma^2$ (Constant variance across all levels of $X$).
                - **Diagnostic Plot**: **Scale-Location Plot** ($\\sqrt{|\\text{Standardized Residuals}|}$ vs. Fitted).
                - **Formal Statistical Test**: **Breusch-Pagan Test**, **White's Test**.
                - **Remediation**: Weighted Least Squares (WLS), Huber-White robust standard errors (HC3).
                """
            )
        
        with st.container(border=True):
            st.markdown("#### 4. Normality of Residuals")
            st.markdown(
                """
                - **Assumption**: $\\epsilon \\sim \\mathcal{N}(0, \\sigma^2 I)$ (Required for valid $t$-tests and $F$-tests in finite samples).
                - **Diagnostic Plot**: **Normal Q-Q Plot** against standard Gaussian quantiles.
                - **Formal Statistical Test**: **Jarque-Bera Test** (Skewness/Kurtosis), **Shapiro-Wilk Test**.
                - **Remediation**: Box-Cox power transform on response, quantile regression.
                """
            )

with st.container(border=True):
    st.markdown("#### 5. No Perfect Multicollinearity")
    st.markdown(
        """
        - **Assumption**: $\\text{rank}(X) = p + 1$ (Columns of $X$ are linearly independent; $(X^TX)$ is invertible).
        - **Diagnostic Metric**: **Variance Inflation Factor (VIF)** ($VIF_j > 5$ indicates high collinearity; $VIF_j > 10$ indicates severe).
        - **Formal Matrix Condition**: **Condition Number** $\\kappa(X) = \\sqrt{\\lambda_{max} / \\lambda_{min}} > 30$.
        - **Remediation**: Drop collinear variables, Ridge regularization ($L_2$), or Principal Component Regression (PCR).
        """
    )

st.divider()

st.header("🎯 Influential Observations & Leverage Analysis")
st.markdown(
    """
    - **Cook's Distance ($D_i$)**: Identifies data points exerting disproportionate influence on the estimated $\\hat{\\beta}$ vector (Cutoff threshold: $D_i > \\frac{4}{n}$).
    - **Hat Matrix Leverage ($h_{ii}$)**: Pinpoints extreme values in feature space ($h_{ii} > \\frac{2(p+1)}{n}$).
    - **Residuals vs. Leverage Plot**: Combines studentized residuals with leverage to locate critical influential outliers.
    """
)

st.caption("Assumption testing suite and diagnostic plots will be modularized under `src/diagnostics/` in Milestone 4.")
