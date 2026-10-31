"""
Statistical Assumption Diagnostics Laboratory for Ordinary Least Squares (OLS) Regression.
Evaluates the five classical Gauss-Markov regression assumptions:
1. Linearity & Functional Form (Ramsey RESET Test, Residuals vs. Fitted with LOWESS)
2. Independence & Autocorrelation (Durbin-Watson, Ljung-Box Portmanteau, Residual ACF)
3. Homoscedasticity (Breusch-Pagan Test, White's Test, Scale-Location Plot)
4. Normality of Residuals (Jarque-Bera Test, Shapiro-Wilk, Normal Q-Q Plot, Histogram)
5. Multicollinearity (Variance Inflation Factor, Condition Number, Matrix Rank)
Plus influential observations analysis (Cook's Distance, Leverage).
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.data.models import DatasetMetadata, SplitResult
from src.models.model_result import OLSModelResult
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.diagnostics.result_models import (
    DiagnosticResult,
    DiagnosticSuiteResult,
    DiagnosticType,
    DiagnosticDecision,
    ApplicabilityStatus
)
from src.diagnostics.runner import (
    validate_active_model_for_diagnostics,
    run_full_diagnostic_suite
)
from src.diagnostics.plots import (
    plot_residuals_vs_fitted_lowess,
    plot_residuals_vs_time,
    plot_residual_acf,
    plot_scale_location,
    plot_normal_qq,
    plot_residuals_histogram_kde,
    plot_vif_horizontal_bar
)
from src.diagnostics.independence import compute_residual_acf
from src.diagnostics.interpretation import (
    EXOGENEITY_EDUCATIONAL_NOTE,
    GAUSS_MARKOV_VS_NORMALITY_NOTE
)


# -----------------------------------------------------------------------------
# PAGE TITLE & ACADEMIC HEADER
# -----------------------------------------------------------------------------
st.title("🧪 Statistical Assumption Diagnostics Laboratory")
st.subheader("Formal Econometric Hypothesis Testing & Residual Diagnostics for OLS Regression")

st.caption(
    "ℹ️ **Milestone 4 Active**: Statistical Assumption Diagnostics Laboratory. "
    "Subsequent comparative model diagnostics and reporting suites are "
    "Under Development and Scheduled for Milestone 5–6."
)

st.divider()

# -----------------------------------------------------------------------------
# SESSION STATE & PREREQUISITE VALIDATION
# -----------------------------------------------------------------------------
split_res: Optional[SplitResult] = st.session_state.get("dataset_split_result", None)
current_meta: Optional[DatasetMetadata] = st.session_state.get("active_dataset_metadata", None)
tracker: RegressionExperimentTracker = st.session_state.get("regression_tracker", RegressionExperimentTracker())

if split_res is None or current_meta is None:
    st.warning("⚠️ **Dataset & Partition Required**: Please select a dataset in the **Dataset Library** and create a split.")
    st.stop()

# Retrieve available models for active split
available_models = tracker.get_valid_models(
    current_split_fingerprint=split_res.split_fingerprint,
    current_dataset_fingerprint=current_meta.dataset_fingerprint
)

if not available_models:
    st.info("ℹ️ **No OLS Models Available**: Please fit an OLS regression model in the **Regression Laboratory** before running diagnostics.")
    st.stop()

# -----------------------------------------------------------------------------
# SECTION A: Active Model Selection & Verification Card
# -----------------------------------------------------------------------------
st.header("1. Model Selection & Specification Verification")

model_options = {m.model_id: f"{m.model_name} (ID: {m.model_id})" for m in available_models}

# Default to active_ols_model from session state if available
active_model_id = None
if "active_ols_model" in st.session_state and st.session_state["active_ols_model"] is not None:
    if st.session_state["active_ols_model"].model_id in model_options:
        active_model_id = st.session_state["active_ols_model"].model_id

selected_model_id = st.selectbox(
    "Select Fitted OLS Model for Assumption Diagnostics:",
    options=list(model_options.keys()),
    format_func=lambda mid: model_options[mid],
    index=list(model_options.keys()).index(active_model_id) if active_model_id else 0,
    key="diag_model_select"
)

active_model: OLSModelResult = tracker.get_model(selected_model_id)

# Validate model compatibility
is_valid_model, val_msg = validate_active_model_for_diagnostics(
    model=active_model,
    current_dataset_id=current_meta.dataset_id,
    current_dataset_fingerprint=current_meta.dataset_fingerprint,
    current_split_fingerprint=split_res.split_fingerprint
)

if not is_valid_model:
    st.error(f"❌ {val_msg}")
    st.stop()

# Model Metadata Card
with st.container(border=True):
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Dataset", current_meta.dataset_name)
    col_m2.metric("Split Strategy", split_res.strategy.capitalize())
    col_m3.metric("Train Samples (n)", f"{len(split_res.X_train):,}")
    col_m4.metric("Parameters (k)", f"{active_model.design_matrix_metadata.num_parameters}")

    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        st.markdown(f"**Feature Source**: `{active_model.feature_source_label}`")
        st.markdown(f"**Selected Original Features** ({len(active_model.selected_original_features)}): {', '.join(active_model.selected_original_features)}")
    with col_sub2:
        inf_badge = "✅ Full Rank (Inferentially Valid)" if active_model.is_valid_for_inference else "⚠️ Rank Deficient (Prediction-Only)"
        st.markdown(f"**Inference Status**: {inf_badge}")
        st.markdown(f"**Condition Number**: `{active_model.design_matrix_metadata.condition_number:.1f}` | **Rank**: `{active_model.design_matrix_metadata.matrix_rank} / {active_model.design_matrix_metadata.num_parameters}`")

st.divider()

# -----------------------------------------------------------------------------
# SECTION B: Econometric Foundations & Exogeneity Education
# -----------------------------------------------------------------------------
st.header("2. Methodological Foundations & Critical Caveats")

st.info(GAUSS_MARKOV_VS_NORMALITY_NOTE)
st.warning(EXOGENEITY_EDUCATIONAL_NOTE)

st.divider()

# -----------------------------------------------------------------------------
# SECTION C: Statistical Test Controls & Execution
# -----------------------------------------------------------------------------
st.header("3. Statistical Diagnostics Configuration")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    alpha_choice = st.selectbox(
        "Significance Level (α):",
        options=[0.01, 0.05, 0.10],
        index=1,
        help="Type I error tolerance. A smaller alpha (0.01) makes rejecting H0 more conservative.",
        key="diag_alpha_select"
    )

with col_ctrl2:
    opt_white = st.checkbox(
        "Include White's General Test",
        value=False,
        help="Evaluates general non-linear heteroscedasticity using squares and cross-products. In high dimensions, degrees of freedom may be exhausted."
    )
    opt_shapiro = st.checkbox(
        "Include Shapiro-Wilk Test",
        value=False,
        help="Performs Shapiro-Wilk normality test for n <= 5,000 observations."
    )

with col_ctrl3:
    opt_influence = st.checkbox(
        "Include Influence & Leverage Diagnostics",
        value=True,
        help="Computes Cook's distance, hat matrix leverage, and studentized residuals."
    )

# Session state diagnostic suite storage
if "diagnostic_suites" not in st.session_state:
    st.session_state["diagnostic_suites"] = {}

btn_run_diagnostics = st.button("🚀 Run Statistical Assumption Diagnostics", type="primary", key="btn_run_diag")

if btn_run_diagnostics:
    with st.spinner("Computing Ramsey RESET, Durbin-Watson, Ljung-Box, Breusch-Pagan, Jarque-Bera, and VIF..."):
        suite_res = run_full_diagnostic_suite(
            model_result=active_model,
            split_result=split_res,
            alpha=alpha_choice,
            include_white=opt_white,
            include_shapiro=opt_shapiro,
            include_influence=opt_influence
        )
        st.session_state["diagnostic_suites"][active_model.model_id] = suite_res
        st.success(f"✅ Diagnostic suite completed for model `{active_model.model_id}` at α = {alpha_choice:.2f}!")

# Retrieve cached suite if available
cached_suite: Optional[DiagnosticSuiteResult] = st.session_state["diagnostic_suites"].get(active_model.model_id, None)

if cached_suite is None:
    st.info("👆 Click **Run Statistical Assumption Diagnostics** to compute formal hypothesis tests and diagnostic plots.")
    st.stop()

st.divider()

# -----------------------------------------------------------------------------
# SECTION D: Five Diagnostic Areas Overview Matrix
# -----------------------------------------------------------------------------
st.header("4. Diagnostic Overview Matrix")

st.markdown(
    """
    The matrix below summarizes the statistical decisions for each regression assumption.
    **Note**: A decision of **Fail to Reject H0** indicates that the empirical data does not present 
    sufficient evidence against the null hypothesis at $\\alpha = {alpha:.2f}$. It is **never** mathematical proof that the assumption holds.
    """.format(alpha=cached_suite.alpha)
)

summary_rows = cached_suite.to_summary_table()
if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

# Warning notices if any
if cached_suite.warnings:
    for w in cached_suite.warnings:
        st.warning(w)

st.divider()

# -----------------------------------------------------------------------------
# SECTION E: Interactive Plotly Diagnostic Visualizations
# -----------------------------------------------------------------------------
st.header("5. Interactive Diagnostic Visualizations")

diag_tabs = st.tabs([
    "📈 Linearity (Residuals vs. Fitted)",
    "⏱️ Independence (Sequence & ACF)",
    "📐 Homoscedasticity (Scale-Location)",
    "📉 Normality (Q-Q & Histogram)",
    "📊 Multicollinearity (VIF Chart)"
])

# Residuals and fitted values
y_train_actual = split_res.y_train.values
train_preds = active_model.train_predictions.values
train_residuals = y_train_actual - train_preds

with diag_tabs[0]:
    st.markdown("#### Residuals vs. Fitted Values (LOWESS Curve)")
    st.caption("A curved or parabolic LOWESS trend indicates unmodeled non-linear relationships or missing interaction terms.")
    fig_res_fit = plot_residuals_vs_fitted_lowess(
        fitted_values=train_preds,
        residuals=train_residuals,
        target_name=current_meta.target_column,
        target_unit=current_meta.target_unit
    )
    st.plotly_chart(fig_res_fit, use_container_width=True)

with diag_tabs[1]:
    st.markdown("#### Independence & Sequential Autocorrelation")
    col_ind1, col_ind2 = st.columns(2)
    with col_ind1:
        st.markdown("**Residual Sequence Plot**")
        timestamps_train = getattr(split_res, "timestamps_train", None)
        fig_time = plot_residuals_vs_time(
            residuals=train_residuals,
            timestamps=timestamps_train,
            dataset_name=current_meta.dataset_name
        )
        st.plotly_chart(fig_time, use_container_width=True)
    with col_ind2:
        st.markdown("**Autocorrelation Function (ACF)**")
        acf_data = compute_residual_acf(train_residuals, nlags=20)
        fig_acf = plot_residual_acf(acf_data, max_lags=20)
        st.plotly_chart(fig_acf, use_container_width=True)

with diag_tabs[2]:
    st.markdown("#### Homoscedasticity: Scale-Location Plot")
    st.caption("The Scale-Location plot checks whether residual dispersion grows or shrinks with fitted values.")
    fig_scale = plot_scale_location(
        fitted_values=train_preds,
        residuals=train_residuals,
        target_name=current_meta.target_column
    )
    st.plotly_chart(fig_scale, use_container_width=True)

with diag_tabs[3]:
    st.markdown("#### Normality of Residuals")
    col_norm1, col_norm2 = st.columns(2)
    with col_norm1:
        st.markdown("**Normal Q-Q Plot**")
        fig_qq = plot_normal_qq(train_residuals)
        st.plotly_chart(fig_qq, use_container_width=True)
    with col_norm2:
        st.markdown("**Residual Distribution Histogram & Normal Density**")
        fig_hist = plot_residuals_histogram_kde(train_residuals, target_unit=current_meta.target_unit)
        st.plotly_chart(fig_hist, use_container_width=True)

with diag_tabs[4]:
    st.markdown("#### Variance Inflation Factor (VIF)")
    st.caption("Thresholds: VIF > 5.0 (Moderate Multi-collinearity) and VIF > 10.0 (Severe Multi-collinearity).")
    vif_result = cached_suite.diagnostics.get("multicollinearity_vif", None)
    if vif_result and "vif_table" in vif_result.details:
        fig_vif = plot_vif_horizontal_bar(vif_result.details["vif_table"])
        st.plotly_chart(fig_vif, use_container_width=True)
    else:
        st.info("No VIF calculation available for this model.")

st.divider()

# -----------------------------------------------------------------------------
# SECTION F: Detailed Hypothesis Tests & Remedial Directions
# -----------------------------------------------------------------------------
st.header("6. Formal Hypothesis Tests & Detailed Interpretations")

for method_id, diag in cached_suite.diagnostics.items():
    with st.expander(f"🔬 **{diag.method_name}** — Decision: `{diag.decision.replace('_', ' ').title()}`", expanded=(diag.decision == DiagnosticDecision.REJECT_NULL.value)):
        c1, c2, c3 = st.columns(3)
        stat_display = f"{diag.statistic_value:.4f}" if (diag.statistic_value is not None and np.isfinite(diag.statistic_value)) else "N/A"
        pval_display = f"{diag.p_value:.4f}" if diag.p_value is not None else "N/A (Descriptive)"
        c1.metric(f"Test Statistic ({diag.statistic_name})", stat_display)
        c2.metric("p-value", pval_display)
        c3.metric("Significance Level (α)", f"{diag.alpha:.2f}" if diag.alpha else "N/A")

        if diag.null_hypothesis:
            st.markdown(f"**Null Hypothesis ($H_0$)**: {diag.null_hypothesis}")
            st.markdown(f"**Alternative Hypothesis ($H_1$)**: {diag.alternative_hypothesis}")

        st.markdown(f"**Statistical Interpretation**:\n\n{diag.interpretation}")

        if diag.limitations:
            st.markdown(f"**Known Limitations & Assumptions**:\n\n{diag.limitations}")

        if diag.recommendations:
            st.markdown("**Potential Remedial Directions (Educational Guidance)**:")
            for rec in diag.recommendations:
                st.markdown(f"- {rec}")

st.divider()

# -----------------------------------------------------------------------------
# SECTION G: Diagnostic History & Comparison Table
# -----------------------------------------------------------------------------
st.header("7. Diagnostic Suite Export & History")

st.markdown(
    """
    Every diagnostic run generates an immutable, strongly-typed `DiagnosticSuiteResult` containing 
    reproducibility hashes, test statistics, degrees of freedom, and educational interpretations.
    """
)

# Export JSON capability
suite_export = {
    "suite_id": cached_suite.suite_id,
    "model_id": cached_suite.model_id,
    "dataset_id": cached_suite.dataset_id,
    "dataset_fingerprint": cached_suite.dataset_fingerprint,
    "split_fingerprint": cached_suite.split_fingerprint,
    "alpha": cached_suite.alpha,
    "timestamp": cached_suite.timestamp,
    "diagnostics": {k: v.to_dict() for k, v in cached_suite.diagnostics.items()}
}

st.download_button(
    "📥 Export Full Diagnostic Suite (JSON)",
    data=pd.Series([suite_export]).to_json(orient="records", indent=2),
    file_name=f"diagnostics_{active_model.model_id}.json",
    mime="application/json"
)
