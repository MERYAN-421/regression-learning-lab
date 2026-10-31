"""
Ordinary Least Squares (OLS) Regression Modeling Laboratory.
Provides interactive model construction, reference-category dummy encoding, rank validation,
econometric coefficient summary tables, and mathematical LaTeX equation formulation.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.data.models import DatasetMetadata, SplitResult
from src.features.result_models import FeatureSelectionResult
from src.experiments.feature_selection_runner import FeatureSelectionExperimentTracker
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model
from src.models.model_result import OLSModelResult
from src.models.model_service import train_ols_experiment


st.title("📈 Regression Model Laboratory")
st.subheader("Ordinary Least Squares (OLS) Estimation & Econometric Summary")

st.caption(
    "ℹ️ **Milestone 3 Active**: Multiple Linear Regression & Model Evaluation Laboratory. "
    "Subsequent econometric diagnostic suites (Gauss-Markov Tests, Influence Diagnostics) are "
    "Under Development and Scheduled for Milestone 4–5."
)

# -----------------------------------------------------------------------------
# 1. PREREQUISITE CHECKS: Active Dataset & Split
# -----------------------------------------------------------------------------
split_res: Optional[SplitResult] = st.session_state.get("active_split_result")
current_meta: Optional[DatasetMetadata] = st.session_state.get("active_dataset_metadata")
fingerprint: str = st.session_state.get("active_dataset_fingerprint", "")

if split_res is None or current_meta is None:
    st.warning("⚠️ **No Active Dataset Split Found**")
    st.info(
        "Please navigate to the **Dataset Library** (`pages/dataset.py`) to select a benchmark "
        "dataset and execute a valid train/val/test data partition before training an OLS model."
    )
    st.stop()

# Initialize or retrieve regression tracker
if "regression_experiment_tracker" not in st.session_state:
    st.session_state["regression_experiment_tracker"] = RegressionExperimentTracker()
tracker: RegressionExperimentTracker = st.session_state["regression_experiment_tracker"]

# Retrieve feature selection tracker
fs_tracker: FeatureSelectionExperimentTracker = st.session_state.get(
    "fs_experiment_tracker", FeatureSelectionExperimentTracker()
)

# -----------------------------------------------------------------------------
# SECTION A: Model & Dataset Context
# -----------------------------------------------------------------------------
with st.container(border=True):
    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    col_meta1.metric("Active Dataset", current_meta.name)
    col_meta2.metric("Target Variable", f"{current_meta.target_column} ({current_meta.target_unit})")
    col_meta3.metric("Splitting Strategy", split_res.strategy.capitalize())
    col_meta4.metric(
        "Sample Sizes",
        f"Train: {len(split_res.X_train):,} | Val: {len(split_res.X_val):,}"
    )

st.divider()

# -----------------------------------------------------------------------------
# SECTION B: Feature Set Selection
# -----------------------------------------------------------------------------
st.header("1. Select Predictor Feature Set")

feature_mode = st.radio(
    "Choose Feature Source:",
    [
        "All Features Baseline (All Usable Predictors in Dataset)",
        "Completed Feature Selection Experiment (from Milestone 2 Lab)"
    ],
    index=0,
    key="reg_feature_mode_radio"
)

selected_features: List[str] = []
source_id: str = "all_features"
source_label: str = "All Features (Baseline)"
selected_fs_result: Optional[FeatureSelectionResult] = None

if feature_mode.startswith("All Features"):
    selected_features = split_res.X_train.columns.tolist()
    source_id = "all_features"
    source_label = f"All Features Baseline (P={len(selected_features)})"
    st.info(
        f"📋 Modeling with all **{len(selected_features)}** original usable predictor features from `{current_meta.name}`."
    )
else:
    # Get valid experiments matching current split fingerprint
    valid_fs_runs = fs_tracker.get_valid_experiments(split_res.split_fingerprint, current_meta.id)
    if not valid_fs_runs:
        st.warning(
            "⚠️ **No Feature Selection Experiments Found for Active Split**\n\n"
            "You haven't run any feature selection experiments on this exact data split yet. "
            "Please go to **Feature Selection** (`pages/feature_selection.py`) to run algorithms "
            "(Pearson, F-test, RFE, SFS, Lasso) or select 'All Features Baseline' above."
        )
        selected_features = split_res.X_train.columns.tolist()
    else:
        run_options = {
            f"{r.method_name} (K={r.selected_original_count}) — Val RMSE: {r.val_rmse:.3f}, R²: {r.val_r2:.3f} [{r.experiment_id}]": r
            for r in valid_fs_runs
        }
        selected_key = st.selectbox(
            "Select Completed Feature Selection Run:",
            list(run_options.keys()),
            key="reg_fs_run_select"
        )
        selected_fs_result = run_options[selected_key]
        selected_features = list(selected_fs_result.selected_original_features)
        source_id = selected_fs_result.experiment_id
        source_label = f"{selected_fs_result.method_name} (K={selected_fs_result.selected_original_count})"

        st.success(
            f"✅ Loaded **{len(selected_features)}** selected original features from `{selected_fs_result.method_name}` "
            f"(K={selected_fs_result.selected_original_count})."
        )

# Display feature badges
st.caption(f"**Original Predictors to Model ({len(selected_features)}):** " + ", ".join([f"`{f}`" for f in selected_features]))

st.divider()

# -----------------------------------------------------------------------------
# SECTION C: OLS Design Matrix Configuration & Diagnostics
# -----------------------------------------------------------------------------
st.header("2. OLS Design Matrix Configuration & Identifiability Check")

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    fit_intercept = st.checkbox(
        "Fit Intercept Term (β₀ / 'const')",
        value=True,
        help="Includes a constant column of 1s. Automatically enables reference-category dummy encoding (drop='first') to prevent the dummy-variable trap."
    )
with col_cfg2:
    scale_numeric = st.checkbox(
        "Standardize Numeric Predictors (StandardScaler)",
        value=False,
        help="Standardizes continuous predictors to zero mean and unit variance. Unchecked uses original natural units for direct physical interpretation."
    )

# Pre-build design matrix to inspect rank and collinearity
try:
    preview_design_res, preview_builder = build_ols_design_matrices(
        X_train=split_res.X_train,
        selected_features=selected_features,
        X_val=split_res.X_val,
        fit_intercept=fit_intercept,
        scale_numeric=scale_numeric,
        dataset_id=current_meta.id,
        dataset_fingerprint=fingerprint,
        split_fingerprint=split_res.split_fingerprint
    )
    meta = preview_design_res.metadata

    # Display structural metrics
    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("Original Features", f"{len(meta.selected_original_features)}")
    dm2.metric("Design Matrix Columns (p)", f"{meta.num_parameters}")
    dm3.metric("Matrix Rank", f"{meta.matrix_rank} / {meta.num_parameters}")
    dm4.metric("Residual df (n - p)", f"{meta.residual_degrees_of_freedom:,}")

    # Rank and collinearity status
    if meta.is_full_rank:
        st.success(
            f"✅ **Full Column Rank Verified**: Matrix rank is {meta.matrix_rank} across {meta.num_parameters} columns. "
            f"Condition number = `{meta.condition_number:.2e}`. OLS normal equations yield a uniquely identifiable solution."
        )
    else:
        st.error(
            f"⚠️ **Rank Deficiency Detected**: Matrix rank is {meta.matrix_rank} but {meta.num_parameters} columns were supplied. "
            "Exact linear dependence prevents unique coefficient identification."
        )
        if meta.collinear_columns:
            st.warning(f"**Identified Linearly Dependent Columns**: {', '.join(meta.collinear_columns)}")
            st.info(
                "💡 **Educational Insight**: In King County House Sales, `sqft_living = sqft_above + sqft_basement`. "
                "Including all three predictors creates an exact linear identity ($X_1 - X_2 - X_3 = 0$), causing $(X^T X)$ "
                "to become singular. In practice, omit one component or model the composite total."
            )

    # Reference category explanation expander
    if meta.categorical_features:
        with st.expander("🏷️ Categorical Reference Level Table (Dummy Variable Trap Prevention)", expanded=False):
            st.markdown(
                "When an intercept ($\\beta_0$) is included, each categorical feature with $C$ levels is encoded into $C - 1$ "
                "indicator dummies by omitting one reference level. This prevents the **dummy variable trap** (exact multicollinearity with $\\beta_0$)."
            )
            cat_rows = []
            for cat_col, ref_cat in meta.reference_categories.items():
                dummies = meta.original_to_encoded_mapping.get(cat_col, [])
                cat_rows.append({
                    "Categorical Feature": cat_col,
                    "Omitted Reference Category": f"'{ref_cat}'",
                    "Active Dummies": len(dummies),
                    "Dummy Column Names": ", ".join([f"`{d}`" for d in dummies])
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Design matrix formulation error: {e}")
    st.stop()

st.divider()

# -----------------------------------------------------------------------------
# SECTION D: Train OLS Model
# -----------------------------------------------------------------------------
st.header("3. Estimate Ordinary Least Squares (OLS) Parameters")

btn_train = st.button("🚀 Train OLS Regression Model", type="primary", key="btn_train_ols")

if btn_train:
    with st.spinner(f"Fitting OLS model on {len(split_res.X_train):,} training observations via statsmodels..."):
        try:
            model_res = train_ols_experiment(
                split_result=split_res,
                feature_source=selected_fs_result if selected_fs_result is not None else "all_features",
                target_name=current_meta.target_column,
                fit_intercept=fit_intercept,
                scale_numeric=scale_numeric,
                dataset_id=current_meta.id,
                dataset_fingerprint=fingerprint
            )
            tracker.add_model(model_res)
            st.session_state["active_ols_model"] = model_res
            st.success(f"✅ OLS Model `{model_res.model_id}` successfully trained in {model_res.trained_timestamp}!")
        except Exception as e:
            st.error(f"Model estimation error: {e}")

# Retrieve latest active model
active_model: Optional[OLSModelResult] = st.session_state.get("active_ols_model")

if active_model is not None and active_model.split_fingerprint == split_res.split_fingerprint:
    # -----------------------------------------------------------------------------
    # SECTION E: Econometric Model Summary
    # -----------------------------------------------------------------------------
    st.divider()
    st.header(f"4. Econometric Summary: {active_model.model_name}")

    if active_model.post_selection_inference_warning:
        st.warning(
            "⚠️ **Post-Selection Inference Caution**: Features were chosen via data-driven feature selection. "
            "Conventional OLS $p$-values, $t$-statistics, and confidence intervals do not account for the selection search "
            "and are presented for descriptive and pedagogical purposes."
        )

    # Scorecards
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Training R²", f"{active_model.train_metrics.r2:.4f}")
    sc2.metric("Train Adj. R²", f"{active_model.train_metrics.adj_r2:.4f}" if active_model.train_metrics.adj_r2 is not None else "N/A")
    sc3.metric("Train RMSE", f"{active_model.train_metrics.rmse:.3f} {current_meta.target_unit}")
    sc4.metric("Val RMSE", f"{active_model.val_metrics.rmse:.3f} {current_meta.target_unit}")
    sc5.metric("Omnibus F-Stat", f"{active_model.f_statistic:.2f}" if np.isfinite(active_model.f_statistic) else "N/A")

    # Fitted Regression Equation
    st.markdown("#### 📐 Mathematical Regression Equation")
    tab_eq_preview, tab_eq_full, tab_eq_text = st.tabs([
        "🔍 Compact Preview",
        "📜 Complete Expanded Formula",
        "📋 Copyable Text Representation"
    ])
    with tab_eq_preview:
        _, preview_fitted_eq = generate_regression_equation_latex(
            target_name=current_meta.target_column,
            params=active_model.fitted_model.params,
            fit_intercept=active_model.fit_intercept,
            max_terms=6
        )
        st.latex(preview_fitted_eq)
    with tab_eq_full:
        st.latex(active_model.equation_fitted)
    with tab_eq_text:
        st.code(active_model.equation_text, language="text")

    # Coefficient Table
    st.markdown("#### 📊 Estimated Parameters & Hypothesis Tests ($H_0: \\beta_j = 0$)")
    st.dataframe(active_model.coef_table, use_container_width=True, hide_index=True)

    # Educational notes on interpretation
    with st.expander("📚 Guide to OLS Statistical Parameter Interpretation", expanded=False):
        st.markdown(
            """
            - **Coefficient ($\\hat{\\beta}_j$)**: Represents the estimated marginal change in the response variable per one-unit increase in predictor $X_j$, holding all other included predictors constant.
            - **Standard Error ($SE$)**: Measures the sampling variability of the estimator across repeated hypothetical samples from the population.
            - **$t$-statistic ($t = \\hat{\\beta}_j / SE$)**: The number of standard errors $\\hat{\\beta}_j$ lies away from zero under the null hypothesis $H_0: \\beta_j = 0$.
            - **$p$-value ($P(>|t|)$)**: The two-tailed probability of observing a $t$-statistic at least as extreme as the calculated value assuming $H_0$ is true. If $p < 0.05$, the predictor is statistically significant at the 5% significance level.
            - **95% Confidence Interval**: In 95% of hypothetical repeated samples, the calculated interval $[\\beta_{lower}, \\beta_{upper}]$ will contain the true population parameter $\\beta_j$.
            - **Omnibus $F$-Statistic**: Tests the joint null hypothesis that all slope coefficients are simultaneously zero ($H_0: \\beta_1 = \\beta_2 = \\dots = \\beta_p = 0$). A small $p$-value confirms that the regression explains significant variation overall.
            """
        )

    # -----------------------------------------------------------------------------
    # SECTION F: Evaluation Handoff
    # -----------------------------------------------------------------------------
    st.divider()
    st.info(
        "💡 **Next Step**: Proceed to the **Model Evaluation Laboratory** (`pages/evaluation.py`) to inspect "
        "Actual vs. Predicted plots, residual distributions, generalization gaps, and perform controlled final Test evaluation."
    )
