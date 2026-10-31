"""
Model Evaluation Laboratory for Regression Learning Lab.
Evaluates OLS predictive accuracy across continuous loss metrics (MSE, RMSE, MAE, R²),
renders interactive Actual vs. Predicted and Residual plots, quantifies generalization gaps,
maintains comparative model leaderboards, and enforces controlled final Test evaluation.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.data.models import DatasetMetadata, SplitResult
from src.models.model_result import OLSModelResult
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.evaluation.plots import (
    plot_actual_vs_predicted,
    plot_residuals_vs_fitted,
    plot_residuals_histogram,
    plot_model_comparison_bar
)


st.title("🎯 Model Evaluation Laboratory")
st.subheader("Predictive Accuracy, Residual Analysis & Generalization Assessment")

st.caption(
    "ℹ️ **Milestone 3 Active**: Predictive Accuracy & Model Evaluation Laboratory. "
    "Subsequent formal econometric diagnostic suites (Gauss-Markov Tests, Influence Diagnostics) are "
    "Under Development and Scheduled for Milestone 4–5."
)

# -----------------------------------------------------------------------------
# 1. PREREQUISITE CHECKS: Active Split & Model
# -----------------------------------------------------------------------------
split_res: Optional[SplitResult] = st.session_state.get("active_split_result")
current_meta: Optional[DatasetMetadata] = st.session_state.get("active_dataset_metadata")

if split_res is None or current_meta is None:
    st.warning("⚠️ **No Active Dataset Split Found**")
    st.info("Please select a benchmark dataset and execute a split in `pages/dataset.py`.")
    st.stop()

# Retrieve or create regression experiment tracker
if "regression_experiment_tracker" not in st.session_state:
    st.session_state["regression_experiment_tracker"] = RegressionExperimentTracker()
tracker: RegressionExperimentTracker = st.session_state["regression_experiment_tracker"]

# Get all valid models matching active split
valid_models = tracker.get_valid_models(split_res.split_fingerprint, current_meta.id)

if not valid_models:
    st.warning("⚠️ **No Trained OLS Models Found for Active Split**")
    st.info(
        "Please navigate to the **Regression Model Laboratory** (`pages/regression.py`) "
        "and train an Ordinary Least Squares (OLS) model before analyzing performance metrics."
    )
    st.stop()

# -----------------------------------------------------------------------------
# SECTION A: Active Model Selection & Context
# -----------------------------------------------------------------------------
st.header("1. Active Model Selection")

model_options = {
    f"{m.model_name} [{m.model_id}] — Val RMSE: {m.val_metrics.rmse:.3f}, R²: {m.val_metrics.r2:.3f}": m
    for m in valid_models
}

# Default to latest trained active_ols_model if available
latest_model: Optional[OLSModelResult] = st.session_state.get("active_ols_model")
default_idx = 0
if latest_model is not None:
    for idx, (lbl, m) in enumerate(model_options.items()):
        if m.model_id == latest_model.model_id:
            default_idx = idx
            break

selected_model_label = st.selectbox(
    "Select Model to Inspect & Evaluate:",
    list(model_options.keys()),
    index=default_idx,
    key="eval_active_model_select"
)
active_model = model_options[selected_model_label]
st.session_state["active_ols_model"] = active_model

with st.container(border=True):
    am1, am2, am3, am4 = st.columns(4)
    am1.metric("Model Identifier", active_model.model_id)
    am2.metric("Feature Source", active_model.feature_source_label)
    am3.metric("Original Features", f"{len(active_model.selected_original_features)}")
    am4.metric("Estimated Parameters", f"{active_model.design_matrix_metadata.num_parameters} (incl. const)")

st.divider()

# -----------------------------------------------------------------------------
# SECTION B: Performance Scorecards (Train vs. Validation)
# -----------------------------------------------------------------------------
st.header("2. Quantitative Performance Scorecards")

col_train, col_val = st.columns(2)

with col_train:
    with st.container(border=True):
        st.markdown(f"#### 🏋️ In-Sample Training Metrics (N = {active_model.train_metrics.sample_count:,})")
        tr1, tr2 = st.columns(2)
        tr1.metric("Training RMSE", f"{active_model.train_metrics.rmse:.3f} {current_meta.target_unit}")
        tr2.metric("Training MAE", f"{active_model.train_metrics.mae:.3f} {current_meta.target_unit}")

        tr3, tr4 = st.columns(2)
        tr3.metric("Training R²", f"{active_model.train_metrics.r2:.4f}")
        adj_str = f"{active_model.train_metrics.adj_r2:.4f}" if active_model.train_metrics.adj_r2 is not None else "N/A"
        tr4.metric("Training Adjusted R²", adj_str)
        st.caption(f"Training MSE: `{active_model.train_metrics.mse:.4f}`")

with col_val:
    with st.container(border=True):
        st.markdown(f"#### 🛡️ Out-of-Sample Validation Metrics (N = {active_model.val_metrics.sample_count:,})")
        vl1, vl2 = st.columns(2)
        vl1.metric("Validation RMSE", f"{active_model.val_metrics.rmse:.3f} {current_meta.target_unit}")
        vl2.metric("Validation MAE", f"{active_model.val_metrics.mae:.3f} {current_meta.target_unit}")

        vl3, vl4 = st.columns(2)
        vl4_delta = round(active_model.val_metrics.r2 - active_model.train_metrics.r2, 4)
        vl3.metric("Validation R²", f"{active_model.val_metrics.r2:.4f}", delta=f"{vl4_delta:+.4f} vs Train")
        vl4.metric("Target Units", current_meta.target_unit or "Units")
        st.caption(f"Validation MSE: `{active_model.val_metrics.mse:.4f}`")

# Generalization Gap Analysis
gen_gap = active_model.val_metrics.rmse - active_model.train_metrics.rmse
pct_gap = (gen_gap / active_model.train_metrics.rmse) * 100 if active_model.train_metrics.rmse > 0 else 0.0

if gen_gap > 0 and pct_gap > 20:
    st.warning(
        f"⚠️ **Generalization Gap Warning**: Validation RMSE is `{gen_gap:.3f}` ({pct_gap:.1f}%) higher than Training RMSE. "
        "The model may be mildly overfitting to training idiosyncrasies. Consider parsimonious feature selection or regularization."
    )
elif gen_gap <= 0:
    st.info(
        f"ℹ️ **Stable Generalization**: Validation RMSE is virtually identical to or lower than Training RMSE "
        f"({gen_gap:.3f} error difference). The specification demonstrates balanced predictive stability."
    )
else:
    st.success(
        f"✅ **Healthy Generalization**: Validation RMSE is within `{pct_gap:.1f}%` of Training RMSE "
        f"(Gap = `{gen_gap:.3f}` {current_meta.target_unit})."
    )

st.divider()

# -----------------------------------------------------------------------------
# SECTION C: Actual vs. Predicted Visualizer
# -----------------------------------------------------------------------------
st.header("3. Actual vs. Predicted Diagnostic Plot")

view_split = st.radio(
    "Select Partition to Plot:",
    ["Validation Set (Out-of-Sample)", "Training Set (In-Sample)"],
    horizontal=True,
    key="eval_scatter_split_radio"
)

if view_split.startswith("Validation"):
    y_plot_true = split_res.y_val
    y_plot_pred = active_model.val_predictions
    curr_rmse = active_model.val_metrics.rmse
    curr_r2 = active_model.val_metrics.r2
    split_lbl = "Validation"
else:
    y_plot_true = split_res.y_train
    y_plot_pred = active_model.train_predictions
    curr_rmse = active_model.train_metrics.rmse
    curr_r2 = active_model.train_metrics.r2
    split_lbl = "Training"

fig_act_pred = plot_actual_vs_predicted(
    y_true=y_plot_true,
    y_pred=y_plot_pred,
    target_name=current_meta.target_column,
    target_unit=current_meta.target_unit,
    split_name=split_lbl,
    rmse=curr_rmse,
    r2=curr_r2
)
st.plotly_chart(fig_act_pred, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# SECTION D: Residual Diagnostics (Descriptive)
# -----------------------------------------------------------------------------
st.header("4. Residual Distribution & Pattern Inspection")

st.info(
    "💡 **Descriptive Error Inspection**: The plots below provide graphical diagnostics of prediction residuals "
    "($e_i = y_i - \\hat{y}_i$). Formal econometric hypothesis testing of Gauss-Markov assumptions "
    "(Breusch-Pagan, Durbin-Watson, Jarque-Bera) is provided in Milestone 4."
)

res_tab1, res_tab2 = st.tabs([
    "📈 Residuals vs. Fitted Values",
    "📊 Residual Distribution Histogram"
])

residuals_series = y_plot_true - y_plot_pred

with res_tab1:
    fig_res_fitted = plot_residuals_vs_fitted(
        y_pred=y_plot_pred,
        residuals=residuals_series,
        target_name=current_meta.target_column,
        target_unit=current_meta.target_unit,
        split_name=split_lbl
    )
    st.plotly_chart(fig_res_fitted, use_container_width=True)

with res_tab2:
    fig_res_hist = plot_residuals_histogram(
        residuals=residuals_series,
        target_name=current_meta.target_column,
        target_unit=current_meta.target_unit,
        split_name=split_lbl
    )
    st.plotly_chart(fig_res_hist, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# SECTION E: Model Comparison Leaderboard
# -----------------------------------------------------------------------------
st.header("5. Model Comparison Leaderboard")

comp_df = tracker.generate_comparison_table(split_res.split_fingerprint)
if not comp_df.empty:
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    if len(comp_df) > 1:
        fig_comp = plot_model_comparison_bar(comp_df)
        st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("Train multiple OLS models to unlock side-by-side comparative leaderboards.")

st.divider()

# -----------------------------------------------------------------------------
# SECTION F: Controlled Final Test Evaluation
# -----------------------------------------------------------------------------
st.header("6. Controlled Final Test Evaluation (Test Set Lock)")

st.warning(
    "🔒 **Methodological Integrity Warning**: In scientific regression workflows, the **Test Set** must remain strictly "
    "unseen during model exploration, feature selection, and specification tuning. Evaluating the test set repeatedly "
    "causes data snooping and produces overly optimistic performance estimates."
)

with st.expander("⚙️ Final Test Evaluation Policy & Authorization", expanded=True):
    st.markdown(
        """
        Choose a finalized model evaluation protocol:
        - **Refit on Train + Validation (Recommended)**: Retains the fixed model specification, refits the imputer, encoder, 
          and OLS parameters on the pooled $(X_{\\text{train}} \\cup X_{\\text{val}})$ sample to maximize sample size, and evaluates on $X_{\\text{test}}$.
        - **Evaluate Train-Only Model**: Evaluates the model trained exclusively on $X_{\\text{train}}$ directly against $X_{\\text{test}}$ without re-estimation.
        """
    )
    refit_choice = st.radio(
        "Refit Policy for Test Finalization:",
        [
            "Refit specification on Train + Validation (Recommended)",
            "Use Train-Only Model Directly"
        ],
        index=0,
        key="eval_test_refit_radio"
    )
    policy_code = "train_val" if refit_choice.startswith("Refit") else "train_only"

    confirm_finalize = st.checkbox(
        "I confirm that feature selection, model comparison, and architectural exploration are complete.",
        value=False,
        key="eval_test_confirm_checkbox"
    )

    btn_eval_test = st.button(
        "🚀 Run Final Test Evaluation",
        type="primary",
        disabled=not confirm_finalize,
        key="btn_run_final_test"
    )

    if btn_eval_test:
        with st.spinner(f"Evaluating model '{active_model.model_id}' on unseen Test partition ({len(split_res.X_test):,} samples)..."):
            try:
                finalized_model = tracker.finalize_test_evaluation(
                    model_id=active_model.model_id,
                    split_result=split_res,
                    refit_policy=policy_code
                )
                st.session_state["active_ols_model"] = finalized_model
                st.success(f"✅ Final Test Evaluation completed for model `{finalized_model.model_id}`!")
            except Exception as e:
                st.error(f"Test evaluation error: {e}")

# Display Test Scorecard if evaluated
if active_model.test_metrics is not None:
    st.markdown("### 🏆 Final Unseen Test Performance")
    tst1, tst2, tst3, tst4 = st.columns(4)
    tst1.metric("Test RMSE", f"{active_model.test_metrics.rmse:.3f} {current_meta.target_unit}")
    tst2.metric("Test MAE", f"{active_model.test_metrics.mae:.3f} {current_meta.target_unit}")
    tst3.metric("Test R²", f"{active_model.test_metrics.r2:.4f}")
    tst4.metric("Test Samples", f"{active_model.test_metrics.sample_count:,}")

    # Plot Actual vs. Predicted on Test
    if active_model.test_predictions is not None:
        fig_test = plot_actual_vs_predicted(
            y_true=split_res.y_test,
            y_pred=active_model.test_predictions,
            target_name=current_meta.target_column,
            target_unit=current_meta.target_unit,
            split_name="Final Test",
            rmse=active_model.test_metrics.rmse,
            r2=active_model.test_metrics.r2
        )
        st.plotly_chart(fig_test, use_container_width=True)
