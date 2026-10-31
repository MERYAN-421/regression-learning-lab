"""
Experiment & Model Comparison Laboratory for Ordinary Least Squares (OLS) Regression.
Milestone 5: Multi-Model Benchmarking, Trade-off Analysis, Diagnostic Comparison & Candidate Selection.

Strict Rules Enforced:
- RULE 1: Reuses existing FeatureSelectionExperimentTracker, RegressionExperimentTracker, and Diagnostic suites.
- RULE 2: Strictly validation-based ranking. Test Set performance is never used for candidate ranking or selection.
- RULE 3: Never converts statistical diagnostics into an artificial total score or assumption pass count.
- RULE 4: Preserves all accepted Milestone 0-4 behaviors.
"""
from datetime import datetime
from typing import Dict, List, Optional
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.models import DatasetMetadata, SplitResult
from src.data.loader import get_dataset_fingerprint
from src.models.model_result import OLSModelResult
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.experiments.comparison_experiment_tracker import ComparisonExperimentTracker
from src.diagnostics.result_models import DiagnosticSuiteResult

from src.comparison.result_models import (
    ExperimentComparisonResult,
    RankingPolicy,
    safe_serialize_numeric
)
from src.comparison.compatibility import validate_model_compatibility
from src.comparison.leaderboard import build_comparison_leaderboard
from src.comparison.normalization import compute_radar_scores, RADAR_DIMENSIONS
from src.comparison.radar import create_radar_chart
from src.comparison.pareto import compute_pareto_frontier, create_pareto_scatter_chart
from src.comparison.diagnostics_matrix import (
    build_diagnostic_comparison_matrix,
    DIAGNOSTIC_COMPARISON_NOTES
)
from src.comparison.feature_overlap import (
    build_feature_inclusion_dataframe,
    compute_pairwise_jaccard_matrix,
    compute_feature_selection_frequencies,
    create_feature_frequency_chart,
    create_jaccard_heatmap
)
from src.comparison.interpretation import (
    RADAR_EDUCATIONAL_DISCLAIMER,
    PARETO_EDUCATIONAL_NOTE,
    GENERALIZATION_GAP_NOTE,
    DIAGNOSTIC_SCORING_WARNING,
    render_candidate_selection_summary
)
from src.comparison.comparison_runner import run_model_comparison


# -----------------------------------------------------------------------------
# PAGE TITLE & ACADEMIC HEADER
# -----------------------------------------------------------------------------
st.title("⚖️ Experiment & Model Comparison Laboratory")
st.subheader("Side-by-Side Model Benchmarking, Multi-Objective Trade-Offs & Candidate Selection")

st.caption(
    "ℹ️ **Milestone 5 Active**: Compare previously fitted OLS models under a fair, "
    "leakage-isolated evaluation protocol. Rankings are strictly based on out-of-sample Validation metrics. "
    "Test Set performance is never revealed or used for candidate selection (Rule 2). "
    "Diagnostic tests are evaluated individually without artificial composite scoring (Rule 3). "
    "Subsequent comprehensive academic laboratory reporting suites are Under Development and Scheduled for Milestone 6."
)

st.divider()

# -----------------------------------------------------------------------------
# SESSION STATE & TRACKERS INITIALIZATION
# -----------------------------------------------------------------------------
split_res: Optional[SplitResult] = st.session_state.get("dataset_split_result", None)
current_meta: Optional[DatasetMetadata] = st.session_state.get("active_dataset_metadata", None)
regression_tracker: RegressionExperimentTracker = st.session_state.get(
    "regression_tracker", RegressionExperimentTracker()
)

if "comparison_tracker" not in st.session_state:
    st.session_state["comparison_tracker"] = ComparisonExperimentTracker()
comparison_tracker: ComparisonExperimentTracker = st.session_state["comparison_tracker"]

# Retrieve diagnostic suites stored from Diagnostics page
raw_diagnostic_suites = st.session_state.get("diagnostic_suites", {})

if split_res is None or current_meta is None:
    st.warning("⚠️ **Dataset & Partition Required**: Please select a dataset in the **Dataset Library** and partition it.")
    st.stop()

active_ds_fp = st.session_state.get("active_dataset_fingerprint") or get_dataset_fingerprint(current_meta)
active_ds_id = getattr(current_meta, "id", "dataset")
active_ds_name = getattr(current_meta, "name", "Dataset")

# Retrieve available models for active split
available_models: List[OLSModelResult] = regression_tracker.get_valid_models(
    current_split_fingerprint=split_res.split_fingerprint,
    current_dataset_fingerprint=active_ds_fp
)

if not available_models:
    st.info(
        "ℹ️ **No Compatible Models Found**: Please fit at least two OLS regression models "
        "in the **Regression Laboratory** on the active dataset partition before running comparisons."
    )
    st.stop()

# Map diagnostic suites to model IDs (taking the latest suite per model if multiple exist)
model_diag_map: Dict[str, DiagnosticSuiteResult] = {}
for suite in raw_diagnostic_suites.values():
    if hasattr(suite, "model_id") and suite.model_id:
        model_diag_map[suite.model_id] = suite


# -----------------------------------------------------------------------------
# SECTION A: EXPERIMENT SCOPE & MODEL SELECTION
# -----------------------------------------------------------------------------
st.header("1. Experiment Scope & Candidate Model Selection")

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Dataset", active_ds_name)
    col2.metric("Target Variable", current_meta.target_column)
    col3.metric("Split Strategy", split_res.strategy.capitalize())
    col4.metric("Available Models", len(available_models))

    st.caption(
        f"**Partition Fingerprint**: `{split_res.split_fingerprint[:12]}` | "
        f"**Dataset Fingerprint**: `{active_ds_fp[:12]}` | "
        f"**Train Observations**: {len(split_res.X_train)} | "
        f"**Validation Observations**: {len(split_res.X_val)}"
    )

model_options = {
    m.model_id: f"{m.feature_source_label} | K={len(m.selected_original_features)}, p={m.design_matrix_metadata.num_parameters} (ID: {m.model_id[:8]})"
    for m in available_models
}

# Select default models (up to 4)
default_selected_ids = list(model_options.keys())[:min(4, len(model_options))]

selected_model_ids = st.multiselect(
    "Select OLS Models to Compare (2–5 recommended for clear visual contrast):",
    options=list(model_options.keys()),
    default=default_selected_ids,
    format_func=lambda mid: model_options[mid],
    key="comparison_selected_models"
)

if len(selected_model_ids) < 1:
    st.warning("⚠️ Please select at least one fitted model to evaluate.")
    st.stop()

selected_models: List[OLSModelResult] = [
    regression_tracker.get_model(mid) for mid in selected_model_ids
]

# Run Compatibility Validation
compat_check = validate_model_compatibility(
    models=selected_models,
    active_dataset_id=active_ds_id,
    active_dataset_fingerprint=active_ds_fp,
    active_split_fingerprint=split_res.split_fingerprint,
    active_target_name=current_meta.target_column,
    active_target_unit=getattr(current_meta, "target_unit", "") or ""
)

if not compat_check.is_compatible:
    st.error(f"❌ **Compatibility Validation Failed**: {'; '.join(compat_check.reasons)}")
    st.stop()

if compat_check.warnings:
    for w in compat_check.warnings:
        st.warning(f"⚠️ {w}")

st.divider()

# -----------------------------------------------------------------------------
# SECTION B: MODEL LEADERBOARD (VALIDATION-ONLY RANKING)
# -----------------------------------------------------------------------------
st.header("2. Model Comparison Leaderboard")

st.markdown(
    """
    > **Rule 2 Protocol**: Models are evaluated and ranked strictly on **Validation partition metrics**. 
    > Final Test performance is never used for candidate selection or development-stage sorting.
    """
)

col_rank, col_filter = st.columns([2, 2])

ranking_options = {
    RankingPolicy.VAL_RMSE_ASC: "Validation RMSE (Ascending — Default)",
    RankingPolicy.VAL_MAE_ASC: "Validation MAE (Ascending)",
    RankingPolicy.VAL_R2_DESC: "Validation R² (Descending)",
    RankingPolicy.PARSIMONY_ASC: "Parsimony (Parameter Count Ascending)",
    RankingPolicy.GEN_GAP_ASC: "Generalization Gap (|Val - Train| Ascending)"
}

selected_ranking_policy = col_rank.selectbox(
    "Rank Leaderboard By:",
    options=list(ranking_options.keys()),
    format_func=lambda p: ranking_options[p],
    index=0
)

# Compute Pareto Frontier and Leaderboard
pareto_ids, pareto_points = compute_pareto_frontier(selected_models)
diag_suite_ids = {m.model_id: model_diag_map[m.model_id].suite_id for m in selected_models if m.model_id in model_diag_map}

leaderboard_records = build_comparison_leaderboard(
    models=selected_models,
    ranking_policy=selected_ranking_policy,
    pareto_model_ids=pareto_ids,
    diagnostic_suite_map=diag_suite_ids
)

leaderboard_df = pd.DataFrame(leaderboard_records)

# Filter columns for clean presentation
display_cols = [
    "Rank", "Model ID", "Feature Source", "Original Features (K)", "Parameters (p)",
    "Val RMSE", "Train RMSE", "Gen. Gap", "Val R²", "Train R²", "Val MAE",
    "AIC", "BIC", "Inference Status", "Pareto Status", "Diagnostic Suite"
]
existing_display_cols = [c for c in display_cols if c in leaderboard_df.columns]
st.dataframe(leaderboard_df[existing_display_cols], use_container_width=True, hide_index=True)

with st.expander("ℹ️ Notes on Information Criteria (AIC/BIC) and Inference Status"):
    st.markdown(
        """
        - **AIC / BIC Comparability**: Akaike (AIC) and Bayesian (BIC) Information Criteria are only directly 
          comparable when candidate models are fitted on the *identical training observations* and use the *identical 
          response variable and likelihood function*.
        - **Inference Status**: 'Full Rank' indicates that $(X^T X)$ is invertible with strictly positive residual degrees 
          of freedom, enabling valid hypothesis tests. 'Prediction Only' models may suffer from rank deficiency or collinearity, 
          meaning individual coefficient p-values are mathematically invalid.
        """
    )

st.divider()

# -----------------------------------------------------------------------------
# SECTION C: PERFORMANCE COMPARISON & GENERALIZATION GAP
# -----------------------------------------------------------------------------
st.header("3. Train vs. Validation Generalization Analysis")

col_p1, col_p2 = st.columns(2)

# RMSE comparison bar chart
rmse_plot_data = []
for m in selected_models:
    rmse_plot_data.append({"Model": m.feature_source_label, "Partition": "Train RMSE", "RMSE": m.train_metrics.rmse})
    rmse_plot_data.append({"Model": m.feature_source_label, "Partition": "Val RMSE", "RMSE": m.val_metrics.rmse})
rmse_df = pd.DataFrame(rmse_plot_data)

fig_rmse = px.bar(
    rmse_df,
    x="Model",
    y="RMSE",
    color="Partition",
    barmode="group",
    title="Train vs. Validation RMSE by Candidate Model",
    color_discrete_map={"Train RMSE": "#2E86AB", "Val RMSE": "#F24236"}
)
fig_rmse.update_layout(xaxis_title="", yaxis_title="RMSE", margin=dict(t=40, b=20, l=20, r=20))
col_p1.plotly_chart(fig_rmse, use_container_width=True)

# Generalization Gap Bar Chart
gap_plot_data = [
    {
        "Model": m.feature_source_label,
        "Generalization Gap (Val - Train)": m.val_metrics.rmse - m.train_metrics.rmse,
        "Val RMSE": m.val_metrics.rmse
    }
    for m in selected_models
]
gap_df = pd.DataFrame(gap_plot_data)

fig_gap = px.bar(
    gap_df,
    x="Model",
    y="Generalization Gap (Val - Train)",
    title="Generalization Gap (Val RMSE - Train RMSE)",
    color="Generalization Gap (Val - Train)",
    color_continuous_scale="Reds"
)
fig_gap.update_layout(xaxis_title="", margin=dict(t=40, b=20, l=20, r=20))
col_p2.plotly_chart(fig_gap, use_container_width=True)

st.info(GENERALIZATION_GAP_NOTE)

st.divider()

# -----------------------------------------------------------------------------
# SECTION D: NORMALIZED RADAR CHART
# -----------------------------------------------------------------------------
st.header("4. Multi-Metric Radar Chart Analysis")

col_radar, col_radar_info = st.columns([3, 2])

radar_scores = compute_radar_scores(selected_models)
fig_radar = create_radar_chart(selected_models, radar_scores=radar_scores, max_models=5)
col_radar.plotly_chart(fig_radar, use_container_width=True)

with col_radar_info:
    st.markdown("#### Radar Chart Normalization Rules")
    st.markdown(
        """
        Each metric dimension is mapped to $[0.0, 1.0]$:
        - **Lower-is-Better** (Val RMSE, Val MAE, Parameters, Gen Gap):
          $$S_i = \\frac{\\max(x) - x_i}{\\max(x) - \\min(x)}$$
        - **Higher-is-Better** (Val R²):
          $$S_i = \\frac{x_i - \\min(x)}{\\max(x) - \\min(x)}$$
        - **Constant / Single Value**: Assigned a neutral score of $0.5$ (or $1.0$ for single model) to avoid division by zero.
        """
    )
    st.warning(RADAR_EDUCATIONAL_DISCLAIMER)

st.divider()

# -----------------------------------------------------------------------------
# SECTION E: PARETO EFFICIENCY & COMPLEXITY TRADE-OFF
# -----------------------------------------------------------------------------
st.header("5. Prediction-Complexity Trade-Off (Pareto Frontier)")

col_pareto_chart, col_pareto_text = st.columns([3, 2])

active_candidate_id = st.session_state.get("selected_candidate_id", None)
fig_pareto = create_pareto_scatter_chart(
    selected_models,
    pareto_model_ids=pareto_ids,
    selected_candidate_id=active_candidate_id
)
col_pareto_chart.plotly_chart(fig_pareto, use_container_width=True)

with col_pareto_text:
    st.markdown("#### Occam's Razor & Efficiency")
    st.markdown(
        f"""
        - **Pareto-Efficient Models**: `{', '.join(pareto_ids)}`
        - A model is on the Pareto Frontier (⭐) if no other candidate simultaneously achieves **fewer parameters** and **lower Validation RMSE**.
        - Choosing between frontier models represents a modeling trade-off between parsimony and predictive error.
        """
    )
    st.info(PARETO_EDUCATIONAL_NOTE)

st.divider()

# -----------------------------------------------------------------------------
# SECTION F: STATISTICAL DIAGNOSTIC COMPARISON MATRIX
# -----------------------------------------------------------------------------
st.header("6. Statistical Diagnostic Comparison Matrix")

st.markdown(
    """
    > **Rule 3 Protocol**: Assumption test p-values are never summed or converted into a 'pass percentage' 
    > or composite quality score. Each hypothesis test is evaluated independently on its econometric merits.
    """
)

diag_matrix_rows = build_diagnostic_comparison_matrix(selected_models, diagnostic_suites=model_diag_map)
diag_matrix_df = pd.DataFrame(diag_matrix_rows)

if not diag_matrix_df.empty:
    st.dataframe(diag_matrix_df, use_container_width=True, hide_index=True)
else:
    st.info("No diagnostic suites found for the selected models.")

st.caption(DIAGNOSTIC_COMPARISON_NOTES)
st.warning(DIAGNOSTIC_SCORING_WARNING)

st.divider()

# -----------------------------------------------------------------------------
# SECTION G: ORIGINAL FEATURE SELECTION OVERLAP
# -----------------------------------------------------------------------------
st.header("7. Feature Selection Consistency & Overlap")

col_ov1, col_ov2 = st.columns(2)

inclusion_df = build_feature_inclusion_dataframe(selected_models)
jaccard_df = compute_pairwise_jaccard_matrix(selected_models)
freq_dict = compute_feature_selection_frequencies(selected_models)

with col_ov1:
    st.markdown("#### Pairwise Jaccard Feature Similarity")
    st.caption("$$J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}$$ (1.0 = identical original feature sets)")
    fig_jaccard = create_jaccard_heatmap(jaccard_df)
    st.plotly_chart(fig_jaccard, use_container_width=True)

with col_ov2:
    st.markdown("#### Candidate Feature Selection Frequency")
    st.caption("Counts how many compared models retained each original predictor.")
    fig_freq = create_feature_frequency_chart(freq_dict, total_models=len(selected_models))
    st.plotly_chart(fig_freq, use_container_width=True)

with st.expander("📋 View Binary Feature Inclusion Matrix"):
    st.dataframe(inclusion_df, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# SECTION H: CANDIDATE MODEL SELECTION & RATIONALE
# -----------------------------------------------------------------------------
st.header("8. Candidate Model Selection & Modeler Rationale")

st.markdown(
    """
    Record the preferred candidate model resulting from this comparison laboratory.
    Documenting your modeling rationale produces an auditable evaluation trail for **Milestone 6 Academic Reporting**.
    """
)

cand_options = {
    m.model_id: f"{m.feature_source_label} | Val RMSE={m.val_metrics.rmse:.4f}, p={m.design_matrix_metadata.num_parameters} ({'⭐ Pareto' if m.model_id in pareto_ids else 'Dominated'})"
    for m in selected_models
}

cand_col1, cand_col2 = st.columns([2, 3])

selected_cand_id = cand_col1.radio(
    "Select Preferred Development Candidate Model:",
    options=list(cand_options.keys()),
    format_func=lambda mid: cand_options[mid],
    key="candidate_model_radio"
)

rationale_categories = cand_col2.multiselect(
    "Primary Selection Rationale Tags:",
    options=[
        "Lowest Validation RMSE",
        "Parsimony & Simplicity (Occam's Razor)",
        "Pareto Frontier Efficient",
        "Clean Statistical Diagnostics",
        "Domain Interpretability",
        "Minimal Generalization Gap"
    ],
    default=["Lowest Validation RMSE"] if selected_cand_id in pareto_ids else ["Parsimony & Simplicity (Occam's Razor)"]
)

rationale_notes = cand_col2.text_area(
    "Modeler Decision Justification & Pedagogical Rationale:",
    value=f"Selected {cand_options[selected_cand_id]} based on {', '.join(rationale_categories)}. "
          f"Balancing predictive out-of-sample error against parameter parsimony and diagnostic validity.",
    height=100
)

if st.button("💾 Record Candidate Selection & Freeze Comparison Experiment", type="primary"):
    # Run full comparison pipeline and record result
    full_comp_result = run_model_comparison(
        models=selected_models,
        active_dataset_id=active_ds_id,
        active_dataset_fingerprint=active_ds_fp,
        active_split_fingerprint=split_res.split_fingerprint,
        active_target_name=current_meta.target_column,
        active_target_unit=getattr(current_meta, "target_unit", "") or "",
        ranking_policy=selected_ranking_policy,
        diagnostic_suites=model_diag_map,
        selected_candidate_id=selected_cand_id,
        candidate_selection_rationale=rationale_notes
    )

    comparison_tracker.add_comparison(full_comp_result)
    st.session_state["active_comparison_result"] = full_comp_result
    st.session_state["selected_candidate_id"] = selected_cand_id

    st.success(
        f"✅ Comparison Experiment `{full_comp_result.comparison_id}` recorded! "
        f"Candidate model `{selected_cand_id}` successfully registered."
    )

    summary_text = render_candidate_selection_summary(
        selected_model_id=selected_cand_id,
        feature_source_label=regression_tracker.get_model(selected_cand_id).feature_source_label,
        parameters=regression_tracker.get_model(selected_cand_id).design_matrix_metadata.num_parameters,
        val_rmse=regression_tracker.get_model(selected_cand_id).val_metrics.rmse,
        val_r2=regression_tracker.get_model(selected_cand_id).val_metrics.r2,
        is_pareto=(selected_cand_id in pareto_ids),
        rationale=rationale_notes
    )
    st.markdown(summary_text)

st.divider()

# -----------------------------------------------------------------------------
# SECTION I: COMPARISON HISTORY & ARTIFACT REPRODUCIBILITY
# -----------------------------------------------------------------------------
st.header("9. Comparison History & Milestone 6 Export")

history = comparison_tracker.get_valid_comparisons(
    current_split_fingerprint=split_res.split_fingerprint,
    current_dataset_id=active_ds_id,
    current_dataset_fingerprint=active_ds_fp
)

if history:
    hist_rows = []
    for comp in history:
        hist_rows.append({
            "Comparison ID": comp.comparison_id,
            "Compared Models": len(comp.compared_model_ids),
            "Ranking Policy": comp.ranking_policy,
            "Pareto Models": len(comp.pareto_model_ids),
            "Selected Candidate": comp.selected_candidate_model_id or "None",
            "Reproducibility Hash": comp.reproducibility_hash,
            "Timestamp": comp.timestamp
        })
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    # Allow downloading the latest comparison artifact JSON
    latest_comp = history[-1]
    st.download_button(
        label=f"⬇️ Download Latest Comparison Artifact ({latest_comp.comparison_id}.json)",
        data=latest_comp.to_json(indent=2),
        file_name=f"{latest_comp.comparison_id}.json",
        mime="application/json"
    )
else:
    st.info("No comparisons registered yet for the current partition. Click 'Record Candidate Selection' above to persist.")
