"""
Feature Selection Laboratory Page.
Configurable execution, comparison, and evaluation of 8 feature selection algorithms:
Baseline, Pearson Correlation, F-test, Mutual Information, RFE, Forward SFS, Backward SFS, and Lasso.
Enforces group-aware selection for categorical variables, train-only preprocessing, and isolated cross-validation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data import (
    get_registry,
    load_dataset_bundle,
    validate_dataset_schema,
    compute_dataset_profile,
    check_dataset_availability,
    get_dataset_fingerprint,
    DatasetMetadata,
    DatasetBundle,
    SplitResult,
)
from src.features import (
    FEATURE_SELECTION_REGISTRY,
    run_feature_selection,
    FeatureSelectionResult,
)
from src.experiments.feature_selection_runner import FeatureSelectionExperimentTracker

# Header
st.title("🎯 Feature Selection Laboratory")
st.subheader("Group-Aware Filter, Wrapper & Embedded Dimensionality Reduction")

st.markdown(
    """
    Select and evaluate optimal subsets of original predictor features using statistical filters,
    model-driven wrappers, and L1 regularization. Cross-validation and validation evaluations
    strictly isolate partitions to prevent lookahead and entity data leakage.
    """
)

st.caption(
    "ℹ️ **Milestone 2 Active**: EDA & Feature Selection Laboratory. "
    "Downstream econometric stages (OLS Inference, Diagnostic Dashboards) are "
    "Under Development and Scheduled for Milestone 3–5."
)

st.divider()

# ---------------------------------------------------------
# Load Active Dataset & Split State
# ---------------------------------------------------------
registry = get_registry()
all_metadata = registry.list_datasets()
dataset_options = {meta.id: f"{meta.name} ({meta.domain})" for meta in all_metadata}

if "selected_dataset_id" not in st.session_state:
    st.session_state["selected_dataset_id"] = all_metadata[0].id

selected_id = st.session_state["selected_dataset_id"]
current_meta = registry.get_dataset(selected_id)

is_available = check_dataset_availability(current_meta)
fingerprint = get_dataset_fingerprint(current_meta) if is_available else "missing"

@st.cache_data(show_spinner=False)
def _load_cached_bundle(ds_id: str, file_fingerprint: str):
    meta = registry.get_dataset(ds_id)
    raw_df, feat_df, target_series, meta_df = load_dataset_bundle(meta)
    val_report = validate_dataset_schema(raw_df, meta)
    profile = compute_dataset_profile(raw_df, feat_df, target_series, meta)
    return DatasetBundle(
        metadata=meta,
        raw_df=raw_df,
        feature_df=feat_df,
        target_series=target_series,
        metadata_df=meta_df,
        profile=profile
    ), val_report

if not is_available:
    st.warning(f"⚠️ Dataset '{current_meta.name}' is not present on disk. Please visit the Dataset Library to download it.")
    st.stop()

with st.spinner("Loading active dataset bundle..."):
    bundle, val_report = _load_cached_bundle(current_meta.id, fingerprint)

# Check active split
has_split = (
    "active_split_result" in st.session_state
    and st.session_state.get("active_split_dataset_id") == current_meta.id
)

# Initialize Experiment Tracker in Session State
if "experiment_tracker" not in st.session_state:
    st.session_state["experiment_tracker"] = FeatureSelectionExperimentTracker()
tracker: FeatureSelectionExperimentTracker = st.session_state["experiment_tracker"]

# Invalidate stale experiments if dataset changed
if has_split:
    split_res: SplitResult = st.session_state["active_split_result"]
    purged_count = tracker.invalidate_stale_experiments(current_meta.id, split_res.split_fingerprint)
    if purged_count > 0:
        st.info(f"ℹ️ Purged {purged_count} stale experiment(s) associated with an earlier dataset split configuration.")

# ---------------------------------------------------------
# SECTION A: Active Dataset Summary & Dependency Check
# ---------------------------------------------------------
st.header("1. Active Dataset & Split Verification")

if not has_split:
    st.error(
        "🚫 **No Active Train / Validation Split Detected**: "
        "Feature Selection requires a partitioned dataset to prevent data leakage and evaluate generalization. "
        "Please navigate to **Dataset Library** to partition this dataset first."
    )
    st.stop()

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Dataset", current_meta.name.split()[0])
m2.metric("Target Variable", current_meta.target_column)
m3.metric("Usable Features", f"{len(bundle.feature_df.columns)}")
m4.metric("Train Samples", f"{len(split_res.X_train):,}")
m5.metric("Validation Samples", f"{len(split_res.X_val):,}")
m6.metric("Split Strategy", split_res.strategy.capitalize())

st.divider()

# ---------------------------------------------------------
# SECTION B: Selection Configuration
# ---------------------------------------------------------
st.header("2. Feature Selection Configuration")

method_keys = list(FEATURE_SELECTION_REGISTRY.keys())
method_labels = {
    m: f"[{FEATURE_SELECTION_REGISTRY[m]['category']}] {FEATURE_SELECTION_REGISTRY[m]['name']}"
    for m in method_keys
}

col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 1, 1])

with col_cfg1:
    selected_method_id = st.selectbox(
        "Select Feature Selection Method:",
        options=method_keys,
        format_func=lambda x: method_labels[x],
        index=1,  # Default to Pearson
        key="fs_method_selector"
    )
    method_info = FEATURE_SELECTION_REGISTRY[selected_method_id]
    st.caption(f"ℹ️ **Description**: {method_info['description']} | **Complexity**: `{method_info['complexity']}`")

total_feats = len(bundle.feature_df.columns)
default_k = min(15, total_feats) if total_feats >= 15 else max(2, total_feats // 2)

with col_cfg2:
    if method_info["requires_k"]:
        selected_k = st.slider(
            "Selected Original Predictors (K):",
            min_value=1,
            max_value=total_feats,
            value=default_k,
            step=1,
            key="fs_k_slider",
            help="Refers to the count of original dataset predictors. Categorical dummy columns are grouped under their parent original feature."
        )
    else:
        st.write("Selected Features:")
        st.info(f"**All {total_feats} original features** (Baseline)")
        selected_k = total_feats

with col_cfg3:
    # CV folds control
    default_folds = 3 if current_meta.recommended_split == "temporal" else 5
    selected_cv_folds = st.number_input(
        "CV Folds:",
        min_value=2,
        max_value=10,
        value=default_folds,
        step=1,
        key="fs_cv_folds_input"
    )

# Advanced Lasso Options if applicable
lasso_alpha = None
if selected_method_id == "lasso":
    with st.expander("⚙️ Advanced Lasso Regularization Settings", expanded=False):
        use_manual_alpha = st.checkbox("Specify custom alpha penalty", value=False)
        if use_manual_alpha:
            lasso_alpha = st.number_input("Alpha (L1 penalty):", min_value=1e-5, max_value=100.0, value=0.01, format="%.4f")
        else:
            st.caption("Default: 5-Fold LassoCV automatically selects the optimal cross-validated alpha parameter.")
        st.info(
            "💡 **Lasso vs. Exact-K**: Standard Lasso selects features via the L1 penalty, creating natural sparsity. "
            "If the natural non-zero count differs from your requested K, features are ranked by group coefficient magnitude."
        )

# Computational Budget Advisory for Wrapper methods
if selected_method_id in ("forward_sfs", "backward_sfs") and len(split_res.X_train) > 5000:
    st.warning(
        f"⚠️ **Computational Complexity Notice**: `{method_info['name']}` iteratively refits LinearRegression across multiple subsets. "
        f"Training set has {len(split_res.X_train):,} observations. Execution may take several seconds."
    )

st.divider()

# ---------------------------------------------------------
# SECTION C: Execute Experiment
# ---------------------------------------------------------
st.header("3. Execute Feature Selection")

run_col1, run_col2 = st.columns([1, 2])
with run_col1:
    btn_execute = st.button("🚀 Run Feature Selection Experiment", type="primary", key="btn_run_fs")

if btn_execute:
    with st.spinner(f"Running {method_info['name']} (K={selected_k}) with {selected_cv_folds}-Fold isolated CV..."):
        try:
            exp_result = run_feature_selection(
                X_train=split_res.X_train,
                y_train=split_res.y_train,
                X_val=split_res.X_val,
                y_val=split_res.y_val,
                method_id=selected_method_id,
                k=selected_k,
                cv_strategy=split_res.strategy,
                cv_folds=int(selected_cv_folds),
                random_seed=current_meta.default_random_seed,
                dataset_id=current_meta.id,
                dataset_fingerprint=fingerprint,
                split_fingerprint=split_res.split_fingerprint,
                groups_train=split_res.groups_train,
                alpha=lasso_alpha
            )
            tracker.add_result(exp_result)
            st.session_state["latest_fs_result"] = exp_result
            st.success(f"✅ Experiment completed in {exp_result.execution_time_seconds:.2f} seconds!")
        except Exception as e:
            st.error(f"Feature selection execution error: {e}")

# Retrieve latest active experiment
active_result: Optional[FeatureSelectionResult] = st.session_state.get("latest_fs_result")

if active_result is not None:
    # ---------------------------------------------------------
    # SECTION D: Selected Features Inspection
    # ---------------------------------------------------------
    st.divider()
    st.header(f"4. Selected Predictor Features ({active_result.selected_original_count} / {active_result.total_original_count})")

    sf1, sf2, sf3, sf4 = st.columns(4)
    sf1.metric("Selected Original Features", f"{active_result.selected_original_count} / {active_result.total_original_count}")
    sf2.metric("Selected Encoded Dimension", f"{active_result.selected_encoded_dimension} / {active_result.full_encoded_dimension}")
    sf3.metric("Validation RMSE", f"{active_result.val_rmse:.3f}")
    sf4.metric("Validation R²", f"{active_result.val_r2:.3f}")

    if active_result.warnings:
        for w in active_result.warnings:
            st.info(f"ℹ️ {w}")

    # Build detailed feature mapping table
    table_rows = []
    for m in active_result.feature_mappings:
        table_rows.append({
            "Original Feature": m.original_name,
            "Type": m.feature_type.capitalize(),
            "Selected": "✅ Selected" if m.is_selected else "— Excluded",
            "Selection Rank": m.rank,
            "Method Score": f"{m.score:.4f}" if m.score != 1e9 else "N/A",
            "Encoded Dummies": m.encoded_dimension,
            "Encoded Names Sample": ", ".join(m.encoded_column_names[:3]) + ("..." if len(m.encoded_column_names) > 3 else "")
        })

    table_df = pd.DataFrame(table_rows).sort_values(by="Selection Rank", ascending=True)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # SECTION E: Selection Visualizations
    # ---------------------------------------------------------
    st.divider()
    st.header("5. Feature Selection Visualizations")

    tab_viz_rank, tab_viz_comp, tab_viz_overlap = st.tabs([
        "📊 Feature Importance / Ranking Chart",
        "⚖️ Method Comparison (RMSE & R²)",
        "🧩 Feature Selection Overlap Matrix"
    ])

    with tab_viz_rank:
        rank_df = table_df.head(25)
        fig_rank = px.bar(
            rank_df,
            x="Original Feature",
            y="Selection Rank",
            color="Selected",
            title=f"Feature Rankings: {active_result.method_name} (Top {len(rank_df)})",
            color_discrete_map={"✅ Selected": "#4F46E5", "— Excluded": "#94A3B8"}
        )
        fig_rank.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_rank, use_container_width=True)

    with tab_viz_comp:
        comp_df = tracker.generate_comparison_table()
        if not comp_df.empty:
            st.markdown("#### Experiment Performance Leaderboard (Sorted by Validation RMSE)")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            fig_comp = px.bar(
                comp_df,
                x="Method",
                y="Val RMSE",
                color="Category",
                title="Validation RMSE across Feature Selection Experiments (Lower is Better)",
                text_auto=".2f"
            )
            fig_comp.update_layout(template="plotly_white")
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Run multiple feature selection methods to view comparative performance.")

    with tab_viz_overlap:
        overlap_matrix = tracker.generate_feature_overlap_matrix()
        if not overlap_matrix.empty and len(overlap_matrix.columns) > 2:
            st.markdown("#### Cross-Method Feature Selection Consistency")
            st.dataframe(overlap_matrix, use_container_width=True, hide_index=True)
            st.caption("Common features selected across multiple distinct paradigms indicate robust predictive signal.")
        else:
            st.info("Run at least two different selection methods to inspect feature overlap consistency.")

    # ---------------------------------------------------------
    # SECTION F: Performance Summary & Evaluation
    # ---------------------------------------------------------
    st.divider()
    st.header("6. Performance Summary & Statistical Interpretation")

    p1, p2, p3 = st.columns(3)
    p1.metric("CV RMSE (Mean ± Std)", f"{active_result.cv_rmse_mean:.3f} ± {active_result.cv_rmse_std:.3f}")
    p2.metric("Validation MSE", f"{active_result.val_mse:.3f}")
    p3.metric("Validation R²", f"{active_result.val_r2:.3f}")

    with st.container(border=True):
        st.markdown("#### 🎓 Metric Interpretation & Statistical Best Practices")
        st.markdown(
            f"""
            - **Root Mean Squared Error (RMSE = {active_result.val_rmse:.2f})**:
              Expressed in the original Target units (*{current_meta.target_unit}*).
              Measures the standard deviation of unexplained validation residuals. Lower values reflect superior out-of-sample fit.
            - **Coefficient of Determination ($R^2$ = {active_result.val_r2:.3f})**:
              Proportion of target variance explained by the selected subset.
              *Note*: On out-of-sample validation data, $R^2$ can be **negative** if the fitted model's predictions have larger variance than simply guessing the training target mean.
            - **No Lookahead Guarantee**:
              Feature selectors and imputers were fitted **strictly on the Training set**. Validation data remained completely untouched until final scoring.
            """
        )
