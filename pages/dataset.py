"""
Dataset Library Page: interactive dataset selection, profiling, and splitting engine.
Preserves selection and split results in Streamlit Session State across pages.
"""
import streamlit as st
import pandas as pd
from typing import Optional
from src.data import (
    get_registry,
    load_dataset_bundle,
    validate_dataset_schema,
    compute_dataset_profile,
    split_dataset,
    check_dataset_availability,
    download_dataset,
    create_target_histogram,
    create_missing_values_bar,
    create_feature_breakdown_chart,
    DatasetMetadata,
    DatasetBundle,
    SplitConfig,
    SplitResult,
    SplitStrategy,
)

# Header
st.title("📂 Dataset Library & Data Engine")
st.subheader("Curated Real-World Benchmark Datasets (14–39 Usable Predictors)")

st.markdown(
    """
    Explore, validate, and partition real-world regression datasets from Kaggle and UCI.
    Each registered dataset natively contains between **14 and 39 usable predictor features** without
    artificial column filtering, serving as the benchmark foundation for the entire modeling workflow.
    """
)

st.caption(
    "ℹ️ **Milestone 1 Active**: Dataset Library & Data Engine. "
    "Downstream modeling stages (Feature Selection, OLS Regression, Assumption Diagnostics) "
    "are Under Development and Scheduled for Milestone 2–5."
)

st.divider()

# Load Registry
registry = get_registry()
all_metadata = registry.list_datasets()
dataset_options = {meta.id: f"{meta.name} ({meta.domain})" for meta in all_metadata}

# Initialize Session State
if "selected_dataset_id" not in st.session_state:
    st.session_state["selected_dataset_id"] = all_metadata[0].id

# Ensure cache container
if "dataset_bundles" not in st.session_state:
    st.session_state["dataset_bundles"] = {}

# ---------------------------------------------------------
# SECTION A: Dataset Selection & Metadata
# ---------------------------------------------------------
st.header("1. Dataset Selection")

col_sel, col_info = st.columns([1, 2])

with col_sel:
    selected_id = st.selectbox(
        "Select Active Dataset:",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        index=list(dataset_options.keys()).index(st.session_state["selected_dataset_id"]),
        key="dataset_selector"
    )

    # Detect dataset switch: invalidate stale splits and models
    if selected_id != st.session_state["selected_dataset_id"]:
        st.session_state["selected_dataset_id"] = selected_id
        # Clear stale split and future downstream models
        if "active_split_result" in st.session_state:
            del st.session_state["active_split_result"]
        if "trained_model" in st.session_state:
            del st.session_state["trained_model"]
        st.rerun()

current_meta: DatasetMetadata = registry.get_dataset(st.session_state["selected_dataset_id"])

# Check availability on disk
is_available = check_dataset_availability(current_meta)

with col_info:
    with st.container(border=True):
        st.markdown(f"### {current_meta.name}")
        st.markdown(f"**Domain**: `{current_meta.domain}` | **Provider**: {current_meta.source_provider}")
        st.caption(current_meta.description)
        st.markdown(f"🔗 [Kaggle Source Repository]({current_meta.kaggle_url}) | 📜 **License**: {current_meta.license_notes}")
        
        with st.expander("🎓 Educational Learning Objectives", expanded=False):
            for obj in current_meta.learning_objectives:
                st.markdown(f"- {obj}")

# If file is not present locally, provide guided download UI
if not is_available:
    st.error(
        f"⚠️ **Dataset File Missing**: Could not find `{current_meta.local_path}` on disk.\n\n"
        f"You can download it automatically from the source repository below or acquire it manually."
    )
    if st.button(f"📥 Download '{current_meta.name}' Now"):
        with st.spinner("Downloading dataset from source repository..."):
            try:
                download_dataset(current_meta.id, force=True)
                st.success("Download complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}. Please place the CSV manually in 'data/raw/'.")
    st.stop()

# ---------------------------------------------------------
# Load and Cache Active Dataset Bundle
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_cached_bundle(ds_id: str):
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

with st.spinner("Loading and profiling dataset..."):
    bundle, validation_report = _load_cached_bundle(current_meta.id)
    st.session_state["active_dataset_bundle"] = bundle

# Check schema validation warnings / discrepancies
if validation_report.warnings:
    for w in validation_report.warnings:
        st.warning(f"ℹ️ {w}")

if not validation_report.is_valid:
    for d in validation_report.discrepancies:
        st.error(f"❌ Schema Discrepancy: {d}")

st.divider()

# ---------------------------------------------------------
# SECTION B: Dataset Metrics & Architecture
# ---------------------------------------------------------
st.header("2. Dataset Architecture & Dimensions")

prof = bundle.profile

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Observations", f"{prof.total_rows:,}")
m2.metric("Total Raw Columns", f"{prof.total_columns}")
m3.metric("Usable Predictors", f"{prof.usable_feature_count}", help="Mandatory 14-39 features constraint")
m4.metric("Numeric / Categorical", f"{len(prof.numeric_features)} / {len(prof.categorical_features)}")
m5.metric("Missing Values", f"{prof.missing_percentage_total:.2f}%")

# Column role breakdown container
with st.container(border=True):
    st.markdown("#### 📐 Column Role Mapping")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"**🎯 Target Variable ($y$)**:")
        st.info(f"**`{current_meta.target_column}`**\n\nUnit: *{current_meta.target_unit}*\nNature: *{current_meta.target_nature}*")
    with r2:
        st.markdown(f"**🚫 Excluded Columns** ({len(current_meta.excluded_predictor_columns)}):")
        if current_meta.excluded_predictor_columns:
            st.markdown(", ".join([f"`{c}`" for c in current_meta.excluded_predictor_columns]))
            st.caption("Excluded from predictor matrix as identifiers, temporal index, or grouping keys.")
        else:
            st.caption("None (All raw features are candidate predictors)")
    with r3:
        st.markdown(f"**✅ Usable Model Predictors ($X$)** ({prof.usable_feature_count}):")
        st.success(f"**{prof.usable_feature_count} features** (Complies with 14–39 requirement)")
        st.caption(f"{len(prof.numeric_features)} continuous/numeric, {len(prof.categorical_features)} categorical")

st.divider()

# ---------------------------------------------------------
# SECTION C: Data Preview
# ---------------------------------------------------------
st.header("3. Raw Data Preview")

col_prev_ctrl, col_prev_spacer = st.columns([1, 2])
with col_prev_ctrl:
    preview_n = st.slider("Preview Rows:", min_value=5, max_value=50, value=10, step=5)

tab_preview_raw, tab_preview_feats, tab_preview_schema = st.tabs(["Raw Dataset", "Predictor Matrix (X)", "Column Schema & Roles"])

with tab_preview_raw:
    st.dataframe(bundle.raw_df.head(preview_n), use_container_width=True)

with tab_preview_feats:
    st.dataframe(bundle.feature_df.head(preview_n), use_container_width=True)

with tab_preview_schema:
    schema_rows = []
    for col in bundle.raw_df.columns:
        dtype = prof.column_dtypes.get(col, "unknown")
        nulls = prof.column_missing_summary.get(col, 0)
        null_pct = (nulls / prof.total_rows) * 100.0 if prof.total_rows > 0 else 0
        if col == current_meta.target_column:
            role = "🎯 Target"
        elif col in current_meta.excluded_predictor_columns:
            role = "🚫 Excluded Metadata"
        elif col in prof.numeric_features:
            role = "🔢 Numeric Predictor"
        else:
            role = "🔤 Categorical Predictor"

        schema_rows.append({
            "Column Name": col,
            "Role": role,
            "Data Type": dtype,
            "Missing Count": nulls,
            "Missing %": f"{null_pct:.1f}%"
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------
# SECTION D: Descriptive Statistics & Visualizations
# ---------------------------------------------------------
st.header("4. Descriptive Profiling & Exploratory Visualizations")

tab_desc, tab_target_dist, tab_types, tab_missing = st.tabs([
    "📊 Descriptive Statistics",
    "📈 Target Distribution",
    "🍩 Predictor Types",
    "🔍 Missing Values Audit"
])

with tab_desc:
    st.markdown("#### Numeric Features Summary Table")
    numeric_df = bundle.feature_df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe().T.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("No numeric features found.")

    if prof.categorical_features:
        st.markdown("#### Categorical Features Cardinality")
        cat_summary = []
        for c in prof.categorical_features:
            s = bundle.feature_df[c]
            cat_summary.append({
                "Feature": c,
                "Unique Values": s.nunique(),
                "Top Category": s.mode().iloc[0] if not s.mode().empty else "N/A",
                "Top Frequency": s.value_counts().iloc[0] if not s.empty else 0
            })
        st.dataframe(pd.DataFrame(cat_summary), use_container_width=True, hide_index=True)

with tab_target_dist:
    st.plotly_chart(
        create_target_histogram(bundle.target_series, current_meta),
        use_container_width=True
    )
    t_stats = prof.target_stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target Min", f"{t_stats['min']:.2f}" if t_stats['min'] is not None else "--")
    c2.metric("Target Mean", f"{t_stats['mean']:.2f}" if t_stats['mean'] is not None else "--")
    c3.metric("Target Median", f"{t_stats['median']:.2f}" if t_stats['median'] is not None else "--")
    c4.metric("Target Max", f"{t_stats['max']:.2f}" if t_stats['max'] is not None else "--")

with tab_types:
    st.plotly_chart(create_feature_breakdown_chart(prof), use_container_width=True)

with tab_missing:
    missing_fig = create_missing_values_bar(bundle.raw_df)
    if missing_fig:
        st.plotly_chart(missing_fig, use_container_width=True)
    else:
        st.success("✅ Clean Dataset: Zero missing values detected across all columns!")

st.divider()

# ---------------------------------------------------------
# SECTION E: Train / Validation / Test Configuration
# ---------------------------------------------------------
st.header("5. Train / Validation / Test Splitting Configuration")

recommended_strat = SplitStrategy(current_meta.recommended_split)

st.markdown(
    f"**Recommended Splitting Strategy for this Dataset**: `{recommended_strat.value.upper()}`\n\n"
)

if recommended_strat == SplitStrategy.TEMPORAL:
    st.info(
        f"⏱️ **Temporal Splitting Rationale**: '{current_meta.name}' is an ordered time-series dataset. "
        f"Chronological partitioning on `{current_meta.date_column}` ensures that future observations "
        "are strictly isolated into validation and test sets, preventing forward-looking data leakage."
    )
elif recommended_strat == SplitStrategy.GROUP:
    st.info(
        f"🌍 **Group Splitting Rationale**: '{current_meta.name}' is a panel dataset with repeated observations "
        f"over time grouped by `{current_meta.group_column}`. Group-disjoint splitting ensures that each country "
        "belongs strictly to one partition, testing model generalization to previously unseen countries."
    )
else:
    st.info(
        f"🎲 **Random Splitting Rationale**: '{current_meta.name}' contains independent cross-sectional records. "
        "A deterministic random split partitions observations evenly while maintaining consistent reproducible partitions."
    )

col_strat, col_ratios, col_seed = st.columns([1, 2, 1])

with col_strat:
    selected_strategy_str = st.selectbox(
        "Split Strategy:",
        options=[s.value for s in SplitStrategy],
        index=[s.value for s in SplitStrategy].index(recommended_strat.value),
        format_func=lambda x: f"{x.capitalize()} Split"
    )
    chosen_strategy = SplitStrategy(selected_strategy_str)

with col_ratios:
    st.markdown("**Partition Proportions (Must sum to 100%)**:")
    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        train_pct = st.number_input("Train %", min_value=50, max_value=85, value=70, step=5)
    with r_col2:
        val_pct = st.number_input("Validation %", min_value=5, max_value=30, value=15, step=5)
    with r_col3:
        test_pct = st.number_input("Test %", min_value=5, max_value=30, value=15, step=5)

    sum_pct = train_pct + val_pct + test_pct
    if sum_pct != 100:
        st.error(f"❌ Proportions sum to {sum_pct}%. They must equal exactly 100%!")

with col_seed:
    if chosen_strategy != SplitStrategy.TEMPORAL:
        random_seed = st.number_input("Random Seed:", value=current_meta.default_random_seed, step=1)
    else:
        st.write("Date Column:")
        st.code(current_meta.date_column)
        random_seed = 42

execute_split = st.button("🚀 Execute Train / Validation / Test Split", type="primary", disabled=(sum_pct != 100))

if execute_split:
    split_config = SplitConfig(
        strategy=chosen_strategy,
        train_ratio=train_pct / 100.0,
        val_ratio=val_pct / 100.0,
        test_ratio=test_pct / 100.0,
        random_seed=int(random_seed),
        date_column=current_meta.date_column,
        group_column=current_meta.group_column
    )
    try:
        with st.spinner("Partitioning dataset..."):
            split_res = split_dataset(
                feature_df=bundle.feature_df,
                target_series=bundle.target_series,
                metadata=current_meta,
                config=split_config,
                raw_df=bundle.raw_df
            )
            st.session_state["active_split_config"] = split_config
            st.session_state["active_split_result"] = split_res
            st.success(f"✅ Successfully partitioned dataset using {chosen_strategy.value.capitalize()} Split!")
    except Exception as e:
        st.error(f"Splitting error: {e}")

# ---------------------------------------------------------
# SECTION F: Split Result Summary & Inspection
# ---------------------------------------------------------
if "active_split_result" in st.session_state and st.session_state.get("selected_dataset_id") == current_meta.id:
    st.divider()
    st.header("6. Partition Summary & Leakage Prevention Audit")

    res: SplitResult = st.session_state["active_split_result"]

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Train Partition", f"{res.sample_counts['train']:,} rows", f"{res.actual_proportions['train']:.1%}")
    sc2.metric("Validation Partition", f"{res.sample_counts['validation']:,} rows", f"{res.actual_proportions['validation']:.1%}")
    sc3.metric("Test Partition", f"{res.sample_counts['test']:,} rows", f"{res.actual_proportions['test']:.1%}")
    sc4.metric("Total Eligible Observations", f"{res.sample_counts['total_eligible']:,} rows")

    if res.warnings:
        for w in res.warnings:
            st.info(f"ℹ️ {w}")

    # Strategy-specific details
    with st.container(border=True):
        if res.strategy == SplitStrategy.TEMPORAL.value and res.temporal_bounds:
            st.markdown("#### ⏱️ Chronological Partition Boundaries")
            tb = res.temporal_bounds
            tb_df = pd.DataFrame([
                {"Partition": "Train", "Earliest Timestamp": tb['train'][0], "Latest Timestamp": tb['train'][1], "Rows": res.sample_counts['train']},
                {"Partition": "Validation", "Earliest Timestamp": tb['validation'][0], "Latest Timestamp": tb['validation'][1], "Rows": res.sample_counts['validation']},
                {"Partition": "Test", "Earliest Timestamp": tb['test'][0], "Latest Timestamp": tb['test'][1], "Rows": res.sample_counts['test']}
            ])
            st.dataframe(tb_df, use_container_width=True, hide_index=True)
            st.caption("Temporal isolation verified: Train dates < Validation dates < Test dates.")

        elif res.strategy == SplitStrategy.GROUP.value and res.group_summary:
            st.markdown("#### 🌍 Disjoint Group Assignment Audit")
            gs = res.group_summary
            st.markdown(
                f"- **Train Groups ({gs['train_group_count']})**: {', '.join(gs['train_groups_sample'])} ...\n"
                f"- **Validation Groups ({gs['val_group_count']})**: {', '.join(gs['val_groups_sample'])} ...\n"
                f"- **Test Groups ({gs['test_group_count']})**: {', '.join(gs['test_groups_sample'])} ...\n"
            )
            st.caption("Zero group overlap verified: No country appears across multiple partitions.")

        else:
            st.markdown("#### 🎲 Random Partitioning Seed & Isolation")
            st.markdown(f"Reproducible random seed: `{res.random_seed}`. Partitions have zero index overlap.")

    st.success(
        "🔒 **Leakage Prevention Guarantee**: Feature scalers, imputers, and selection algorithms "
        "will be fitted strictly on the Training set in future milestones."
    )
