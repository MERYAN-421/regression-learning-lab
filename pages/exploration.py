"""
Exploratory Data Analysis (EDA) Laboratory Page.
Interactive inspection of univariate distributions, target behavior, bivariate relationships,
correlation matrices (Pearson vs. Spearman), missing values, and data quality audits.
Operates on the active DatasetBundle and Train/Validation partitions established in Milestone 1.
"""
import streamlit as st
import pandas as pd
import numpy as np

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
from src.exploration import (
    compute_numeric_distribution,
    compute_categorical_distribution,
    compute_correlation_matrix,
    compute_numeric_bivariate_stats,
    compute_categorical_target_summary,
    plot_numeric_distribution,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_bivariate_scatter,
    plot_categorical_target_boxplot,
)

# Header & Milestone Status
st.title("📊 Exploratory Data Analysis (EDA Lab)")
st.subheader("Statistical Distributions, Relationships & Correlation Analysis")

st.markdown(
    """
    Explore empirical data distributions, examine pairwise correlations, inspect feature-target
    associations, and audit data quality before feature selection and regression modeling.
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

with st.spinner("Loading dataset bundle..."):
    bundle, val_report = _load_cached_bundle(current_meta.id, fingerprint)

# Check active split
has_split = (
    "active_split_result" in st.session_state
    and st.session_state.get("active_split_dataset_id") == current_meta.id
)

# ---------------------------------------------------------
# SECTION A: Dataset Overview & Partition Scope Selector
# ---------------------------------------------------------
st.header("1. Active Dataset & Partition Scope")

c_scope1, c_scope2 = st.columns([2, 1])

with c_scope1:
    scope_options = ["Training Partition (Leakage-Safe Default)"]
    if has_split:
        scope_options.extend(["Full Raw Dataset (Descriptive Audit)", "Validation Partition", "Test Partition"])
    else:
        scope_options.append("Full Raw Dataset (Unpartitioned)")

    selected_scope = st.radio(
        "Select Active Analysis Scope:",
        options=scope_options,
        index=0,
        horizontal=True,
        help="Supervised EDA should default to Training data to prevent confirmation bias and data leakage."
    )

with c_scope2:
    if has_split:
        split_res: SplitResult = st.session_state["active_split_result"]
        st.success(f"🔒 **Active Split**: `{split_res.strategy.upper()}` ({split_res.sample_counts['train']} Train rows)")
    else:
        st.info("ℹ️ **Unpartitioned Mode**: Showing full dataset. Configure partitions in **Dataset Library**.")

# Derive active data frames based on selected scope
if "Training" in selected_scope and has_split:
    active_feat_df = split_res.X_train
    active_target_s = split_res.y_train
    active_scope_name = "Training Partition"
elif "Validation" in selected_scope and has_split:
    active_feat_df = split_res.X_val
    active_target_s = split_res.y_val
    active_scope_name = "Validation Partition"
elif "Test" in selected_scope and has_split:
    active_feat_df = split_res.X_test
    active_target_s = split_res.y_test
    active_scope_name = "Test Partition"
else:
    active_feat_df = bundle.feature_df
    active_target_s = bundle.target_series
    active_scope_name = "Full Raw Dataset"

# Combined working dataframe for bivariate and relationship analysis
active_working_df = active_feat_df.copy()
active_working_df[current_meta.target_column] = active_target_s

numeric_features = active_feat_df.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [c for c in active_feat_df.columns if c not in numeric_features]

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Active Scope", active_scope_name)
m2.metric("Observations (N)", f"{len(active_working_df):,}")
m3.metric("Usable Features", f"{len(active_feat_df.columns)}")
m4.metric("Numeric Features", f"{len(numeric_features)}")
m5.metric("Categorical Features", f"{len(categorical_features)}")
m6.metric("Target Variable", current_meta.target_column)

st.divider()

# ---------------------------------------------------------
# SECTION B: Feature Distribution
# ---------------------------------------------------------
st.header("2. Predictor Feature Distributions")

col_feat_sel, col_feat_info = st.columns([1, 2])

with col_feat_sel:
    selected_feature = st.selectbox(
        "Select Feature to Inspect:",
        options=list(active_feat_df.columns),
        key="eda_feature_selector"
    )

is_num = selected_feature in numeric_features

if is_num:
    num_stats = compute_numeric_distribution(active_feat_df[selected_feature])
    
    with col_feat_info:
        st.markdown(f"#### Summary Statistics: `{selected_feature}`")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean", f"{num_stats.mean:.2f}")
        s2.metric("Median", f"{num_stats.median:.2f}")
        s3.metric("Std Dev", f"{num_stats.std:.2f}")
        s4.metric("IQR", f"{num_stats.iqr:.2f}")

        s5, s6, s7, s8 = st.columns(4)
        s5.metric("Min", f"{num_stats.min:.2f}")
        s6.metric("Max", f"{num_stats.max:.2f}")
        s7.metric("Skewness", f"{num_stats.skewness:.2f}")
        s8.metric("Kurtosis", f"{num_stats.kurtosis:.2f}")

        if num_stats.outlier_count_iqr > 0:
            st.caption(
                f"ℹ️ **IQR Outlier Indicator**: Detected {num_stats.outlier_count_iqr:,} values ({num_stats.outlier_pct_iqr:.1f}%) "
                f"beyond $[Q_1 - 1.5\\times\\text{{IQR}}, Q_3 + 1.5\\times\\text{{IQR}}]$. "
                "These are noted for descriptive diagnosis and are not automatically removed."
            )

    st.plotly_chart(
        plot_numeric_distribution(active_feat_df[selected_feature], selected_feature),
        use_container_width=True
    )
else:
    cat_stats = compute_categorical_distribution(active_feat_df[selected_feature])

    with col_feat_info:
        st.markdown(f"#### Cardinality Breakdown: `{selected_feature}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Categories", f"{cat_stats.cardinality}")
        c2.metric("Top Category (Mode)", f"{cat_stats.mode}")
        c3.metric("Mode Frequency", f"{cat_stats.mode_frequency:,} ({cat_stats.mode_pct:.1f}%)")

    cf1, cf2 = st.columns([2, 1])
    with cf1:
        st.plotly_chart(
            plot_categorical_distribution(active_feat_df[selected_feature], selected_feature),
            use_container_width=True
        )
    with cf2:
        st.markdown("**Category Frequency Table**")
        st.dataframe(cat_stats.frequency_table, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------
# SECTION C: Target Variable Distribution
# ---------------------------------------------------------
st.header("3. Target Variable ($y$) Distribution")

tgt_stats = compute_numeric_distribution(active_target_s)

t_c1, t_c2 = st.columns([1, 2])
with t_c1:
    with st.container(border=True):
        st.markdown(f"**Target Variable**: `{current_meta.target_column}`")
        st.markdown(f"**Unit**: *{current_meta.target_unit}*")
        st.markdown(f"**Nature**: *{current_meta.target_nature}*")
        st.metric("Mean", f"{tgt_stats.mean:.2f}")
        st.metric("Median", f"{tgt_stats.median:.2f}")
        st.metric("Std Dev", f"{tgt_stats.std:.2f}")
        st.metric("Skewness", f"{tgt_stats.skewness:.2f}")

        # Interpretation of skewness
        if abs(tgt_stats.skewness) > 1.0:
            st.warning(
                f"⚠️ **High Target Skewness ({tgt_stats.skewness:.2f})**: "
                "A highly skewed response variable can cause high-leverage predictions "
                "and heteroscedastic residuals in OLS regression. "
                "In Milestone 3, variance-stabilizing transformations (such as logarithmic or Box-Cox) may be examined."
            )
        else:
            st.success(f"✅ Target distribution exhibits mild skewness ({tgt_stats.skewness:.2f}), suitable for linear modeling.")

with t_c2:
    st.plotly_chart(
        plot_numeric_distribution(active_target_s, current_meta.target_column),
        use_container_width=True
    )

st.divider()

# ---------------------------------------------------------
# SECTION D: Feature Relationships & Associations
# ---------------------------------------------------------
st.header("4. Feature Relationships & Associations")

tab_rel_num, tab_rel_cat, tab_rel_pair = st.tabs([
    "📈 Numeric Feature vs. Target",
    "📦 Categorical Feature vs. Target",
    "🔄 Numeric Predictor Pair"
])

with tab_rel_num:
    if numeric_features:
        sel_num_rel = st.selectbox(
            "Select Numeric Predictor ($X$):",
            options=numeric_features,
            key="rel_num_x_selector"
        )
        biv_stats = compute_numeric_bivariate_stats(active_working_df, sel_num_rel, current_meta.target_column)

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Pearson Correlation (r)", f"{biv_stats.pearson_r:.3f}")
        bc2.metric("Pearson p-value", f"{biv_stats.pearson_p_value:.4e}")
        bc3.metric("Spearman Rank (ρ)", f"{biv_stats.spearman_rho:.3f}")
        bc4.metric("Bivariate OLS Slope", f"{biv_stats.ols_slope:.4f}")

        fig_scatter, sample_size, was_sampled = plot_bivariate_scatter(
            active_working_df, sel_num_rel, current_meta.target_column, add_trendline=True
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.caption(
            "🎓 **Statistical Guidance**: Correlation assesses linear association between two variables, "
            "but does **not** imply a causal relationship. Even strong correlations can arise from confounding variables."
        )
    else:
        st.info("No numeric predictor features available for bivariate correlation scatter.")

with tab_rel_cat:
    if categorical_features:
        sel_cat_rel = st.selectbox(
            "Select Categorical Predictor ($X$):",
            options=categorical_features,
            key="rel_cat_x_selector"
        )
        cat_summary_df = compute_categorical_target_summary(
            active_working_df, sel_cat_rel, current_meta.target_column
        )

        st.markdown(f"#### Target Distribution by `{sel_cat_rel}`")
        fig_cat_box, _, _ = plot_categorical_target_boxplot(
            active_working_df, sel_cat_rel, current_meta.target_column
        )
        st.plotly_chart(fig_cat_box, use_container_width=True)

        st.markdown("**Grouped Summary Statistics Table**")
        st.dataframe(cat_summary_df.style.format({"Mean": "{:.2f}", "Std": "{:.2f}", "Median": "{:.2f}", "Min": "{:.2f}", "Max": "{:.2f}"}), use_container_width=True, hide_index=True)
    else:
        st.info("No categorical predictor features in this dataset.")

with tab_rel_pair:
    if len(numeric_features) >= 2:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_x = st.selectbox("Predictor 1 ($X_1$):", options=numeric_features, index=0, key="pair_p_x")
        with col_p2:
            p_y = st.selectbox("Predictor 2 ($X_2$):", options=numeric_features, index=min(1, len(numeric_features)-1), key="pair_p_y")

        pair_stats = compute_numeric_bivariate_stats(active_working_df, p_x, p_y)
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Pearson r", f"{pair_stats.pearson_r:.3f}")
        pc2.metric("Spearman ρ", f"{pair_stats.spearman_rho:.3f}")
        pc3.metric("Pair Linear Slope", f"{pair_stats.ols_slope:.4f}")

        fig_pair, _, _ = plot_bivariate_scatter(active_working_df, p_x, p_y, add_trendline=True)
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.info("Requires at least two numeric features to compare.")

st.divider()

# ---------------------------------------------------------
# SECTION E: Correlation Matrix Analysis
# ---------------------------------------------------------
st.header("5. Pairwise Correlation Analysis")

c_opt1, c_opt2 = st.columns([1, 2])
with c_opt1:
    corr_method = st.radio(
        "Correlation Metric:",
        options=["pearson", "spearman"],
        format_func=lambda x: "Pearson (Linear Correlation)" if x == "pearson" else "Spearman (Rank Monotonic Correlation)",
        horizontal=False,
        key="corr_method_radio"
    )

with c_opt2:
    all_num_with_target = numeric_features + [current_meta.target_column] if current_meta.target_column not in numeric_features else numeric_features
    # Default selection: target + up to 12 features for readable heatmap
    default_features = [current_meta.target_column] + [f for f in numeric_features[:10] if f != current_meta.target_column]
    selected_corr_features = st.multiselect(
        "Select Features for Heatmap:",
        options=all_num_with_target,
        default=default_features,
        key="corr_features_multiselect"
    )

corr_result = compute_correlation_matrix(
    active_working_df,
    columns=selected_corr_features,
    method=corr_method
)

tab_heat, tab_pairs, tab_table = st.tabs(["🔥 Correlation Heatmap", "🏆 Top Correlated Pairs", "📋 Full Correlation Table"])

with tab_heat:
    st.plotly_chart(
        plot_correlation_heatmap(corr_result.matrix, f"{corr_method.capitalize()} Correlation Heatmap"),
        use_container_width=True
    )
    st.caption(
        "💡 **Interpretation Guide**: Values close to $+1.0$ indicate strong positive linear association; "
        "values close to $-1.0$ indicate strong negative association. "
        "High pairwise correlation between predictors signals potential multicollinearity."
    )

with tab_pairs:
    if corr_result.top_correlated_pairs:
        pairs_data = [
            {
                "Feature 1": p.feature_a,
                "Feature 2": p.feature_b,
                f"{corr_method.capitalize()} r": f"{p.correlation:.3f}",
                "|r| Magnitude": f"{p.abs_correlation:.3f}"
            }
            for p in corr_result.top_correlated_pairs
        ]
        st.dataframe(pd.DataFrame(pairs_data), use_container_width=True, hide_index=True)
    else:
        st.info("No correlated pairs found.")

with tab_table:
    st.dataframe(corr_result.matrix.style.format("{:.2f}").background_gradient(cmap="coolwarm", vmin=-1.0, vmax=1.0), use_container_width=True)

st.divider()

# ---------------------------------------------------------
# SECTION F: Missing Values & Data Quality Audit
# ---------------------------------------------------------
st.header("6. Data Quality & Missingness Audit")

prof = bundle.profile

qc1, qc2, qc3 = st.columns(3)
qc1.metric("Total Missing Values", f"{prof.missing_count_total:,} cells", f"{prof.missing_percentage_total:.2f}%")
qc2.metric("Duplicate Rows", f"{prof.duplicate_rows:,}")
qc3.metric("Constant Predictors", f"{len(corr_result.constant_columns_excluded)}")

if corr_result.constant_columns_excluded:
    st.warning(f"⚠️ Constant features detected: {', '.join(corr_result.constant_columns_excluded)}")

with st.container(border=True):
    st.markdown("#### Missing Values per Feature")
    miss_records = []
    for col in active_working_df.columns:
        cnt = int(active_working_df[col].isnull().sum())
        pct = (cnt / len(active_working_df)) * 100.0 if len(active_working_df) > 0 else 0
        if cnt > 0:
            miss_records.append({"Feature": col, "Missing Count": cnt, "Missing %": f"{pct:.2f}%"})
    
    if miss_records:
        st.dataframe(pd.DataFrame(miss_records), use_container_width=True, hide_index=True)
    else:
        st.success("✅ Zero missing values in current active partition.")
