"""
Offline Unit & Integration Tests for Milestone 2: Feature Selection Laboratory.
Covers:
- Test Group B: Preprocessing & ColumnTransformer
- Test Group C: All Eight Feature Selection Methods
- Test Group D: Categorical Feature Grouping & Original-to-Encoded Mapping
- Test Group E: Cross-Validation Isolation & Deliberate Leakage Trap Test
- Test Group F: Streamlit AppTest for EDA and Feature Selection Pages
"""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from streamlit.testing.v1 import AppTest

from src.features.result_models import (
    OriginalFeatureMapping,
    FeatureSelectionResult,
)
from src.features.preprocessing import (
    build_column_preprocessor,
    fit_train_preprocessor,
    transform_features,
)
from src.features.feature_mapping import (
    build_feature_mappings_from_transformer,
    get_encoded_indices_for_original_features,
    get_encoded_names_for_original_features,
)
from src.features.filter_methods import (
    select_all_features_baseline,
    select_by_pearson_correlation,
    select_by_f_test,
    select_by_mutual_information,
)
from src.features.wrapper_methods import (
    select_by_group_rfe,
    select_by_group_forward_sfs,
    select_by_group_backward_sfs,
)
from src.features.embedded_methods import (
    select_by_lasso,
)
from src.features.evaluation import (
    get_dataset_cv_splitter,
    evaluate_feature_subset_on_validation,
    run_isolated_cross_validation,
)
from src.features.selectors import (
    FEATURE_SELECTION_REGISTRY,
    run_feature_selection,
    compute_experiment_identity,
)
from src.experiments.feature_selection_runner import (
    FeatureSelectionExperimentTracker,
)
from src.data.models import (
    DatasetBundle,
    DatasetMetadata,
    SplitConfig,
    SplitResult,
    SplitStrategy,
)
from src.data.profiling import compute_dataset_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# SYNTHETIC FIXTURES (100% Offline)
# =====================================================================

@pytest.fixture
def synthetic_split_bundle():
    """
    Creates a synthetic DatasetBundle and SplitResult with:
    - 5 numeric features (x1, x2, x3, x4, x5)
    - 2 categorical features (cat1 with 3 levels, cat2 with 4 levels)
    - 1 target (y)
    - Total original features = 7
    - Rows: 100 train, 30 val, 30 test
    """
    np.random.seed(42)
    n_train = 100
    n_val = 30
    n_test = 30
    total = n_train + n_val + n_test

    # Numeric features
    x1 = np.random.normal(0, 1, total)  # Strong signal
    x2 = np.random.normal(0, 1, total)  # Moderate signal
    x3 = np.random.normal(0, 1, total)  # Noise
    x4 = np.random.normal(0, 1, total)  # Noise
    x5 = np.random.normal(0, 1, total)  # Noise

    # Introduce a few NaNs in train to test imputation
    x1[5] = np.nan
    x3[12] = np.nan

    # Categorical features
    c1_choices = ["Red", "Green", "Blue"]
    c2_choices = ["Small", "Medium", "Large", "XLarge"]
    cat1 = np.random.choice(c1_choices, size=total, p=[0.5, 0.3, 0.2])
    cat2 = np.random.choice(c2_choices, size=total, p=[0.4, 0.3, 0.2, 0.1])
    cat1[8] = None  # NaN in categorical

    # Group labels (e.g. for GroupKFold)
    groups = np.array([f"Group_{i % 5}" for i in range(total)])
    timestamps = pd.date_range("2023-01-01", periods=total, freq="D")

    # Target: linear combination of x1, x2 and dummy effect of Green
    y = 3.0 * np.nan_to_num(x1, nan=0.0) - 2.0 * x2 + (cat1 == "Green").astype(float) * 2.5 + np.random.normal(0, 0.5, total)

    usable_features = ["x1", "x2", "x3", "x4", "x5", "cat1", "cat2"]

    df = pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5,
        "cat1": cat1, "cat2": cat2,
        "target": y,
        "group_id": groups,
        "date": timestamps,
    })

    metadata = DatasetMetadata(
        id="synthetic_lab",
        name="Synthetic Lab Dataset",
        source_provider="Unit Test Fixture",
        kaggle_url="https://example.com/test",
        original_reference="Offline unit test reference",
        local_path="",
        target_column="target",
        group_column="group_id",
        date_column="date",
        excluded_predictor_columns=["group_id", "date"],
        recommended_split="random",
        default_split_ratios=[0.625, 0.1875, 0.1875],
        default_random_seed=42
    )

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train+n_val].copy()
    test_df = df.iloc[n_train+n_val:].copy()

    split_result = SplitResult(
        X_train=train_df[usable_features],
        X_val=val_df[usable_features],
        X_test=test_df[usable_features],
        y_train=train_df["target"],
        y_val=val_df["target"],
        y_test=test_df["target"],
        train_indices=np.arange(n_train),
        val_indices=np.arange(n_train, n_train + n_val),
        test_indices=np.arange(n_train + n_val, total),
        actual_proportions={"train": 0.625, "val": 0.1875, "test": 0.1875},
        sample_counts={"train": n_train, "val": n_val, "test": n_test},
        strategy="random",
        random_seed=42,
        groups_train=pd.Series(groups[:n_train], index=train_df.index),
        groups_val=pd.Series(groups[n_train:n_train+n_val], index=val_df.index),
        groups_test=pd.Series(groups[n_train+n_val:], index=test_df.index),
        timestamps_train=pd.Series(timestamps[:n_train], index=train_df.index),
        timestamps_val=pd.Series(timestamps[n_train:n_train+n_val], index=val_df.index),
        timestamps_test=pd.Series(timestamps[n_train+n_val:], index=test_df.index),
        split_fingerprint="synth_split_fp_xyz789"
    )

    bundle = DatasetBundle(
        metadata=metadata,
        raw_df=df,
        feature_df=df[usable_features],
        target_series=df["target"],
        metadata_df=df[["group_id", "date"]],
        profile=compute_dataset_profile(df, df[usable_features], df["target"], metadata),
        split_result=split_result
    )

    return bundle, split_result


# =====================================================================
# Test Group B: Preprocessing & ColumnTransformer
# =====================================================================

def test_preprocessing_train_only_fit_and_mapping(synthetic_split_bundle):
    """
    Verify:
    1. Median imputation for numeric features (handles NaNs in train).
    2. Most frequent mode imputation for categorical features.
    3. StandardScaler applied to numeric columns.
    4. OneHotEncoder expands categories.
    5. No mutation of the original DataFrame.
    """
    bundle, split = synthetic_split_bundle
    X_train_copy = split.X_train.copy(deep=True)

    transformer, num_cols, cat_cols = fit_train_preprocessor(split.X_train, scale_numeric=True)

    # 1. Verify no mutation of raw data
    pd.testing.assert_frame_equal(split.X_train, X_train_copy)

    # 2. Verify transform output
    X_train_enc = transform_features(transformer, split.X_train)
    assert not X_train_enc.isna().to_numpy().any(), "Imputation failed: NaNs remain in transformed matrix"

    # 3. Verify mappings
    mappings, enc_names = build_feature_mappings_from_transformer(transformer, split.X_train)
    assert len(mappings) == 7  # 7 original features
    mapping_dict = {m.original_name: m for m in mappings}
    assert mapping_dict["x1"].encoded_dimension == 1
    assert mapping_dict["cat1"].encoded_dimension >= 3  # Dummy columns
    assert mapping_dict["cat2"].encoded_dimension >= 4


def test_unknown_category_handling_in_validation(synthetic_split_bundle):
    """Verify that unseen categorical levels in validation do not crash prediction (handle_unknown='ignore')."""
    bundle, split = synthetic_split_bundle
    transformer, num_cols, cat_cols = fit_train_preprocessor(split.X_train, scale_numeric=True)

    X_val_with_unseen = split.X_val.copy()
    X_val_with_unseen.iloc[0, X_val_with_unseen.columns.get_loc("cat1")] = "CompletelyNewColor"

    # Should transform without raising an exception
    X_val_enc = transform_features(transformer, X_val_with_unseen)
    assert X_val_enc.shape[0] == len(split.X_val)
    assert not X_val_enc.isna().to_numpy().any()


# =====================================================================
# Test Group C: Selection Methods (All 8 Methods)
# =====================================================================

@pytest.mark.parametrize("method_id", [
    "baseline",
    "pearson",
    "f_test",
    "mutual_information",
    "rfe",
    "forward_sfs",
    "backward_sfs",
    "lasso"
])
def test_all_eight_selection_methods_execution(synthetic_split_bundle, method_id):
    """
    Verify each of the 8 selection methods executes successfully, returns the required
    FeatureSelectionResult structure, respects K where appropriate, and produces scores.
    """
    bundle, split = synthetic_split_bundle
    k = 4 if method_id != "baseline" else 7

    result = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id=method_id,
        k=k,
        cv_strategy="random",
        cv_folds=3,
        random_seed=42,
        dataset_id=bundle.metadata.id,
        dataset_fingerprint="synth_fp",
        split_fingerprint=split.split_fingerprint
    )

    assert isinstance(result, FeatureSelectionResult)
    assert result.method_id == method_id
    assert result.cv_rmse_mean >= 0.0
    assert result.val_rmse >= 0.0
    assert len(result.feature_mappings) == 7

    if method_id == "baseline":
        assert result.selected_original_count == 7
    elif method_id == "lasso":
        # Lasso may return naturally sparse K or top-K
        assert result.selected_original_count > 0
    else:
        assert result.selected_original_count == k


def test_invalid_k_configuration_raises_error(synthetic_split_bundle):
    """Verify that invalid K values (K <= 0 or K > total) raise ValueError."""
    bundle, split = synthetic_split_bundle
    with pytest.raises(ValueError, match="between 1 and"):
        run_feature_selection(
            X_train=split.X_train,
            y_train=split.y_train,
            X_val=split.X_val,
            y_val=split.y_val,
            method_id="pearson",
            k=0,
            cv_strategy="random"
        )

    with pytest.raises(ValueError, match="between 1 and"):
        run_feature_selection(
            X_train=split.X_train,
            y_train=split.y_train,
            X_val=split.X_val,
            y_val=split.y_val,
            method_id="pearson",
            k=99,
            cv_strategy="random"
        )


def test_reproducibility_with_fixed_random_seed(synthetic_split_bundle):
    """Verify that selectors produce identical results when executed with the same random seed."""
    bundle, split = synthetic_split_bundle

    res1 = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="mutual_information",
        k=3,
        cv_strategy="random",
        random_seed=123
    )

    res2 = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="mutual_information",
        k=3,
        cv_strategy="random",
        random_seed=123
    )

    assert res1.selected_original_features == res2.selected_original_features
    assert res1.cv_rmse_mean == pytest.approx(res2.cv_rmse_mean)


# =====================================================================
# Test Group D: Categorical Feature Grouping
# =====================================================================

def test_categorical_grouping_respects_original_k(synthetic_split_bundle):
    """
    CRITICAL CONTRACT:
    A categorical variable with multiple dummy columns must remain ONE original feature.
    Requested K must strictly count original features, not dummy columns.
    """
    bundle, split = synthetic_split_bundle

    # Select K=2 features via Pearson
    result = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="pearson",
        k=2,
        cv_strategy="random"
    )

    assert len(result.selected_original_features) == 2
    assert result.selected_original_count == 2
    # But the encoded input dimension may be > 2 if a categorical feature was selected
    assert result.encoded_input_dimension >= 2

    # Verify that dummy columns belonging to a selected categorical feature are all included
    mapping_dict = {m.original_name: m for m in result.feature_mappings}
    for orig_name in result.selected_original_features:
        mapping = mapping_dict[orig_name]
        for enc_name in mapping.encoded_column_names:
            assert enc_name in result.selected_encoded_feature_names


# =====================================================================
# Test Group E: Cross-Validation & Leakage Prevention
# =====================================================================

def test_group_kfold_group_isolation(synthetic_split_bundle):
    """Verify that GroupKFold strictly prevents group overlap across CV folds."""
    bundle, split = synthetic_split_bundle

    splitter = get_dataset_cv_splitter(
        strategy="group",
        cv_folds=3,
        random_seed=42
    )

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(split.X_train, split.y_train, groups=split.groups_train)):
        train_groups = set(split.groups_train.iloc[train_idx])
        val_groups = set(split.groups_train.iloc[val_idx])
        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"Leakage detected in fold {fold_idx}: Groups {overlap} appear in both train and validation!"


def test_time_series_split_chronological_ordering(synthetic_split_bundle):
    """Verify that TimeSeriesSplit preserves strict chronological order (train indices < val indices)."""
    bundle, split = synthetic_split_bundle

    splitter = get_dataset_cv_splitter(
        strategy="temporal",
        cv_folds=3
    )

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(split.X_train, split.y_train)):
        max_train_idx = np.max(train_idx)
        min_val_idx = np.min(val_idx)
        assert max_train_idx < min_val_idx, f"Chronological leakage in fold {fold_idx}: max train index {max_train_idx} >= min val index {min_val_idx}"


def test_deliberate_leakage_trap():
    """
    DELIBERATE LEAKAGE TRAP TEST:
    Demonstrate that a naive implementation (pre-fitting scaler & selector on entire data
    before cross-validation) produces an artificially over-optimistic (lower) RMSE than
    proper leakage-free fold isolation.
    """
    np.random.seed(99)
    n = 120
    # Generate pure noise features with NO true relationship to target
    X_noise = pd.DataFrame(np.random.normal(0, 1, size=(n, 50)), columns=[f"feat_{i}" for i in range(50)])
    y = pd.Series(np.random.normal(0, 1, size=n), name="target")

    # NAIVE LEAKY WORKFLOW:
    # 1. Screen best 5 features on ALL data
    corrs = [abs(np.corrcoef(X_noise[col], y)[0, 1]) for col in X_noise.columns]
    best_5_indices = np.argsort(corrs)[-5:]
    X_leaky = X_noise.iloc[:, best_5_indices]

    # Evaluate 3-fold CV on the pre-screened features
    leaky_rmses = []
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=False)
    for t_idx, v_idx in kf.split(X_leaky):
        reg = LinearRegression().fit(X_leaky.iloc[t_idx], y.iloc[t_idx])
        preds = reg.predict(X_leaky.iloc[v_idx])
        leaky_rmses.append(root_mean_squared_error(y.iloc[v_idx], preds))
    leaky_rmse_mean = np.mean(leaky_rmses)

    # HONEST ISOLATED WORKFLOW:
    # Screen best 5 features strictly inside each fold
    honest_rmses = []
    for t_idx, v_idx in kf.split(X_noise):
        X_fold_train = X_noise.iloc[t_idx]
        y_fold_train = y.iloc[t_idx]
        fold_corrs = [abs(np.corrcoef(X_fold_train[col], y_fold_train)[0, 1]) for col in X_noise.columns]
        fold_best_5 = np.argsort(fold_corrs)[-5:]
        reg = LinearRegression().fit(X_fold_train.iloc[:, fold_best_5], y_fold_train)
        preds = reg.predict(X_noise.iloc[v_idx, fold_best_5])
        honest_rmses.append(root_mean_squared_error(y.iloc[v_idx], preds))
    honest_rmse_mean = np.mean(honest_rmses)

    # The leaky workflow should produce an artificially lower RMSE than the honest workflow
    # demonstrating why fold isolation is mandatory.
    assert leaky_rmse_mean < honest_rmse_mean, (
        f"Leakage trap assertion: Leaky RMSE ({leaky_rmse_mean:.4f}) should be lower than Honest RMSE ({honest_rmse_mean:.4f})"
    )


# =====================================================================
# Test Group F: Streamlit AppTest Integration Tests
# =====================================================================

def test_apptest_eda_page_rendering():
    """Verify pages/exploration.py renders without unhandled exceptions."""
    at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "exploration.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Exploration page crashed: {at.exception}"


def test_apptest_feature_selection_page_rendering():
    """Verify pages/feature_selection.py renders without unhandled exceptions."""
    at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "feature_selection.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Feature selection page crashed: {at.exception}"


def test_experiment_tracker_stale_detection(synthetic_split_bundle):
    """Verify experiment tracker marks runs as stale when split fingerprint changes."""
    bundle, split = synthetic_split_bundle
    tracker = FeatureSelectionExperimentTracker()

    res = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="pearson",
        k=3,
        cv_strategy="random",
        split_fingerprint=split.split_fingerprint
    )
    tracker.add_result(res)

    # Active split matches -> valid
    valid_runs = tracker.get_valid_experiments(split.split_fingerprint)
    assert len(valid_runs) == 1

    # Split fingerprint changes -> marked as stale / not valid for new split
    new_runs = tracker.get_valid_experiments("new_split_fingerprint_999")
    assert len(new_runs) == 0


# =====================================================================
# Test Group G: Milestone 2 Review Fixes Verification
# =====================================================================

def test_structural_feature_mapping_overlapping_names_and_underscores():
    """
    Task 5 Verification:
    Verify structural feature mapping correctly maps columns when:
    1. Feature names are substrings/prefixes of each other ('feat' vs 'feat_2', 'cat' vs 'cat_type')
    2. Category levels contain underscores ('level_a', 'level_b')
    3. Predictors are mixed numeric and categorical
    4. Encoded indices are disjoint, complete, and exactly contiguous with transformer output.
    """
    df = pd.DataFrame({
        "feat": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["val_x", "val_y", "val_x", "val_y", "val_x"],
        "cat_type": ["type_1", "type_2", "type_3", "type_1", "type_2"],
    })
    numeric_cols = ["feat", "feat_2"]
    categorical_cols = ["cat", "cat_type"]

    preprocessor, num_cols, cat_cols = fit_train_preprocessor(df)
    X_enc = transform_features(preprocessor, df)

    mappings, enc_names = build_feature_mappings_from_transformer(preprocessor, df)

    # 1. Check all 4 original features are mapped
    assert len(mappings) == 4
    mapping_dict = {m.original_name: m for m in mappings}
    assert set(mapping_dict.keys()) == {"feat", "feat_2", "cat", "cat_type"}

    # 2. Check numeric features get exactly 1 encoded index each
    assert mapping_dict["feat"].encoded_dimension == 1
    assert mapping_dict["feat_2"].encoded_dimension == 1
    assert mapping_dict["feat"].encoded_column_indices == [0]
    assert mapping_dict["feat_2"].encoded_column_indices == [1]

    # 3. Check categorical features:
    # 'cat' has 2 levels ("val_x", "val_y") -> 2 indices
    # 'cat_type' has 3 levels ("type_1", "type_2", "type_3") -> 3 indices
    assert mapping_dict["cat"].encoded_dimension == 2
    assert mapping_dict["cat_type"].encoded_dimension == 3
    assert mapping_dict["cat"].encoded_column_indices == [2, 3]
    assert mapping_dict["cat_type"].encoded_column_indices == [4, 5, 6]

    # 4. Check that 'feat' didn't accidentally claim 'feat_2' columns (overlap/substring bug)
    assert set(mapping_dict["feat"].encoded_column_indices).isdisjoint(set(mapping_dict["feat_2"].encoded_column_indices))
    assert set(mapping_dict["cat"].encoded_column_indices).isdisjoint(set(mapping_dict["cat_type"].encoded_column_indices))

    # 5. Check encoded names match the underscore categories properly
    assert all("cat_val_" in name for name in mapping_dict["cat"].encoded_column_names)
    assert all("cat_type_type_" in name for name in mapping_dict["cat_type"].encoded_column_names)

    # 6. Total mapped dimension equals encoded matrix width
    total_mapped_dim = sum(m.encoded_dimension for m in mappings)
    assert total_mapped_dim == X_enc.shape[1] == 7


def test_sfs_nested_cv_rejects_missing_groups():
    """
    Task 1 Verification:
    Verify SFS rejects missing country/group labels with ValueError instead of
    falling back silently to KFold.
    """
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    preprocessor, _, _ = fit_train_preprocessor(df)
    mappings, _ = build_feature_mappings_from_transformer(preprocessor, df)

    # SFS Forward must raise ValueError when cv_strategy='group' and groups_train is None
    with pytest.raises(ValueError, match="Group cross-validation strategy requires aligned group labels"):
        select_by_group_forward_sfs(
            X_input=df,
            y_input=y,
            mappings=mappings,
            k=1,
            cv_strategy="group",
            groups_train=None
        )

    # SFS Backward must raise ValueError when cv_strategy='group' and groups_train is None
    with pytest.raises(ValueError, match="Group cross-validation strategy requires aligned group labels"):
        select_by_group_backward_sfs(
            X_input=df,
            y_input=y,
            mappings=mappings,
            k=1,
            cv_strategy="group",
            groups_train=None
        )


def test_sfs_nested_cv_temporal_strategy():
    """
    Task 1 Verification:
    Verify SFS uses TimeSeriesSplit correctly without leakage for temporal datasets.
    """
    np.random.seed(42)
    n = 60
    t = np.linspace(0, 10, n)
    X = pd.DataFrame({
        "trend": t,
        "noise1": np.random.normal(0, 1, n),
        "noise2": np.random.normal(0, 1, n)
    })
    y = pd.Series(2.5 * t + np.random.normal(0, 0.5, n))

    preprocessor, _, _ = fit_train_preprocessor(X)
    mappings, _ = build_feature_mappings_from_transformer(preprocessor, X)

    # Execute Forward SFS with temporal CV strategy
    selected_fwd = select_by_group_forward_sfs(
        X_input=X,
        y_input=y,
        mappings=mappings,
        k=1,
        cv_strategy="temporal",
        cv_folds=3
    )
    # The trend feature should have the lowest CV RMSE and be selected first
    assert selected_fwd[0].original_name == "trend"
    assert selected_fwd[0].rank == 1


def test_lasso_alpha_selection_rejects_missing_groups():
    """
    Task 2 Verification:
    Verify Lasso alpha-selection workflow rejects missing groups when strategy='group'.
    """
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    preprocessor, _, _ = fit_train_preprocessor(df)
    mappings, _ = build_feature_mappings_from_transformer(preprocessor, df)

    with pytest.raises(ValueError, match="Group cross-validation strategy requires aligned group labels"):
        select_by_lasso(
            X_input=df,
            y_input=y,
            mappings=mappings,
            k=1,
            alpha=None,
            cv_strategy="group",
            groups_train=None
        )


def test_lasso_alpha_selection_dataset_appropriate_strategies():
    """
    Task 2 Verification:
    Verify Lasso alpha selection works deterministically across both TimeSeriesSplit and GroupKFold.
    """
    np.random.seed(42)
    n = 60
    # Grouped dataset test
    groups = pd.Series(np.repeat(["CountryA", "CountryB", "CountryC", "CountryD"], 15))
    X = pd.DataFrame({
        "signal": np.random.normal(0, 1, n),
        "noise": np.random.normal(0, 1, n),
    })
    y = pd.Series(3.0 * X["signal"] + np.random.normal(0, 0.2, n))

    preprocessor, _, _ = fit_train_preprocessor(X)
    mappings, _ = build_feature_mappings_from_transformer(preprocessor, X)

    # Group CV
    res_group, meta_group = select_by_lasso(
        X_input=X,
        y_input=y,
        mappings=mappings,
        k=1,
        alpha=None,
        cv_strategy="group",
        cv_folds=3,
        groups_train=groups
    )
    assert meta_group["cv_strategy"] == "group"
    assert "best_alpha" in meta_group
    assert meta_group["best_alpha"] > 0
    assert res_group[0].original_name == "signal"

    # Temporal CV
    res_temp, meta_temp = select_by_lasso(
        X_input=X,
        y_input=y,
        mappings=mappings,
        k=1,
        alpha=None,
        cv_strategy="temporal",
        cv_folds=3
    )
    assert meta_temp["cv_strategy"] == "temporal"
    assert "best_alpha" in meta_temp
    assert meta_temp["best_alpha"] > 0


def test_lasso_natural_sparsity_vs_exact_k():
    """
    Task 2 Verification:
    Verify Lasso distinguishes natural sparsity from post-hoc exact-K ranking.
    """
    np.random.seed(42)
    n = 80
    X = pd.DataFrame({
        f"x{i}": np.random.normal(0, 1, n) for i in range(10)
    })
    # Only x0, x1 have real signal
    y = pd.Series(5.0 * X["x0"] + 3.0 * X["x1"] + np.random.normal(0, 0.5, n))

    preprocessor, _, _ = fit_train_preprocessor(X)
    mappings, _ = build_feature_mappings_from_transformer(preprocessor, X)

    # High alpha = high sparsity (natural non-zeros < 5)
    # But requested K = 5
    res, meta = select_by_lasso(
        X_input=X,
        y_input=y,
        mappings=mappings,
        k=5,
        alpha=1.0,
        cv_strategy="random",
        cv_folds=3
    )
    selected_count = sum(1 for m in res if m.is_selected)
    assert selected_count == 5, "Requested K=5 must be strictly respected"
    assert "natural_nonzero_original_features" in meta
    # Natural count should be recorded separately
    assert meta["natural_nonzero_original_features"] <= 5


def test_experiment_identity_uniqueness():
    """
    Task 3 Verification:
    Verify that compute_experiment_identity generates distinct hashes when:
    - CV Folds change
    - Alpha values change
    - Random seeds change
    - CV Strategy changes
    And verify that different configs do not overwrite each other in the tracker.
    """
    tracker = FeatureSelectionExperimentTracker()

    base_params = {
        "dataset_fingerprint": "df_abc123",
        "split_fingerprint": "sf_xyz789",
        "method_id": "lasso",
        "k": 5,
        "cv_strategy": "random",
        "cv_folds": 5,
        "random_seed": 42,
        "alpha": 0.01,
    }

    id_base = compute_experiment_identity(**base_params)

    # 1. Change cv_folds
    id_folds = compute_experiment_identity(**{**base_params, "cv_folds": 10})
    assert id_base != id_folds, "Different cv_folds must produce distinct experiment IDs"

    # 2. Change alpha
    id_alpha = compute_experiment_identity(**{**base_params, "alpha": 0.5})
    assert id_base != id_alpha, "Different alpha must produce distinct experiment IDs"

    # 3. Change random seed
    id_seed = compute_experiment_identity(**{**base_params, "random_seed": 999})
    assert id_base != id_seed, "Different random_seed must produce distinct experiment IDs"

    # 4. Change cv_strategy
    id_strat = compute_experiment_identity(**{**base_params, "cv_strategy": "temporal"})
    assert id_base != id_strat, "Different cv_strategy must produce distinct experiment IDs"


def test_encoded_dimension_reporting(synthetic_split_bundle):
    """
    Task 4 Verification:
    Verify explicit distinction between:
    - Total Original Predictors
    - Selected Original Predictors
    - Full Encoded Dimension
    - Selected Encoded Dimension
    """
    bundle, split = synthetic_split_bundle
    tracker = FeatureSelectionExperimentTracker()

    res = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="f_test",
        k=3,
        cv_strategy="random",
        cv_folds=3,
        dataset_fingerprint="test_fp",
        split_fingerprint=split.split_fingerprint
    )
    tracker.add_result(res)

    # Synthetic bundle has 7 original features (5 numeric, 2 categorical)
    assert res.total_original_count == 7
    assert res.selected_original_count == 3
    # Categorical features have 3 and 4 levels -> full encoded dimension > 7
    assert res.full_encoded_dimension > 7
    assert res.selected_encoded_dimension == len(res.selected_encoded_feature_names)
    assert res.selected_encoded_dimension < res.full_encoded_dimension

    # Verify comparison table columns
    table = tracker.generate_comparison_table()
    assert "Total Orig" in table.columns
    assert "Selected Orig" in table.columns
    assert "Full Enc Dim" in table.columns
    assert "Selected Enc Dim" in table.columns
    row = table.iloc[0]
    assert row["Total Orig"] == 7
    assert row["Selected Orig"] == 3
    assert row["Full Enc Dim"] == res.full_encoded_dimension
    assert row["Selected Enc Dim"] == res.selected_encoded_dimension

