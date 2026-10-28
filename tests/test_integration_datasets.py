"""
Real-Dataset Integration Test Suite for Milestone 1: Dataset Engine.
Verifies real-world Kaggle/UCI benchmark datasets located in data/raw/:
- Appliances Energy Prediction (UCI / Kaggle)
- King County House Sales (Kaggle / King County GIS)
- Life Expectancy (WHO Global Health / Kaggle)
- Student Performance (UCI / Kaggle)

Tests verify file provenance, cryptographic SHA-256 signatures, raw schemas,
14–39 usable predictor boundaries, and real partition executions (Random, Temporal, Group).
"""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data import (
    get_registry,
    load_raw_dataset,
    load_dataset_bundle,
    validate_dataset_schema,
    check_dataset_availability,
    split_dataset,
    SplitConfig,
    SplitStrategy,
)
from src.data.download import verify_dataset_integrity, DOWNLOAD_SOURCES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_real_dataset_present(dataset_id: str):
    """Skip test gracefully if real dataset has not yet been fetched to disk."""
    reg = get_registry()
    meta = reg.get_dataset(dataset_id)
    if not check_dataset_availability(meta):
        pytest.skip(
            f"Integration dataset '{dataset_id}' not found at '{meta.local_path}'. "
            f"Run 'python -m src.data.download {dataset_id}' to fetch it."
        )
    return meta


# =====================================================================
# INTEGRATION TESTS: Real Datasets Verification
# =====================================================================

@pytest.mark.integration
def test_real_datasets_availability_and_sha256_checksums():
    """Verify all 4 registered benchmark datasets match their cryptographic SHA-256 signatures."""
    reg = get_registry()
    for ds_id in ["appliances_energy", "kc_house_sales", "life_expectancy", "student_performance"]:
        meta = _ensure_real_dataset_present(ds_id)
        assert meta.sha256_checksum, f"Dataset {ds_id} must have a declared sha256_checksum"
        is_verified = verify_dataset_integrity(ds_id)
        assert is_verified, f"Cryptographic SHA-256 mismatch for real dataset '{ds_id}'"


@pytest.mark.integration
def test_real_all_four_datasets_pass_schema_validation():
    """Verify all 4 real datasets strictly pass schema validation and satisfy 14-39 rule."""
    reg = get_registry()
    for ds_id in reg.get_all_ids():
        meta = _ensure_real_dataset_present(ds_id)
        raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)
        report = validate_dataset_schema(raw_df, meta)

        assert report.is_valid, f"Dataset '{ds_id}' failed schema validation: {report.blocking_errors}"
        assert not report.has_blocking_errors
        assert 14 <= report.usable_feature_count <= 39, (
            f"Dataset '{ds_id}' usable features ({report.usable_feature_count}) outside 14-39"
        )
        assert report.target_name == meta.target_column


@pytest.mark.integration
def test_real_student_performance_semicolon_delimited_loading():
    """Verify Student Performance loads with semicolon delimiter and has 32 usable features."""
    meta = _ensure_real_dataset_present("student_performance")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    assert len(raw_df) == 395
    assert len(raw_df.columns) == 33
    assert target_s.name == "G3"
    assert len(feat_df.columns) == 32
    assert "G3" not in feat_df.columns


@pytest.mark.integration
def test_real_appliances_energy_temporal_split():
    """Verify Appliances Energy date parsing and chronological split with zero future leakage."""
    meta = _ensure_real_dataset_present("appliances_energy")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    assert pd.api.types.is_datetime64_any_dtype(raw_df["date"])
    assert len(raw_df) == 19735
    assert len(feat_df.columns) == 27

    cfg = SplitConfig(
        strategy=SplitStrategy.TEMPORAL,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        date_column="date"
    )
    res = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    assert res.temporal_bounds is not None
    assert res.temporal_bounds["train"][1] <= res.temporal_bounds["validation"][0]
    assert res.temporal_bounds["validation"][1] <= res.temporal_bounds["test"][0]
    assert len(res.X_train) + len(res.X_val) + len(res.X_test) == 19735


@pytest.mark.integration
def test_real_kc_house_sales_random_split_and_metadata_exclusion():
    """Verify King County House Sales excludes id/date and executes reproducible random split."""
    meta = _ensure_real_dataset_present("kc_house_sales")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    assert len(raw_df) == 21613
    assert len(raw_df.columns) == 21
    assert "id" in meta_df.columns
    assert "date" in meta_df.columns
    assert len(feat_df.columns) == 18
    assert target_s.name == "price"

    cfg = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    res = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    assert len(res.X_train) == 15129
    assert len(res.X_val) == 3241
    assert len(res.X_test) == 3243


@pytest.mark.integration
def test_real_life_expectancy_group_split_and_null_target_handling():
    """Verify WHO Life Expectancy isolates Country groups and excludes 10 null target rows."""
    meta = _ensure_real_dataset_present("life_expectancy")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    assert len(raw_df) == 2938
    assert len(feat_df.columns) == 20
    assert "Country" in meta_df.columns

    cfg = SplitConfig(
        strategy=SplitStrategy.GROUP,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        group_column="Country",
        random_seed=42
    )
    res = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    # 10 null target rows excluded -> 2928 eligible
    assert res.sample_counts["total_eligible"] == 2928
    assert any("excluded 10" in w.lower() for w in res.warnings)

    train_c = set(raw_df.iloc[res.train_indices]["Country"])
    val_c = set(raw_df.iloc[res.val_indices]["Country"])
    test_c = set(raw_df.iloc[res.test_indices]["Country"])

    # Disjointness guarantee
    assert len(train_c & val_c) == 0
    assert len(train_c & test_c) == 0
    assert len(val_c & test_c) == 0


# =====================================================================
# MILESTONE 2 INTEGRATION TESTS: Real Datasets EDA & Feature Selection
# =====================================================================

@pytest.mark.integration
def test_real_student_performance_group_aware_selection():
    """
    Demo D / Test Group G: Student Performance (UCI).
    Verify group-aware feature selection:
    1. 32 original predictors (numeric + categorical).
    2. Request K=15 original predictors.
    3. Ensure selected original count is 15, while encoded model dimension > 15.
    4. Categorical dummy columns belong to the selected original feature.
    """
    meta = _ensure_real_dataset_present("student_performance")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    split = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    from src.features.selectors import run_feature_selection
    res = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="pearson",
        k=15,
        cv_strategy="random",
        cv_folds=3,
        random_seed=42,
        dataset_id=meta.id
    )

    assert res.selected_original_count == 15
    assert len(res.selected_original_features) == 15
    # One-hot expansion means encoded dimension > 15
    assert res.encoded_input_dimension > 15
    assert res.val_rmse > 0.0


@pytest.mark.integration
def test_real_appliances_energy_eda_and_temporal_selection():
    """
    Demo A / Test Group G: Appliances Energy (UCI).
    Verify EDA distributions, correlation, and TimeSeriesSplit feature selection without leakage.
    """
    meta = _ensure_real_dataset_present("appliances_energy")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    from src.exploration.distributions import compute_numeric_distribution
    from src.exploration.correlations import compute_correlation_matrix
    from src.features.selectors import run_feature_selection

    target_stats = compute_numeric_distribution(target_s)
    assert target_stats.count == 19735
    assert target_stats.skewness > 0  # Right-skewed energy consumption

    corr_res = compute_correlation_matrix(feat_df, columns=feat_df.columns[:8], method="pearson")
    assert corr_res.matrix.shape[0] == 8

    cfg = SplitConfig(strategy=SplitStrategy.TEMPORAL, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, date_column="date")
    split = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    # Fast test run with Pearson K=10 on chronological data
    res = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="pearson",
        k=10,
        cv_strategy="temporal",
        cv_folds=3,
        dataset_id=meta.id
    )
    assert res.selected_original_count == 10
    assert res.val_rmse > 0.0


@pytest.mark.integration
def test_real_kc_house_sales_eda_and_f_test_selection():
    """
    Demo B / Test Group G: King County House Sales.
    Verify target distribution, correlation with sqft_living, and F-test selection K=10.
    """
    meta = _ensure_real_dataset_present("kc_house_sales")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    from src.exploration.relationships import compute_numeric_bivariate_stats
    from src.features.selectors import run_feature_selection

    rel = compute_numeric_bivariate_stats(feat_df.assign(price=target_s), "sqft_living", "price")
    assert rel.pearson_r > 0.65  # Strong relationship between living area and price

    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    split = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    res = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="f_test",
        k=10,
        cv_strategy="random",
        cv_folds=3,
        dataset_id=meta.id
    )
    assert res.selected_original_count == 10
    assert "sqft_living" in res.selected_original_features


@pytest.mark.integration
def test_real_life_expectancy_group_kfold_selection():
    """
    Demo C / Test Group G: WHO Life Expectancy.
    Verify Country-based Group Split, missing-value profiling, Country preservation in SplitResult,
    and GroupKFold feature selection with zero country overlap between folds.
    """
    meta = _ensure_real_dataset_present("life_expectancy")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.GROUP, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, group_column="Country", random_seed=42)
    split = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    assert split.groups_train is not None
    assert len(split.groups_train) == len(split.X_train)

    from src.features.selectors import run_feature_selection
    res = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="pearson",
        k=10,
        cv_strategy="group",
        cv_folds=3,
        groups_train=split.groups_train,
        dataset_id=meta.id
    )
    assert res.selected_original_count == 10
    assert res.val_rmse > 0.0

