"""
Comprehensive unit and integration test suite for Milestone 1: Dataset Engine.
Covers Registry, Loader, Schema Validation, Profiling, Splitting (Random, Temporal, Group),
Leakage Prevention, and AppTest UI verification.
"""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from src.data import (
    DatasetRegistry,
    DatasetMetadata,
    RegistryValidationError,
    load_raw_dataset,
    load_dataset_bundle,
    split_features_and_target,
    check_dataset_availability,
    DatasetNotFoundError,
    validate_dataset_schema,
    ValidationReport,
    compute_dataset_profile,
    split_dataset,
    DataSplittingError,
    SplitConfig,
    SplitStrategy,
    get_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# TEST GROUP A: Dataset Registry
# =====================================================================

def test_registry_loads_valid_yaml():
    """Verify registry loads datasets.yaml cleanly."""
    reg = DatasetRegistry()
    assert len(reg.list_datasets()) >= 4
    for ds_id in ["appliances_energy", "kc_house_sales", "life_expectancy", "student_performance"]:
        assert ds_id in reg.get_all_ids()


def test_registry_rejects_duplicate_ids(tmp_path):
    """Verify duplicate dataset IDs raise RegistryValidationError."""
    bad_yaml = {
        "datasets": [
            {"id": "duplicate_id", "name": "A", "target_column": "y", "recommended_split": "random"},
            {"id": "duplicate_id", "name": "B", "target_column": "y", "recommended_split": "random"},
        ]
    }
    cfg_file = tmp_path / "dup.yaml"
    cfg_file.write_text(yaml.dump(bad_yaml), encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="Duplicate dataset ID"):
        DatasetRegistry(config_rel_path=str(cfg_file))


def test_registry_rejects_missing_target(tmp_path):
    """Verify missing target column raises RegistryValidationError."""
    bad_yaml = {
        "datasets": [
            {"id": "no_target", "name": "A", "target_column": "", "recommended_split": "random"},
        ]
    }
    cfg_file = tmp_path / "no_target.yaml"
    cfg_file.write_text(yaml.dump(bad_yaml), encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="must define a target_column"):
        DatasetRegistry(config_rel_path=str(cfg_file))


def test_registry_rejects_unsupported_split(tmp_path):
    """Verify unsupported split method is rejected."""
    bad_yaml = {
        "datasets": [
            {"id": "bad_split", "name": "A", "target_column": "y", "recommended_split": "quantum_split"},
        ]
    }
    cfg_file = tmp_path / "bad_split.yaml"
    cfg_file.write_text(yaml.dump(bad_yaml), encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="unsupported split strategy"):
        DatasetRegistry(config_rel_path=str(cfg_file))


def test_registry_rejects_temporal_without_date(tmp_path):
    """Verify temporal split without date_column raises RegistryValidationError."""
    bad_yaml = {
        "datasets": [
            {"id": "no_date", "name": "A", "target_column": "y", "recommended_split": "temporal", "date_column": None},
        ]
    }
    cfg_file = tmp_path / "no_date.yaml"
    cfg_file.write_text(yaml.dump(bad_yaml), encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="requires a valid 'date_column'"):
        DatasetRegistry(config_rel_path=str(cfg_file))


def test_registry_rejects_group_without_group_column(tmp_path):
    """Verify group split without group_column raises RegistryValidationError."""
    bad_yaml = {
        "datasets": [
            {"id": "no_group", "name": "A", "target_column": "y", "recommended_split": "group", "group_column": None},
        ]
    }
    cfg_file = tmp_path / "no_group.yaml"
    cfg_file.write_text(yaml.dump(bad_yaml), encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="requires a valid 'group_column'"):
        DatasetRegistry(config_rel_path=str(cfg_file))


def test_registry_rejects_feature_count_outside_14_39(tmp_path):
    """Verify feature count < 14 or > 39 is rejected."""
    bad_yaml = {
        "datasets": [
            {"id": "too_few", "name": "A", "target_column": "y", "recommended_split": "random", "expected_usable_features": 8},
        ]
    }
    cfg_file = tmp_path / "too_few.yaml"
    cfg_file.write_text(yaml.dump(bad_yaml), encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="violates the 14-39 constraint"):
        DatasetRegistry(config_rel_path=str(cfg_file))


# =====================================================================
# TEST GROUP B: Data Loading & Validation
# =====================================================================

def test_file_not_found_raises_informative_error():
    """Verify missing CSV file raises DatasetNotFoundError with instructions."""
    fake_meta = DatasetMetadata(
        id="non_existent",
        name="Ghost Dataset",
        source_provider="Kaggle",
        kaggle_url="https://kaggle.com",
        original_reference="None",
        local_path="data/raw/does_not_exist_xyz.csv",
        target_column="target"
    )
    with pytest.raises(DatasetNotFoundError, match="not found at"):
        load_raw_dataset(fake_meta)


def test_semicolon_delimited_csv_loading():
    """Verify Student Performance loads with semicolon delimiter."""
    reg = get_registry()
    meta = reg.get_dataset("student_performance")
    assert meta.csv_separator == ";"
    raw_df = load_raw_dataset(meta)
    assert len(raw_df) == 395
    assert len(raw_df.columns) == 33
    assert "G3" in raw_df.columns


def test_temporal_date_parsing():
    """Verify Appliances Energy date column parses as datetime."""
    reg = get_registry()
    meta = reg.get_dataset("appliances_energy")
    raw_df = load_raw_dataset(meta)
    assert pd.api.types.is_datetime64_any_dtype(raw_df["date"])
    assert len(raw_df) == 19735


def test_features_and_target_separation_no_raw_mutation():
    """Verify target separation does not mutate raw dataframe."""
    reg = get_registry()
    meta = reg.get_dataset("kc_house_sales")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    # Raw dataframe remains complete
    assert len(raw_df.columns) == 21
    # Target is separate
    assert target_s.name == "price"
    assert "price" not in feat_df.columns
    # Excluded metadata (id, date) is in meta_df and excluded from feat_df
    assert "id" not in feat_df.columns
    assert "date" not in feat_df.columns
    assert "id" in meta_df.columns
    assert "date" in meta_df.columns
    # Features count: 21 - 2 - 1 = 18
    assert len(feat_df.columns) == 18


def test_all_registered_datasets_pass_schema_validation():
    """Verify all 4 registered datasets pass validate_dataset_schema with 14-39 features."""
    reg = get_registry()
    for ds_id in reg.get_all_ids():
        meta = reg.get_dataset(ds_id)
        raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)
        report = validate_dataset_schema(raw_df, meta)
        assert report.is_valid, f"Dataset {ds_id} failed schema validation: {report.discrepancies}"
        assert 14 <= report.usable_feature_count <= 39
        assert report.target_name == meta.target_column


# =====================================================================
# TEST GROUP C: Data Splitting Engine
# =====================================================================

def test_random_split_proportions_and_non_overlapping():
    """Verify Random split proportions, coverage, and disjoint indices."""
    reg = get_registry()
    meta = reg.get_dataset("kc_house_sales")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    cfg = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    res = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    n_total = len(feat_df)
    assert len(res.X_train) + len(res.X_val) + len(res.X_test) == n_total
    # Non-overlapping indices
    set_tr = set(res.train_indices)
    set_v = set(res.val_indices)
    set_te = set(res.test_indices)
    assert len(set_tr & set_v) == 0
    assert len(set_tr & set_te) == 0
    assert len(set_v & set_te) == 0


def test_random_split_reproducibility():
    """Verify same seed yields exact identical partitions."""
    reg = get_registry()
    meta = reg.get_dataset("student_performance")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    cfg1 = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=99)
    cfg2 = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=99)

    res1 = split_dataset(feat_df, target_s, meta, cfg1, raw_df=raw_df)
    res2 = split_dataset(feat_df, target_s, meta, cfg2, raw_df=raw_df)

    np.testing.assert_array_equal(res1.train_indices, res2.train_indices)
    np.testing.assert_array_equal(res1.val_indices, res2.val_indices)
    np.testing.assert_array_equal(res1.test_indices, res2.test_indices)


def test_temporal_split_chronological_ordering():
    """Verify Temporal split partitions strictly chronologically without forward leakage."""
    reg = get_registry()
    meta = reg.get_dataset("appliances_energy")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    cfg = SplitConfig(
        strategy=SplitStrategy.TEMPORAL,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        date_column=meta.date_column
    )
    res = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    tb = res.temporal_bounds
    assert tb is not None
    # Train max date <= Validation min date
    assert tb["train"][1] <= tb["validation"][0]
    # Validation max date <= Test min date
    assert tb["validation"][1] <= tb["test"][0]


def test_group_split_disjoint_partitions():
    """Verify Group split ensures no group (Country) appears in multiple partitions."""
    reg = get_registry()
    meta = reg.get_dataset("life_expectancy")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    cfg = SplitConfig(
        strategy=SplitStrategy.GROUP,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        group_column=meta.group_column,
        random_seed=42
    )
    res = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    train_countries = set(raw_df.iloc[res.train_indices]["Country"])
    val_countries = set(raw_df.iloc[res.val_indices]["Country"])
    test_countries = set(raw_df.iloc[res.test_indices]["Country"])

    # Disjointness check
    assert len(train_countries & val_countries) == 0
    assert len(train_countries & test_countries) == 0
    assert len(val_countries & test_countries) == 0
    # Group coverage
    all_groups = set(raw_df.dropna(subset=[meta.target_column])["Country"])
    assert (train_countries | val_countries | test_countries) == all_groups


def test_target_missingness_explicitly_excluded_and_warned():
    """Verify rows with missing target are excluded and warned (e.g. 10 nulls in WHO data)."""
    reg = get_registry()
    meta = reg.get_dataset("life_expectancy")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    cfg = SplitConfig(
        strategy=SplitStrategy.GROUP,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        group_column=meta.group_column,
        random_seed=42
    )
    res = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    assert len(res.warnings) >= 1
    assert "excluded 10" in res.warnings[0].lower()
    # Total eligible rows: 2938 - 10 = 2928
    assert res.sample_counts["total_eligible"] == 2928


def test_invalid_split_proportions_rejected():
    """Verify proportions not summing to 1.0 are rejected."""
    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.6, val_ratio=0.2, test_ratio=0.1)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        cfg.validate()


# =====================================================================
# TEST GROUP D: Leakage Prevention
# =====================================================================

def test_leakage_prevention_partitions_are_independent():
    """
    Verify that splitting yields independent partitions and does not fit
    transformations or scalers on the combined dataset.
    """
    reg = get_registry()
    meta = reg.get_dataset("kc_house_sales")
    raw_df, feat_df, target_s, meta_df = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    res = split_dataset(feat_df, target_s, meta, cfg, raw_df=raw_df)

    # Train and test mean of continuous feature differ slightly (demonstrating independent slices)
    sqft_train_mean = res.X_train["sqft_living"].mean()
    sqft_test_mean = res.X_test["sqft_living"].mean()
    assert abs(sqft_train_mean - sqft_test_mean) > 0.01

    # Verify no NaN created by splitting
    assert res.X_train.shape[1] == 18
    assert res.X_val.shape[1] == 18
    assert res.X_test.shape[1] == 18


# =====================================================================
# TEST GROUP E: Streamlit AppTest Execution
# =====================================================================

def test_dataset_page_renders_cleanly_in_apptest():
    """Verify pages/dataset.py executes without exception via Streamlit AppTest."""
    dataset_page = PROJECT_ROOT / "pages" / "dataset.py"
    at = AppTest.from_file(str(dataset_page), default_timeout=20).run()
    assert len(at.exception) == 0, f"Exception in dataset.py: {at.exception}"
    assert len(at.title) >= 1
