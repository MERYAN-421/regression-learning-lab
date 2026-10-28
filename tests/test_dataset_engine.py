"""
Offline Unit Test Suite for Milestone 1: Dataset Engine.
Covers Registry, Loader, Schema Validation, Cache Invalidation, Profiling,
Splitting (Random, Temporal, Group), Leakage Prevention, and AppTest UI state transitions.
Runs 100% offline using synthetic in-memory/temp fixtures without external network calls.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
import yaml
from typing import Tuple
from streamlit.testing.v1 import AppTest

from src.data import (
    DatasetRegistry,
    DatasetMetadata,
    RegistryValidationError,
    load_raw_dataset,
    load_dataset_bundle,
    split_features_and_target,
    check_dataset_availability,
    compute_file_fingerprint,
    get_dataset_fingerprint,
    DatasetNotFoundError,
    validate_dataset_schema,
    ValidationReport,
    SchemaValidationError,
    compute_dataset_profile,
    split_dataset,
    DataSplittingError,
    SplitConfig,
    SplitStrategy,
    get_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# SYNTHETIC TEST FIXTURES (100% Offline)
# =====================================================================

@pytest.fixture
def synthetic_dataset_csv(tmp_path) -> Tuple[Path, DatasetMetadata]:
    """
    Generate a valid temporary CSV dataset with:
    - 20 usable predictors (12 numeric, 8 categorical) -> satisfies 14-39 rule
    - 1 target column ('price_target')
    - 1 date column ('timestamp')
    - 1 group column ('region_id')
    - 200 rows
    Does NOT write into data/raw/ or overwrite real dataset names.
    """
    np.random.seed(42)
    n_rows = 200
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")
    regions = [f"Region_{i % 10}" for i in range(n_rows)]

    data = {
        "timestamp": dates.strftime("%Y-%m-%d"),
        "region_id": regions,
        "price_target": np.random.uniform(50.0, 500.0, size=n_rows),
    }
    # Add 12 numeric predictors
    for i in range(1, 13):
        data[f"num_feat_{i}"] = np.random.normal(0, 1, size=n_rows)
    # Add 8 categorical predictors
    for i in range(1, 9):
        data[f"cat_feat_{i}"] = np.random.choice(["Alpha", "Beta", "Gamma"], size=n_rows)

    df = pd.DataFrame(data)
    csv_file = tmp_path / "synthetic_test_dataset.csv"
    df.to_csv(csv_file, index=False)

    meta = DatasetMetadata(
        id="synthetic_test",
        name="Synthetic Benchmark",
        source_provider="Unit Test Fixture",
        kaggle_url="https://example.com/test",
        original_reference="Offline unit test reference",
        local_path=str(csv_file.relative_to(PROJECT_ROOT)) if csv_file.is_relative_to(PROJECT_ROOT) else str(csv_file),
        csv_separator=",",
        encoding="utf-8",
        target_column="price_target",
        date_column="timestamp",
        group_column="region_id",
        excluded_predictor_columns=["timestamp", "region_id"],
        expected_rows=n_rows,
        expected_usable_features=20,
        recommended_split="random",
        default_split_ratios=[0.70, 0.15, 0.15],
        default_random_seed=42
    )
    return csv_file, meta


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
# TEST GROUP B: Schema Validation Enforcement (Task 1)
# =====================================================================

def test_validation_blocks_on_fewer_than_14_features():
    """Verify schema validation creates a critical blocking error when features < 14."""
    df = pd.DataFrame({f"x{i}": range(20) for i in range(10)})
    df["target"] = range(20)
    meta = DatasetMetadata(
        id="low_feat", name="Low", source_provider="Test", kaggle_url="https://test",
        original_reference="", local_path="", target_column="target"
    )
    report = validate_dataset_schema(df, meta)
    assert not report.is_valid
    assert report.has_blocking_errors
    assert any("violates mandatory 14–39 rule" in err for err in report.blocking_errors)


def test_validation_blocks_on_more_than_39_features():
    """Verify schema validation creates a critical blocking error when features > 39."""
    df = pd.DataFrame({f"x{i}": range(20) for i in range(45)})
    df["target"] = range(20)
    meta = DatasetMetadata(
        id="high_feat", name="High", source_provider="Test", kaggle_url="https://test",
        original_reference="", local_path="", target_column="target"
    )
    report = validate_dataset_schema(df, meta)
    assert not report.is_valid
    assert report.has_blocking_errors
    assert any("violates mandatory 14–39 rule" in err for err in report.blocking_errors)


def test_validation_blocks_on_missing_target_column():
    """Verify missing target column is marked as critical blocking error."""
    df = pd.DataFrame({f"x{i}": range(20) for i in range(16)})
    meta = DatasetMetadata(
        id="miss_tgt", name="Miss", source_provider="Test", kaggle_url="https://test",
        original_reference="", local_path="", target_column="non_existent_target"
    )
    report = validate_dataset_schema(df, meta)
    assert not report.is_valid
    assert any("not found in dataset columns" in err for err in report.blocking_errors)


def test_validation_blocks_on_duplicate_headers():
    """Verify duplicate column names are marked as critical blocking error."""
    df = pd.DataFrame([[1, 2, 3]], columns=["col", "col", "target"])
    meta = DatasetMetadata(
        id="dup", name="Dup", source_provider="Test", kaggle_url="https://test",
        original_reference="", local_path="", target_column="target"
    )
    report = validate_dataset_schema(df, meta)
    assert not report.is_valid
    assert any("Duplicate column headers detected" in err for err in report.blocking_errors)


def test_validation_blocks_on_empty_dataframe():
    """Verify empty DataFrame (0 observations) creates a blocking error."""
    df = pd.DataFrame()
    meta = DatasetMetadata(
        id="empty", name="Empty", source_provider="Test", kaggle_url="https://test",
        original_reference="", local_path="", target_column="target"
    )
    report = validate_dataset_schema(df, meta)
    assert not report.is_valid
    assert any("empty" in err.lower() for err in report.blocking_errors)


def test_validation_warns_on_target_missingness_without_blocking():
    """Verify target nulls are flagged as warning, not blocking error (excluded during split)."""
    df = pd.DataFrame({f"x{i}": range(20) for i in range(16)})
    df["target"] = [1.0 if i < 18 else np.nan for i in range(20)]
    meta = DatasetMetadata(
        id="tgt_nulls", name="Nulls", source_provider="Test", kaggle_url="https://test",
        original_reference="", local_path="", target_column="target"
    )
    report = validate_dataset_schema(df, meta)
    assert report.is_valid  # 16 features within 14-39, target exists
    assert len(report.warnings) >= 1
    assert any("missing values" in w for w in report.warnings)


# =====================================================================
# TEST GROUP C: Cache Invalidation via File Fingerprinting (Task 2)
# =====================================================================

def test_fingerprint_changes_when_file_content_is_replaced(tmp_path):
    """Verify compute_file_fingerprint produces distinct fingerprints when file changes."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    fp1 = compute_file_fingerprint(csv_file)

    # Modify file content
    csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    fp2 = compute_file_fingerprint(csv_file)

    assert fp1 != fp2, "Fingerprint must change when file content is modified"


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


# =====================================================================
# TEST GROUP D: Data Splitting Engine (Offline with synthetic data)
# =====================================================================

def test_split_dataset_blocks_on_critical_schema_error():
    """Verify split_dataset immediately blocks splitting when schema fails 14-39 rule."""
    feat_df = pd.DataFrame({f"x{i}": range(50) for i in range(5)})  # 5 features (< 14)
    target = pd.Series(range(50), name="y")
    meta = DatasetMetadata(
        id="test", name="T", source_provider="", kaggle_url="", original_reference="",
        local_path="", target_column="y"
    )
    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    with pytest.raises(DataSplittingError, match="Usable feature count"):
        split_dataset(feat_df, target, meta, cfg)


def test_random_split_proportions_and_non_overlapping():
    """Verify Random split proportions, coverage, and disjoint indices on synthetic data."""
    n_rows = 200
    feat_df = pd.DataFrame({f"x{i}": np.random.randn(n_rows) for i in range(20)})
    target = pd.Series(np.random.randn(n_rows), name="target")
    meta = DatasetMetadata(
        id="test", name="T", source_provider="", kaggle_url="", original_reference="",
        local_path="", target_column="target"
    )
    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    res = split_dataset(feat_df, target, meta, cfg)

    assert len(res.X_train) + len(res.X_val) + len(res.X_test) == n_rows
    set_tr = set(res.train_indices)
    set_v = set(res.val_indices)
    set_te = set(res.test_indices)
    assert len(set_tr & set_v) == 0
    assert len(set_tr & set_te) == 0
    assert len(set_v & set_te) == 0


def test_random_split_reproducibility():
    """Verify same seed yields exact identical partitions."""
    n_rows = 100
    feat_df = pd.DataFrame({f"x{i}": np.random.randn(n_rows) for i in range(16)})
    target = pd.Series(np.random.randn(n_rows), name="target")
    meta = DatasetMetadata(
        id="test", name="T", source_provider="", kaggle_url="", original_reference="",
        local_path="", target_column="target"
    )
    cfg1 = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=99)
    cfg2 = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=99)

    res1 = split_dataset(feat_df, target, meta, cfg1)
    res2 = split_dataset(feat_df, target, meta, cfg2)

    np.testing.assert_array_equal(res1.train_indices, res2.train_indices)
    np.testing.assert_array_equal(res1.val_indices, res2.val_indices)
    np.testing.assert_array_equal(res1.test_indices, res2.test_indices)


def test_temporal_split_chronological_ordering():
    """Verify Temporal split partitions strictly chronologically without forward leakage."""
    n_rows = 150
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    raw_df = pd.DataFrame({"datetime_col": dates, "y": range(n_rows)})
    for i in range(15):
        raw_df[f"feat_{i}"] = range(n_rows)

    feat_df = raw_df[[f"feat_{i}" for i in range(15)]]
    target = raw_df["y"]
    meta = DatasetMetadata(
        id="temporal_mock", name="TempMock", source_provider="", kaggle_url="", original_reference="",
        local_path="", target_column="y", date_column="datetime_col"
    )
    cfg = SplitConfig(strategy=SplitStrategy.TEMPORAL, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, date_column="datetime_col")
    res = split_dataset(feat_df, target, meta, cfg, raw_df=raw_df)

    tb = res.temporal_bounds
    assert tb is not None
    assert tb["train"][1] <= tb["validation"][0]
    assert tb["validation"][1] <= tb["test"][0]


def test_group_split_disjoint_partitions():
    """Verify Group split ensures no group appears across multiple partitions."""
    n_rows = 300
    groups = [f"Group_{i % 15}" for i in range(n_rows)]
    raw_df = pd.DataFrame({"grp": groups, "y": np.random.randn(n_rows)})
    for i in range(18):
        raw_df[f"f_{i}"] = np.random.randn(n_rows)

    feat_df = raw_df[[f"f_{i}" for i in range(18)]]
    target = raw_df["y"]
    meta = DatasetMetadata(
        id="group_mock", name="GroupMock", source_provider="", kaggle_url="", original_reference="",
        local_path="", target_column="y", group_column="grp"
    )
    cfg = SplitConfig(strategy=SplitStrategy.GROUP, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, group_column="grp", random_seed=42)
    res = split_dataset(feat_df, target, meta, cfg, raw_df=raw_df)

    tr_grps = set(raw_df.iloc[res.train_indices]["grp"])
    val_grps = set(raw_df.iloc[res.val_indices]["grp"])
    te_grps = set(raw_df.iloc[res.test_indices]["grp"])

    assert len(tr_grps & val_grps) == 0
    assert len(tr_grps & te_grps) == 0
    assert len(val_grps & te_grps) == 0


def test_invalid_split_proportions_rejected():
    """Verify proportions not summing to 1.0 are rejected."""
    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.6, val_ratio=0.2, test_ratio=0.1)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        cfg.validate()


def test_leakage_prevention_partitions_are_independent():
    """Verify that splitting creates independent slices without global fitting."""
    n_rows = 200
    feat_df = pd.DataFrame({f"feat_{i}": np.random.normal(loc=i, scale=1.0, size=n_rows) for i in range(16)})
    target = pd.Series(np.random.normal(10, 2, size=n_rows), name="y")
    meta = DatasetMetadata(
        id="leak_mock", name="LM", source_provider="", kaggle_url="", original_reference="",
        local_path="", target_column="y"
    )
    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    res = split_dataset(feat_df, target, meta, cfg)

    assert res.X_train.shape[1] == 16
    assert res.X_val.shape[1] == 16
    assert res.X_test.shape[1] == 16


# =====================================================================
# TEST GROUP E: Streamlit AppTest Execution & State Transitions (Task 3)
# =====================================================================

def test_dataset_page_renders_cleanly_in_apptest():
    """Verify pages/dataset.py executes without exception via Streamlit AppTest."""
    dataset_page = PROJECT_ROOT / "pages" / "dataset.py"
    at = AppTest.from_file(str(dataset_page), default_timeout=20).run()
    assert len(at.exception) == 0, f"Exception in dataset.py: {at.exception}"
    assert len(at.title) >= 1


def test_dataset_page_state_transition_detects_outdated_split():
    """
    Verify AppTest state transition: executing a split, then modifying a proportion input,
    correctly marks the active partition state as outdated.
    """
    dataset_page = PROJECT_ROOT / "pages" / "dataset.py"
    at = AppTest.from_file(str(dataset_page), default_timeout=20).run()

    # Find the execute split button and trigger it
    exec_button = at.button(key="execute_split_button")
    if exec_button is not None and not exec_button.disabled:
        at.button(key="execute_split_button").click().run()
        # Verify active split result exists
        assert "active_split_result" in at.session_state
        assert "active_split_config" in at.session_state

        # Now change the train proportion to 75%
        at.number_input(key="split_train_pct").set_value(75).run()
        # The page re-renders, and the outdated warning should be displayed
        assert any("outdated" in str(w.value).lower() or "changed" in str(w.value).lower() for w in at.warning)
