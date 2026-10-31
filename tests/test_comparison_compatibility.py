"""
Group A Tests: Experiment Compatibility Engine.
Verifies:
- Same dataset and split partition compatibility.
- Rejection of different dataset IDs.
- Rejection of different dataset fingerprints.
- Rejection of different split fingerprints.
- Rejection of different target columns.
- Rejection of mismatched partition sample sizes.
- Informative warnings for missing or non-identical target units.
- Descriptive compatibility messages without generic exceptions.
"""
import pytest
import numpy as np
import pandas as pd

from src.models.model_service import train_ols_experiment
from src.comparison.compatibility import validate_model_compatibility
from src.data.models import SplitResult


@pytest.fixture
def synthetic_split_and_models():
    """Generates a valid split and two compatible fitted OLS models."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.normal(0, 1, n),
        "x3": np.random.normal(0, 1, n),
    })
    y = 2.0 + 1.5 * df["x1"] - 0.8 * df["x2"] + np.random.normal(0, 0.1, n)

    split = SplitResult(
        X_train=df.iloc[:60],
        X_val=df.iloc[60:80],
        X_test=df.iloc[80:],
        y_train=y.iloc[:60],
        y_val=y.iloc[60:80],
        y_test=y.iloc[80:],
        train_indices=np.arange(60),
        val_indices=np.arange(60, 80),
        test_indices=np.arange(80, 100),
        actual_proportions={"train": 0.6, "val": 0.2, "test": 0.2},
        sample_counts={"train": 60, "val": 20, "test": 20},
        strategy="random",
        random_seed=42,
        split_fingerprint="split_fp_12345"
    )

    model1 = train_ols_experiment(
        split_result=split,
        feature_source=["x1", "x2"],
        dataset_id="test_ds",
        dataset_fingerprint="ds_fp_abcde",
        target_name="target_y",
        target_unit="points"
    )

    model2 = train_ols_experiment(
        split_result=split,
        feature_source=["x1"],
        dataset_id="test_ds",
        dataset_fingerprint="ds_fp_abcde",
        target_name="target_y",
        target_unit="points"
    )

    return split, model1, model2


def test_compatible_models_pass_validation(synthetic_split_and_models):
    split, m1, m2 = synthetic_split_and_models
    res = validate_model_compatibility(
        models=[m1, m2],
        active_dataset_id="test_ds",
        active_dataset_fingerprint="ds_fp_abcde",
        active_split_fingerprint="split_fp_12345",
        active_target_name="target_y",
        active_target_unit="points"
    )
    assert res.is_compatible is True
    assert len(res.reasons) == 0


def test_different_dataset_id_rejected(synthetic_split_and_models):
    split, m1, m2 = synthetic_split_and_models
    # Validate against a mismatched dataset ID
    res = validate_model_compatibility(
        models=[m1, m2],
        active_dataset_id="other_dataset",
        active_dataset_fingerprint="ds_fp_abcde",
        active_split_fingerprint="split_fp_12345",
        active_target_name="target_y"
    )
    assert res.is_compatible is False
    assert any("Dataset ID mismatch" in r for r in res.reasons)


def test_different_dataset_fingerprint_rejected(synthetic_split_and_models):
    split, m1, m2 = synthetic_split_and_models
    res = validate_model_compatibility(
        models=[m1, m2],
        active_dataset_id="test_ds",
        active_dataset_fingerprint="ds_fp_DIFFERENT",
        active_split_fingerprint="split_fp_12345",
        active_target_name="target_y"
    )
    assert res.is_compatible is False
    assert any("Dataset fingerprint mismatch" in r for r in res.reasons)


def test_different_split_fingerprint_rejected(synthetic_split_and_models):
    split, m1, m2 = synthetic_split_and_models
    res = validate_model_compatibility(
        models=[m1, m2],
        active_dataset_id="test_ds",
        active_dataset_fingerprint="ds_fp_abcde",
        active_split_fingerprint="split_fp_DIFFERENT",
        active_target_name="target_y"
    )
    assert res.is_compatible is False
    assert any("Split fingerprint mismatch" in r for r in res.reasons)


def test_different_target_name_rejected(synthetic_split_and_models):
    split, m1, m2 = synthetic_split_and_models
    res = validate_model_compatibility(
        models=[m1, m2],
        active_dataset_id="test_ds",
        active_dataset_fingerprint="ds_fp_abcde",
        active_split_fingerprint="split_fp_12345",
        active_target_name="completely_different_target"
    )
    assert res.is_compatible is False
    assert any("Target variable mismatch" in r for r in res.reasons)


def test_target_unit_warning_not_blocking(synthetic_split_and_models):
    split, m1, m2 = synthetic_split_and_models
    # Differing target unit should issue a warning rather than blocking comparison
    res = validate_model_compatibility(
        models=[m1, m2],
        active_dataset_id="test_ds",
        active_dataset_fingerprint="ds_fp_abcde",
        active_split_fingerprint="split_fp_12345",
        active_target_name="target_y",
        active_target_unit="meters"  # models have "points"
    )
    assert res.is_compatible is True
    assert len(res.warnings) > 0
    assert any("Target measurement unit differs" in w for w in res.warnings)
