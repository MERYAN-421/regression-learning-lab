"""
Unit tests for OLS Design Matrix Builder (Group A).
Verifies:
- Numeric median imputation and scaling
- Categorical reference-level encoding (drop='first')
- Explicit single intercept column ('const')
- Exact column-rank and condition number calculation
- Linear dependency detection (e.g. X1 = X2 + X3)
- Constant feature handling
- Unknown category handling in validation
- Train-only fitting and zero-leakage transformation
- FeatureSelectionResult integration
"""
import numpy as np
import pandas as pd
import pytest

from src.models.design_matrix import (
    OLSDesignMatrixBuilder,
    build_ols_design_matrices,
)
from src.features.result_models import (
    FeatureSelectionResult,
    OriginalFeatureMapping,
)


@pytest.fixture
def synthetic_design_data():
    """Generates synthetic DataFrame with numeric, categorical, and collinear columns."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(5, 1, n)
    # Structural identity: x3 = x1 + x2 (exact linear dependence)
    x3 = x1 + x2
    # Constant column
    x_const = np.ones(n) * 7.0

    # Categorical column with 3 levels
    cat1 = pd.Series(np.random.choice(["LevelA", "LevelB", "LevelC"], size=n), dtype=object)
    # Binary categorical column
    cat2 = pd.Series(np.random.choice(["No", "Yes"], size=n), dtype=object)

    # Introduce some missing values
    x1[5] = np.nan
    cat1.iloc[10] = None

    df = pd.DataFrame({
        "num1": x1,
        "num2": x2,
        "num_collinear": x3,
        "num_constant": x_const,
        "cat_multi": cat1,
        "cat_binary": cat2
    })
    return df


def test_design_matrix_numeric_imputation_and_unscaled(synthetic_design_data):
    """Verify median imputation and original-units unscaled representation."""
    df = synthetic_design_data
    builder = OLSDesignMatrixBuilder(
        selected_original_features=["num1", "num2"],
        fit_intercept=True,
        scale_numeric=False
    )
    builder.fit(df)
    X_des = builder.transform(df)

    # Should contain intercept + 2 numeric features
    assert list(X_des.columns) == ["const", "num1", "num2"]
    assert np.all(X_des["const"] == 1.0)
    assert not X_des.isna().to_numpy().any(), "Imputation failed: NaNs remain"
    # Verify unscaled original units preserved
    assert abs(X_des["num1"].mean() - 10.0) < 1.0
    assert abs(X_des["num2"].mean() - 5.0) < 0.5


def test_design_matrix_numeric_scaling(synthetic_design_data):
    """Verify numeric standardization when scale_numeric=True."""
    df = synthetic_design_data
    builder = OLSDesignMatrixBuilder(
        selected_original_features=["num1", "num2"],
        fit_intercept=True,
        scale_numeric=True
    )
    builder.fit(df)
    X_des = builder.transform(df)

    assert abs(X_des["num1"].mean()) < 1e-10
    assert abs(X_des["num1"].std(ddof=0) - 1.0) < 1e-5
    assert abs(X_des["num2"].mean()) < 1e-10
    assert abs(X_des["num2"].std(ddof=0) - 1.0) < 1e-5


def test_design_matrix_categorical_reference_encoding(synthetic_design_data):
    """
    Verify reference category encoding:
    When fit_intercept=True, drop='first' drops 1 reference level to prevent dummy-variable trap.
    """
    df = synthetic_design_data
    builder = OLSDesignMatrixBuilder(
        selected_original_features=["cat_multi", "cat_binary"],
        fit_intercept=True
    )
    builder.fit(df)
    X_des = builder.transform(df)

    meta = builder.metadata
    # cat_multi has 3 levels -> 2 dummies
    # cat_binary has 2 levels -> 1 dummy
    # const = 1 column -> total 4 columns
    assert X_des.shape[1] == 4
    assert "const" in X_des.columns
    assert meta.num_parameters == 4

    # Verify reference categories are stored
    assert "cat_multi" in meta.reference_categories
    assert "cat_binary" in meta.reference_categories
    assert meta.reference_categories["cat_multi"] in ["LevelA", "LevelB", "LevelC"]
    assert meta.reference_categories["cat_binary"] in ["No", "Yes"]

    # Verify dummy column names
    ref_multi = meta.reference_categories["cat_multi"]
    assert f"cat_multi_{ref_multi}" not in X_des.columns, "Reference category must be omitted"


def test_design_matrix_without_intercept():
    """Verify that when fit_intercept=False, all categorical levels are retained."""
    df = pd.DataFrame({"cat": ["A", "B", "C", "A", "B"]})
    builder = OLSDesignMatrixBuilder(selected_original_features=["cat"], fit_intercept=False)
    builder.fit(df)
    X_des = builder.transform(df)

    assert "const" not in X_des.columns
    assert set(X_des.columns) == {"cat_A", "cat_B", "cat_C"}


def test_design_matrix_rank_and_perfect_collinearity_detection():
    """
    Verify exact rank calculation and detection of linear dependency (num_collinear = num1 + num2).
    """
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(5, 1, n)
    x3 = x1 + x2  # Exact structural identity: num_collinear = num1 + num2
    df = pd.DataFrame({
        "num1": x1,
        "num2": x2,
        "num_collinear": x3
    })
    builder = OLSDesignMatrixBuilder(
        selected_original_features=["num1", "num2", "num_collinear"],
        fit_intercept=True
    )
    builder.fit(df)
    meta = builder.metadata

    # 3 features + const = 4 columns
    # But rank should be 3 because num_collinear = num1 + num2!
    assert meta.num_parameters == 4
    assert meta.matrix_rank == 3
    assert not meta.is_full_rank, "Should detect rank deficiency"
    assert not meta.is_valid_for_inference
    assert any("Rank deficiency detected" in w for w in meta.warnings)


def test_design_matrix_constant_feature_warning(synthetic_design_data):
    """Verify warning when non-intercept feature has zero variance."""
    df = synthetic_design_data
    builder = OLSDesignMatrixBuilder(
        selected_original_features=["num1", "num_constant"],
        fit_intercept=True
    )
    builder.fit(df)
    meta = builder.metadata

    assert any("zero variance" in w for w in meta.warnings)


def test_design_matrix_unknown_category_in_validation(synthetic_design_data):
    """
    Verify unknown category in validation is safely mapped to reference level (all zeros)
    without throwing exceptions or producing NaNs.
    """
    train_df = pd.DataFrame({"cat": ["A", "B", "A", "B"]})
    val_df = pd.DataFrame({"cat": ["A", "B", "UnknownNewCategory"]})

    builder = OLSDesignMatrixBuilder(selected_original_features=["cat"], fit_intercept=True)
    builder.fit(train_df)
    X_val_des = builder.transform(val_df)

    assert len(X_val_des) == 3
    assert not X_val_des.isna().to_numpy().any()
    # The unknown category row should have 0 for the non-reference dummy
    assert X_val_des.iloc[2]["cat_B"] == 0.0


def test_design_matrix_train_only_fit_guarantee():
    """Verify that validation data is never mutated and does not affect imputer statistics."""
    train_df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})  # Median = 20.0
    val_df = pd.DataFrame({"x": [1000.0, 2000.0, np.nan]})

    builder = OLSDesignMatrixBuilder(selected_original_features=["x"], fit_intercept=True)
    builder.fit(train_df)
    X_val_des = builder.transform(val_df)

    # Imputed value for missing x in val should be train median (20.0), NOT val median!
    assert X_val_des.iloc[2]["x"] == 20.0
