"""
Offline Unit Test Suite for Milestone 2: Exploratory Data Analysis (EDA) Engine.
Covers:
- Test Group A: EDA
  * Numeric distribution summary metrics (mean, median, std, skewness, kurtosis, quantiles)
  * Categorical distribution summary metrics (cardinality, frequencies, missingness)
  * Missing value and data quality checks
  * Constant variable handling and exclusion from correlation
  * Pearson vs Spearman correlation calculations and ranking
  * Non-linear monotonic relationship behavior (Spearman vs Pearson)
  * Missing value handling in correlations (min_periods & pairwise dropna)
  * Outlier flagging with IQR fences
  * Empty dataset and invalid method error handling
  * Bivariate relationships (numeric-to-target & categorical-to-target)
  * Plotly figure generation for distributions, correlations, and relationships
"""
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.exploration.distributions import (
    compute_numeric_distribution,
    compute_categorical_distribution,
    NumericDistributionStats,
    CategoricalDistributionStats,
)
from src.exploration.correlations import (
    compute_correlation_matrix,
    CorrelationMatrixResult,
)
from src.exploration.relationships import (
    compute_numeric_bivariate_stats,
    compute_categorical_target_summary,
    NumericBivariateStats,
)
from src.exploration.plots import (
    plot_numeric_distribution,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_bivariate_scatter,
    plot_categorical_target_boxplot,
)


@pytest.fixture
def synthetic_eda_df():
    """Generates a clean synthetic DataFrame for EDA tests."""
    np.random.seed(42)
    n = 150
    x_num = np.random.normal(loc=50.0, scale=10.0, size=n)
    # Add a couple deliberate outliers beyond 1.5 * IQR
    x_num[0] = 130.0
    x_num[1] = -30.0

    # Non-linear monotonic variable: y_mono = exp(x_mono / 2)
    x_mono = np.linspace(1, 10, n)
    y_mono = np.exp(x_mono / 2.0)

    # Constant column
    const_col = np.ones(n) * 42.0

    # Categorical column with some NaNs
    categories = ["TypeA", "TypeB", "TypeC"]
    cat_col = np.random.choice(categories, size=n, p=[0.5, 0.35, 0.15])
    # Introduce explicit NaNs
    cat_s = pd.Series(cat_col, dtype=object)
    cat_s.iloc[0:5] = np.nan

    # Target column
    target = 2.5 * x_num + np.random.normal(0, 5, n)

    df = pd.DataFrame({
        "num_feat": x_num,
        "x_mono": x_mono,
        "y_mono": y_mono,
        "const_feat": const_col,
        "cat_feat": cat_s,
        "target": target,
    })
    return df


# =====================================================================
# Test Group A: EDA Unit Tests
# =====================================================================

def test_numeric_distribution_calculation(synthetic_eda_df):
    """Verify numeric distribution stats: mean, median, std, skewness, kurtosis, quantiles."""
    s = synthetic_eda_df["num_feat"]
    stats = compute_numeric_distribution(s)
    assert isinstance(stats, NumericDistributionStats)
    assert stats.feature_name == "num_feat"
    assert stats.count == len(synthetic_eda_df)
    assert stats.missing_count == 0
    assert stats.q25 < stats.median < stats.q75
    assert stats.iqr == pytest.approx(stats.q75 - stats.q25)
    assert stats.outlier_count_iqr >= 2  # At least the 2 deliberate outliers
    assert stats.outlier_pct_iqr > 0.0
    assert stats.std > 0
    assert isinstance(stats.skewness, float)
    assert isinstance(stats.kurtosis, float)


def test_categorical_distribution_calculation(synthetic_eda_df):
    """Verify categorical distribution stats: category frequency, counts, missingness."""
    s = synthetic_eda_df["cat_feat"]
    stats = compute_categorical_distribution(s)
    assert isinstance(stats, CategoricalDistributionStats)
    assert stats.feature_name == "cat_feat"
    assert stats.cardinality == 3  # TypeA, TypeB, TypeC
    assert stats.missing_count > 0
    assert stats.mode == "TypeA"
    assert not stats.frequency_table.empty
    assert "Category" in stats.frequency_table.columns
    assert "Count" in stats.frequency_table.columns


def test_constant_variable_handling(synthetic_eda_df):
    """Verify that constant variables are flagged and handled safely."""
    stats = compute_numeric_distribution(synthetic_eda_df["const_feat"])
    assert stats.is_constant is True
    assert stats.std == 0.0
    assert stats.outlier_count_iqr == 0


def test_correlation_matrix_pearson_and_spearman(synthetic_eda_df):
    """Verify correlation matrix computation for Pearson and Spearman methods and constant exclusion."""
    cols = ["num_feat", "target", "x_mono", "y_mono", "const_feat"]
    
    # Pearson
    pearson_res = compute_correlation_matrix(synthetic_eda_df, cols, method="pearson")
    assert isinstance(pearson_res, CorrelationMatrixResult)
    assert pearson_res.method == "pearson"
    # const_feat has zero variance and should be cleanly excluded to avoid div by zero
    assert "const_feat" in pearson_res.constant_columns_excluded
    assert pearson_res.matrix.shape == (4, 4)
    # Diagonal should be 1.0
    assert pearson_res.matrix.loc["num_feat", "num_feat"] == pytest.approx(1.0)
    assert len(pearson_res.top_correlated_pairs) > 0

    # Spearman
    spearman_res = compute_correlation_matrix(synthetic_eda_df, cols, method="spearman")
    assert spearman_res.method == "spearman"
    assert spearman_res.matrix.loc["num_feat", "num_feat"] == pytest.approx(1.0)


def test_pearson_vs_spearman_monotonic_behavior(synthetic_eda_df):
    """
    Verify that Spearman correlation correctly detects monotonic non-linear association
    (y = exp(x/2)) with rho = 1.0, while Pearson r < 1.0.
    """
    spearman_res = compute_correlation_matrix(synthetic_eda_df, ["x_mono", "y_mono"], method="spearman")
    pearson_res = compute_correlation_matrix(synthetic_eda_df, ["x_mono", "y_mono"], method="pearson")

    spearman_rho = spearman_res.matrix.loc["x_mono", "y_mono"]
    pearson_r = pearson_res.matrix.loc["x_mono", "y_mono"]

    # Spearman should be 1.0 for perfect monotonic curve
    assert spearman_rho == pytest.approx(1.0, rel=1e-4)
    # Pearson should be strictly less than 1.0 due to non-linearity
    assert pearson_r < 0.95
    assert spearman_rho > pearson_r


def test_correlation_missing_value_handling():
    """Verify pairwise dropna behavior when computing correlations with NaNs."""
    df_nan = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, np.nan, 10.0],
    })
    # Use min_periods=2 for this small 5-row test
    res = compute_correlation_matrix(df_nan, ["a", "b"], method="pearson", min_periods=2)
    # Rows 0, 1, 4 have pairwise non-nulls: ([1, 2, 5], [2, 4, 10]) -> slope=2, r=1.0
    assert res.matrix.loc["a", "b"] == pytest.approx(1.0)


def test_empty_dataset_and_invalid_method():
    """Verify graceful handling for empty series and invalid correlation method."""
    empty_s = pd.Series([], dtype=float, name="empty_col")
    stats = compute_numeric_distribution(empty_s)
    assert stats.count == 0
    assert stats.is_constant is True

    cat_stats = compute_categorical_distribution(pd.Series([], dtype=object, name="empty_cat"))
    assert cat_stats.count == 0
    assert cat_stats.cardinality == 0

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Unsupported correlation method"):
        compute_correlation_matrix(df, ["a", "b"], method="kendall")


def test_outlier_flagging_logic():
    """Verify IQR-based outlier threshold math."""
    data = [10, 10, 11, 12, 10, 11, 10, 12, 100, -50]
    s = pd.Series(data, name="outlier_test")
    stats = compute_numeric_distribution(s)
    assert stats.outlier_count_iqr == 2
    assert stats.q75 + 1.5 * stats.iqr < 100
    assert stats.q25 - 1.5 * stats.iqr > -50


def test_numeric_bivariate_relationship(synthetic_eda_df):
    """Verify numeric vs target bivariate analysis (Pearson r, Spearman rho, OLS slope)."""
    rel = compute_numeric_bivariate_stats(
        synthetic_eda_df,
        "num_feat",
        "target"
    )
    assert isinstance(rel, NumericBivariateStats)
    assert rel.feature_x == "num_feat"
    assert rel.feature_y == "target"
    assert rel.ols_slope == pytest.approx(2.5, abs=0.5)
    assert rel.pearson_r > 0.8


def test_categorical_target_summary(synthetic_eda_df):
    """Verify categorical vs target grouped statistics."""
    summary = compute_categorical_target_summary(
        synthetic_eda_df,
        "cat_feat",
        "target"
    )
    assert isinstance(summary, pd.DataFrame)
    assert "Count" in summary.columns
    assert "Mean" in summary.columns
    assert "Median" in summary.columns
    assert "Std" in summary.columns
    assert "TypeA" in summary["cat_feat"].values


def test_plotly_figures_creation(synthetic_eda_df):
    """Verify that all EDA Plotly visualization functions return valid go.Figure objects."""
    fig_num = plot_numeric_distribution(synthetic_eda_df["num_feat"], "num_feat")
    assert isinstance(fig_num, go.Figure)

    fig_cat = plot_categorical_distribution(synthetic_eda_df["cat_feat"], "cat_feat")
    assert isinstance(fig_cat, go.Figure)

    corr_res = compute_correlation_matrix(synthetic_eda_df, ["num_feat", "target"], method="pearson")
    fig_corr = plot_correlation_heatmap(corr_res.matrix)
    assert isinstance(fig_corr, go.Figure)

    fig_scatter, n_pts, sampled = plot_bivariate_scatter(synthetic_eda_df, "num_feat", "target")
    assert isinstance(fig_scatter, go.Figure)
    assert n_pts == len(synthetic_eda_df)
    assert sampled is False

    fig_box, n_pts_b, sampled_b = plot_categorical_target_boxplot(synthetic_eda_df, "cat_feat", "target")
    assert isinstance(fig_box, go.Figure)
