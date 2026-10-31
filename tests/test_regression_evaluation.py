"""
Unit tests for Regression Evaluation Metrics (Group C).
Tests:
- MSE, RMSE, MAE mathematical accuracy
- R² and negative R² detection
- Perfect prediction limits
- Zero variance target edge case
- In-sample vs. out-of-sample metric separation
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.regression_metrics import compute_regression_metrics


def test_metrics_perfect_prediction():
    """Verify metrics when predictions perfectly equal actual values."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_regression_metrics(y_true=y, y_pred=y, num_non_intercept_parameters=1, is_in_sample=True)

    assert metrics.mse == 0.0
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.r2 == 1.0
    assert metrics.adj_r2 == 1.0


def test_metrics_standard_linear_errors():
    """Verify standard arithmetic error metrics with known offsets."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    # Constant error of +2.0
    y_pred = np.array([12.0, 22.0, 32.0, 42.0])

    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred, num_non_intercept_parameters=1)

    assert metrics.mae == 2.0
    assert metrics.mse == 4.0
    assert metrics.rmse == 2.0
    # TSS = 500, RSS = 16 -> R² = 1 - 16/500 = 0.968
    assert abs(metrics.r2 - 0.968) < 1e-4


def test_metrics_negative_r2_handling():
    """Verify negative R² when model performs worse than simple mean baseline."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # mean = 3.0, TSS = 10.0
    # Terrible predictions
    y_pred = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred, num_non_intercept_parameters=1)
    assert metrics.r2 < 0.0, "R² must be negative when RSS > TSS"


def test_metrics_in_sample_vs_out_of_sample_adjusted_r2():
    """Verify that Adjusted R² is calculated in-sample, but omitted/annotated out-of-sample."""
    y_true = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
    y_pred = y_true + np.random.normal(0, 1, len(y_true))

    in_sample = compute_regression_metrics(y_true, y_pred, num_non_intercept_parameters=2, is_in_sample=True)
    out_of_sample = compute_regression_metrics(y_true, y_pred, num_non_intercept_parameters=2, is_in_sample=False)

    assert in_sample.adj_r2 is not None
    assert out_of_sample.adj_r2 is None, "Out-of-sample Adjusted R² should not be reported"
    assert "in-sample" in out_of_sample.notes


def test_metrics_zero_variance_target():
    """Verify edge case where target variable has zero variance."""
    y_const = np.array([5.0, 5.0, 5.0, 5.0])
    metrics = compute_regression_metrics(y_true=y_const, y_pred=y_const, num_non_intercept_parameters=1)
    assert metrics.r2 == 1.0
    assert metrics.rmse == 0.0


def test_metrics_intercept_only_adjusted_r2():
    """
    Verify Task 4 edge case:
    Intercept-only model (num_non_intercept_parameters = 0).
    Adjusted R² should be 0.0 by mathematical definition (adding no predictors beyond mean).
    """
    y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y_pred = np.full_like(y_true, fill_value=np.mean(y_true))  # Mean predictions

    metrics = compute_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_non_intercept_parameters=0,
        is_in_sample=True,
        fit_intercept=True
    )
    assert metrics.r2 == 0.0
    assert metrics.adj_r2 == 0.0


def test_metrics_no_intercept_adjusted_r2_against_statsmodels():
    """
    Verify Task 4 edge case:
    No-intercept model (fit_intercept = False).
    Uses uncentered degrees of freedom: df_total = n, df_resid = n - p.
    Verify exact match with statsmodels uncentered rsquared_adj.
    """
    import statsmodels.api as sm

    np.random.seed(42)
    n = 50
    X = np.random.normal(5, 2, (n, 2))
    # True relationship with zero intercept
    y = 2.5 * X[:, 0] - 1.2 * X[:, 1] + np.random.normal(0, 0.5, n)

    # Fit statsmodels without constant
    res = sm.OLS(y, X).fit()
    y_pred = res.predict(X)

    metrics = compute_regression_metrics(
        y_true=y,
        y_pred=y_pred,
        num_non_intercept_parameters=2,
        is_in_sample=True,
        fit_intercept=False
    )

    # Check match within numerical tolerance
    assert abs(metrics.r2 - res.rsquared) < 1e-6
    assert abs(metrics.adj_r2 - res.rsquared_adj) < 1e-6

