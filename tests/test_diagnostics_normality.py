"""
Group D: Normality of Residuals Diagnostics Tests.
Verifies:
- Jarque-Bera test numerical accuracy against statsmodels jarque_bera
- Detection of non-normality on skewed and heavy-tailed error distributions
- Non-rejection on Gaussian i.i.d. errors
- Critical econometric explanation: Rejecting normality does NOT invalidate BLUE
- Normal Q-Q plot and histogram computation and rendering
- Insufficient data handling (n < 8)
- Optional Shapiro-Wilk test execution
"""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.stattools import jarque_bera

from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model
from src.diagnostics.normality import (
    run_jarque_bera_test,
    run_shapiro_wilk_test,
    compute_qq_plot_data
)
from src.diagnostics.result_models import DiagnosticDecision, ApplicabilityStatus
from src.diagnostics.plots import plot_normal_qq, plot_residuals_histogram_kde


@pytest.fixture
def normal_ols_fixture():
    """Synthetic OLS model with Gaussian errors eps ~ N(0, 1.0)."""
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, n)
    eps = np.random.normal(0, 1.0, n)
    y = 4.0 + 2.0 * x + eps

    df_x = pd.DataFrame({"x": x})
    y_ser = pd.Series(y, name="y")

    design_res, builder = build_ols_design_matrices(df_x, selected_features=["x"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser, target_name="y")
    return model, df_x, y_ser


@pytest.fixture
def skewed_nonnormal_ols_fixture():
    """Synthetic OLS model with heavily right-skewed exponential errors."""
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, n)
    # Exponential distribution produces heavy positive skewness
    eps = np.random.exponential(scale=2.0, size=n) - 2.0
    y = 1.0 + 3.0 * x + eps

    df_x = pd.DataFrame({"x": x})
    y_ser = pd.Series(y, name="y")

    design_res, builder = build_ols_design_matrices(df_x, selected_features=["x"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser, target_name="y")
    return model, df_x, y_ser


def test_jarque_bera_numerical_accuracy(normal_ols_fixture):
    """Verify JB statistic, p-value, skewness, and kurtosis match statsmodels jarque_bera."""
    model, _, _ = normal_ols_fixture
    diag_res = run_jarque_bera_test(model, alpha=0.05)

    ref_jb = jarque_bera(model.fitted_model.resid)
    assert abs(diag_res.statistic_value - float(ref_jb[0])) < 1e-6
    assert abs(diag_res.p_value - float(ref_jb[1])) < 1e-6
    assert abs(diag_res.details["sample_skewness"] - float(ref_jb[2])) < 1e-6
    assert abs(diag_res.details["sample_kurtosis"] - float(ref_jb[3])) < 1e-6
    assert diag_res.method_id == "jarque_bera"
    assert diag_res.decision == DiagnosticDecision.FAIL_TO_REJECT_NULL.value


def test_jarque_bera_detects_severe_skewness(skewed_nonnormal_ols_fixture):
    """Verify Jarque-Bera rejects H0 on exponential errors with positive skewness."""
    model, _, _ = skewed_nonnormal_ols_fixture
    diag_res = run_jarque_bera_test(model, alpha=0.05)

    assert diag_res.decision == DiagnosticDecision.REJECT_NULL.value
    assert diag_res.p_value < 0.001
    assert diag_res.details["sample_skewness"] > 0.8
    # Econometric correctness: BLUE is not invalidated
    assert "BLUE" in diag_res.interpretation
    assert "Central Limit Theorem" in diag_res.interpretation


def test_jarque_bera_insufficient_sample():
    """Verify Jarque-Bera returns INSUFFICIENT_DATA when n < 8."""
    df_x = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})  # n = 5
    y_ser = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0], name="y")
    design_res, _ = build_ols_design_matrices(df_x, selected_features=["x"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser)

    diag_res = run_jarque_bera_test(model)
    assert diag_res.decision == DiagnosticDecision.INSUFFICIENT_DATA.value
    assert diag_res.applicability_status == ApplicabilityStatus.NOT_APPLICABLE.value


def test_qq_plot_data_and_figure(normal_ols_fixture):
    """Verify Normal Q-Q plot data computation and figure rendering."""
    model, _, y_ser = normal_ols_fixture
    resid = y_ser.values - model.train_predictions.values

    qq_data = compute_qq_plot_data(resid)
    assert "theoretical_quantiles" in qq_data
    assert "sample_quantiles" in qq_data
    assert qq_data["correlation_r"] > 0.95

    fig_qq = plot_normal_qq(resid)
    assert fig_qq is not None

    fig_hist = plot_residuals_histogram_kde(resid, target_unit="units")
    assert fig_hist is not None


def test_shapiro_wilk_execution(normal_ols_fixture):
    """Verify optional Shapiro-Wilk test runs on moderate samples."""
    model, _, _ = normal_ols_fixture
    sw_res = run_shapiro_wilk_test(model, alpha=0.05)
    assert sw_res.method_id == "shapiro_wilk"
    assert sw_res.statistic_value > 0.90
