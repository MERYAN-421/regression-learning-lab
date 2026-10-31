"""
Group B: Independence and Autocorrelation Diagnostics Tests.
Verifies:
- Durbin-Watson numerical match against statsmodels
- Ljung-Box Q-statistic and p-value match against statsmodels acorr_ljungbox
- Detection of positive serial correlation in simulated AR(1) errors
- Non-rejection in i.i.d. white noise errors
- Dataset-specific applicability guards:
  - Appliances Energy: Applicable (temporal split)
  - Life Expectancy: Not Applicable (country panel)
  - KC House & Student Performance: Not Applicable (cross-sectional)
- ACF confidence bands calculation
"""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.data.models import SplitResult
from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model
from src.diagnostics.independence import (
    run_durbin_watson_test,
    run_ljung_box_test,
    compute_residual_acf
)
from src.diagnostics.result_models import DiagnosticDecision, ApplicabilityStatus
from src.diagnostics.plots import plot_residuals_vs_time, plot_residual_acf


@pytest.fixture
def independent_ols_fixture():
    """Synthetic time-series OLS model with independent (i.i.d.) Gaussian errors."""
    np.random.seed(42)
    n = 150
    x = np.random.normal(0, 1, n)
    eps = np.random.normal(0, 0.5, n)
    y = 5.0 + 2.5 * x + eps

    df_x = pd.DataFrame({"x": x})
    y_ser = pd.Series(y, name="y")

    design_res, builder = build_ols_design_matrices(df_x, selected_features=["x"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser, target_name="y")
    model.dataset_id = "appliances_energy"  # temporal dataset ID
    return model, df_x, y_ser


@pytest.fixture
def ar1_autocorrelated_ols_fixture():
    """Synthetic time-series OLS model with strong AR(1) serial correlation (phi = 0.85)."""
    np.random.seed(42)
    n = 150
    x = np.random.normal(0, 1, n)

    # Generate AR(1) error terms: e_t = 0.85 * e_{t-1} + u_t
    eps = np.zeros(n)
    u = np.random.normal(0, 0.3, n)
    for t in range(1, n):
        eps[t] = 0.85 * eps[t - 1] + u[t]

    y = 5.0 + 2.5 * x + eps

    df_x = pd.DataFrame({"x": x})
    y_ser = pd.Series(y, name="y")

    design_res, builder = build_ols_design_matrices(df_x, selected_features=["x"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser, target_name="y")
    model.dataset_id = "appliances_energy"
    return model, df_x, y_ser


def test_durbin_watson_numerical_accuracy(independent_ols_fixture):
    """Verify DW statistic matches direct statsmodels durbin_watson computation."""
    model, _, y_ser = independent_ols_fixture
    diag_res = run_durbin_watson_test(model)

    ref_dw = float(durbin_watson(model.fitted_model.resid))
    assert abs(diag_res.statistic_value - ref_dw) < 1e-6
    assert diag_res.method_id == "durbin_watson"
    assert diag_res.decision == DiagnosticDecision.DESCRIPTIVE_ONLY.value
    # For i.i.d. errors, DW should be close to 2.0
    assert 1.6 < diag_res.statistic_value < 2.4


def test_durbin_watson_detects_positive_autocorrelation(ar1_autocorrelated_ols_fixture):
    """Verify DW statistic drops substantially below 1.5 on positive AR(1) correlation."""
    model, _, _ = ar1_autocorrelated_ols_fixture
    diag_res = run_durbin_watson_test(model)

    assert diag_res.statistic_value < 1.0, f"Expected DW << 1.5 on strong AR(1), got {diag_res.statistic_value}"
    assert "positive first-order serial correlation" in diag_res.interpretation


def test_ljung_box_numerical_accuracy(independent_ols_fixture):
    """Verify Ljung-Box Q-statistic and p-value match statsmodels acorr_ljungbox."""
    model, _, _ = independent_ols_fixture
    diag_res = run_ljung_box_test(model, max_lag=10, alpha=0.05)

    ref_lb = acorr_ljungbox(model.fitted_model.resid, lags=[10], return_df=True)
    ref_q = float(ref_lb["lb_stat"].iloc[-1])
    ref_pval = float(ref_lb["lb_pvalue"].iloc[-1])

    assert abs(diag_res.statistic_value - ref_q) < 1e-6
    assert abs(diag_res.p_value - ref_pval) < 1e-6
    assert diag_res.decision == DiagnosticDecision.FAIL_TO_REJECT_NULL.value


def test_ljung_box_detects_autocorrelation_rejection(ar1_autocorrelated_ols_fixture):
    """Verify Ljung-Box rejects H0 on AR(1) autocorrelated errors."""
    model, _, _ = ar1_autocorrelated_ols_fixture
    diag_res = run_ljung_box_test(model, max_lag=10, alpha=0.05)

    assert diag_res.decision == DiagnosticDecision.REJECT_NULL.value
    assert diag_res.p_value < 0.001
    assert "rejects H0" in diag_res.interpretation


def test_independence_applicability_guards():
    """Verify dataset-specific applicability enforcement: Life Expectancy and KC House."""
    np.random.seed(42)
    n = 60
    df_x = pd.DataFrame({"x": np.random.normal(0, 1, n)})
    y_ser = pd.Series(np.random.normal(0, 1, n), name="y")
    design_res, _ = build_ols_design_matrices(df_x, selected_features=["x"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser)

    # 1. Life Expectancy (Country Panel)
    model.dataset_id = "life_expectancy"
    dw_life = run_durbin_watson_test(model)
    lb_life = run_ljung_box_test(model)
    assert dw_life.applicability_status == ApplicabilityStatus.NOT_APPLICABLE.value
    assert lb_life.applicability_status == ApplicabilityStatus.NOT_APPLICABLE.value
    assert "panel" in dw_life.applicability_reason.lower()

    # 2. King County (Cross-Sectional)
    model.dataset_id = "kc_house_sales"
    dw_kc = run_durbin_watson_test(model)
    assert dw_kc.applicability_status == ApplicabilityStatus.NOT_APPLICABLE.value
    assert "cross-sectional" in dw_kc.applicability_reason.lower()


def test_residual_acf_computation():
    """Verify Autocorrelation Function (ACF) computation and Bartlett confidence band."""
    residuals = np.random.normal(0, 1, 100)
    acf_data = compute_residual_acf(residuals, nlags=15)

    assert len(acf_data["acf_values"]) == 16  # lag 0 to 15
    assert abs(acf_data["acf_values"][0] - 1.0) < 1e-6  # lag 0 is 1.0
    assert abs(acf_data["bartlett_threshold"] - 1.96 / 10.0) < 1e-4

    # Test ACF Plot rendering
    fig = plot_residual_acf(acf_data, max_lags=15)
    assert fig is not None
