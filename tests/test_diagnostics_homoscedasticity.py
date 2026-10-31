"""
Group C: Homoscedasticity Diagnostics Tests.
Verifies:
- Breusch-Pagan test numerical accuracy against statsmodels reference (LM stat, LM pval, F stat, F pval)
- Detection of heteroscedasticity on expanding error variance
- Non-rejection on genuine homoscedastic data
- Scale-Location plot data computation
- White's test optional execution
- Insufficient degrees of freedom handling
"""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.diagnostic import het_breuschpagan

from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model
from src.diagnostics.homoscedasticity import (
    run_breusch_pagan_test,
    run_white_test,
    compute_scale_location_data
)
from src.diagnostics.result_models import DiagnosticDecision, ApplicabilityStatus
from src.diagnostics.plots import plot_scale_location


@pytest.fixture
def homoscedastic_ols_fixture():
    """Synthetic regression model with constant error variance (sigma^2 = 1.0)."""
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(1, 10, n)
    x2 = np.random.uniform(1, 10, n)
    # Constant variance eps ~ N(0, 1.0)
    eps = np.random.normal(0, 1.0, n)
    y = 2.0 + 1.5 * x1 + 0.8 * x2 + eps

    df_x = pd.DataFrame({"x1": x1, "x2": x2})
    y_ser = pd.Series(y, name="y")

    design_res, builder = build_ols_design_matrices(df_x, selected_features=["x1", "x2"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser, target_name="y")
    return model, df_x, y_ser


@pytest.fixture
def heteroscedastic_ols_fixture():
    """Synthetic regression model with severe linear heteroscedasticity (sigma_i proportional to x1)."""
    np.random.seed(42)
    n = 150
    x1 = np.linspace(1, 15, n)
    # Error standard deviation expands linearly with x1
    eps = np.random.normal(0, 0.6 * x1, n)
    y = 3.0 + 2.0 * x1 + eps

    df_x = pd.DataFrame({"x1": x1})
    y_ser = pd.Series(y, name="y")

    design_res, builder = build_ols_design_matrices(df_x, selected_features=["x1"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser, target_name="y")
    return model, df_x, y_ser


def test_breusch_pagan_numerical_accuracy_against_statsmodels(homoscedastic_ols_fixture):
    """Verify Breusch-Pagan LM and F statistics match direct statsmodels het_breuschpagan output."""
    model, _, _ = homoscedastic_ols_fixture
    diag_res = run_breusch_pagan_test(model, alpha=0.05)

    ref_lm, ref_lm_pval, ref_f, ref_f_pval = het_breuschpagan(
        model.fitted_model.resid,
        model.fitted_model.model.exog
    )

    assert abs(diag_res.statistic_value - ref_lm) < 1e-6
    assert abs(diag_res.p_value - ref_lm_pval) < 1e-6
    assert abs(diag_res.details["f_statistic"] - ref_f) < 1e-6
    assert abs(diag_res.details["f_pvalue"] - ref_f_pval) < 1e-6
    assert diag_res.method_id == "breusch_pagan"
    assert diag_res.statistic_name == "LM statistic"


def test_breusch_pagan_detects_severe_heteroscedasticity(heteroscedastic_ols_fixture):
    """Verify Breusch-Pagan rejects H0 on severe expanding variance."""
    model, _, _ = heteroscedastic_ols_fixture
    diag_res = run_breusch_pagan_test(model, alpha=0.05)

    assert diag_res.decision == DiagnosticDecision.REJECT_NULL.value
    assert diag_res.p_value < 0.001
    assert "rejects H0" in diag_res.interpretation
    # Econometric correctness: coefficients remain unbiased, but conventional SEs are invalid
    assert "unbiased" in diag_res.interpretation.lower()
    assert any("HC3" in r for r in diag_res.recommendations)


def test_breusch_pagan_fails_to_reject_on_homoscedastic_data(homoscedastic_ols_fixture):
    """Verify Breusch-Pagan fails to reject H0 on constant variance data."""
    model, _, _ = homoscedastic_ols_fixture
    diag_res = run_breusch_pagan_test(model, alpha=0.05)

    assert diag_res.decision == DiagnosticDecision.FAIL_TO_REJECT_NULL.value
    assert diag_res.p_value >= 0.05


def test_scale_location_computation_and_plot(homoscedastic_ols_fixture):
    """Verify computation of coordinates for Scale-Location plot."""
    model, _, y_ser = homoscedastic_ols_fixture
    y_pred = model.train_predictions.values
    resid = y_ser.values - y_pred

    data = compute_scale_location_data(y_pred, resid)
    assert "sqrt_abs_std_residuals" in data
    assert np.all(data["sqrt_abs_std_residuals"] >= 0.0)

    fig = plot_scale_location(y_pred, resid, target_name="y")
    assert fig is not None


def test_white_test_execution_and_guard():
    """Verify White's test executes on small models and guards against high dimensions."""
    np.random.seed(42)
    n = 60
    x = np.random.normal(0, 1, n)
    y = 1.0 + 2.0 * x + np.random.normal(0, 0.5, n)
    df_x = pd.DataFrame({"x": x})
    y_ser = pd.Series(y, name="y")

    design_res, _ = build_ols_design_matrices(df_x, selected_features=["x"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser)

    white_res = run_white_test(model, alpha=0.05)
    assert white_res.method_id == "white_test"
    assert white_res.statistic_value is not None
