"""
Group A: Linearity and Functional-Form Diagnostics Tests.
Verifies:
- Ramsey RESET test numerical accuracy against statsmodels reference
- Configurable powers [2, 3]
- Non-linear data misspecification detection
- Linear data non-rejection
- Invalid configurations (insufficient sample, rank deficient)
- LOWESS figure generation
"""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.diagnostic import linear_reset
import statsmodels.api as sm

from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model
from src.diagnostics.linearity import run_ramsey_reset_test
from src.diagnostics.result_models import DiagnosticDecision, ApplicabilityStatus
from src.diagnostics.plots import plot_residuals_vs_fitted_lowess


@pytest.fixture
def linear_ols_fixture():
    """Synthetic linear dataset with genuine linear data-generating process."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = 3.0 + 2.0 * x1 - 1.5 * x2 + np.random.normal(0, 0.5, n)

    df_x = pd.DataFrame({"x1": x1, "x2": x2})
    y_ser = pd.Series(y, name="y")

    design_res, builder = build_ols_design_matrices(df_x, selected_features=["x1", "x2"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser, target_name="y")
    return model, df_x, y_ser


@pytest.fixture
def nonlinear_ols_fixture():
    """Synthetic non-linear dataset with severe quadratic misspecification."""
    np.random.seed(42)
    n = 120
    x = np.linspace(-3, 3, n)
    # True relationship is highly non-linear (quadratic + cubic)
    y = 2.0 + 1.5 * x + 3.0 * (x ** 2) - 0.5 * (x ** 3) + np.random.normal(0, 0.5, n)

    df_x = pd.DataFrame({"x": x})
    y_ser = pd.Series(y, name="y")

    # Fit linear-only model
    design_res, builder = build_ols_design_matrices(df_x, selected_features=["x"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser, target_name="y")
    return model, df_x, y_ser


def test_ramsey_reset_numerical_accuracy_against_statsmodels(linear_ols_fixture):
    """Verify Ramsey RESET statistic and p-value match direct statsmodels linear_reset output."""
    model, _, _ = linear_ols_fixture

    # Diagnostic module execution
    diag_res = run_ramsey_reset_test(model, powers=[2, 3], alpha=0.05)

    # Reference calculation using statsmodels
    ref_reset = linear_reset(model.fitted_model, power=3, test_type="fitted", use_f=True)

    ref_f = float(ref_reset.fvalue if hasattr(ref_reset, "fvalue") else ref_reset.statistic)
    ref_pval = float(ref_reset.pvalue)

    assert abs(diag_res.statistic_value - ref_f) < 1e-6
    assert abs(diag_res.p_value - ref_pval) < 1e-6
    assert diag_res.method_id == "ramsey_reset"
    assert diag_res.statistic_name == "F-statistic"
    assert diag_res.applicability_status == ApplicabilityStatus.APPLICABLE.value


def test_ramsey_reset_detects_severe_nonlinearity(nonlinear_ols_fixture):
    """Verify Ramsey RESET rejects H0 on severe polynomial misspecification."""
    model, _, _ = nonlinear_ols_fixture

    diag_res = run_ramsey_reset_test(model, powers=[2, 3], alpha=0.05)

    assert diag_res.decision == DiagnosticDecision.REJECT_NULL.value
    assert diag_res.p_value < 0.05
    assert "functional-form misspecification" in diag_res.interpretation
    assert len(diag_res.recommendations) > 0


def test_ramsey_reset_fails_to_reject_on_correct_linear_model(linear_ols_fixture):
    """Verify Ramsey RESET fails to reject H0 on well-specified linear data."""
    model, _, _ = linear_ols_fixture

    diag_res = run_ramsey_reset_test(model, powers=[2, 3], alpha=0.05)

    assert diag_res.decision == DiagnosticDecision.FAIL_TO_REJECT_NULL.value
    assert diag_res.p_value >= 0.05
    # Strict semantic check: Must not state that model is proven correct
    assert "not prove" in diag_res.interpretation.lower()


def test_ramsey_reset_rank_deficient_handling():
    """Verify that Ramsey RESET returns NOT_APPLICABLE on rank-deficient prediction-only models."""
    np.random.seed(42)
    n = 50
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = x1 + x2  # Exact collinearity
    df_x = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    y_ser = pd.Series(1.0 + x1 + x2, name="y")

    design_res, _ = build_ols_design_matrices(df_x, selected_features=["x1", "x2", "x3"], fit_intercept=True)
    # Fit with allow_rank_deficient=True
    model = fit_ols_model(design_res, y_train=y_ser, allow_rank_deficient=True)

    diag_res = run_ramsey_reset_test(model, powers=[2, 3], alpha=0.05)

    assert diag_res.decision == DiagnosticDecision.NOT_APPLICABLE.value
    assert diag_res.applicability_status == ApplicabilityStatus.NOT_APPLICABLE.value
    assert not diag_res.is_valid_for_inference
    assert "rank-deficient" in diag_res.applicability_reason.lower()


def test_residuals_vs_fitted_lowess_plotting(linear_ols_fixture):
    """Verify Residuals vs. Fitted plot generation with LOWESS curve."""
    model, _, y_ser = linear_ols_fixture
    y_pred = model.train_predictions.values
    resid = y_ser.values - y_pred

    fig = plot_residuals_vs_fitted_lowess(
        fitted_values=y_pred,
        residuals=resid,
        target_name="y"
    )

    assert fig is not None
    # Check traces: Scatter of residuals + LOWESS smooth line
    trace_names = [t.name for t in fig.data if t.name]
    assert "Residuals" in trace_names
    assert "LOWESS Smooth" in trace_names
