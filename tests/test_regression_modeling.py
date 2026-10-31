"""
Unit tests for OLS Regression Modeling & Econometric Accuracy (Group B).
Validates statsmodels numerical outputs against known ground-truth parameters:
- Coefficient point estimates and intercept
- Standard errors, t-statistics, p-values, 95% confidence intervals
- R², Adjusted R², F-statistic, degrees of freedom
- LaTeX equations formulation
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model, generate_regression_equation_latex


@pytest.fixture
def synthetic_linear_dataset():
    """
    Creates a controlled linear regression dataset with known ground-truth coefficients:
    y = 5.0 + 2.5 * x1 - 1.8 * x2 + epsilon
    where epsilon ~ N(0, 0.5^2)
    """
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    eps = np.random.normal(0, 0.5, n)
    y = 5.0 + 2.5 * x1 - 1.8 * x2 + eps

    df = pd.DataFrame({"x1": x1, "x2": x2})
    target = pd.Series(y, name="response")
    return df, target


def test_ols_parameter_estimation_accuracy(synthetic_linear_dataset):
    """Verify estimated coefficients closely match true population values and statsmodels."""
    X, y = synthetic_linear_dataset
    design_res, builder = build_ols_design_matrices(X, selected_features=["x1", "x2"], fit_intercept=True)
    model_res = fit_ols_model(design_res, y_train=y, target_name="response")

    # True values: beta_0 = 5.0, beta_1 = 2.5, beta_2 = -1.8
    params = model_res.fitted_model.params
    assert abs(params["const"] - 5.0) < 0.15
    assert abs(params["x1"] - 2.5) < 0.15
    assert abs(params["x2"] - (-1.8)) < 0.15

    # Direct statsmodels comparison
    direct_sm = sm.OLS(y.values, sm.add_constant(X.values)).fit()
    np.testing.assert_allclose(params.values, direct_sm.params, rtol=1e-5)
    np.testing.assert_allclose(model_res.fitted_model.bse.values, direct_sm.bse, rtol=1e-5)
    np.testing.assert_allclose(model_res.fitted_model.tvalues.values, direct_sm.tvalues, rtol=1e-5)
    np.testing.assert_allclose(model_res.fitted_model.pvalues.values, direct_sm.pvalues, rtol=1e-5)


def test_ols_goodness_of_fit_and_degrees_of_freedom(synthetic_linear_dataset):
    """Verify R², Adjusted R², F-statistic, and degrees of freedom."""
    X, y = synthetic_linear_dataset
    design_res, builder = build_ols_design_matrices(X, selected_features=["x1", "x2"], fit_intercept=True)
    model_res = fit_ols_model(design_res, y_train=y, target_name="response")

    # Strong linear signal -> R² should be high (> 0.90)
    assert model_res.r2 > 0.90
    assert model_res.adj_r2 is not None
    assert model_res.adj_r2 <= model_res.r2  # Adjusted R² <= R²
    assert model_res.f_statistic > 100.0
    assert model_res.f_pvalue < 1e-10

    # df_resid should be n - p = 200 - 3 = 197
    assert model_res.df_resid == 197
    assert model_res.df_model == 2


def test_ols_latex_equation_formulation(synthetic_linear_dataset):
    """Verify generation of general and fitted LaTeX equations."""
    X, y = synthetic_linear_dataset
    design_res, builder = build_ols_design_matrices(X, selected_features=["x1", "x2"], fit_intercept=True)
    model_res = fit_ols_model(design_res, y_train=y, target_name="response")

    gen_eq = model_res.equation_general
    fitted_eq = model_res.equation_fitted

    assert r"\hat{response}" in gen_eq
    assert r"\hat{\beta}_0" in gen_eq
    assert r"\hat{response}" in fitted_eq
    assert "x1" in fitted_eq
    assert "x2" in fitted_eq
    assert model_res.equation_text.startswith("Estimated response =")


def test_ols_coefficient_table_structure(synthetic_linear_dataset):
    """Verify coefficient table contains all mandatory columns and valid confidence intervals."""
    X, y = synthetic_linear_dataset
    design_res, builder = build_ols_design_matrices(X, selected_features=["x1", "x2"], fit_intercept=True)
    model_res = fit_ols_model(design_res, y_train=y, target_name="response")

    df_coef = model_res.coef_table
    expected_cols = [
        "Feature", "Coefficient", "Std. Error", "t-statistic",
        "p-value", "CI Lower (95%)", "CI Upper (95%)", "Interpretation"
    ]
    assert list(df_coef.columns) == expected_cols
    assert len(df_coef) == 3

    # For both x1 and x2, CI Lower < Coef < CI Upper
    for _, row in df_coef.iterrows():
        if row["CI Lower (95%)"] != "N/A":
            assert float(row["CI Lower (95%)"]) <= float(row["Coefficient"]) <= float(row["CI Upper (95%)"])
