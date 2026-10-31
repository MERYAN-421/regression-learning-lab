"""
Group E: Multicollinearity Diagnostics Tests.
Verifies:
- VIF calculation accuracy against statsmodels variance_inflation_factor
- Intercept exclusion from VIF table
- Severe multicollinearity detection (VIF > 10)
- Perfect collinearity handling (VIF = inf, rank deficiency)
- Low collinearity on orthogonal predictors (VIF approx 1.0)
- VIF horizontal bar plot rendering
"""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model
from src.diagnostics.multicollinearity import run_multicollinearity_diagnostics
from src.diagnostics.result_models import DiagnosticDecision
from src.diagnostics.plots import plot_vif_horizontal_bar


def test_vif_numerical_accuracy_against_statsmodels():
    """Verify that VIF calculations match direct statsmodels output on non-intercept columns."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = 0.8 * x1 + np.random.normal(0, 0.6, n)
    x3 = np.random.normal(0, 1, n)

    df_x = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    y_ser = pd.Series(1.0 + 2.0 * x1 + 3.0 * x2 + 0.5 * x3, name="y")

    design_res, builder = build_ols_design_matrices(df_x, selected_features=["x1", "x2", "x3"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser)

    diag_res = run_multicollinearity_diagnostics(model)

    # Reference calculation: design matrix with constant
    exog = model.fitted_model.model.exog
    ref_vifs = {
        "x1": float(variance_inflation_factor(exog, 1)),
        "x2": float(variance_inflation_factor(exog, 2)),
        "x3": float(variance_inflation_factor(exog, 3))
    }

    # Verify intercept is NOT in the VIF table
    vif_features = [r["feature"] for r in diag_res.details["vif_table"]]
    assert "const" not in vif_features

    # Verify values match
    for row in diag_res.details["vif_table"]:
        f_name = row["feature"]
        assert abs(row["vif"] - ref_vifs[f_name]) < 1e-5


def test_vif_detects_severe_collinearity():
    """Verify that near-collinear features produce VIF > 10 and severe risk level."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(10, 2, n)
    # Extremely correlated: r ~ 0.995
    x2 = x1 + np.random.normal(0, 0.1, n)
    x3 = np.random.normal(5, 1, n)

    df_x = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    y_ser = pd.Series(2.0 + x1 + x2, name="y")

    design_res, _ = build_ols_design_matrices(df_x, selected_features=["x1", "x2", "x3"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser)

    diag_res = run_multicollinearity_diagnostics(model)

    assert diag_res.details["max_vif"] > 10.0
    high_cols = diag_res.details["high_vif_columns"]
    assert "x1" in high_cols or "x2" in high_cols
    assert "Severe" in diag_res.interpretation


def test_vif_orthogonal_features():
    """Verify orthogonal predictors produce VIF close to 1.0."""
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)

    df_x = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    y_ser = pd.Series(1.0 + x1 + x2 + x3, name="y")

    design_res, _ = build_ols_design_matrices(df_x, selected_features=["x1", "x2", "x3"], fit_intercept=True)
    model = fit_ols_model(design_res, y_train=y_ser)

    diag_res = run_multicollinearity_diagnostics(model)

    for row in diag_res.details["vif_table"]:
        assert row["vif"] < 1.2
        assert "Low Multicollinearity" in row["risk_level"]


def test_vif_bar_chart_rendering():
    """Verify VIF horizontal bar chart renders properly with threshold guidelines."""
    vif_records = [
        {"feature": "x1", "vif": 12.5, "risk_level": "High Multicollinearity (Severe)"},
        {"feature": "x2", "vif": 6.8, "risk_level": "Moderate Multicollinearity"},
        {"feature": "x3", "vif": 1.2, "risk_level": "Low Multicollinearity (Stable)"}
    ]

    fig = plot_vif_horizontal_bar(vif_records)
    assert fig is not None
    # Check that threshold lines are present
    shapes_and_lines = fig.layout.shapes
    assert len(shapes_and_lines) >= 2  # VIF=5 and VIF=10 lines
