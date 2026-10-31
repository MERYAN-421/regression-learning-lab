"""
Model evaluation package for Regression Learning Lab.
"""
from src.evaluation.regression_metrics import (
    RegressionMetrics,
    compute_regression_metrics,
)
from src.evaluation.predictions import predict_ols
from src.evaluation.plots import (
    plot_actual_vs_predicted,
    plot_residuals_vs_fitted,
    plot_residuals_histogram,
    plot_model_comparison_bar,
)
from src.evaluation.comparison import build_model_comparison_table

__all__ = [
    "RegressionMetrics",
    "compute_regression_metrics",
    "predict_ols",
    "plot_actual_vs_predicted",
    "plot_residuals_vs_fitted",
    "plot_residuals_histogram",
    "plot_model_comparison_bar",
    "build_model_comparison_table",
]
