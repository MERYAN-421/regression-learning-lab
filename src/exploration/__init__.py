"""
Exploratory Data Analysis (EDA) module for Regression Learning Lab.
Provides univariate distributions, correlation matrices, bivariate relationship metrics,
and interactive Plotly visualizations.
"""
from src.exploration.distributions import (
    NumericDistributionStats,
    CategoricalDistributionStats,
    compute_numeric_distribution,
    compute_categorical_distribution,
)
from src.exploration.correlations import (
    CorrelationPair,
    CorrelationMatrixResult,
    compute_correlation_matrix,
)
from src.exploration.relationships import (
    NumericBivariateStats,
    compute_numeric_bivariate_stats,
    compute_categorical_target_summary,
)
from src.exploration.plots import (
    plot_numeric_distribution,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_bivariate_scatter,
    plot_categorical_target_boxplot,
)

__all__ = [
    "NumericDistributionStats",
    "CategoricalDistributionStats",
    "compute_numeric_distribution",
    "compute_categorical_distribution",
    "CorrelationPair",
    "CorrelationMatrixResult",
    "compute_correlation_matrix",
    "NumericBivariateStats",
    "compute_numeric_bivariate_stats",
    "compute_categorical_target_summary",
    "plot_numeric_distribution",
    "plot_categorical_distribution",
    "plot_correlation_heatmap",
    "plot_bivariate_scatter",
    "plot_categorical_target_boxplot",
]
