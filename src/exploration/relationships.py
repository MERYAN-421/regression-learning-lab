"""
Bivariate feature relationships module for exploratory data analysis.
Computes relationship metrics for numeric-to-target and categorical-to-target analysis.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class NumericBivariateStats:
    feature_x: str
    feature_y: str
    sample_count: int
    pearson_r: float
    pearson_p_value: float
    spearman_rho: float
    spearman_p_value: float
    ols_slope: float
    ols_intercept: float


def compute_numeric_bivariate_stats(
    df: pd.DataFrame,
    x_col: str,
    y_col: str
) -> NumericBivariateStats:
    """Compute bivariate association metrics between two numeric variables."""
    valid_data = df[[x_col, y_col]].dropna()
    n = len(valid_data)
    if n < 3:
        return NumericBivariateStats(
            feature_x=x_col,
            feature_y=y_col,
            sample_count=n,
            pearson_r=0.0,
            pearson_p_value=1.0,
            spearman_rho=0.0,
            spearman_p_value=1.0,
            ols_slope=0.0,
            ols_intercept=0.0
        )

    x = valid_data[x_col].values
    y = valid_data[y_col].values

    # Check for zero variance
    if np.std(x) == 0 or np.std(y) == 0:
        return NumericBivariateStats(
            feature_x=x_col,
            feature_y=y_col,
            sample_count=n,
            pearson_r=0.0,
            pearson_p_value=1.0,
            spearman_rho=0.0,
            spearman_p_value=1.0,
            ols_slope=0.0,
            ols_intercept=float(np.mean(y))
        )

    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)

    return NumericBivariateStats(
        feature_x=x_col,
        feature_y=y_col,
        sample_count=n,
        pearson_r=float(r_p),
        pearson_p_value=float(p_p),
        spearman_rho=float(r_s),
        spearman_p_value=float(p_s),
        ols_slope=float(slope),
        ols_intercept=float(intercept)
    )


def compute_categorical_target_summary(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    top_n_categories: int = 20
) -> pd.DataFrame:
    """
    Compute grouped target statistics across categories of a categorical predictor.
    """
    valid_data = df[[cat_col, target_col]].dropna()
    if valid_data.empty:
        return pd.DataFrame()

    grouped = valid_data.groupby(cat_col)[target_col].agg(
        Count="count",
        Mean="mean",
        Std="std",
        Median="median",
        Min="min",
        Max="max"
    ).reset_index()

    # Sort descending by count, then take top N
    grouped = grouped.sort_values(by="Count", ascending=False).head(top_n_categories)
    grouped["Std"] = grouped["Std"].fillna(0.0)
    return grouped
