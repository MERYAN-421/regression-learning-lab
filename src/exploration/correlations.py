"""
Correlation analysis module for exploratory data analysis.
Computes Pearson and Spearman correlation matrices with robust handling of missing
values and zero-variance (constant) features.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class CorrelationPair:
    feature_a: str
    feature_b: str
    correlation: float
    abs_correlation: float
    method: str


@dataclass
class CorrelationMatrixResult:
    method: str
    numeric_columns: List[str]
    matrix: pd.DataFrame
    top_correlated_pairs: List[CorrelationPair] = field(default_factory=list)
    constant_columns_excluded: List[str] = field(default_factory=list)


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    min_periods: int = 10,
    top_n_pairs: int = 10
) -> CorrelationMatrixResult:
    """
    Compute pairwise correlation matrix for numeric columns in DataFrame.
    
    Parameters:
        df: Input DataFrame.
        columns: Optional list of numeric columns to analyze.
        method: 'pearson' (linear) or 'spearman' (monotonic rank).
        min_periods: Minimum non-null overlapping observations required.
        top_n_pairs: Number of top pairwise correlated pairs to return.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unsupported correlation method: '{method}'. Choose 'pearson' or 'spearman'.")

    # Filter to numeric columns only
    if columns is not None:
        target_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        target_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not target_cols:
        return CorrelationMatrixResult(
            method=method,
            numeric_columns=[],
            matrix=pd.DataFrame(),
            top_correlated_pairs=[],
            constant_columns_excluded=[]
        )

    # Exclude constant features (zero variance causes division by zero in correlation)
    valid_cols = []
    constant_cols = []
    for c in target_cols:
        clean_s = df[c].dropna()
        if clean_s.nunique() <= 1:
            constant_cols.append(c)
        else:
            valid_cols.append(c)

    if not valid_cols:
        return CorrelationMatrixResult(
            method=method,
            numeric_columns=target_cols,
            matrix=pd.DataFrame(index=target_cols, columns=target_cols),
            top_correlated_pairs=[],
            constant_columns_excluded=constant_cols
        )

    sub_df = df[valid_cols]
    corr_matrix = sub_df.corr(method=method, min_periods=min_periods)

    # Fill diagonal with 1.0 where applicable
    for c in valid_cols:
        corr_matrix.loc[c, c] = 1.0

    # Extract top pairwise correlations (upper triangle excluding diagonal)
    pairs = []
    for i in range(len(valid_cols)):
        col_a = valid_cols[i]
        for j in range(i + 1, len(valid_cols)):
            col_b = valid_cols[j]
            val = corr_matrix.loc[col_a, col_b]
            if not np.isnan(val):
                pairs.append(CorrelationPair(
                    feature_a=col_a,
                    feature_b=col_b,
                    correlation=float(val),
                    abs_correlation=float(abs(val)),
                    method=method
                ))

    # Sort descending by absolute correlation
    pairs.sort(key=lambda p: p.abs_correlation, reverse=True)

    return CorrelationMatrixResult(
        method=method,
        numeric_columns=valid_cols,
        matrix=corr_matrix,
        top_correlated_pairs=pairs[:top_n_pairs],
        constant_columns_excluded=constant_cols
    )
