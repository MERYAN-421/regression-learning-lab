"""
Model comparison and leaderboard generation for Regression Learning Lab.
"""
from typing import List, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from src.models.model_result import OLSModelResult


def build_model_comparison_table(
    models: List["OLSModelResult"],
    target_split_fingerprint: Optional[str] = None
) -> pd.DataFrame:
    """
    Constructs a comparative leaderboard across multiple fitted OLS models.
    Filters by split_fingerprint if provided to prevent comparing models from different data partitions.
    """
    if not models:
        return pd.DataFrame()

    filtered_models = [
        m for m in models
        if target_split_fingerprint is None or m.split_fingerprint == target_split_fingerprint
    ]

    if not filtered_models:
        return pd.DataFrame()

    rows = []
    for m in filtered_models:
        rows.append(m.to_summary_dict())

    df = pd.DataFrame(rows)
    # Sort by Validation RMSE ascending
    if "Val RMSE" in df.columns:
        df = df.sort_values(by="Val RMSE", ascending=True).reset_index(drop=True)
    return df
