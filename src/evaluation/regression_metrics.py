"""
Regression Metrics Engine for Regression Learning Lab.

Calculates standard continuous error metrics (MSE, RMSE, MAE) and variance-explained
metrics (R², Adjusted R²) for regression evaluation.
Explicitly distinguishes in-sample training metrics from out-of-sample validation/test metrics.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class RegressionMetrics:
    """Encapsulates evaluated performance metrics for a specific dataset split."""
    split_name: str  # "Train", "Validation", "Test"
    sample_count: int
    num_parameters: int  # p non-intercept columns
    mse: float
    rmse: float
    mae: float
    r2: float
    adj_r2: Optional[float]  # Only meaningful in-sample
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "Split": self.split_name,
            "Samples": self.sample_count,
            "MSE": round(self.mse, 4),
            "RMSE": round(self.rmse, 4),
            "MAE": round(self.mae, 4),
            "R²": round(self.r2, 4),
        }
        if self.adj_r2 is not None:
            d["Adjusted R²"] = round(self.adj_r2, 4)
        return d


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    num_non_intercept_parameters: int = 1,
    split_name: str = "Train",
    is_in_sample: bool = True
) -> RegressionMetrics:
    """
    Compute MSE, RMSE, MAE, R², and in-sample Adjusted R².
    
    Parameters:
        y_true: Ground truth target observations.
        y_pred: Predicted target values from model.
        num_non_intercept_parameters: Number of slope parameters (p).
        split_name: Name of the partition ("Train", "Validation", or "Test").
        is_in_sample: True if computing on training set; False for out-of-sample validation/test.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true has {len(y_t)} items, y_pred has {len(y_p)} items.")

    n = len(y_t)
    if n == 0:
        raise ValueError("Cannot calculate regression metrics on an empty dataset.")

    # Check for NaN / Inf
    if not (np.all(np.isfinite(y_t)) and np.all(np.isfinite(y_p))):
        raise ValueError("Target or predictions contain non-finite (NaN or Inf) values.")

    # Residuals
    residuals = y_t - y_p

    # Error metrics
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    # Total Sum of Squares (TSS) around actual mean
    y_mean = np.mean(y_t)
    tss = float(np.sum((y_t - y_mean) ** 2))
    rss = float(np.sum(residuals ** 2))

    if tss < 1e-12:
        # Zero variance in target
        r2 = 1.0 if rss < 1e-12 else 0.0
    else:
        r2 = float(1.0 - (rss / tss))

    # Adjusted R² is an in-sample goodness-of-fit criterion
    adj_r2: Optional[float] = None
    notes = ""
    p = max(1, num_non_intercept_parameters)

    if is_in_sample:
        df_resid = n - p - 1
        if df_resid > 0:
            adj_r2 = float(1.0 - (1.0 - r2) * ((n - 1) / df_resid))
        else:
            notes = "Adjusted R² undefined: non-positive residual degrees of freedom (n <= p + 1)."
    else:
        notes = "Adjusted R² is an in-sample penalization formula; reporting standard out-of-sample R²."

    return RegressionMetrics(
        split_name=split_name,
        sample_count=n,
        num_parameters=p,
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        adj_r2=adj_r2,
        notes=notes
    )
