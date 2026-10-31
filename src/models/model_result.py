"""
Data models for Ordinary Least Squares (OLS) regression models and evaluation results.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from src.models.design_matrix import DesignMatrixMetadata, OLSDesignMatrixBuilder
from src.evaluation.regression_metrics import RegressionMetrics


@dataclass
class OLSModelResult:
    """
    Immutable artifact encapsulating a fitted OLS regression model,
    its econometric parameter estimates, statistical inference tests,
    mathematical equation strings, and cross-split performance metrics.
    """
    model_id: str
    model_name: str
    dataset_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    feature_source_id: str  # "all_features" or Milestone 2 experiment_id
    feature_source_label: str  # Human-readable method label
    selected_original_features: List[str]
    design_matrix_metadata: DesignMatrixMetadata
    fit_intercept: bool
    scale_numeric: bool
    coef_table: pd.DataFrame
    equation_general: str
    equation_fitted: str
    equation_text: str
    train_metrics: RegressionMetrics
    val_metrics: RegressionMetrics
    test_metrics: Optional[RegressionMetrics]
    r2: float
    adj_r2: Optional[float]
    f_statistic: float
    f_pvalue: float
    log_likelihood: float
    aic: float
    bic: float
    rss: float
    df_resid: int
    df_model: int
    post_selection_inference_warning: bool
    warnings: List[str] = field(default_factory=list)
    test_evaluation_status: str = "untested"  # "untested", "evaluated_train_only", "evaluated_train_val_refit"
    trained_timestamp: str = ""
    # In-memory fitted references
    fitted_model: Optional[Any] = None
    builder: Optional[OLSDesignMatrixBuilder] = None
    train_predictions: Optional[pd.Series] = None
    val_predictions: Optional[pd.Series] = None
    test_predictions: Optional[pd.Series] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Summary dictionary for side-by-side leaderboard comparison."""
        return {
            "Model ID": self.model_id,
            "Feature Source": self.feature_source_label,
            "Original K": len(self.selected_original_features),
            "OLS Parameters (p)": self.design_matrix_metadata.num_parameters,
            "Train RMSE": round(self.train_metrics.rmse, 3),
            "Val RMSE": round(self.val_metrics.rmse, 3),
            "Gen. Gap": round(self.val_metrics.rmse - self.train_metrics.rmse, 3),
            "Train R²": round(self.train_metrics.r2, 4),
            "Val R²": round(self.val_metrics.r2, 4),
            "Train Adj R²": round(self.train_metrics.adj_r2, 4) if self.train_metrics.adj_r2 is not None else "N/A",
            "F-stat": round(self.f_statistic, 2) if np_finite(self.f_statistic) else "N/A",
            "Full Rank": "✅ Yes" if self.design_matrix_metadata.is_full_rank else "⚠️ Rank Deficient",
            "Test Status": self.test_evaluation_status
        }


def np_finite(val: Any) -> bool:
    try:
        import numpy as np
        return bool(np.isfinite(val))
    except Exception:
        return False
