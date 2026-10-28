"""
Data models for Feature Selection results, feature mappings, and experiment metadata.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class OriginalFeatureMapping:
    """
    Maps an original dataset predictor feature to its encoded representations.
    Crucial for grouping dummy one-hot columns back into a single original feature.
    """
    original_name: str
    feature_type: str  # "numeric" or "categorical"
    encoded_column_names: List[str]
    encoded_column_indices: List[int]
    is_selected: bool = False
    rank: int = 0
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def encoded_dimension(self) -> int:
        return len(self.encoded_column_names)


@dataclass
class FeatureSelectionResult:
    """
    Complete immutable artifact of a Feature Selection experiment.
    Contains everything needed to evaluate, visualize, and reconstruct the pipeline.
    """
    experiment_id: str
    dataset_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    method_id: str
    method_name: str
    category: str  # "Baseline", "Filter", "Wrapper", "Embedded"
    requested_k: int
    selected_original_features: List[str]
    selected_original_count: int
    all_original_features: List[str]
    encoded_feature_names: List[str]
    selected_encoded_feature_names: List[str]
    encoded_input_dimension: int
    feature_mappings: List[OriginalFeatureMapping]
    cv_rmse_mean: float
    cv_rmse_std: float
    val_mse: float
    val_rmse: float
    val_r2: float
    execution_time_seconds: float
    cv_strategy_used: str
    cv_folds: int
    random_seed: int
    warnings: List[str] = field(default_factory=list)
    notes: str = ""

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "Method": self.method_name,
            "Category": self.category,
            "Requested K": self.requested_k,
            "Selected Orig K": self.selected_original_count,
            "Encoded Dim": len(self.selected_encoded_feature_names),
            "CV RMSE (Mean ± Std)": f"{self.cv_rmse_mean:.3f} ± {self.cv_rmse_std:.3f}",
            "Val RMSE": f"{self.val_rmse:.3f}",
            "Val R²": f"{self.val_r2:.3f}",
            "Runtime (s)": f"{self.execution_time_seconds:.2f}s"
        }
