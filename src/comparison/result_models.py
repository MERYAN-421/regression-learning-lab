"""
Data models and typed artifacts for the Experiment & Model Comparison Laboratory.
Defines ExperimentComparisonResult, compatibility models, and structured comparison data contracts.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import numpy as np
import pandas as pd


class RankingPolicy(str, Enum):
    """Supported validation-based model ranking policies (strictly no Test Set ranking)."""
    VAL_RMSE_ASC = "val_rmse_asc"
    VAL_MAE_ASC = "val_mae_asc"
    VAL_R2_DESC = "val_r2_desc"
    PARSIMONY_ASC = "parsimony_asc"
    GEN_GAP_ASC = "gen_gap_asc"


@dataclass
class ModelComparisonRecord:
    """Structured development-stage metrics record for a single model in comparison."""
    model_id: str
    model_name: str
    feature_source_id: str
    feature_source_label: str
    original_feature_count: int
    parameter_count: int
    train_sample_count: int
    val_sample_count: int
    train_rmse: float
    val_rmse: float
    train_mae: float
    val_mae: float
    train_r2: float
    val_r2: float
    train_adj_r2: Optional[float]
    generalization_gap: float  # val_rmse - train_rmse
    aic: Optional[float]
    bic: Optional[float]
    is_valid_for_inference: bool
    is_pareto_efficient: bool = False
    diagnostic_suite_id: Optional[str] = None
    test_evaluation_status: str = "untested"  # Informative flag, strictly excluded from ranking


@dataclass
class CompatibilityResult:
    """Result of checking whether a collection of models can be fairly compared."""
    is_compatible: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dataset_id: str = ""
    dataset_fingerprint: str = ""
    split_fingerprint: str = ""
    target_name: str = ""
    target_unit: str = ""


@dataclass
class ExperimentComparisonResult:
    """
    Immutable artifact encapsulating a comprehensive multi-model comparison.
    Contains leaderboard rankings, normalized radar scores, Pareto frontier points,
    assumption diagnostic matrices, original feature overlap, and candidate selection audit metadata.
    """
    comparison_id: str
    dataset_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    target_name: str
    target_unit: str
    compared_model_ids: List[str]
    referenced_feature_selection_ids: Dict[str, Optional[str]]
    referenced_diagnostic_suite_ids: Dict[str, Optional[str]]
    evaluation_scope: str = "validation"  # Strictly development-stage validation
    ranking_policy: str = RankingPolicy.VAL_RMSE_ASC.value
    leaderboard: List[Dict[str, Any]] = field(default_factory=list)
    model_complexity: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    normalization_config: Dict[str, Any] = field(default_factory=dict)
    radar_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)  # model_id -> metric -> score [0, 1]
    pareto_model_ids: List[str] = field(default_factory=list)
    diagnostic_matrix: List[Dict[str, Any]] = field(default_factory=list)
    feature_overlap_matrix: Dict[str, Any] = field(default_factory=dict)
    selected_candidate_model_id: Optional[str] = None
    candidate_selection_rationale: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    methodological_limitations: List[str] = field(default_factory=list)
    timestamp: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    reproducibility_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison result to a JSON-serializable dictionary with safe numeric handling."""
        return {
            "comparison_id": self.comparison_id,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "target_name": self.target_name,
            "target_unit": self.target_unit,
            "compared_model_ids": self.compared_model_ids,
            "referenced_feature_selection_ids": self.referenced_feature_selection_ids,
            "referenced_diagnostic_suite_ids": self.referenced_diagnostic_suite_ids,
            "evaluation_scope": self.evaluation_scope,
            "ranking_policy": self.ranking_policy,
            "leaderboard": safe_serialize_numeric(self.leaderboard),
            "model_complexity": safe_serialize_numeric(self.model_complexity),
            "normalization_config": safe_serialize_numeric(self.normalization_config),
            "radar_scores": safe_serialize_numeric(self.radar_scores),
            "pareto_model_ids": self.pareto_model_ids,
            "diagnostic_matrix": safe_serialize_numeric(self.diagnostic_matrix),
            "feature_overlap_matrix": safe_serialize_numeric(self.feature_overlap_matrix),
            "selected_candidate_model_id": self.selected_candidate_model_id,
            "candidate_selection_rationale": self.candidate_selection_rationale,
            "warnings": self.warnings,
            "methodological_limitations": self.methodological_limitations,
            "timestamp": self.timestamp,
            "configuration": safe_serialize_numeric(self.configuration),
            "reproducibility_hash": self.reproducibility_hash
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize comparison result to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


def safe_serialize_numeric(val: Any) -> Any:
    """
    Recursively converts NumPy scalars, DataFrames, Series, and non-finite numbers
    into standard JSON-compliant Python primitives.
    """
    if val is None:
        return None
    elif isinstance(val, (bool, np.bool_)):
        return bool(val)
    elif isinstance(val, (int, np.integer)):
        return int(val)
    elif isinstance(val, (float, np.floating)):
        if np.isnan(val) or not np.isfinite(val):
            return None
        return float(val)
    elif isinstance(val, (np.ndarray, list, tuple)):
        return [safe_serialize_numeric(item) for item in val]
    elif isinstance(val, pd.DataFrame):
        return [safe_serialize_numeric(row) for row in val.to_dict(orient="records")]
    elif isinstance(val, pd.Series):
        return safe_serialize_numeric(val.to_dict())
    elif isinstance(val, dict):
        return {str(k): safe_serialize_numeric(v) for k, v in val.items()}
    else:
        return str(val)


def generate_comparison_id(
    split_fingerprint: str,
    model_ids: List[str],
    ranking_policy: str,
    extra_config: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """Generates a unique deterministic identifier and reproducibility hash for a comparison."""
    raw_payload = {
        "split_fingerprint": split_fingerprint,
        "model_ids": sorted(model_ids),
        "ranking_policy": ranking_policy,
        "extra_config": extra_config or {}
    }
    raw_str = json.dumps(safe_serialize_numeric(raw_payload), sort_keys=True)
    h = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:12]
    return f"comp_{h}", h
