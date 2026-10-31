"""
Data models and typed artifacts for the Statistical Assumption Diagnostics Laboratory.
Represents individual diagnostic outcomes and aggregated diagnostic suites for OLS models.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json


class DiagnosticType(str, Enum):
    """The core categories of linear regression assumption diagnostics."""
    LINEARITY = "linearity"
    INDEPENDENCE = "independence"
    HOMOSCEDASTICITY = "homoscedasticity"
    NORMALITY = "normality"
    MULTICOLLINEARITY = "multicollinearity"
    INFLUENCE = "influence"


class DiagnosticDecision(str, Enum):
    """
    Formal statistical hypothesis test decisions and analytical states.
    Strictly adheres to scientific hypothesis testing conventions:
    'fail_to_reject_null' indicates insufficient evidence against H0 at alpha,
    NEVER proof that H0 is true.
    """
    REJECT_NULL = "reject_null"
    FAIL_TO_REJECT_NULL = "fail_to_reject_null"
    DESCRIPTIVE_ONLY = "descriptive_only"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DATA = "insufficient_data"
    NUMERICAL_FAILURE = "numerical_failure"


class ApplicabilityStatus(str, Enum):
    """Applicability status of a diagnostic given the model and data structure."""
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    EXPLORATORY = "exploratory"
    CONDITIONAL = "conditional"


@dataclass
class DiagnosticResult:
    """
    Strongly typed, immutable artifact encapsulating a single assumption diagnostic execution.
    Contains statistical test values, hypothesis definitions, p-values, decision outcomes,
    educational interpretations, limitations, and reproducible configuration hashes.
    """
    diagnostic_id: str
    model_id: str
    dataset_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    diagnostic_type: str  # from DiagnosticType
    method_id: str
    method_name: str
    data_scope: str = "train"  # Diagnostics evaluate training data
    is_valid_for_inference: bool = True
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    statistic_name: str = ""
    statistic_value: float = float("nan")
    p_value: Optional[float] = None
    alpha: Optional[float] = 0.05
    decision: str = DiagnosticDecision.DESCRIPTIVE_ONLY.value
    applicability_status: str = ApplicabilityStatus.APPLICABLE.value
    applicability_reason: str = ""
    assumptions_prerequisites: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    limitations: str = ""
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    configuration: Dict[str, Any] = field(default_factory=dict)
    reproducibility_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic result to a JSON-serializable dictionary with safe numeric handling."""
        return {
            "diagnostic_id": self.diagnostic_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "diagnostic_type": self.diagnostic_type,
            "method_id": self.method_id,
            "method_name": self.method_name,
            "data_scope": self.data_scope,
            "is_valid_for_inference": self.is_valid_for_inference,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "statistic_name": self.statistic_name,
            "statistic_value": safe_serialize_numeric(self.statistic_value),
            "p_value": safe_serialize_numeric(self.p_value),
            "alpha": self.alpha,
            "decision": self.decision,
            "applicability_status": self.applicability_status,
            "applicability_reason": self.applicability_reason,
            "assumptions_prerequisites": self.assumptions_prerequisites,
            "details": safe_serialize_numeric(self.details),
            "interpretation": self.interpretation,
            "limitations": self.limitations,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "runtime_seconds": self.runtime_seconds,
            "configuration": safe_serialize_numeric(self.configuration),
            "reproducibility_hash": self.reproducibility_hash
        }


@dataclass
class DiagnosticSuiteResult:
    """
    Aggregated artifact containing all diagnostics executed for an OLS model specification.
    Maintains full configuration metadata and safe JSON serialization.
    Methodological note: An individual suite CANNOT prove all assumptions hold.
    """
    suite_id: str
    model_id: str
    dataset_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    diagnostics: Dict[str, DiagnosticResult] = field(default_factory=dict)  # method_id -> DiagnosticResult
    alpha: float = 0.05
    timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: DiagnosticResult) -> None:
        """Register a diagnostic result into the suite."""
        self.diagnostics[result.method_id] = result

    def get_by_type(self, diag_type: DiagnosticType) -> List[DiagnosticResult]:
        """Retrieve all results for a specific diagnostic category."""
        return [r for r in self.diagnostics.values() if r.diagnostic_type == diag_type.value]

    def to_summary_table(self) -> List[Dict[str, Any]]:
        """
        Generate a structured summary table for the entire diagnostic suite.
        Adheres to statistical reporting conventions without misleading 'All Passed' labels.
        """
        rows = []
        for r in self.diagnostics.values():
            stat_val = r.statistic_value
            is_nan = stat_val is None or (isinstance(stat_val, float) and (stat_val != stat_val))
            stat_str = f"{stat_val:.4f}" if not is_nan else "N/A"
            pval_str = f"{r.p_value:.4f}" if r.p_value is not None else "N/A"
            rows.append({
                "Category": r.diagnostic_type.capitalize(),
                "Diagnostic Method": r.method_name,
                "Statistic": f"{r.statistic_name} = {stat_str}" if stat_str != "N/A" else "N/A",
                "p-value": pval_str,
                "Decision": r.decision.replace("_", " ").title(),
                "Applicability": r.applicability_status.title(),
                "Primary Finding": r.interpretation[:100] + "..." if len(r.interpretation) > 100 else r.interpretation
            })
        return rows

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire suite to a JSON-serializable dictionary."""
        return {
            "suite_id": self.suite_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "alpha": self.alpha,
            "timestamp": self.timestamp,
            "warnings": self.warnings,
            "configuration": safe_serialize_numeric(self.configuration),
            "diagnostics": {k: res.to_dict() for k, res in self.diagnostics.items()}
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize suite to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


def safe_serialize_numeric(val: Any) -> Any:
    """
    Recursively converts NumPy scalars, DataFrames, Series, and non-finite numbers
    into standard JSON-compliant Python primitives.
    """
    import numpy as np
    import pandas as pd

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


def generate_diagnostic_id(
    model_id: str,
    method_id: str,
    config: Dict[str, Any]
) -> Tuple[str, str]:
    """Generate a deterministic identifier and reproducibility hash for a diagnostic."""
    raw_str = f"{model_id}:{method_id}:{json.dumps(safe_serialize_numeric(config), sort_keys=True)}"
    h = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:12]
    return f"diag_{method_id}_{h}", h
