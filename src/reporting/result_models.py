"""
Data models and typed contracts for Academic Report Generation Laboratory.
Defines AcademicReportResult, ReportTemplateType, ReadinessStatus, EvidenceStatus,
EvidenceRecord, EvidenceManifest, ReportSection, FigureReference, and AuthorContent.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json
import numpy as np
import pandas as pd


class ReportTemplateType(str, Enum):
    """Supported academic report templates."""
    FULL_ACADEMIC = "full_academic"
    EXECUTIVE_SUMMARY = "executive_summary"
    STATISTICAL_DIAGNOSTICS = "statistical_diagnostics"


class ReadinessStatus(str, Enum):
    """Readiness status of an academic report prior to rendering."""
    READY = "ready"
    READY_WITH_LIMITATIONS = "ready_with_limitations"
    BLOCKED = "blocked"


class EvidenceStatus(str, Enum):
    """Validation status for individual pieces of experimental evidence."""
    VERIFIED = "verified"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    NOT_APPLICABLE = "not_applicable"
    UNVERIFIED = "unverified"


class SectionID(str, Enum):
    """Canonical section IDs for academic reports."""
    SEC_EXEC_SUMMARY = "sec_1_exec_summary"
    SEC_DATASET = "sec_2_dataset"
    SEC_EDA_SELECTION = "sec_3_eda_selection"
    SEC_OLS_METHODOLOGY = "sec_4_ols_methodology"
    SEC_COEFFICIENTS = "sec_5_coefficients"
    SEC_DIAGNOSTICS = "sec_6_diagnostics"
    SEC_COMPARISON = "sec_7_comparison"
    SEC_DISCUSSION = "sec_8_discussion"
    SEC_REPRODUCIBILITY = "sec_9_reproducibility"
    SEC_APPENDIX_FINAL_TEST = "sec_appendix_final_test"


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
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    elif isinstance(val, np.ndarray):
        return [safe_serialize_numeric(x) for x in val.tolist()]
    elif isinstance(val, pd.DataFrame):
        return safe_serialize_numeric(val.to_dict(orient="records"))
    elif isinstance(val, pd.Series):
        return safe_serialize_numeric(val.to_dict())
    elif isinstance(val, dict):
        return {str(k): safe_serialize_numeric(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple, set)):
        return [safe_serialize_numeric(x) for x in val]
    return str(val)


@dataclass
class EvidenceRecord:
    """
    Strongly typed evidence claim referencing a validated experiment artifact.
    Guarantees no numerical claim is fabricated or disconnected from raw artifacts.
    """
    claim_id: str
    claim_type: str
    source_artifact_type: str  # "dataset", "split", "model", "diagnostic_suite", "comparison", "feature_selection", "final_test"
    source_artifact_id: str
    source_field_path: str
    dataset_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    model_id: Optional[str] = None
    diagnostic_suite_id: Optional[str] = None
    diagnostic_method_id: Optional[str] = None
    analysis_data_scope: str = "train"  # "train", "validation", "test", "cv", "dataset"
    observed_value: Any = None
    display_value: str = ""
    unit: Optional[str] = None
    evidence_status: str = EvidenceStatus.VERIFIED.value
    validation_reason: str = ""
    methodological_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "source_artifact_type": self.source_artifact_type,
            "source_artifact_id": self.source_artifact_id,
            "source_field_path": self.source_field_path,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "model_id": self.model_id,
            "diagnostic_suite_id": self.diagnostic_suite_id,
            "diagnostic_method_id": self.diagnostic_method_id,
            "analysis_data_scope": self.analysis_data_scope,
            "observed_value": safe_serialize_numeric(self.observed_value),
            "display_value": self.display_value,
            "unit": self.unit,
            "evidence_status": self.evidence_status,
            "validation_reason": self.validation_reason,
            "methodological_warnings": self.methodological_warnings
        }


@dataclass
class EvidenceManifest:
    """
    Comprehensive collection of verified evidence records supporting an academic report.
    Serves as an immutable audit trail for peer review and grading.
    """
    report_id: str
    records: Dict[str, EvidenceRecord] = field(default_factory=dict)
    verified_count: int = 0
    missing_count: int = 0
    incompatible_count: int = 0
    not_applicable_count: int = 0
    unverified_count: int = 0
    warnings: List[str] = field(default_factory=list)
    audit_timestamp: str = ""

    def add_record(self, record: EvidenceRecord) -> None:
        self.records[record.claim_id] = record
        if record.evidence_status == EvidenceStatus.VERIFIED.value:
            self.verified_count += 1
        elif record.evidence_status == EvidenceStatus.MISSING.value:
            self.missing_count += 1
        elif record.evidence_status == EvidenceStatus.INCOMPATIBLE.value:
            self.incompatible_count += 1
        elif record.evidence_status == EvidenceStatus.NOT_APPLICABLE.value:
            self.not_applicable_count += 1
        elif record.evidence_status == EvidenceStatus.UNVERIFIED.value:
            self.unverified_count += 1

    def get_record(self, claim_id: str) -> Optional[EvidenceRecord]:
        return self.records.get(claim_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "audit_timestamp": self.audit_timestamp,
            "verified_count": self.verified_count,
            "missing_count": self.missing_count,
            "incompatible_count": self.incompatible_count,
            "not_applicable_count": self.not_applicable_count,
            "unverified_count": self.unverified_count,
            "warnings": self.warnings,
            "records": {k: r.to_dict() for k, r in self.records.items()}
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class FigureReference:
    """Metadata and provenance reference for a figure embedded in the report."""
    figure_id: str
    title: str
    caption: str
    figure_type: str
    model_id: Optional[str] = None
    experiment_id: Optional[str] = None
    diagnostic_suite_id: Optional[str] = None
    data_scope: str = "validation"
    provenance: str = ""
    caveats: List[str] = field(default_factory=list)
    is_available: bool = True
    image_data_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "title": self.title,
            "caption": self.caption,
            "figure_type": self.figure_type,
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "diagnostic_suite_id": self.diagnostic_suite_id,
            "data_scope": self.data_scope,
            "provenance": self.provenance,
            "caveats": self.caveats,
            "is_available": self.is_available
        }


@dataclass
class ReportSection:
    """Structured academic section component."""
    section_id: str
    title: str
    order: int
    content_blocks: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    figure_ids: List[str] = field(default_factory=list)
    evidence_reference_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    author_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "order": self.order,
            "content_blocks": self.content_blocks,
            "tables": safe_serialize_numeric(self.tables),
            "figure_ids": self.figure_ids,
            "evidence_reference_ids": self.evidence_reference_ids,
            "warnings": self.warnings,
            "author_notes": self.author_notes
        }


@dataclass
class AuthorContent:
    """Editable student / researcher narrative and reflection fields."""
    author_name: str = "Anonymous Researcher"
    research_question: str = ""
    main_findings: str = ""
    practical_significance: str = ""
    modeling_limitations: str = ""
    statistical_limitations: str = ""
    alternative_explanations: str = ""
    future_improvements: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author_name": self.author_name,
            "research_question": self.research_question,
            "main_findings": self.main_findings,
            "practical_significance": self.practical_significance,
            "modeling_limitations": self.modeling_limitations,
            "statistical_limitations": self.statistical_limitations,
            "alternative_explanations": self.alternative_explanations,
            "future_improvements": self.future_improvements
        }


@dataclass
class AcademicReportResult:
    """
    Immutable, fully auditable artifact of an academic report generation run.
    Contains structured sections, verified evidence manifest, reproducibility manifest,
    and references to all underlying experimental artifacts.
    """
    report_id: str
    report_title: str
    template_type: str  # from ReportTemplateType
    dataset_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    comparison_id: Optional[str] = None
    selected_model_id: str = ""
    feature_selection_artifact_ids: Dict[str, Optional[str]] = field(default_factory=dict)
    diagnostic_suite_ids: Dict[str, Optional[str]] = field(default_factory=dict)
    report_sections: List[ReportSection] = field(default_factory=list)
    evidence_manifest: Optional[EvidenceManifest] = None
    readiness_status: str = ReadinessStatus.READY.value
    readiness_reasons: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    report_configuration: Dict[str, Any] = field(default_factory=dict)
    reproducibility_manifest: Dict[str, Any] = field(default_factory=dict)
    author_content: AuthorContent = field(default_factory=AuthorContent)
    figure_references: List[FigureReference] = field(default_factory=list)
    final_test_appendix_status: str = "excluded"  # "excluded", "authorized", "unauthorized_blocked"
    generated_timestamp: str = ""
    reproducibility_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_title": self.report_title,
            "template_type": self.template_type,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "comparison_id": self.comparison_id,
            "selected_model_id": self.selected_model_id,
            "feature_selection_artifact_ids": self.feature_selection_artifact_ids,
            "diagnostic_suite_ids": self.diagnostic_suite_ids,
            "readiness_status": self.readiness_status,
            "readiness_reasons": self.readiness_reasons,
            "missing_evidence": self.missing_evidence,
            "warnings": self.warnings,
            "final_test_appendix_status": self.final_test_appendix_status,
            "generated_timestamp": self.generated_timestamp,
            "reproducibility_hash": self.reproducibility_hash,
            "report_configuration": safe_serialize_numeric(self.report_configuration),
            "reproducibility_manifest": safe_serialize_numeric(self.reproducibility_manifest),
            "author_content": self.author_content.to_dict(),
            "figure_references": [f.to_dict() for f in self.figure_references],
            "report_sections": [s.to_dict() for s in self.report_sections],
            "evidence_manifest": self.evidence_manifest.to_dict() if self.evidence_manifest else None
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
