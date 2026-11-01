"""
Academic Report Generation Laboratory package.
Provides structured synthesis, evidence manifest provenance, and export services.
"""
from src.reporting.result_models import (
    AcademicReportResult,
    ReportTemplateType,
    ReadinessStatus,
    EvidenceStatus,
    EvidenceRecord,
    EvidenceManifest,
    ReportSection,
    FigureReference,
    AuthorContent,
    SectionID
)
from src.reporting.source_resolver import ReportSourceResolver, ResolvedReportSources
from src.reporting.compatibility import validate_report_artifacts, ReportCompatibilityResult
from src.reporting.readiness import evaluate_report_readiness, ReportReadinessEvaluation
from src.reporting.evidence_manifest import EvidenceManifestBuilder
from src.reporting.figure_registry import ReportFigureRegistry
from src.reporting.section_builder import AcademicSectionBuilder
from src.reporting.markdown_renderer import render_report_markdown
from src.reporting.html_renderer import render_report_html
from src.reporting.reproduction_script import (
    generate_reproduction_script,
    generate_reproduction_readme,
    generate_requirements_txt
)
from src.reporting.export_service import ReportExportService
from src.reporting.report_runner import run_academic_report, compute_deterministic_report_id

__all__ = [
    "AcademicReportResult",
    "ReportTemplateType",
    "ReadinessStatus",
    "EvidenceStatus",
    "EvidenceRecord",
    "EvidenceManifest",
    "ReportSection",
    "FigureReference",
    "AuthorContent",
    "SectionID",
    "ReportSourceResolver",
    "ResolvedReportSources",
    "validate_report_artifacts",
    "ReportCompatibilityResult",
    "evaluate_report_readiness",
    "ReportReadinessEvaluation",
    "EvidenceManifestBuilder",
    "ReportFigureRegistry",
    "AcademicSectionBuilder",
    "render_report_markdown",
    "render_report_html",
    "generate_reproduction_script",
    "generate_reproduction_readme",
    "generate_requirements_txt",
    "ReportExportService",
    "run_academic_report",
    "compute_deterministic_report_id"
]
