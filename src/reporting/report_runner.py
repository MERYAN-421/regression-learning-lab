"""
Academic Report Runner and Orchestrator.
Coordinates source resolution, readiness evaluation, evidence manifest construction,
figure generation, and section compilation into an immutable AcademicReportResult.
"""
from datetime import datetime
import hashlib
import json
from typing import Any, Dict, List, Optional

from src.reporting.result_models import (
    AcademicReportResult,
    ReportTemplateType,
    ReadinessStatus,
    AuthorContent,
    safe_serialize_numeric
)
from src.reporting.source_resolver import ResolvedReportSources
from src.reporting.readiness import evaluate_report_readiness, ReportReadinessEvaluation
from src.reporting.evidence_manifest import EvidenceManifestBuilder
from src.reporting.figure_registry import ReportFigureRegistry
from src.reporting.section_builder import AcademicSectionBuilder


def compute_deterministic_report_id(
    model_id: str,
    dataset_fingerprint: str,
    split_fingerprint: str,
    template_type: str,
    comparison_id: Optional[str] = None,
    diagnostic_suite_id: Optional[str] = None,
    author_text: str = "",
    include_final_test: bool = False
) -> str:
    """
    Computes a deterministic hash ID identifying the exact report configuration and sources.
    """
    hasher = hashlib.sha256()
    hasher.update(model_id.encode("utf-8"))
    hasher.update(dataset_fingerprint.encode("utf-8"))
    hasher.update(split_fingerprint.encode("utf-8"))
    hasher.update(template_type.encode("utf-8"))
    if comparison_id:
        hasher.update(comparison_id.encode("utf-8"))
    if diagnostic_suite_id:
        hasher.update(diagnostic_suite_id.encode("utf-8"))
    hasher.update(author_text.encode("utf-8"))
    hasher.update(str(include_final_test).encode("utf-8"))
    return f"rep_{hasher.hexdigest()[:12]}"


def run_academic_report(
    sources: ResolvedReportSources,
    template_type: ReportTemplateType = ReportTemplateType.FULL_ACADEMIC,
    report_title: str = "Academic Regression Learning Laboratory Report",
    author_content: Optional[AuthorContent] = None,
    enabled_section_ids: Optional[List[str]] = None,
    include_final_test_authorized: bool = False,
    expected_dataset_id: Optional[str] = None,
    expected_dataset_fingerprint: Optional[str] = None,
    expected_split_fingerprint: Optional[str] = None,
    expected_target_unit: Optional[str] = None
) -> AcademicReportResult:
    """
    Executes the report generation pipeline without retraining models or modifying trackers.
    """
    author = author_content or AuthorContent()

    # 1. Evaluate Readiness
    readiness: ReportReadinessEvaluation = evaluate_report_readiness(
        sources=sources,
        template_type=template_type,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
        expected_split_fingerprint=expected_split_fingerprint,
        expected_target_unit=expected_target_unit
    )

    now_iso = datetime.utcnow().isoformat() + "Z"

    # If blocked, return a blocked result artifact detailing reasons
    if readiness.is_blocked:
        return AcademicReportResult(
            report_id=f"rep_blocked_{hashlib.sha256(now_iso.encode()).hexdigest()[:8]}",
            report_title=report_title,
            template_type=template_type.value,
            dataset_id=sources.model.dataset_id if sources.model else (expected_dataset_id or "unknown"),
            dataset_fingerprint=sources.model.dataset_fingerprint if sources.model else (expected_dataset_fingerprint or "unknown"),
            split_fingerprint=sources.model.split_fingerprint if sources.model else (expected_split_fingerprint or "unknown"),
            readiness_status=ReadinessStatus.BLOCKED.value,
            readiness_reasons=readiness.blockers,
            warnings=readiness.warnings,
            author_content=author,
            generated_timestamp=now_iso
        )

    model = sources.model
    diag_suite_id = sources.diagnostic_suite.suite_id if sources.diagnostic_suite else None
    comp_id = sources.comparison.comparison_id if sources.comparison else None

    # Compute stable report ID
    author_composite = f"{author.author_name}_{author.research_question}_{author.main_findings}"
    report_id = compute_deterministic_report_id(
        model_id=model.model_id,
        dataset_fingerprint=model.dataset_fingerprint,
        split_fingerprint=model.split_fingerprint,
        template_type=template_type.value,
        comparison_id=comp_id,
        diagnostic_suite_id=diag_suite_id,
        author_text=author_composite,
        include_final_test=include_final_test_authorized
    )

    # 2. Build Evidence Manifest (built BEFORE rendering any report text)
    manifest_builder = EvidenceManifestBuilder(report_id=report_id)
    manifest = manifest_builder.build_from_sources(
        sources=sources,
        include_final_test_authorized=include_final_test_authorized
    )

    # 3. Build Figures
    fig_registry = ReportFigureRegistry(sources=sources)
    figure_refs = fig_registry.build_all_available_figures()

    # 4. Build Structured Sections
    section_builder = AcademicSectionBuilder(
        sources=sources,
        manifest=manifest,
        author_content=author,
        include_final_test_authorized=include_final_test_authorized
    )
    sections = section_builder.build_sections_for_template(
        template_type=template_type,
        section_ids_enabled=enabled_section_ids
    )

    # 5. Build Reproducibility Hash
    hash_payload = {
        "report_id": report_id,
        "model_id": model.model_id,
        "dataset_fingerprint": model.dataset_fingerprint,
        "split_fingerprint": model.split_fingerprint,
        "template": template_type.value,
        "verified_evidence_count": manifest.verified_count
    }
    repro_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode()).hexdigest()

    repro_manifest = {
        "dataset_source": getattr(sources.dataset_metadata, "original_reference", getattr(sources.dataset_metadata, "provenance_notes", "Verified Registry")) if sources.dataset_metadata else "Verified Registry",
        "dataset_fingerprint": model.dataset_fingerprint,
        "split_fingerprint": model.split_fingerprint,
        "model_id": model.model_id,
        "feature_selection_source": model.feature_source_id,
        "diagnostic_suite_id": diag_suite_id,
        "comparison_id": comp_id,
        "python_version": "3.11",
        "libraries": {
            "statsmodels": "0.14+",
            "scikit-learn": "1.3+",
            "numpy": "1.24+",
            "pandas": "2.0+",
            "plotly": "5.15+"
        }
    }

    report_config = {
        "report_title": report_title,
        "template_type": template_type.value,
        "include_final_test": include_final_test_authorized,
        "enabled_sections": enabled_section_ids or [s.section_id for s in sections]
    }

    final_test_status = "excluded"
    if include_final_test_authorized:
        final_test_status = "authorized" if model.test_evaluation_status != "untested" else "untested_omitted"

    all_warnings = list(readiness.warnings)
    if readiness.limitations:
        all_warnings.extend(readiness.limitations)

    return AcademicReportResult(
        report_id=report_id,
        report_title=report_title,
        template_type=template_type.value,
        dataset_id=model.dataset_id,
        dataset_fingerprint=model.dataset_fingerprint,
        split_fingerprint=model.split_fingerprint,
        comparison_id=comp_id,
        selected_model_id=model.model_id,
        feature_selection_artifact_ids={model.model_id: model.feature_source_id},
        diagnostic_suite_ids={model.model_id: diag_suite_id} if diag_suite_id else {},
        report_sections=sections,
        evidence_manifest=manifest,
        readiness_status=readiness.status.value,
        readiness_reasons=readiness.blockers,
        missing_evidence=readiness.limitations,
        warnings=all_warnings,
        report_configuration=report_config,
        reproducibility_manifest=repro_manifest,
        author_content=author,
        figure_references=figure_refs,
        final_test_appendix_status=final_test_status,
        generated_timestamp=now_iso,
        reproducibility_hash=repro_hash
    )
