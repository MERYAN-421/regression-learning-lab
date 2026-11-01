"""
Report readiness evaluation engine.
Classifies report readiness as READY, READY_WITH_LIMITATIONS, or BLOCKED based on
strict evidence availability and compatibility criteria.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.reporting.result_models import ReadinessStatus, ReportTemplateType
from src.reporting.source_resolver import ResolvedReportSources
from src.reporting.compatibility import validate_report_artifacts, ReportCompatibilityResult


@dataclass
class ReportReadinessEvaluation:
    """Detailed readiness assessment outcome."""
    status: ReadinessStatus
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    compatibility: Optional[ReportCompatibilityResult] = None

    @property
    def is_blocked(self) -> bool:
        return self.status == ReadinessStatus.BLOCKED

    @property
    def can_generate(self) -> bool:
        return self.status in (ReadinessStatus.READY, ReadinessStatus.READY_WITH_LIMITATIONS)


def evaluate_report_readiness(
    sources: ResolvedReportSources,
    template_type: ReportTemplateType = ReportTemplateType.FULL_ACADEMIC,
    expected_dataset_id: Optional[str] = None,
    expected_dataset_fingerprint: Optional[str] = None,
    expected_split_fingerprint: Optional[str] = None,
    expected_target_unit: Optional[str] = None,
    require_comparison: bool = False,
    allow_alternative_model: bool = False
) -> ReportReadinessEvaluation:
    """
    Evaluates whether the provided experimental artifacts satisfy the integrity requirements
    for generating the requested report template.
    """
    blockers: List[str] = []
    warnings: List[str] = []
    limitations: List[str] = []

    # 1. Check Source Resolution Errors
    if sources.resolution_errors:
        blockers.extend(sources.resolution_errors)

    if sources.model is None:
        blockers.append("Primary OLS model artifact is missing. Report generation is blocked.")
        return ReportReadinessEvaluation(
            status=ReadinessStatus.BLOCKED,
            blockers=blockers,
            warnings=sources.resolution_warnings
        )

    # 2. Check Comparison Requirement
    if require_comparison and sources.comparison is None:
        blockers.append("This report configuration requires a completed model comparison, but none was provided.")

    # 3. Check Artifact Compatibility
    compat = validate_report_artifacts(
        model=sources.model,
        comparison=sources.comparison,
        diagnostic_suite=sources.diagnostic_suite,
        feature_selection_results=sources.feature_selection_results,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
        expected_split_fingerprint=expected_split_fingerprint,
        expected_target_unit=expected_target_unit,
        allow_alternative_model=allow_alternative_model
    )

    if not compat.is_compatible:
        blockers.extend(compat.reasons)

    warnings.extend(compat.warnings)
    limitations.extend(compat.limitations)
    warnings.extend(sources.resolution_warnings)

    # 4. Template-Specific Checks
    if template_type == ReportTemplateType.STATISTICAL_DIAGNOSTICS:
        if sources.diagnostic_suite is None:
            blockers.append("Template C (Statistical Diagnostics) requires an active DiagnosticSuiteResult, but none was resolved.")
        elif len(sources.diagnostic_suite.diagnostics) == 0:
            blockers.append("Diagnostic suite is empty. Cannot generate a Statistical Diagnostics report without diagnostic tests.")

    # 5. Missing Optional Artifacts -> Ready with Limitations
    if sources.model is not None:
        # Check if feature selection artifact is referenced but missing
        fs_id = sources.model.feature_source_id
        if fs_id != "all_features" and fs_id not in sources.feature_selection_results:
            limitations.append(
                f"Referenced feature selection artifact '{fs_id}' could not be retrieved from tracker. "
                "Feature selection CV metrics will be noted as unavailable."
            )

        # Check if diagnostic suite is missing for templates other than template C
        if template_type != ReportTemplateType.STATISTICAL_DIAGNOSTICS and sources.diagnostic_suite is None:
            limitations.append(
                "Statistical assumption diagnostics have not been run for the selected model. "
                "Diagnostics section will document this omission."
            )

        # Rank-deficient / prediction-only inference check
        if not sources.model.is_valid_for_inference:
            limitations.append(
                "Selected model is rank-deficient and flagged for prediction only. "
                "Conventional coefficient hypothesis tests and p-values are blocked from inference tables."
            )

    # Determine Final Status
    if len(blockers) > 0:
        status = ReadinessStatus.BLOCKED
    elif len(limitations) > 0 or len(warnings) > 0:
        status = ReadinessStatus.READY_WITH_LIMITATIONS
    else:
        status = ReadinessStatus.READY

    return ReportReadinessEvaluation(
        status=status,
        blockers=blockers,
        warnings=warnings,
        limitations=limitations,
        compatibility=compat
    )
