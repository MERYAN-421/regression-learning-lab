"""
Compatibility checking and verification rules for academic report generation.
Enforces strict provenance, fingerprint validation, and prevents cross-model or cross-split contamination.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from src.models.model_result import OLSModelResult
from src.comparison.result_models import ExperimentComparisonResult
from src.diagnostics.result_models import DiagnosticSuiteResult
from src.features.result_models import FeatureSelectionResult


@dataclass
class ReportCompatibilityResult:
    """Evaluation of source artifact compatibility for report generation."""
    is_compatible: bool
    reasons: List[str] = field(default_factory=list)  # Blocker reasons
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    dataset_id: str = ""
    dataset_fingerprint: str = ""
    split_fingerprint: str = ""
    target_name: str = ""
    target_unit: Optional[str] = None


from src.reporting.comparison_adapter import adapt_leaderboard_row


def validate_report_artifacts(
    model: Optional[OLSModelResult],
    comparison: Optional[ExperimentComparisonResult] = None,
    diagnostic_suite: Optional[DiagnosticSuiteResult] = None,
    feature_selection_results: Optional[Dict[str, FeatureSelectionResult]] = None,
    expected_dataset_id: Optional[str] = None,
    expected_dataset_fingerprint: Optional[str] = None,
    expected_split_fingerprint: Optional[str] = None,
    expected_target_unit: Optional[str] = None,
    allow_alternative_model: bool = False
) -> ReportCompatibilityResult:
    """
    Validates structural and methodological compatibility among all experimental artifacts
    destined for an academic report.
    """
    reasons = []
    warnings = []
    limitations = []

    if model is None:
        return ReportCompatibilityResult(
            is_compatible=False,
            reasons=["Selected primary OLS model artifact is missing or None."],
            dataset_id=expected_dataset_id or "",
            dataset_fingerprint=expected_dataset_fingerprint or "",
            split_fingerprint=expected_split_fingerprint or ""
        )

    base_ds_id = expected_dataset_id or model.dataset_id
    base_ds_fp = expected_dataset_fingerprint or model.dataset_fingerprint
    base_split_fp = expected_split_fingerprint or model.split_fingerprint
    base_target_name = model.target_name
    base_target_unit = expected_target_unit if expected_target_unit is not None else model.target_unit

    # 1. Check Primary Model Fingerprints
    if base_ds_id and model.dataset_id != base_ds_id:
        reasons.append(
            f"Dataset ID mismatch: Model '{model.model_id}' trained on '{model.dataset_id}', expected '{base_ds_id}'."
        )
    if base_ds_fp and model.dataset_fingerprint != base_ds_fp:
        reasons.append(
            f"Dataset fingerprint mismatch: Model '{model.model_id}' fingerprint '{model.dataset_fingerprint[:8]}' "
            f"does not match expected active dataset fingerprint '{base_ds_fp[:8]}'."
        )
    if base_split_fp and model.split_fingerprint != base_split_fp:
        reasons.append(
            f"Split fingerprint mismatch: Model '{model.model_id}' split fingerprint '{model.split_fingerprint[:8]}' "
            f"does not match expected active split partition '{base_split_fp[:8]}'."
        )

    # 2. Check Comparison Result Compatibility (if present)
    if comparison is not None:
        if comparison.dataset_id != model.dataset_id:
            reasons.append(
                f"Comparison dataset ID '{comparison.dataset_id}' does not match model dataset ID '{model.dataset_id}'."
            )
        if comparison.dataset_fingerprint != model.dataset_fingerprint:
            reasons.append(
                f"Comparison dataset fingerprint '{comparison.dataset_fingerprint[:8]}' does not match "
                f"model dataset fingerprint '{model.dataset_fingerprint[:8]}'."
            )
        if comparison.split_fingerprint != model.split_fingerprint:
            reasons.append(
                f"Comparison split partition '{comparison.split_fingerprint[:8]}' does not match "
                f"model split partition '{model.split_fingerprint[:8]}'."
            )
        if comparison.target_name != model.target_name:
            reasons.append(
                f"Comparison target '{comparison.target_name}' does not match model target '{model.target_name}'."
            )

        # Check Target Unit Mismatch
        if comparison.target_unit != model.target_unit:
            reasons.append(
                f"Target measurement unit mismatch: Model '{model.model_id}' has unit '{model.target_unit}', "
                f"but Comparison '{comparison.comparison_id}' has unit '{comparison.target_unit}'."
            )

        # Verify candidate model membership
        if model.model_id not in comparison.compared_model_ids:
            reasons.append(
                f"Selected model '{model.model_id}' is not among the compared models in comparison '{comparison.comparison_id}'."
            )

        # Verify if candidate matches comparison's recorded candidate
        if comparison.selected_candidate_model_id and comparison.selected_candidate_model_id != model.model_id:
            if not allow_alternative_model:
                reasons.append(
                    f"Selected model '{model.model_id}' does not match comparison candidate "
                    f"'{comparison.selected_candidate_model_id}' in comparison '{comparison.comparison_id}'. "
                    "Set allow_alternative_model=True to analyze a non-candidate comparison model."
                )
            else:
                warnings.append(
                    f"Reporting model '{model.model_id}' differs from the formally selected candidate "
                    f"'{comparison.selected_candidate_model_id}' in comparison '{comparison.comparison_id}'."
                )

        # Validate metric consistency between model and comparison leaderboard
        for raw_row in comparison.leaderboard:
            try:
                row = adapt_leaderboard_row(raw_row, pareto_model_ids=comparison.pareto_model_ids)
            except ValueError as e:
                reasons.append(f"Comparison leaderboard schema error: {str(e)}")
                continue

            if row.model_id == model.model_id:
                if abs(row.val_rmse - model.val_metrics.rmse) > 1e-4:
                    reasons.append(
                        f"Inconsistent Validation RMSE: Model '{model.model_id}' artifact records {model.val_metrics.rmse:.6f}, "
                        f"but Comparison leaderboard records {row.val_rmse:.6f}. Evidence is stale or altered."
                    )
                if abs(row.val_r2 - model.val_metrics.r2) > 1e-4:
                    reasons.append(
                        f"Inconsistent Validation R²: Model '{model.model_id}' artifact records {model.val_metrics.r2:.6f}, "
                        f"but Comparison leaderboard records {row.val_r2:.6f}. Evidence is stale or altered."
                    )
                break

    # 3. Check Diagnostic Suite Compatibility (if present)
    if diagnostic_suite is not None:
        if diagnostic_suite.model_id != model.model_id:
            reasons.append(
                f"Diagnostic Suite '{diagnostic_suite.suite_id}' was computed for model '{diagnostic_suite.model_id}', "
                f"which does not match selected model '{model.model_id}'."
            )
        if diagnostic_suite.dataset_fingerprint != model.dataset_fingerprint:
            reasons.append(
                f"Diagnostic Suite dataset fingerprint '{diagnostic_suite.dataset_fingerprint[:8]}' does not match "
                f"model dataset fingerprint '{model.dataset_fingerprint[:8]}'."
            )
        if diagnostic_suite.split_fingerprint != model.split_fingerprint:
            reasons.append(
                f"Diagnostic Suite split fingerprint '{diagnostic_suite.split_fingerprint[:8]}' does not match "
                f"model split fingerprint '{model.split_fingerprint[:8]}'."
            )
        # Verify individual diagnostic results
        for m_id, d_res in diagnostic_suite.diagnostics.items():
            if d_res.model_id != model.model_id:
                reasons.append(
                    f"Diagnostic result '{m_id}' references model '{d_res.model_id}' instead of '{model.model_id}'."
                )
    else:
        limitations.append("No statistical assumption diagnostic suite is attached for the selected model.")

    # 4. Check Feature Selection Artifacts
    if feature_selection_results:
        for fs_id, fs_res in feature_selection_results.items():
            if fs_res.dataset_fingerprint != model.dataset_fingerprint:
                reasons.append(
                    f"Feature Selection '{fs_id}' dataset fingerprint does not match model."
                )
            if fs_res.split_fingerprint != model.split_fingerprint:
                reasons.append(
                    f"Feature Selection '{fs_id}' split fingerprint does not match model."
                )

    # 5. Methodological Inference Warnings
    if not model.is_valid_for_inference:
        warnings.append(
            f"Model '{model.model_id}' is flagged as invalid for conventional statistical inference (rank deficient). "
            "Report must restrict coefficient hypothesis testing and label inference as unavailable."
        )
    if model.post_selection_inference_warning:
        warnings.append(
            "Model predictors were selected via data-driven feature selection. "
            "Report must state that conventional standard errors and p-values are subject to selection bias."
        )

    is_compatible = (len(reasons) == 0)

    return ReportCompatibilityResult(
        is_compatible=is_compatible,
        reasons=reasons,
        warnings=warnings,
        limitations=limitations,
        dataset_id=model.dataset_id,
        dataset_fingerprint=model.dataset_fingerprint,
        split_fingerprint=model.split_fingerprint,
        target_name=model.target_name,
        target_unit=model.target_unit
    )
