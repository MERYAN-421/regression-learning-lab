"""
Source resolution engine for academic reporting.
Retrieves and links verified experiment artifacts from existing Milestone 1-5 registries and trackers.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.data.models import DatasetMetadata, SplitResult
from src.models.model_result import OLSModelResult
from src.comparison.result_models import ExperimentComparisonResult
from src.diagnostics.result_models import DiagnosticSuiteResult
from src.features.result_models import FeatureSelectionResult
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.experiments.comparison_experiment_tracker import ComparisonExperimentTracker


@dataclass
class ResolvedReportSources:
    """Container for experimental artifacts resolved for report generation."""
    model: Optional[OLSModelResult] = None
    comparison: Optional[ExperimentComparisonResult] = None
    diagnostic_suite: Optional[DiagnosticSuiteResult] = None
    feature_selection_results: Dict[str, FeatureSelectionResult] = field(default_factory=dict)
    all_comparison_models: Dict[str, OLSModelResult] = field(default_factory=dict)
    dataset_metadata: Optional[DatasetMetadata] = None
    split_result: Optional[SplitResult] = None
    resolution_errors: List[str] = field(default_factory=list)
    resolution_warnings: List[str] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.model is not None and len(self.resolution_errors) == 0


class ReportSourceResolver:
    """
    Resolves artifact dependencies for academic reports without mutating trackers or fitting models.
    """

    def __init__(
        self,
        regression_tracker: RegressionExperimentTracker,
        comparison_tracker: Optional[ComparisonExperimentTracker] = None,
        diagnostic_suites: Optional[Dict[str, DiagnosticSuiteResult]] = None,
        feature_selection_tracker: Optional[Any] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        split_result: Optional[SplitResult] = None
    ):
        self.regression_tracker = regression_tracker
        self.comparison_tracker = comparison_tracker
        self.diagnostic_suites = diagnostic_suites or {}
        self.feature_selection_tracker = feature_selection_tracker
        self.dataset_metadata = dataset_metadata
        self.split_result = split_result

    def resolve(
        self,
        selected_model_id: Optional[str] = None,
        comparison_id: Optional[str] = None,
        diagnostic_suite_id: Optional[str] = None
    ) -> ResolvedReportSources:
        """
        Resolves all artifacts required for report generation based on user selection.
        """
        errors = []
        warnings = []
        comparison: Optional[ExperimentComparisonResult] = None
        model: Optional[OLSModelResult] = None
        diag_suite: Optional[DiagnosticSuiteResult] = None
        comparison_models: Dict[str, OLSModelResult] = {}
        fs_results: Dict[str, FeatureSelectionResult] = {}

        # 1. Resolve Comparison if specified
        if comparison_id:
            if self.comparison_tracker is None:
                errors.append("Comparison ID was provided, but no ComparisonExperimentTracker is configured.")
            else:
                comparison = self.comparison_tracker.get_comparison(comparison_id)
                if comparison is None:
                    errors.append(f"Comparison record with ID '{comparison_id}' was not found in tracker.")
                else:
                    # Resolve models participating in comparison
                    for m_id in comparison.compared_model_ids:
                        c_model = self.regression_tracker.get_model(m_id)
                        if c_model is not None:
                            comparison_models[m_id] = c_model
                        else:
                            warnings.append(f"Model '{m_id}' referenced in comparison was not found in tracker.")

                    # If model_id not given, default to comparison's selected candidate
                    if not selected_model_id:
                        selected_model_id = comparison.selected_candidate_model_id

        # 2. Resolve Primary Model
        if not selected_model_id:
            errors.append("No primary OLS model was specified and no candidate model could be inferred.")
        else:
            model = self.regression_tracker.get_model(selected_model_id)
            if model is None:
                errors.append(f"Primary OLS model with ID '{selected_model_id}' was not found in tracker.")

        # 3. Resolve Diagnostic Suite
        if model is not None:
            if diagnostic_suite_id:
                # Explicit diagnostic suite requested: never silently substitute
                if diagnostic_suite_id in self.diagnostic_suites:
                    diag_suite = self.diagnostic_suites[diagnostic_suite_id]
                    if diag_suite.model_id != model.model_id:
                        warnings.append(
                            f"Explicit Diagnostic Suite '{diagnostic_suite_id}' was evaluated for model "
                            f"'{diag_suite.model_id}', not selected model '{model.model_id}'."
                        )
                else:
                    warnings.append(
                        f"Explicit Diagnostic Suite '{diagnostic_suite_id}' was not found in available suites."
                    )
                    diag_suite = None
            elif comparison and model.model_id in comparison.referenced_diagnostic_suite_ids:
                ref_suite_id = comparison.referenced_diagnostic_suite_ids[model.model_id]
                if ref_suite_id:
                    if ref_suite_id in self.diagnostic_suites:
                        diag_suite = self.diagnostic_suites[ref_suite_id]
                    else:
                        warnings.append(
                            f"Diagnostic Suite '{ref_suite_id}' referenced in comparison was not found in available suites."
                        )
                        diag_suite = None
            else:
                # Search available suites matching model ID only when no specific suite was requested/referenced
                for s_id, s_obj in self.diagnostic_suites.items():
                    if s_obj.model_id == model.model_id:
                        diag_suite = s_obj
                        break

            if diag_suite is None and not diagnostic_suite_id and not (comparison and model.model_id in comparison.referenced_diagnostic_suite_ids):
                warnings.append(f"No statistical diagnostic suite resolved for model '{model.model_id}'.")

        # 4. Resolve Feature Selection Results
        if self.feature_selection_tracker is not None:
            if hasattr(self.feature_selection_tracker, "get_all_experiments"):
                all_fs = self.feature_selection_tracker.get_all_experiments()
                if isinstance(all_fs, list):
                    for fs in all_fs:
                        fs_results[fs.experiment_id] = fs
                elif isinstance(all_fs, dict):
                    fs_results.update(all_fs)

        return ResolvedReportSources(
            model=model,
            comparison=comparison,
            diagnostic_suite=diag_suite,
            feature_selection_results=fs_results,
            all_comparison_models=comparison_models,
            dataset_metadata=self.dataset_metadata,
            split_result=self.split_result,
            resolution_errors=errors,
            resolution_warnings=warnings
        )
