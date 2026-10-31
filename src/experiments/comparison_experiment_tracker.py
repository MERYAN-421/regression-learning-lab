"""
Lightweight Comparison Experiment Tracker.
Stores and indexes ExperimentComparisonResult artifacts across user sessions.
Ensures stale comparisons are invalidated upon dataset or split changes.
Does not duplicate model fitting, feature selection, or diagnostic storage responsibilities.
"""
from typing import Dict, List, Optional
from src.comparison.result_models import ExperimentComparisonResult


class ComparisonExperimentTracker:
    """
    Session-level registry for multi-model comparison experiments.
    Maintains reproducibility metadata and candidate model selection rationale.
    """

    def __init__(self):
        self._comparisons: Dict[str, ExperimentComparisonResult] = {}

    def add_comparison(self, comparison: ExperimentComparisonResult) -> None:
        """Register a completed comparison experiment."""
        self._comparisons[comparison.comparison_id] = comparison

    def get_comparison(self, comparison_id: str) -> Optional[ExperimentComparisonResult]:
        """Retrieve a comparison record by its unique deterministic ID."""
        return self._comparisons.get(comparison_id)

    def get_all_comparisons(self) -> List[ExperimentComparisonResult]:
        """Return all stored comparisons."""
        return list(self._comparisons.values())

    def get_valid_comparisons(
        self,
        current_split_fingerprint: str,
        current_dataset_id: Optional[str] = None,
        current_dataset_fingerprint: Optional[str] = None
    ) -> List[ExperimentComparisonResult]:
        """
        Retrieves comparisons that strictly match the active dataset and split partition.
        Invalidates stale comparisons from previous splits or different datasets.
        """
        valid = []
        for comp in self._comparisons.values():
            if comp.split_fingerprint != current_split_fingerprint:
                continue
            if current_dataset_id and comp.dataset_id != current_dataset_id:
                continue
            if current_dataset_fingerprint and comp.dataset_fingerprint != current_dataset_fingerprint:
                continue
            valid.append(comp)
        return valid

    def clear(self) -> None:
        """Clear all comparison records."""
        self._comparisons.clear()
