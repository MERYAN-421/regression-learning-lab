"""
Report Experiment Tracker.
Maintains an immutable session-level registry of generated AcademicReportResult artifacts.
Invalidates or flags stale reports when the active dataset or split partition changes.
Preserves existing experiment trackers and does not duplicate modeling or diagnostic responsibilities.
"""
from typing import Dict, List, Optional
from src.reporting.result_models import AcademicReportResult


class ReportExperimentTracker:
    """
    Session-level registry for academic laboratory reports.
    Provides indexing by report_id and validation against active dataset/split fingerprints.
    """

    def __init__(self):
        self._reports: Dict[str, AcademicReportResult] = {}

    def add_report(self, report: AcademicReportResult) -> None:
        """Registers an academic report into the tracker."""
        self._reports[report.report_id] = report

    def get_report(self, report_id: str) -> Optional[AcademicReportResult]:
        """Retrieves a report record by its unique deterministic ID."""
        return self._reports.get(report_id)

    def get_all_reports(self) -> List[AcademicReportResult]:
        """Returns all reports in chronological order of generation."""
        return list(self._reports.values())

    def get_valid_reports(
        self,
        current_split_fingerprint: str,
        current_dataset_id: Optional[str] = None,
        current_dataset_fingerprint: Optional[str] = None
    ) -> List[AcademicReportResult]:
        """
        Retrieves reports that match the active dataset and split partition.
        Excludes stale reports generated from earlier splits or different datasets.
        """
        valid = []
        for rep in self._reports.values():
            if rep.split_fingerprint != current_split_fingerprint:
                continue
            if current_dataset_id and rep.dataset_id != current_dataset_id:
                continue
            if current_dataset_fingerprint and rep.dataset_fingerprint != current_dataset_fingerprint:
                continue
            valid.append(rep)
        return valid

    def clear(self) -> None:
        """Clears all report records from the tracker."""
        self._reports.clear()
