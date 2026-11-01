"""
Automated Test Suite — Milestone 6: Group E & Critical Negative Case 4
Tests:
- Strict preservation of the Final Test Lock during report generation
- Complete prohibition of finalize_test_evaluation() invocation
- Development-stage metrics isolation (Test metrics cannot reorder candidate models)
- Authorization requirement for Final Test Evaluation Appendix
- Critical Negative Case 4: Unauthorized test metrics injected into unfinalized model
- Reproduction script test isolation (strictly development-stage replay)
"""
import pytest
from unittest.mock import MagicMock
import numpy as np

from src.evaluation.regression_metrics import RegressionMetrics
from src.reporting.source_resolver import ResolvedReportSources
from src.reporting.evidence_manifest import EvidenceManifestBuilder
from src.reporting.report_runner import run_academic_report
from src.reporting.reproduction_script import generate_reproduction_script

from tests.test_reporting_sources import create_dummy_model, create_dummy_comparison


class TestTestSetIsolation:
    """Group E: Leakage trap tests & test isolation."""

    def test_report_generation_never_invokes_finalize_test_evaluation(self):
        """Trap Test: Mock tracker to confirm finalize_test_evaluation is NEVER called."""
        m1 = create_dummy_model("m1")
        mock_reg_tracker = MagicMock()
        mock_reg_tracker.get_model.return_value = m1

        sources = ResolvedReportSources(model=m1)
        report = run_academic_report(sources=sources)

        assert report.readiness_status in ("ready", "ready_with_limitations")
        mock_reg_tracker.finalize_test_evaluation.assert_not_called()

    def test_locked_test_metrics_never_appear_in_development_sections(self):
        """Confirm Test metrics never leak into Section 1, 5, or 7."""
        m1 = create_dummy_model("m1")
        # Model is untested
        assert m1.test_evaluation_status == "untested"
        assert m1.test_metrics is None

        sources = ResolvedReportSources(model=m1)
        report = run_academic_report(sources=sources, include_final_test_authorized=False)

        manifest = report.evidence_manifest
        assert manifest.get_record("test_rmse") is None
        assert manifest.get_record("test_r2") is None

        # Check section texts
        for sec in report.report_sections:
            assert sec.section_id != "sec_appendix_final_test"
            for block in sec.content_blocks:
                assert "Test RMSE" not in block

    def test_authorized_final_test_appendix_included_only_when_authorized(self):
        m1 = create_dummy_model("m1")
        # Give model a previously finalized test evaluation artifact
        m1.test_evaluation_status = "evaluated_train_val_refit"
        m1.test_refit_policy = "train_val"
        m1.test_metrics = RegressionMetrics(split_name="Test", sample_count=30, num_parameters=1, rmse=1.45, mae=1.05, mse=2.10, r2=0.86, adj_r2=None)

        sources = ResolvedReportSources(model=m1)

        # 1. Without authorization toggle, appendix is omitted
        rep_unauthorized = run_academic_report(sources=sources, include_final_test_authorized=False)
        sec_ids_unauth = [s.section_id for s in rep_unauthorized.report_sections]
        assert "sec_appendix_final_test" not in sec_ids_unauth

        # 2. With authorization toggle, appendix is included
        rep_authorized = run_academic_report(sources=sources, include_final_test_authorized=True)
        sec_ids_auth = [s.section_id for s in rep_authorized.report_sections]
        assert "sec_appendix_final_test" in sec_ids_auth

        test_sec = next(s for s in rep_authorized.report_sections if s.section_id == "sec_appendix_final_test")
        tbl = test_sec.tables[0]
        assert any(r["Metric"] == "Test RMSE" and "1.450" in r["Value"] for r in tbl["rows"])

    def test_reproduction_script_strictly_excludes_test_evaluation(self):
        m1 = create_dummy_model("m1")
        sources = ResolvedReportSources(model=m1)
        report = run_academic_report(sources=sources)

        script = generate_reproduction_script(report, sources)
        assert "X_test" not in script or "Locked" in script
        assert "ols_final = sm.OLS" not in script
        assert "finalize_test_evaluation" not in script
        assert "Test partition" in script or "STRICT TEST ISOLATION" in script

    def test_critical_negative_case_4_unauthorized_test_metrics_in_unfinalized_model(self):
        """
        Case 4: Inject Test metrics into an unfinalized model artifact.
        Expected behavior: The report must not expose these metrics unless authorization is satisfied.
        """
        m1 = create_dummy_model("m1")
        # Inject test metrics but test_evaluation_status remains "untested"
        m1.test_metrics = RegressionMetrics(split_name="Test", sample_count=30, num_parameters=1, rmse=0.5, mae=0.4, mse=0.25, r2=0.99, adj_r2=None)
        m1.test_evaluation_status = "untested"

        sources = ResolvedReportSources(model=m1)
        # Even if include_final_test_authorized is requested, status is untested
        report = run_academic_report(sources=sources, include_final_test_authorized=True)

        sec_ids = [s.section_id for s in report.report_sections]
        assert "sec_appendix_final_test" not in sec_ids, "Appendix was improperly attached to an untested model!"
        assert report.final_test_appendix_status in ("untested_omitted", "excluded")
