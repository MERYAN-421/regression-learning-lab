"""
Automated Test Suite — Milestone 6: Group C & Critical Negative Case 5
Tests:
- All 9 sections generated for Template A (Full Academic Report)
- Abbreviated structure for Template B (Executive Summary)
- Specialized structure for Template C (Statistical Diagnostics Report)
- User-toggled section omission
- Correct coefficient table generation
- Preservation of candidate selection rationale
- Critical Negative Case 5: Prediction-Only / Rank-Deficient Model guards
"""
import pytest
import numpy as np
import pandas as pd

from src.reporting.result_models import (
    ReportTemplateType,
    SectionID,
    AuthorContent,
    ReadinessStatus
)
from src.reporting.source_resolver import ResolvedReportSources
from src.reporting.report_runner import run_academic_report
from src.diagnostics.result_models import DiagnosticSuiteResult, DiagnosticResult, DiagnosticDecision

from tests.test_reporting_sources import create_dummy_model, create_dummy_comparison


class TestAcademicSections:
    """Group C: Report sections & templates tests."""

    def test_full_academic_report_has_all_nine_sections(self):
        m1 = create_dummy_model("m1")
        comp = create_dummy_comparison("c1", model_ids=["m1"], candidate_id="m1")

        # Create dummy diagnostic suite
        d_suite = DiagnosticSuiteResult(
            suite_id="s1",
            model_id="m1",
            dataset_id="ds_test",
            dataset_fingerprint="ds_fp_123456",
            split_fingerprint="split_fp_123456"
        )
        d_suite.add_result(DiagnosticResult(
            diagnostic_id="d1", model_id="m1", dataset_id="ds_test",
            dataset_fingerprint="ds_fp_123456", split_fingerprint="split_fp_123456",
            diagnostic_type="independence", method_id="durbin_watson",
            method_name="Durbin-Watson", statistic_name="DW", statistic_value=1.98
        ))

        sources = ResolvedReportSources(model=m1, comparison=comp, diagnostic_suite=d_suite)
        report = run_academic_report(sources=sources, template_type=ReportTemplateType.FULL_ACADEMIC)

        assert report.readiness_status in ("ready", "ready_with_limitations")
        sec_ids = [s.section_id for s in report.report_sections]

        assert len(sec_ids) == 9
        assert sec_ids == [
            SectionID.SEC_EXEC_SUMMARY.value,
            SectionID.SEC_DATASET.value,
            SectionID.SEC_EDA_SELECTION.value,
            SectionID.SEC_OLS_METHODOLOGY.value,
            SectionID.SEC_COEFFICIENTS.value,
            SectionID.SEC_DIAGNOSTICS.value,
            SectionID.SEC_COMPARISON.value,
            SectionID.SEC_DISCUSSION.value,
            SectionID.SEC_REPRODUCIBILITY.value
        ]

    def test_executive_summary_template_structure(self):
        m1 = create_dummy_model("m1")
        comp = create_dummy_comparison("c1", model_ids=["m1"], candidate_id="m1")
        sources = ResolvedReportSources(model=m1, comparison=comp)

        report = run_academic_report(sources=sources, template_type=ReportTemplateType.EXECUTIVE_SUMMARY)
        sec_ids = [s.section_id for s in report.report_sections]

        assert SectionID.SEC_EXEC_SUMMARY.value in sec_ids
        assert SectionID.SEC_COEFFICIENTS.value in sec_ids
        assert SectionID.SEC_COMPARISON.value in sec_ids
        # Detailed diagnostics and methodology are excluded from Executive Summary
        assert SectionID.SEC_DIAGNOSTICS.value not in sec_ids
        assert SectionID.SEC_OLS_METHODOLOGY.value not in sec_ids

    def test_statistical_diagnostics_template_structure(self):
        m1 = create_dummy_model("m1")
        d_suite = DiagnosticSuiteResult(
            suite_id="s1", model_id="m1", dataset_id="ds_test",
            dataset_fingerprint="ds_fp_123456", split_fingerprint="split_fp_123456"
        )
        d_suite.add_result(DiagnosticResult(
            diagnostic_id="d1", model_id="m1", dataset_id="ds_test",
            dataset_fingerprint="ds_fp_123456", split_fingerprint="split_fp_123456",
            diagnostic_type="linearity", method_id="ramsey_reset",
            method_name="Ramsey RESET", statistic_name="F", statistic_value=1.2, p_value=0.28
        ))
        sources = ResolvedReportSources(model=m1, diagnostic_suite=d_suite)

        report = run_academic_report(sources=sources, template_type=ReportTemplateType.STATISTICAL_DIAGNOSTICS)
        sec_ids = [s.section_id for s in report.report_sections]

        assert SectionID.SEC_DIAGNOSTICS.value in sec_ids
        assert SectionID.SEC_OLS_METHODOLOGY.value in sec_ids
        # Multi-model comparison is omitted in specialized diagnostics report
        assert SectionID.SEC_COMPARISON.value not in sec_ids

    def test_candidate_rationale_preservation(self):
        m1 = create_dummy_model("m1")
        custom_rationale = "Candidate chosen due to optimal trade-off on Pareto frontier and stable residual variance."
        comp = create_dummy_comparison("c1", model_ids=["m1"], candidate_id="m1")
        comp.candidate_selection_rationale = custom_rationale

        sources = ResolvedReportSources(model=m1, comparison=comp)
        report = run_academic_report(sources=sources)

        sec7 = next(s for s in report.report_sections if s.section_id == SectionID.SEC_COMPARISON.value)
        assert any(custom_rationale in block for block in sec7.content_blocks)

    def test_critical_negative_case_5_prediction_only_model(self):
        """
        Case 5: Use a rank-deficient model explicitly configured for prediction only.
        Expected behavior: Prediction results may appear.
        Conventional coefficient significance claims must be blocked.
        """
        m_rank_def = create_dummy_model("m_rank_def", is_valid_for_inference=False)
        m_rank_def.design_matrix_metadata.is_full_rank = False
        m_rank_def.design_matrix_metadata.matrix_rank = 1

        sources = ResolvedReportSources(model=m_rank_def)
        report = run_academic_report(sources=sources)

        # Confirm prediction metrics (RMSE, MAE, R2) are present
        manifest = report.evidence_manifest
        val_rmse_rec = manifest.get_record("val_rmse")
        assert val_rmse_rec is not None
        assert val_rmse_rec.evidence_status == "verified"
        assert abs(val_rmse_rec.observed_value - 1.5) < 1e-4

        # Confirm inferential standard error and p-value records are marked NOT_APPLICABLE
        se_rec = manifest.get_record("stderr_x1")
        assert se_rec is not None
        assert se_rec.evidence_status == "not_applicable"
        assert "rank-deficient" in se_rec.validation_reason

        p_rec = manifest.get_record("pvalue_x1")
        assert p_rec is not None
        assert p_rec.evidence_status == "not_applicable"

        # Inspect Section 5 table: ensure p-values and SE display "N/A (Prediction Only)"
        sec5 = next(s for s in report.report_sections if s.section_id == SectionID.SEC_COEFFICIENTS.value)
        coef_tbl = next(t for t in sec5.tables if t["table_id"] == "tbl_coefficients")
        row_x1 = next(r for r in coef_tbl["rows"] if r["Regressor"] == "x1")
        assert "N/A" in row_x1["p-value"]
        assert "N/A" in row_x1["Std. Error"]
