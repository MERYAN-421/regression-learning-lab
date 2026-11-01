"""
Automated Test Suite — Milestone 6: Group B & Critical Negative Cases 1, 2, 3
Tests:
- Correct field-to-artifact mapping in Evidence Manifest
- Verified, Missing, Incompatible, and Not Applicable states
- Absolute refusal to fabricate statistics when artifacts are missing
- Manifest JSON serialization with safe numerics
- Critical Negative Case 1: Wrong Diagnostic Suite
- Critical Negative Case 2: Altered Validation Metric
- Critical Negative Case 3: Missing FeatureSelectionResult
"""
import json
import pytest
import numpy as np
import pandas as pd

from src.reporting.result_models import (
    EvidenceRecord,
    EvidenceManifest,
    EvidenceStatus,
    ReportTemplateType,
    ReadinessStatus,
    safe_serialize_numeric
)
from src.reporting.evidence_manifest import EvidenceManifestBuilder
from src.reporting.source_resolver import ResolvedReportSources, ReportSourceResolver
from src.reporting.compatibility import validate_report_artifacts
from src.reporting.readiness import evaluate_report_readiness
from src.reporting.report_runner import run_academic_report
from src.diagnostics.result_models import DiagnosticSuiteResult, DiagnosticResult, DiagnosticDecision
from src.features.result_models import FeatureSelectionResult, OriginalFeatureMapping
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.experiments.comparison_experiment_tracker import ComparisonExperimentTracker

from tests.test_reporting_sources import create_dummy_model, create_dummy_comparison


class TestEvidenceManifestEngine:
    """Group B: Evidence Manifest verification tests."""

    def test_evidence_manifest_exact_field_mapping(self):
        m1 = create_dummy_model("m1", val_rmse=1.2345, val_r2=0.8765)
        sources = ResolvedReportSources(model=m1)

        builder = EvidenceManifestBuilder(report_id="rep_test")
        manifest = builder.build_from_sources(sources)

        assert manifest.report_id == "rep_test"
        assert manifest.verified_count > 0

        # Check val_rmse record
        rmse_rec = manifest.get_record("val_rmse")
        assert rmse_rec is not None
        assert rmse_rec.evidence_status == EvidenceStatus.VERIFIED.value
        assert abs(rmse_rec.observed_value - 1.2345) < 1e-6
        assert rmse_rec.display_value == "1.2345"
        assert rmse_rec.analysis_data_scope == "validation"
        assert rmse_rec.source_artifact_id == "m1"
        assert rmse_rec.source_field_path == "val_metrics.rmse"

        # Check train_r2 record
        r2_rec = manifest.get_record("train_r2")
        assert r2_rec is not None
        assert abs(r2_rec.observed_value - 0.88) < 1e-4

    def test_no_fabricated_statistics_when_diagnostic_missing(self):
        m1 = create_dummy_model("m1")
        # No diagnostic suite attached
        sources = ResolvedReportSources(model=m1, diagnostic_suite=None)

        builder = EvidenceManifestBuilder(report_id="rep_test")
        manifest = builder.build_from_sources(sources)

        # Confirm no fake diagnostic records exist in manifest
        dw_rec = manifest.get_record("diag_stat_durbin_watson")
        assert dw_rec is None, "Fabricated Durbin-Watson statistic was improperly created!"
        vif_rec = manifest.get_record("diag_stat_vif")
        assert vif_rec is None, "Fabricated VIF statistic was improperly created!"

    def test_safe_serialization_of_non_finite_values(self):
        raw_dict = {
            "valid_float": 3.14,
            "nan_val": float("nan"),
            "inf_val": float("inf"),
            "neg_inf": float("-inf"),
            "int_val": 42,
            "np_int": np.int64(10),
            "np_float": np.float64(2.718)
        }
        cleaned = safe_serialize_numeric(raw_dict)
        json_str = json.dumps(cleaned)
        assert json_str is not None
        assert cleaned["nan_val"] is None
        assert cleaned["inf_val"] is None
        assert cleaned["neg_inf"] is None
        assert cleaned["np_int"] == 10
        assert abs(cleaned["np_float"] - 2.718) < 1e-4


class TestCriticalNegativeCases:
    """Critical Negative Cases 1, 2, and 3."""

    def test_critical_negative_case_1_wrong_diagnostic_suite(self):
        """
        Case 1: Attach a DiagnosticSuite belonging to a different OLS model.
        Expected behavior: Block or exclude the invalid evidence with a clear explanation.
        Never report the unrelated diagnostic statistic as valid.
        """
        m1 = create_dummy_model("m1")
        # Diagnostic suite belongs to "m_different"
        d_suite = DiagnosticSuiteResult(
            suite_id="suite_other",
            model_id="m_different",
            dataset_id="ds_test",
            dataset_fingerprint="ds_fp_123456",
            split_fingerprint="split_fp_123456"
        )
        d_res = DiagnosticResult(
            diagnostic_id="diag_1",
            model_id="m_different",
            dataset_id="ds_test",
            dataset_fingerprint="ds_fp_123456",
            split_fingerprint="split_fp_123456",
            diagnostic_type="independence",
            method_id="durbin_watson",
            method_name="Durbin-Watson",
            statistic_name="DW",
            statistic_value=1.95
        )
        d_suite.add_result(d_res)

        compat = validate_report_artifacts(model=m1, diagnostic_suite=d_suite)
        assert not compat.is_compatible
        assert any("was computed for model 'm_different'" in r for r in compat.reasons)

        sources = ResolvedReportSources(model=m1, diagnostic_suite=d_suite)
        readiness = evaluate_report_readiness(sources=sources)
        assert readiness.is_blocked
        assert any("m_different" in b for b in readiness.blockers)

    def test_critical_negative_case_2_altered_validation_metric(self):
        """
        Case 2: Inconsistent comparison and model Validation metrics.
        Expected behavior: Identify inconsistent or stale evidence rather than silently choosing one.
        """
        # Model records val_rmse = 1.5
        m1 = create_dummy_model("m1", val_rmse=1.5)
        # Comparison leaderboard erroneously records val_rmse = 0.9 (tampered or stale)
        comp = create_dummy_comparison("c1", model_ids=["m1"], candidate_id="m1", val_rmse=0.9)

        compat = validate_report_artifacts(model=m1, comparison=comp)
        assert not compat.is_compatible
        assert any("Inconsistent Validation RMSE" in r for r in compat.reasons)
        assert any("Evidence is stale or altered" in r for r in compat.reasons)

        sources = ResolvedReportSources(model=m1, comparison=comp)
        readiness = evaluate_report_readiness(sources=sources)
        assert readiness.is_blocked
        assert any("Inconsistent Validation RMSE" in b for b in readiness.blockers)

    def test_critical_negative_case_3_missing_feature_selection_result(self):
        """
        Case 3: Remove the referenced feature selection artifact.
        Expected behavior: Report the missing Feature Selection details.
        Do not invent selection scores, K, or CV statistics.
        Allow generation with limitations if other core evidence remains valid.
        """
        # Model references "fs_lasso_exp_123", but it is NOT in feature_selection_results
        m1 = create_dummy_model("m1")
        m1.feature_source_id = "fs_lasso_exp_123"
        m1.feature_source_label = "Lasso Regularized Selection"

        sources = ResolvedReportSources(model=m1, feature_selection_results={})
        readiness = evaluate_report_readiness(sources=sources)

        # Must not be blocked (it's optional details), but must be READY_WITH_LIMITATIONS
        assert readiness.status == ReadinessStatus.READY_WITH_LIMITATIONS
        assert any("fs_lasso_exp_123" in lim for lim in readiness.limitations)

        # Generate report and confirm missing state is recorded
        report = run_academic_report(sources=sources)
        assert report.readiness_status == ReadinessStatus.READY_WITH_LIMITATIONS.value

        manifest = report.evidence_manifest
        fs_rec = manifest.get_record("fs_method_name")
        assert fs_rec is not None
        assert fs_rec.evidence_status == EvidenceStatus.MISSING.value
        assert fs_rec.observed_value is None
        assert fs_rec.display_value == "Unavailable"

        # Check section 3 content disclosure
        sec3 = next(s for s in report.report_sections if s.section_id == "sec_3_eda_selection")
        assert any("was not found in the active session registry" in b for b in sec3.content_blocks)
