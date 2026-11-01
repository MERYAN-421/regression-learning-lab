"""
Automated Test Suite — Milestone 6: Group A & Group H
Tests:
- Source resolution (Valid Model ID, Missing Model ID, Valid Comparison ID, Invalid Comparison ID)
- Dataset & split fingerprint validation
- Target name & target unit validation
- Stale artifact invalidation
- Deterministic Report ID computation
- Report Experiment Tracker history and immutability
"""
import pytest
import numpy as np
import pandas as pd

from src.data.models import DatasetMetadata, SplitResult
from src.evaluation.regression_metrics import RegressionMetrics
from src.models.design_matrix import DesignMatrixMetadata
from src.models.model_result import OLSModelResult
from src.comparison.result_models import ExperimentComparisonResult, RankingPolicy
from src.diagnostics.result_models import DiagnosticSuiteResult
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.experiments.comparison_experiment_tracker import ComparisonExperimentTracker
from src.experiments.report_experiment_tracker import ReportExperimentTracker

from src.reporting.source_resolver import ReportSourceResolver, ResolvedReportSources
from src.reporting.compatibility import validate_report_artifacts
from src.reporting.comparison_adapter import adapt_leaderboard_row, adapt_comparison_leaderboard
from src.reporting.readiness import evaluate_report_readiness
from src.reporting.result_models import AcademicReportResult, ReportTemplateType, ReadinessStatus
from src.reporting.report_runner import compute_deterministic_report_id, run_academic_report


def create_dummy_model(
    model_id: str = "ols_base",
    dataset_id: str = "ds_test",
    dataset_fp: str = "ds_fp_123456",
    split_fp: str = "split_fp_123456",
    target_name: str = "y",
    target_unit: str = "units",
    val_rmse: float = 1.5,
    val_r2: float = 0.85,
    is_valid_for_inference: bool = True
) -> OLSModelResult:
    train_m = RegressionMetrics(split_name="Train", sample_count=100, num_parameters=1, rmse=1.2, mae=0.9, mse=1.44, r2=0.88, adj_r2=0.87)
    val_m = RegressionMetrics(split_name="Validation", sample_count=30, num_parameters=1, rmse=val_rmse, mae=1.1, mse=val_rmse**2, r2=val_r2, adj_r2=None)
    coef_df = pd.DataFrame({
        "coef": [2.5, 0.75],
        "std_err": [0.2, 0.1],
        "t_stat": [12.5, 7.5],
        "p_value": [0.0001, 0.0002],
        "ci_lower": [2.1, 0.55],
        "ci_upper": [2.9, 0.95]
    }, index=["const", "x1"])

    d_meta = DesignMatrixMetadata(
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fp,
        split_fingerprint=split_fp,
        selected_original_features=["x1"],
        encoded_feature_names=["const", "x1"],
        numeric_features=["x1"],
        categorical_features=[],
        reference_categories={},
        original_to_encoded_mapping={"x1": ["x1"]},
        imputed_numeric_values={},
        imputed_categorical_values={},
        scaling_applied=False,
        scaling_params=None,
        fit_intercept=True,
        matrix_rank=2,
        num_parameters=2,
        residual_degrees_of_freedom=98,
        condition_number=12.4,
        is_full_rank=True,
        is_valid_for_inference=is_valid_for_inference,
        collinear_columns=[]
    )

    return OLSModelResult(
        model_id=model_id,
        model_name=f"Model {model_id}",
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fp,
        split_fingerprint=split_fp,
        feature_source_id="all_features",
        feature_source_label="All Features",
        selected_original_features=["x1"],
        design_matrix_metadata=d_meta,
        fit_intercept=True,
        scale_numeric=False,
        coef_table=coef_df,
        equation_general="y = b0 + b1*x1",
        equation_fitted="y = 2.5 + 0.75*x1",
        equation_text="y = 2.5 + 0.75*x1",
        train_metrics=train_m,
        val_metrics=val_m,
        test_metrics=None,
        r2=0.88,
        adj_r2=0.87,
        f_statistic=56.2,
        f_pvalue=0.00001,
        log_likelihood=-120.5,
        aic=245.0,
        bic=250.2,
        rss=144.0,
        df_resid=98,
        df_model=1,
        post_selection_inference_warning=False,
        is_valid_for_inference=is_valid_for_inference,
        target_name=target_name,
        target_unit=target_unit
    )


def create_dummy_comparison(
    comparison_id: str = "comp_test_1",
    model_ids: List[str] = None,
    candidate_id: str = "ols_base",
    dataset_id: str = "ds_test",
    dataset_fp: str = "ds_fp_123456",
    split_fp: str = "split_fp_123456",
    target_name: str = "y",
    target_unit: str = "units",
    val_rmse: float = 1.5,
    val_r2: float = 0.85
) -> ExperimentComparisonResult:
    if model_ids is None:
        model_ids = ["ols_base", "ols_alt"]

    lb = [
        {"model_id": candidate_id, "model_name": f"Model {candidate_id}", "val_rmse": val_rmse, "val_r2": val_r2, "parameter_count": 2, "generalization_gap": 0.3, "is_pareto_efficient": True},
        {"model_id": "ols_alt", "model_name": "Model ols_alt", "val_rmse": val_rmse + 0.5, "val_r2": val_r2 - 0.05, "parameter_count": 3, "generalization_gap": 0.4, "is_pareto_efficient": False}
    ]

    return ExperimentComparisonResult(
        comparison_id=comparison_id,
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fp,
        split_fingerprint=split_fp,
        target_name=target_name,
        target_unit=target_unit,
        compared_model_ids=model_ids,
        referenced_feature_selection_ids={m: None for m in model_ids},
        referenced_diagnostic_suite_ids={m: None for m in model_ids},
        leaderboard=lb,
        selected_candidate_model_id=candidate_id,
        candidate_selection_rationale="Achieved lowest validation RMSE with parsimonious design."
    )


class TestReportSourceResolution:
    """Group A: Source resolution & compatibility tests."""

    def test_valid_model_id_resolution(self):
        reg_tracker = RegressionExperimentTracker()
        model = create_dummy_model("m1")
        reg_tracker.add_model(model)

        resolver = ReportSourceResolver(regression_tracker=reg_tracker)
        sources = resolver.resolve(selected_model_id="m1")

        assert sources.is_resolved
        assert sources.model.model_id == "m1"
        assert len(sources.resolution_errors) == 0

    def test_missing_model_id_resolution(self):
        reg_tracker = RegressionExperimentTracker()
        resolver = ReportSourceResolver(regression_tracker=reg_tracker)
        sources = resolver.resolve(selected_model_id="nonexistent_model")

        assert not sources.is_resolved
        assert any("nonexistent_model" in err for err in sources.resolution_errors)

    def test_valid_comparison_id_resolution(self):
        reg_tracker = RegressionExperimentTracker()
        comp_tracker = ComparisonExperimentTracker()

        m1 = create_dummy_model("m1")
        m2 = create_dummy_model("m2")
        reg_tracker.add_model(m1)
        reg_tracker.add_model(m2)

        comp = create_dummy_comparison("comp_1", model_ids=["m1", "m2"], candidate_id="m1")
        comp_tracker.add_comparison(comp)

        resolver = ReportSourceResolver(
            regression_tracker=reg_tracker,
            comparison_tracker=comp_tracker
        )
        sources = resolver.resolve(comparison_id="comp_1")

        assert sources.is_resolved
        assert sources.comparison.comparison_id == "comp_1"
        assert sources.model.model_id == "m1"  # defaults to comparison's candidate
        assert "m1" in sources.all_comparison_models
        assert "m2" in sources.all_comparison_models

    def test_invalid_comparison_id_resolution(self):
        comp_tracker = ComparisonExperimentTracker()
        reg_tracker = RegressionExperimentTracker()
        resolver = ReportSourceResolver(regression_tracker=reg_tracker, comparison_tracker=comp_tracker)
        sources = resolver.resolve(comparison_id="comp_missing")

        assert not sources.is_resolved
        assert any("comp_missing" in err for err in sources.resolution_errors)

    def test_dataset_fingerprint_mismatch(self):
        m1 = create_dummy_model("m1", dataset_fp="fp_original")
        compat = validate_report_artifacts(
            model=m1,
            expected_dataset_fingerprint="fp_different"
        )
        assert not compat.is_compatible
        assert any("Dataset fingerprint mismatch" in r for r in compat.reasons)

    def test_split_fingerprint_mismatch(self):
        m1 = create_dummy_model("m1", split_fp="split_orig")
        compat = validate_report_artifacts(
            model=m1,
            expected_split_fingerprint="split_diff"
        )
        assert not compat.is_compatible
        assert any("Split fingerprint mismatch" in r for r in compat.reasons)

    def test_target_measurement_unit_mismatch(self):
        m1 = create_dummy_model("m1", target_unit="USD")
        comp = create_dummy_comparison("c1", model_ids=["m1"], candidate_id="m1", target_unit="EUR")
        compat = validate_report_artifacts(model=m1, comparison=comp)

        assert not compat.is_compatible
        assert any("Target measurement unit mismatch" in r for r in compat.reasons)

    def test_candidate_not_in_comparison(self):
        m1 = create_dummy_model("m_external")
        comp = create_dummy_comparison("c1", model_ids=["m_internal_1", "m_internal_2"])
        compat = validate_report_artifacts(model=m1, comparison=comp)

        assert not compat.is_compatible
        assert any("not among the compared models" in r for r in compat.reasons)

    def test_candidate_mismatch_blocks_without_override(self):
        m1 = create_dummy_model("m1")
        comp = create_dummy_comparison("c1", model_ids=["m1", "m2"], candidate_id="m2")
        compat = validate_report_artifacts(model=m1, comparison=comp, allow_alternative_model=False)

        assert not compat.is_compatible
        assert any("does not match comparison candidate" in r for r in compat.reasons)

        # When explicitly authorized, it warns instead of blocking
        compat_allowed = validate_report_artifacts(model=m1, comparison=comp, allow_alternative_model=True)
        assert compat_allowed.is_compatible
        assert any("differs from the formally selected candidate" in w for w in compat_allowed.warnings)

    def test_deliberately_altered_comparison_val_rmse_blocks_report(self):
        m1 = create_dummy_model("m1", val_rmse=1.2345, val_r2=0.8500)
        # Create comparison leaderboard with display-style keys matching Milestone 5
        lb = [{
            "Rank": 1,
            "Model ID": "m1",
            "Feature Source": "All Features",
            "Original Features (K)": 5,
            "Parameters (p)": 6,
            "Val RMSE": 2.5000,  # Deliberately altered from 1.2345!
            "Train RMSE": 1.1000,
            "Gen. Gap": 1.4000,
            "Val R²": 0.8500,
            "Train R²": 0.8800,
            "Pareto Status": "★ Optimal (Pareto)"
        }]
        comp = create_dummy_comparison("c1", model_ids=["m1"], candidate_id="m1")
        comp.leaderboard = lb

        compat = validate_report_artifacts(model=m1, comparison=comp)
        assert not compat.is_compatible
        assert any("Inconsistent Validation RMSE" in r for r in compat.reasons)

        # Confirm report readiness blocks report generation
        sources = ResolvedReportSources(model=m1, comparison=comp)
        rep = run_academic_report(sources=sources)
        assert rep.readiness_status == "blocked"
        assert any("Inconsistent Validation RMSE" in b for b in rep.readiness_reasons)

    def test_missing_leaderboard_keys_raise_error_never_fabricate_zero(self):
        incomplete_row = {
            "Model ID": "m1",
            # Val RMSE and Val R² are completely missing
            "Parameters (p)": 5
        }
        with pytest.raises(ValueError) as excinfo:
            adapt_leaderboard_row(incomplete_row)
        assert "missing mandatory Validation RMSE" in str(excinfo.value)

    def test_missing_or_malformed_parameter_count_raises_error(self):
        # 1. Missing parameter count completely
        missing_p_row = {
            "Model ID": "m1",
            "Val RMSE": 1.25,
            "Val R²": 0.80
            # Parameters (p) missing
        }
        with pytest.raises(ValueError) as exc:
            adapt_leaderboard_row(missing_p_row)
        assert "missing mandatory parameter count (p)" in str(exc.value)

        # 2. Non-positive parameter count (p <= 0)
        zero_p_row = {
            "Model ID": "m1",
            "Val RMSE": 1.25,
            "Val R²": 0.80,
            "Parameters (p)": 0
        }
        with pytest.raises(ValueError) as exc:
            adapt_leaderboard_row(zero_p_row)
        assert "non-positive parameter count" in str(exc.value)

        # 3. Malformed string parameter count
        string_p_row = {
            "Model ID": "m1",
            "Val RMSE": 1.25,
            "Val R²": 0.80,
            "Parameters (p)": "five_params"
        }
        with pytest.raises(ValueError) as exc:
            adapt_leaderboard_row(string_p_row)
        assert "non-numeric or invalid parameter count" in str(exc.value)

    def test_malformed_original_feature_count_raises_error(self):
        # Negative feature count
        negative_k_row = {
            "Model ID": "m1",
            "Val RMSE": 1.25,
            "Val R²": 0.80,
            "Parameters (p)": 4,
            "Original Features (K)": -2
        }
        with pytest.raises(ValueError) as exc:
            adapt_leaderboard_row(negative_k_row)
        assert "negative original feature count" in str(exc.value)

        # Non-numeric string feature count
        malformed_k_row = {
            "Model ID": "m1",
            "Val RMSE": 1.25,
            "Val R²": 0.80,
            "Parameters (p)": 4,
            "Original Features (K)": "three_features"
        }
        with pytest.raises(ValueError) as exc:
            adapt_leaderboard_row(malformed_k_row)
        assert "non-numeric or invalid original feature count" in str(exc.value)

    def test_optional_complexity_and_metrics_allow_explicit_none_without_fabrication(self):
        row_with_unavailables = {
            "Model ID": "m1",
            "Val RMSE": 1.50,
            "Val R²": 0.75,
            "Parameters (p)": 3,
            "Original Features (K)": "Unavailable",
            "Train RMSE": "Missing",
            "Gen. Gap": "N/A"
        }
        adapted = adapt_leaderboard_row(row_with_unavailables)
        # Must be None, never fabricated zero!
        assert adapted.original_features_k is None
        assert adapted.train_rmse is None
        assert adapted.generalization_gap is None
        assert adapted.val_rmse == 1.50
        assert adapted.val_r2 == 0.75
        assert adapted.parameter_count == 3

    def test_explicit_diagnostic_suite_id_never_silently_replaced(self):
        reg_tracker = RegressionExperimentTracker()
        m1 = create_dummy_model("m1")
        reg_tracker.add_model(m1)

        diag_suite_real = DiagnosticSuiteResult(
            suite_id="diag_suite_real",
            model_id="m1",
            dataset_id=m1.dataset_id,
            dataset_fingerprint=m1.dataset_fingerprint,
            split_fingerprint=m1.split_fingerprint,
            diagnostics={}
        )
        suites = {"diag_suite_real": diag_suite_real}

        resolver = ReportSourceResolver(regression_tracker=reg_tracker, diagnostic_suites=suites)
        # Request a non-existent explicit suite ID
        sources = resolver.resolve(selected_model_id="m1", diagnostic_suite_id="nonexistent_suite_id")

        # Crucial test: It must NOT silently substitute "diag_suite_real"!
        assert sources.diagnostic_suite is None
        assert any("nonexistent_suite_id" in w for w in sources.resolution_warnings)


class TestReportTrackingAndIdentity:
    """Group H: Report tracking, immutability, and deterministic ID tests."""

    def test_deterministic_report_id(self):
        id1 = compute_deterministic_report_id(
            model_id="m1",
            dataset_fingerprint="ds_fp",
            split_fingerprint="split_fp",
            template_type="full_academic",
            report_title="Title A",
            comparison_id="comp_1",
            author_text="Author text A"
        )
        id2 = compute_deterministic_report_id(
            model_id="m1",
            dataset_fingerprint="ds_fp",
            split_fingerprint="split_fp",
            template_type="full_academic",
            report_title="Title A",
            comparison_id="comp_1",
            author_text="Author text A"
        )
        assert id1 == id2
        assert id1.startswith("rep_")

    def test_distinct_report_ids_for_changed_title_and_sections(self):
        id_title_a = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "full_academic", report_title="Title A")
        id_title_b = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "full_academic", report_title="Title B")
        assert id_title_a != id_title_b

        id_sec_1 = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "full_academic", enabled_section_ids=["s1", "s2"])
        id_sec_2 = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "full_academic", enabled_section_ids=["s1", "s3"])
        assert id_sec_1 != id_sec_2

    def test_distinct_report_ids_for_changed_diagnostic_suite(self):
        id_suite_1 = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "full_academic", diagnostic_suite_id="suite_1")
        id_suite_2 = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "full_academic", diagnostic_suite_id="suite_2")
        assert id_suite_1 != id_suite_2

    def test_distinct_report_ids_for_changed_author_text(self):
        id1 = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "full_academic", author_text="Notes A")
        id2 = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "full_academic", author_text="Notes B")
        assert id1 != id2

    def test_distinct_report_ids_for_changed_template(self):
        id_full = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "full_academic")
        id_exec = compute_deterministic_report_id("m1", "ds_fp", "sp_fp", "executive_summary")
        assert id_full != id_exec

    def test_report_tracker_history_and_stale_detection(self):
        tracker = ReportExperimentTracker()
        rep1 = AcademicReportResult(
            report_id="rep_1",
            report_title="Report 1",
            template_type="full_academic",
            dataset_id="ds1",
            dataset_fingerprint="fp1",
            split_fingerprint="split_active"
        )
        rep2 = AcademicReportResult(
            report_id="rep_2",
            report_title="Report 2",
            template_type="full_academic",
            dataset_id="ds1",
            dataset_fingerprint="fp1",
            split_fingerprint="split_stale"
        )

        tracker.add_report(rep1)
        tracker.add_report(rep2)

        assert len(tracker.get_all_reports()) == 2
        assert tracker.get_report("rep_1") == rep1

        valid_reports = tracker.get_valid_reports(current_split_fingerprint="split_active")
        assert len(valid_reports) == 1
        assert valid_reports[0].report_id == "rep_1"

    def test_source_artifact_immutability(self):
        reg_tracker = RegressionExperimentTracker()
        m1 = create_dummy_model("m1")
        reg_tracker.add_model(m1)

        orig_rmse = m1.val_metrics.rmse
        resolver = ReportSourceResolver(regression_tracker=reg_tracker)
        sources = resolver.resolve(selected_model_id="m1")

        report = run_academic_report(sources=sources)
        assert report.readiness_status in ("ready", "ready_with_limitations")

        # Confirm original model artifact in tracker was not mutated
        persisted = reg_tracker.get_model("m1")
        assert persisted.val_metrics.rmse == orig_rmse
        assert persisted is m1
