"""
Automated Test Suite — Milestone 6: Group I (AppTest) & Group J (Real Datasets)
Tests:
- Group I: Streamlit AppTest for pages/report.py
- Group J: Complete end-to-end report generation across all 4 benchmark datasets:
    1. Appliances Energy (Temporal split, chronological evaluation, autocorrelation caveats)
    2. King County House Sales (Multicollinearity, VIF, Pareto trade-off)
    3. Life Expectancy (Country Group split, panel independence caveats)
    4. Student Performance (G1/G2 timing caveats, categorical reference encoding)
- Metric comparison between generated reports and raw model results
- Python reproduction replay test
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data.registry import get_registry
from src.data.loader import load_dataset_bundle, get_dataset_fingerprint
from src.data.splitters import split_dataset
from src.data.models import SplitConfig, SplitStrategy
from src.models.model_service import train_ols_experiment
from src.diagnostics.runner import run_full_diagnostic_suite
from src.comparison.comparison_runner import run_model_comparison
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.experiments.comparison_experiment_tracker import ComparisonExperimentTracker
from src.experiments.report_experiment_tracker import ReportExperimentTracker

from src.reporting.source_resolver import ReportSourceResolver
from src.reporting.report_runner import run_academic_report
from src.reporting.export_service import ReportExportService
from src.reporting.result_models import ReportTemplateType, ReadinessStatus, AuthorContent

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestStreamlitReportAppTest:
    """Group I: Streamlit AppTest tests for pages/report.py."""

    def test_report_page_no_dataset_state(self):
        """When no dataset is partitioned in session state, warn and stop."""
        at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "report.py"), default_timeout=30)
        at.run()
        assert at.warning
        assert any("Dataset & Split Required" in w.value for w in at.warning)

    def test_report_page_no_models_state(self):
        """When dataset is partitioned but no models have been fitted, inform user."""
        reg = get_registry()
        meta = reg.get_dataset("student_performance")
        raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
        cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

        at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "report.py"), default_timeout=30)
        at.session_state["dataset_split_result"] = split
        at.session_state["active_dataset_metadata"] = meta
        at.session_state["regression_tracker"] = RegressionExperimentTracker()
        at.run()

        assert at.info
        assert any("No Trained Models Found" in i.value for i in at.info)

    def test_report_page_renders_with_model(self):
        """When valid model exists, report page renders dashboard, preview, and export controls."""
        reg = get_registry()
        meta = reg.get_dataset("student_performance")
        raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
        ds_fp = get_dataset_fingerprint(meta)
        cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

        m = train_ols_experiment(
            split_result=split,
            feature_source=["studytime", "failures", "absences"],
            dataset_id=meta.id,
            dataset_fingerprint=ds_fp,
            target_name=meta.target_column,
            target_unit=meta.target_unit
        )
        reg_tracker = RegressionExperimentTracker()
        reg_tracker.add_model(m)

        at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "report.py"), default_timeout=30)
        at.session_state["dataset_split_result"] = split
        at.session_state["active_dataset_metadata"] = meta
        at.session_state["regression_tracker"] = reg_tracker
        at.session_state["active_dataset_fingerprint"] = ds_fp
        at.run()

        # Should not have fatal exception
        assert not at.exception
        assert len(at.title) >= 1


class TestRealDatasetReportGeneration:
    """Group J: Complete academic report generation on all 4 Kaggle benchmark datasets."""

    def test_dataset_1_appliances_energy(self):
        """
        Dataset 1: Appliances Energy
        Temporal Split, chronological evaluation, autocorrelation caveats, Durbin-Watson.
        """
        reg = get_registry()
        meta = reg.get_dataset("appliances_energy")
        raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
        ds_fp = get_dataset_fingerprint(meta)

        cfg = SplitConfig(strategy=SplitStrategy.TEMPORAL, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42, date_column="date")
        split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

        feats = ["lights", "T1", "RH_1", "T2", "RH_2", "T_out", "Windspeed"]
        m = train_ols_experiment(
            split_result=split,
            feature_source=feats,
            dataset_id=meta.id,
            dataset_fingerprint=ds_fp,
            target_name=meta.target_column,
            target_unit=meta.target_unit
        )
        diag = run_full_diagnostic_suite(m, split_result=split, alpha=0.05)

        reg_tracker = RegressionExperimentTracker()
        reg_tracker.add_model(m)

        resolver = ReportSourceResolver(
            regression_tracker=reg_tracker,
            diagnostic_suites={diag.suite_id: diag},
            dataset_metadata=meta,
            split_result=split
        )
        sources = resolver.resolve(selected_model_id=m.model_id, diagnostic_suite_id=diag.suite_id)

        report = run_academic_report(
            sources=sources,
            template_type=ReportTemplateType.FULL_ACADEMIC,
            report_title="Appliances Energy Consumption Analysis"
        )

        assert report.readiness_status in ("ready", "ready_with_limitations")
        manifest = report.evidence_manifest
        assert manifest.verified_count > 15

        # Check that reported RMSE matches actual model RMSE
        val_rmse_rec = manifest.get_record("val_rmse")
        assert abs(val_rmse_rec.observed_value - m.val_metrics.rmse) < 1e-4

        # Verify temporal split is documented
        split_rec = manifest.get_record("split_strategy")
        assert split_rec.observed_value == "temporal"

        # Check export
        svc = ReportExportService(report=report, sources=sources)
        md = svc.export_markdown()
        assert "Appliances Energy" in md
        assert f"{m.val_metrics.rmse:.3f}" in md

    def test_dataset_2_kc_house_sales(self):
        """
        Dataset 2: King County House Sales
        Price target (USD), VIF multicollinearity, Pareto frontier, OLS equations.
        """
        reg = get_registry()
        meta = reg.get_dataset("kc_house_sales")
        raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
        ds_fp = get_dataset_fingerprint(meta)

        cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

        m_pruned = train_ols_experiment(
            split_result=split,
            feature_source=["bedrooms", "bathrooms", "sqft_living", "grade"],
            dataset_id=meta.id,
            dataset_fingerprint=ds_fp,
            target_name=meta.target_column,
            target_unit=meta.target_unit
        )
        m_min = train_ols_experiment(
            split_result=split,
            feature_source=["sqft_living", "grade"],
            dataset_id=meta.id,
            dataset_fingerprint=ds_fp,
            target_name=meta.target_column,
            target_unit=meta.target_unit
        )

        comp = run_model_comparison(
            models=[m_pruned, m_min],
            active_dataset_id=meta.id,
            active_dataset_fingerprint=ds_fp,
            active_split_fingerprint=split.split_fingerprint,
            active_target_name=meta.target_column,
            active_target_unit=meta.target_unit
        )

        reg_tracker = RegressionExperimentTracker()
        reg_tracker.add_model(m_pruned)
        reg_tracker.add_model(m_min)
        comp_tracker = ComparisonExperimentTracker()
        comp_tracker.add_comparison(comp)

        resolver = ReportSourceResolver(
            regression_tracker=reg_tracker,
            comparison_tracker=comp_tracker,
            dataset_metadata=meta,
            split_result=split
        )
        sources = resolver.resolve(comparison_id=comp.comparison_id, selected_model_id=m_pruned.model_id)

        report = run_academic_report(
            sources=sources,
            template_type=ReportTemplateType.FULL_ACADEMIC,
            report_title="King County House Sales Hedonic Pricing Model"
        )

        assert report.readiness_status in ("ready", "ready_with_limitations")
        manifest = report.evidence_manifest
        assert "USD" in manifest.get_record("dataset_target_unit").observed_value
        assert "USD" in manifest.get_record("val_rmse").unit

        # Verify Section 7 Comparison Leaderboard table content
        sec7 = next(s for s in report.report_sections if s.section_id == "sec_7_comparison")
        assert len(sec7.tables) >= 1
        comp_table = sec7.tables[0]
        assert "Model ID" in comp_table["headers"]
        assert "Val RMSE" in comp_table["headers"]
        assert "Pareto Status" in comp_table["headers"]
        # Candidate model should be in the table with [Selected Candidate]
        cand_rows = [r for r in comp_table["rows"] if m_pruned.model_id in r["Model ID"]]
        assert len(cand_rows) == 1
        assert "[Selected Candidate]" in cand_rows[0]["Model ID"]
        # Numeric values must match actual model metrics, not fabricated 0.0
        val_rmse_val = float(str(cand_rows[0]["Val RMSE"]).split()[0])
        assert val_rmse_val > 0.0
        assert abs(val_rmse_val - m_pruned.val_metrics.rmse) < 1e-2

    def test_dataset_3_life_expectancy(self):
        """
        Dataset 3: Life Expectancy (WHO)
        Country Group Split, panel dependence caveats.
        """
        reg = get_registry()
        meta = reg.get_dataset("life_expectancy")
        raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
        ds_fp = get_dataset_fingerprint(meta)

        cfg = SplitConfig(strategy=SplitStrategy.GROUP, group_column="Country", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

        feats = ["Adult Mortality", "infant deaths", "Alcohol", "BMI", "Schooling"]
        m = train_ols_experiment(
            split_result=split,
            feature_source=feats,
            dataset_id=meta.id,
            dataset_fingerprint=ds_fp,
            target_name=meta.target_column,
            target_unit=meta.target_unit
        )

        reg_tracker = RegressionExperimentTracker()
        reg_tracker.add_model(m)

        resolver = ReportSourceResolver(
            regression_tracker=reg_tracker,
            dataset_metadata=meta,
            split_result=split
        )
        sources = resolver.resolve(selected_model_id=m.model_id)

        report = run_academic_report(
            sources=sources,
            template_type=ReportTemplateType.EXECUTIVE_SUMMARY,
            report_title="WHO Life Expectancy Country Group Evaluation"
        )

        assert report.readiness_status in ("ready", "ready_with_limitations")
        manifest = report.evidence_manifest
        split_rec = manifest.get_record("split_strategy")
        assert split_rec.observed_value == "group"

    def test_dataset_4_student_performance(self):
        """
        Dataset 4: Student Performance
        Categorical dummy reference encoding, G1/G2 caveats.
        """
        reg = get_registry()
        meta = reg.get_dataset("student_performance")
        raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
        ds_fp = get_dataset_fingerprint(meta)

        cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

        feats = ["studytime", "failures", "absences", "Medu", "Fedu"]
        m = train_ols_experiment(
            split_result=split,
            feature_source=feats,
            dataset_id=meta.id,
            dataset_fingerprint=ds_fp,
            target_name=meta.target_column,
            target_unit=meta.target_unit
        )

        reg_tracker = RegressionExperimentTracker()
        reg_tracker.add_model(m)

        resolver = ReportSourceResolver(
            regression_tracker=reg_tracker,
            dataset_metadata=meta,
            split_result=split
        )
        sources = resolver.resolve(selected_model_id=m.model_id)

        report = run_academic_report(
            sources=sources,
            template_type=ReportTemplateType.FULL_ACADEMIC,
            report_title="Student Academic Achievement OLS Modeling"
        )

        assert report.readiness_status in ("ready", "ready_with_limitations")
        manifest = report.evidence_manifest
        assert manifest.get_record("dataset_target_name").observed_value == "G3"
        assert abs(manifest.get_record("val_r2").observed_value - m.val_metrics.r2) < 1e-4

        # Test reproduction script generation for Student Performance
        svc = ReportExportService(report=report, sources=sources)
        script_text = svc.export_reproduction_script()
        assert "student_performance" in script_text
        assert "run_reproduction" in script_text

        # Verify reproduction script syntax compiles cleanly
        compiled = compile(script_text, "reproduce_student.py", "exec")
        assert compiled is not None
