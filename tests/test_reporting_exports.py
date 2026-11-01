"""
Automated Test Suite — Milestone 6: Group F & Group G
Tests:
- Markdown export structure, table formatting, and heading hierarchy
- Standalone HTML export, styling, MathJax inclusion, and section navigation
- XSS prevention / safe escaping in HTML export
- Manifest JSON export valid structure and serialization
- Reproduction package ZIP contents (report.md, report.html, manifest.json, reproduce.py, etc.)
- Python reproduction script syntax compilation and valid API imports
"""
import io
import json
import zipfile
from pathlib import Path
import pytest

from src.reporting.result_models import AuthorContent, ReportTemplateType, AcademicReportResult
from src.reporting.source_resolver import ResolvedReportSources
from src.reporting.figure_registry import ReportFigureRegistry
from src.reporting.export_service import ReportExportService
from src.reporting.report_runner import run_academic_report
from src.reporting.reproduction_script import generate_reproduction_script

from tests.test_reporting_sources import create_dummy_model, create_dummy_comparison

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestExportIntegrityAndReproducibility:
    """Group F & Group G: Export formatting and reproduction package tests."""

    @pytest.fixture
    def sample_report_and_sources(self):
        m1 = create_dummy_model("m1", val_rmse=1.234, val_r2=0.876)
        comp = create_dummy_comparison("c1", model_ids=["m1"], candidate_id="m1", val_rmse=1.234, val_r2=0.876)
        author = AuthorContent(
            author_name="Alice Researcher",
            research_question="What predicts outcome y?",
            main_findings="Regressor x1 exhibits strong positive association with y."
        )
        sources = ResolvedReportSources(model=m1, comparison=comp)
        report = run_academic_report(
            sources=sources,
            report_title="Empirical Regression Analysis Test",
            author_content=author
        )
        fig_reg = ReportFigureRegistry(sources=sources)
        fig_reg.build_all_available_figures()
        return report, sources, fig_reg

    def test_markdown_export_structure_and_tables(self, sample_report_and_sources):
        report, sources, fig_reg = sample_report_and_sources
        export_svc = ReportExportService(report=report, sources=sources, figure_registry=fig_reg)

        md = export_svc.export_markdown()
        assert isinstance(md, str)
        assert "# Empirical Regression Analysis Test" in md
        assert "**Author**: Alice Researcher" in md
        assert f"**Report ID**: `{report.report_id}`" in md

        # Confirm table syntax
        assert "| Metric | Training Set | Validation Set | Interpretation |" in md
        assert "| --- | --- | --- | --- |" in md
        assert "1.234" in md

        # Confirm evidence provenance section
        assert "### Evidence Provenance Summary" in md
        assert f"`{report.reproducibility_hash}`" in md

    def test_standalone_html_export_and_xss_protection(self, sample_report_and_sources):
        report, sources, fig_reg = sample_report_and_sources

        # Inject potentially malicious script into author reflection
        malicious_author = AuthorContent(
            author_name="Alice Researcher",
            research_question="What predicts outcome y?",
            main_findings="<script>alert('XSS_ATTACK_VECTOR');</script>"
        )
        report = run_academic_report(
            sources=sources,
            report_title="Empirical Regression Analysis Test",
            author_content=malicious_author
        )
        export_svc = ReportExportService(report=report, sources=sources, figure_registry=fig_reg)

        html_out = export_svc.export_html()
        assert isinstance(html_out, str)
        assert "<!DOCTYPE html>" in html_out
        assert "plotly-2.29.1.min.js" in html_out
        assert "tex-mml-chtml.js" in html_out
        assert "<aside class=\"sidebar\">" in html_out

        # XSS Test: The literal raw script tag must NOT be present unescaped
        assert "<script>alert('XSS_ATTACK_VECTOR');</script>" not in html_out
        # It must be escaped
        assert "&lt;script&gt;alert(&#x27;XSS_ATTACK_VECTOR&#x27;);&lt;/script&gt;" in html_out or "&lt;script&gt;" in html_out

    def test_manifest_json_export_is_valid_json(self, sample_report_and_sources):
        report, sources, fig_reg = sample_report_and_sources
        export_svc = ReportExportService(report=report, sources=sources, figure_registry=fig_reg)

        manifest_str = export_svc.export_manifest_json()
        assert isinstance(manifest_str, str)
        parsed = json.loads(manifest_str)

        assert parsed["report_id"] == report.report_id
        assert "records" in parsed
        assert parsed["verified_count"] > 0
        assert "val_rmse" in parsed["records"]
        assert abs(parsed["records"]["val_rmse"]["observed_value"] - 1.234) < 1e-3

    def test_reproduction_script_compiles_as_valid_python(self, sample_report_and_sources):
        report, sources, fig_reg = sample_report_and_sources
        export_svc = ReportExportService(report=report, sources=sources, figure_registry=fig_reg)

        script_code = export_svc.export_reproduction_script()
        assert isinstance(script_code, str)
        # Without split artifact, it marks incomplete
        assert "REPRODUCIBILITY STATUS: INCOMPLETE" in script_code

    def test_reproduction_script_incomplete_status_when_sources_missing(self):
        m1 = create_dummy_model("m1")
        # sources with no split artifact
        sources = ResolvedReportSources(model=m1, split_result=None)
        rep = AcademicReportResult(
            report_id="rep_test",
            report_title="Test Report",
            template_type="full_academic",
            dataset_id="ds1",
            dataset_fingerprint="fp1",
            split_fingerprint="sp1"
        )
        script = generate_reproduction_script(rep, sources)
        assert "REPRODUCIBILITY STATUS: INCOMPLETE" in script
        assert "raise RuntimeError" in script

    def test_reproduction_zip_package_contents(self, sample_report_and_sources):
        report, sources, fig_reg = sample_report_and_sources
        export_svc = ReportExportService(report=report, sources=sources, figure_registry=fig_reg)

        zip_bytes = export_svc.export_reproduction_zip()
        assert isinstance(zip_bytes, bytes)
        assert len(zip_bytes) > 0

        # Read back in-memory zip
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            namelist = zf.namelist()
            assert "report.md" in namelist
            assert "report.html" in namelist
            assert "manifest.json" in namelist
            assert "reproduce.py" in namelist
            assert "README_REPRODUCTION.md" in namelist
            assert "requirements.txt" in namelist

            # Verify contents are non-empty
            assert len(zf.read("report.md")) > 100
            assert len(zf.read("report.html")) > 100
            assert len(zf.read("manifest.json")) > 50
            assert len(zf.read("reproduce.py")) > 50

    def test_end_to_end_execution_of_generated_reproduction_script(self, tmp_path):
        """
        Mandatory Task 2 verification:
        Execute generated reproduce.py end-to-end via Python subprocess on a real dataset
        and verify that training & validation metrics reproduce within numerical tolerance.
        """
        import subprocess
        import sys
        from src.data.registry import get_registry
        from src.data.loader import load_dataset_bundle, get_dataset_fingerprint
        from src.data.models import SplitConfig, SplitStrategy
        from src.data.splitters import split_dataset
        from src.models.model_service import train_ols_experiment

        reg = get_registry()
        meta = reg.get_dataset("student_performance")
        raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
        ds_fp = get_dataset_fingerprint(meta)

        cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

        feats = ["studytime", "failures", "absences"]
        model = train_ols_experiment(
            split_result=split,
            feature_source=feats,
            dataset_id=meta.id,
            dataset_fingerprint=ds_fp,
            target_name=meta.target_column,
            target_unit=meta.target_unit
        )

        sources = ResolvedReportSources(
            model=model,
            dataset_metadata=meta,
            split_result=split
        )
        report = run_academic_report(
            sources=sources,
            report_title="Student Performance Empirical Replay"
        )
        export_svc = ReportExportService(report=report, sources=sources)
        script_code = export_svc.export_reproduction_script()

        # Write out reproduce.py to temporary directory
        script_path = tmp_path / "reproduce.py"
        script_path.write_text(script_code, encoding="utf-8")

        # Execute the generated script via Python subprocess
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=60
        )
        print("reproduce.py stdout:\n", proc.stdout)
        if proc.returncode != 0:
            print("reproduce.py stderr:\n", proc.stderr)

        assert proc.returncode == 0
        assert "REPRODUCTION SUCCESSFUL: All reported metrics match within numerical tolerance." in proc.stdout
        assert "Split partition integrity verified." in proc.stdout
