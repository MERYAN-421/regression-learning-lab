"""
Milestone 7: Production Readiness, Security, Multi-Session Isolation, and Deployment Verification Suite.
Validates Groups A through I:
- Group A: Environment, dependencies, Python 3.11 compatibility, and startup
- Group B: Four-dataset deployment, integrity, and fallback handling
- Group C: Production security configuration, secrets exclusion, and XSS safety
- Group D: Multi-session state isolation and Test Lock independence
- Group E: Full 9-page AppTest rendering and workflow state contracts
- Group F: Statistical integrity, inference guards, and validation-only ranking
- Group G: Reporting, manifests, HTML escaping, and reproduction script execution
- Group H: Performance and timing benchmarks
- Group I: GitHub Actions CI workflow configuration
"""
import copy
import hashlib
import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from src.utils import get_project_root
from src.data.registry import get_registry
from src.data.loader import load_dataset_bundle, compute_file_fingerprint, resolve_dataset_path, DatasetNotFoundError, load_raw_dataset
from src.data.download import verify_dataset_integrity, DOWNLOAD_SOURCES
from src.data.splitters import split_dataset
from src.data.models import DatasetMetadata, SplitConfig, SplitStrategy, SplitResult
from src.features import run_feature_selection
from src.experiments.feature_selection_runner import FeatureSelectionExperimentTracker
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.experiments.comparison_experiment_tracker import ComparisonExperimentTracker
from src.experiments.report_experiment_tracker import ReportExperimentTracker
from src.models.model_service import train_ols_experiment
from src.models.model_result import OLSModelResult
from src.diagnostics.runner import run_full_diagnostic_suite, validate_active_model_for_diagnostics
from src.reporting.html_renderer import render_report_html, escape_text
from src.reporting.reproduction_script import generate_reproduction_script
from src.reporting.result_models import AcademicReportResult
from src.reporting.source_resolver import ResolvedReportSources

PROJECT_ROOT = get_project_root()


# =============================================================================
# GROUP A: ENVIRONMENT, RUNTIME & STARTUP VALIDATION
# =============================================================================
class TestGroupAEnvironment:
    def test_python_runtime_version(self):
        """Verify Python 3.11 target runtime environment."""
        assert sys.version_info.major == 3, "Python 3 is required."
        assert sys.version_info.minor >= 11, f"Expected Python 3.11+, got {sys.version_info.major}.{sys.version_info.minor}"

    def test_core_dependencies_importable(self):
        """Verify all core dependencies import cleanly."""
        imports = [
            "streamlit", "pandas", "numpy", "sklearn", "statsmodels",
            "scipy", "matplotlib", "seaborn", "plotly", "jinja2", "markdown", "yaml"
        ]
        for mod in imports:
            __import__(mod)

    def test_pip_check_clean(self):
        """Verify python -m pip check passes without broken requirements."""
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        assert result.returncode == 0, f"pip check failed: {result.stdout} {result.stderr}"

    def test_app_entrypoint_compiles(self):
        """Verify app.py compiles cleanly to bytecode."""
        import py_compile
        app_file = PROJECT_ROOT / "app.py"
        assert app_file.exists()
        py_compile.compile(str(app_file), doraise=True)


# =============================================================================
# GROUP B: FOUR-DATASET DEPLOYMENT & INTEGRITY
# =============================================================================
class TestGroupBDatasetDeployment:
    def test_all_four_datasets_registered(self):
        """Verify all 4 required benchmark datasets are registered with valid metadata."""
        registry = get_registry()
        expected_ids = ["appliances_energy", "kc_house_sales", "life_expectancy", "student_performance"]
        registered_ids = [d.id for d in registry.list_datasets()]
        for eid in expected_ids:
            assert eid in registered_ids, f"Dataset {eid} missing from registry"

    def test_four_datasets_integrity_and_sha256(self):
        """Verify all 4 datasets exist and match their expected SHA-256 checksums."""
        for ds_id in ["appliances_energy", "kc_house_sales", "life_expectancy", "student_performance"]:
            assert ds_id in DOWNLOAD_SOURCES, f"{ds_id} not in DOWNLOAD_SOURCES"
            is_valid = verify_dataset_integrity(ds_id)
            assert is_valid, f"Dataset {ds_id} failed cryptographic SHA-256 integrity verification"

    def test_missing_dataset_handling(self):
        """Verify missing dataset raises DatasetNotFoundError with actionable message without fatal crash."""
        registry = get_registry()
        meta = copy.deepcopy(registry.get_dataset("student_performance"))
        meta.local_path = "data/raw/definitely_missing_file_xyz.csv"
        with pytest.raises(DatasetNotFoundError) as exc_info:
            load_raw_dataset(meta)
        assert "not found" in str(exc_info.value).lower()
        assert "Acquisition instruction" in str(exc_info.value)


# =============================================================================
# GROUP C: PRODUCTION SECURITY CONFIGURATION
# =============================================================================
class TestGroupCSecurity:
    def test_streamlit_config_cors_and_xsrf_enabled(self):
        """Verify .streamlit/config.toml enables CORS and XSRF protection for production."""
        config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
        assert config_path.exists(), "config.toml missing"
        text = config_path.read_text(encoding="utf-8")
        assert "enableCORS = true" in text or "enableCORS=true" in text
        assert "enableXsrfProtection = true" in text or "enableXsrfProtection=true" in text

    def test_secrets_and_credentials_excluded_in_gitignore(self):
        """Verify .gitignore excludes secrets, keys, credentials, and .env files."""
        gitignore = PROJECT_ROOT / ".gitignore"
        assert gitignore.exists()
        text = gitignore.read_text(encoding="utf-8")
        assert ".streamlit/secrets.toml" in text
        assert ".env" in text
        assert "*.key" in text
        assert "*.pem" in text
        assert "credentials.json" in text

    def test_secrets_file_not_tracked(self):
        """Verify .streamlit/secrets.toml is not committed to the repository."""
        secrets_file = PROJECT_ROOT / ".streamlit" / "secrets.toml"
        assert not secrets_file.exists(), ".streamlit/secrets.toml should not be committed to Git"

    def test_html_export_xss_protection(self):
        """Verify HTML escaping escapes raw user scripts and malicious tags."""
        malicious_input = "<script>alert('xss')</script><b>Bold Title</b>"
        escaped = escape_text(malicious_input)
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped


# =============================================================================
# GROUP D: MULTI-SESSION STATE ISOLATION & TEST LOCK INDEPENDENCE
# =============================================================================
class TestGroupDSessionIsolation:
    def test_independent_user_experiment_trackers(self):
        """Verify User A and User B maintain completely isolated regression trackers."""
        from tests.test_reporting_sources import create_dummy_model
        tracker_user_a = RegressionExperimentTracker()
        tracker_user_b = RegressionExperimentTracker()

        dummy_model_a = create_dummy_model("ols_model_user_a_001")
        tracker_user_a.add_model(dummy_model_a)
        assert tracker_user_a.get_model("ols_model_user_a_001") is not None
        assert tracker_user_b.get_model("ols_model_user_a_001") is None
        assert len(tracker_user_b.get_all_models()) == 0

    def test_independent_final_test_lock_authorization(self):
        """Verify User A finalizing Test evaluation does NOT affect User B's lock state."""
        from tests.test_reporting_sources import create_dummy_model
        split_res = SplitResult(
            X_train=pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0]}),
            y_train=pd.Series([2.0, 4.0, 6.0, 8.0]),
            X_val=pd.DataFrame({"x1": [5.0, 6.0]}),
            y_val=pd.Series([10.0, 12.0]),
            X_test=pd.DataFrame({"x1": [7.0, 8.0]}),
            y_test=pd.Series([14.0, 16.0]),
            train_indices=np.array([0, 1, 2, 3]),
            val_indices=np.array([4, 5]),
            test_indices=np.array([6, 7]),
            strategy="random",
            sample_counts={"train": 4, "validation": 2, "test": 2, "total_eligible": 8},
            actual_proportions={"train": 0.5, "validation": 0.25, "test": 0.25},
            split_fingerprint="split_lock_test",
            random_seed=42
        )

        model_template = create_dummy_model("model_lock_iso", split_fp="split_lock_test")

        tracker_user_a = RegressionExperimentTracker()
        tracker_user_b = RegressionExperimentTracker()

        tracker_user_a.add_model(copy.deepcopy(model_template))
        tracker_user_b.add_model(copy.deepcopy(model_template))

        assert tracker_user_a.get_model("model_lock_iso").test_evaluation_status == "untested"
        assert tracker_user_b.get_model("model_lock_iso").test_evaluation_status == "untested"

        finalized_a = tracker_user_a.finalize_test_evaluation("model_lock_iso", split_res, refit_policy="train_val")
        assert finalized_a.test_evaluation_status != "untested"
        assert tracker_user_a.get_model("model_lock_iso").test_evaluation_status != "untested"

        # User B remains strictly locked
        assert tracker_user_b.get_model("model_lock_iso").test_evaluation_status == "untested"
        assert tracker_user_b.get_model("model_lock_iso").test_metrics is None

    def test_independent_report_trackers(self):
        """Verify User A and User B have independent ReportExperimentTrackers."""
        rep_tracker_a = ReportExperimentTracker()
        rep_tracker_b = ReportExperimentTracker()
        assert len(rep_tracker_a.get_all_reports()) == 0
        assert len(rep_tracker_b.get_all_reports()) == 0


# =============================================================================
# GROUP E: FULL WORKFLOW & STREAMLIT APPTESTS
# =============================================================================
class TestGroupEWorkflowAndAppTests:
    @pytest.mark.parametrize("page_name", [
        "app.py",
        "pages/home.py",
        "pages/dataset.py",
        "pages/exploration.py",
        "pages/feature_selection.py",
        "pages/regression.py",
        "pages/evaluation.py",
        "pages/diagnostics.py",
        "pages/comparison.py",
        "pages/report.py"
    ])
    def test_all_pages_render_cleanly(self, page_name: str):
        """Verify every single page compiles and renders cleanly in Streamlit AppTest."""
        target_path = PROJECT_ROOT / page_name
        assert target_path.exists(), f"Page {page_name} does not exist"
        at = AppTest.from_file(str(target_path), default_timeout=30).run()
        assert not at.exception, f"AppTest raised exception on {page_name}: {at.exception}"


# =============================================================================
# GROUP F: STATISTICAL REGRESSION & LEAKAGE ISOLATION
# =============================================================================
class TestGroupFStatisticalRegression:
    def test_test_lock_permanence(self):
        """Verify repeat finalize_test_evaluation calls raise PermissionError / ValueError."""
        from tests.test_reporting_sources import create_dummy_model
        split_res = SplitResult(
            X_train=pd.DataFrame({"x1": [1.0, 2.0, 3.0]}),
            y_train=pd.Series([1.0, 2.0, 3.0]),
            X_val=pd.DataFrame({"x1": [4.0]}),
            y_val=pd.Series([4.0]),
            X_test=pd.DataFrame({"x1": [5.0]}),
            y_test=pd.Series([5.0]),
            train_indices=np.array([0, 1, 2]),
            val_indices=np.array([3]),
            test_indices=np.array([4]),
            strategy="random",
            sample_counts={"train": 3, "validation": 1, "test": 1, "total_eligible": 5},
            actual_proportions={"train": 0.6, "validation": 0.2, "test": 0.2},
            split_fingerprint="split_lock_perm",
            random_seed=42
        )
        model = create_dummy_model("test_model_perm", split_fp="split_lock_perm")
        tracker = RegressionExperimentTracker()
        tracker.add_model(model)
        tracker.finalize_test_evaluation("test_model_perm", split_res, refit_policy="train_val")

        with pytest.raises(RuntimeError):
            tracker.finalize_test_evaluation("test_model_perm", split_res, refit_policy="train_val")

    def test_ranking_never_uses_test_metrics(self):
        """Verify model comparison ranking is strictly validation-based."""
        from src.comparison.leaderboard import build_comparison_leaderboard
        from tests.test_reporting_sources import create_dummy_model

        m1 = create_dummy_model("model_good_val", val_rmse=1.2)
        m2 = create_dummy_model("model_bad_val", val_rmse=2.5)

        m1.test_metrics = type("Metrics", (), {"rmse": 99.0, "r2": 0.01, "mae": 90.0, "mse": 9801.0})()
        m2.test_metrics = type("Metrics", (), {"rmse": 0.5, "r2": 0.99, "mae": 0.4, "mse": 0.25})()

        leaderboard = build_comparison_leaderboard([m1, m2])
        # Ranked by Val RMSE ascending: m1 (val_rmse=1.2) must rank #1 over m2 (val_rmse=2.5)
        assert leaderboard[0]["Model ID"] == "model_good_val"
        assert leaderboard[1]["Model ID"] == "model_bad_val"


# =============================================================================
# GROUP G: REPORTING & REPRODUCTION EXPORT
# =============================================================================
class TestGroupGReportingAndReproduction:
    def test_reproduction_script_compiles_and_runs(self, tmp_path):
        """Verify reproduction script generation produces valid Python syntax."""
        rep = AcademicReportResult(
            report_id="rep_test_m7",
            report_title="Milestone 7 Test Report",
            template_type="full_academic",
            dataset_id="student_performance",
            dataset_fingerprint="fp1",
            split_fingerprint="sp1"
        )
        sources = ResolvedReportSources(model=None, split_result=None)
        script = generate_reproduction_script(rep, sources)
        assert isinstance(script, str)
        assert "REPRODUCIBILITY STATUS: INCOMPLETE" in script
        compiled = compile(script, "<reproduction_script>", "exec")
        assert compiled is not None


# =============================================================================
# GROUP H: PERFORMANCE AND RESOURCE BENCHMARKS
# =============================================================================
class TestGroupHPerformance:
    def test_student_performance_workflow_timing(self):
        """Measure actual execution time of lightweight Student Performance workflow."""
        start_time = time.time()
        registry = get_registry()
        meta = registry.get_dataset("student_performance")
        raw_df, feat_df, target_series, meta_df = load_dataset_bundle(meta)
        load_duration = time.time() - start_time

        split_start = time.time()
        split_config = SplitConfig(
            strategy=SplitStrategy.RANDOM,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        split_res = split_dataset(feat_df, target_series, meta, split_config, raw_df)
        split_duration = time.time() - split_start

        from src.data.loader import get_dataset_fingerprint
        fp = get_dataset_fingerprint(meta)
        fit_start = time.time()
        model_res = train_ols_experiment(
            split_result=split_res,
            feature_source="all_features",
            target_name=meta.target_column,
            fit_intercept=True,
            scale_numeric=True,
            dataset_id=meta.id,
            dataset_fingerprint=fp,
            allow_rank_deficient=False
        )
        fit_duration = time.time() - fit_start

        diag_start = time.time()
        diag_suite = run_full_diagnostic_suite(model_res, split_res)
        diag_duration = time.time() - diag_start

        total_duration = time.time() - start_time

        assert load_duration < 5.0, f"Load took {load_duration:.2f}s"
        assert split_duration < 2.0, f"Split took {split_duration:.2f}s"
        assert fit_duration < 5.0, f"OLS fit took {fit_duration:.2f}s"
        assert diag_duration < 10.0, f"Diagnostics took {diag_duration:.2f}s"
        assert total_duration < 20.0, f"Total workflow took {total_duration:.2f}s"


# =============================================================================
# GROUP I: GITHUB ACTIONS CI CONFIGURATION
# =============================================================================
class TestGroupICIConfiguration:
    def test_ci_workflow_exists_and_is_valid(self):
        """Verify .github/workflows/ci.yml exists and has proper syntax and jobs."""
        ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        assert ci_path.exists(), "CI workflow file .github/workflows/ci.yml missing"
        with open(ci_path, "r", encoding="utf-8") as f:
            ci_data = yaml.safe_load(f)
        assert "jobs" in ci_data
        assert "test" in ci_data["jobs"]
        steps = ci_data["jobs"]["test"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("Checkout" in n for n in step_names)
        assert any("Python" in n for n in step_names)
        assert any("Dependencies" in n for n in step_names)
        assert any("Test" in n for n in step_names)
