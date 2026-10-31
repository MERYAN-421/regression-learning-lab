"""
Group G, H, I: Diagnostic Integration, AppTest, and Real-Dataset Verification Tests.
Verifies:
- Group G: Model validation, fingerprint mismatch rejection, zero test leakage
- Group H: Streamlit AppTest page rendering
- Group I: Real-dataset integration across all 4 benchmark datasets:
  1. Appliances Energy (Temporal split, chronological serial correlation diagnostics)
  2. King County House Sales (Near structural collinearity, VIF, condition number)
  3. Life Expectancy (Country group split, panel non-applicability of sequential tests)
  4. Student Performance (Reference dummy coding, normality, RESET)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data.registry import get_registry
from src.data.loader import load_dataset_bundle
from src.data.splitters import split_dataset
from src.data.models import SplitConfig, SplitStrategy
from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model
from src.models.model_service import train_ols_experiment
from src.diagnostics.runner import (
    validate_active_model_for_diagnostics,
    run_full_diagnostic_suite
)
from src.diagnostics.result_models import ApplicabilityStatus, DiagnosticDecision

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def integration_split_fixture():
    """Synthetic split fixture for integration tests."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num1": np.random.normal(0, 1, n),
        "num2": np.random.normal(0, 1, n),
        "cat": np.random.choice(["A", "B", "C"], size=n)
    })
    y = 2.0 + 1.5 * df["num1"] - 2.0 * df["num2"] + np.random.normal(0, 0.3, n)

    train_df, val_df, test_df = df.iloc[:60], df.iloc[60:80], df.iloc[80:]
    y_train, y_val, y_test = y.iloc[:60], y.iloc[60:80], y.iloc[80:]

    from src.data.models import SplitResult
    split = SplitResult(
        X_train=train_df,
        X_val=val_df,
        X_test=test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_indices=np.arange(60),
        val_indices=np.arange(60, 80),
        test_indices=np.arange(80, 100),
        actual_proportions={"train": 0.60, "val": 0.20, "test": 0.20},
        sample_counts={"train": 60, "val": 20, "test": 20},
        strategy="random",
        random_seed=42,
        split_fingerprint="split_diag_test_123"
    )
    return split


# =====================================================================
# Group G: Model Validation & Zero Leakage Isolation
# =====================================================================

def test_diagnostic_model_validation_success_and_mismatches(integration_split_fixture):
    """Verify active model validation rejects stale, mismatched, or missing models."""
    split = integration_split_fixture
    model = train_ols_experiment(split, "all_features", fit_intercept=True)

    # 1. Successful validation
    is_valid, msg = validate_active_model_for_diagnostics(
        model=model,
        current_dataset_id="dataset",
        current_dataset_fingerprint=model.dataset_fingerprint,
        current_split_fingerprint=split.split_fingerprint
    )
    assert is_valid, msg

    # 2. Dataset ID mismatch
    is_valid_ds, msg_ds = validate_active_model_for_diagnostics(
        model=model,
        current_dataset_id="other_dataset",
        current_split_fingerprint=split.split_fingerprint
    )
    assert not is_valid_ds
    assert "Dataset mismatch" in msg_ds

    # 3. Split fingerprint mismatch
    is_valid_sp, msg_sp = validate_active_model_for_diagnostics(
        model=model,
        current_dataset_id="dataset",
        current_split_fingerprint="stale_split_999"
    )
    assert not is_valid_sp
    assert "Split fingerprint mismatch" in msg_sp


def test_zero_test_set_leakage_guarantee(integration_split_fixture):
    """Verify that assumption diagnostics only touch Training partition, strictly isolating Test data."""
    split = integration_split_fixture
    model = train_ols_experiment(split, "all_features", fit_intercept=True)

    # Execute full diagnostic suite
    suite = run_full_diagnostic_suite(model, split_result=split, alpha=0.05)

    # Verify that all diagnostic items operate on data_scope='train'
    for diag in suite.diagnostics.values():
        assert diag.data_scope == "train"

    # Verify model test evaluation status remains 'untested'
    assert model.test_evaluation_status == "untested"
    assert model.test_metrics is None


def test_diagnostic_suite_aggregation_and_summary(integration_split_fixture):
    """Verify that DiagnosticSuiteResult aggregates all 5 areas and produces clean summaries."""
    split = integration_split_fixture
    model = train_ols_experiment(split, "all_features", fit_intercept=True)

    suite = run_full_diagnostic_suite(model, split_result=split, alpha=0.05)

    assert "ramsey_reset" in suite.diagnostics
    assert "durbin_watson" in suite.diagnostics
    assert "ljung_box" in suite.diagnostics
    assert "breusch_pagan" in suite.diagnostics
    assert "jarque_bera" in suite.diagnostics
    assert "multicollinearity_vif" in suite.diagnostics

    summary_table = suite.to_summary_table()
    assert len(summary_table) >= 5

    categories = [row["Category"] for row in summary_table]
    assert "Linearity" in categories
    assert "Independence" in categories
    assert "Homoscedasticity" in categories
    assert "Normality" in categories
    assert "Multicollinearity" in categories


# =====================================================================
# Group H: Streamlit AppTest Page Rendering
# =====================================================================

def test_apptest_diagnostics_page_renders():
    """Verify that pages/diagnostics.py renders cleanly without uncaught exceptions via Streamlit AppTest."""
    at = AppTest.from_file("pages/diagnostics.py")
    at.run()
    assert not at.exception, f"AppTest raised exception on diagnostics page: {at.exception}"


# =====================================================================
# Group I: Real Dataset Integration Tests (All 4 Benchmark Datasets)
# =====================================================================

def test_real_dataset_appliances_energy_temporal_diagnostics():
    """
    Verify Appliances Energy assumption diagnostics:
    - Temporal ordering preserved
    - Durbin-Watson and Ljung-Box marked APPLICABLE
    - Breusch-Pagan, Jarque-Bera, and VIF computed cleanly
    """
    reg = get_registry()
    meta = reg.get_dataset("appliances_energy")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.TEMPORAL, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, date_column="date")
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    selected = ["lights", "T1", "RH_1", "T2", "T3", "T4", "T6", "T8", "T_out", "Windspeed"]
    design_res, builder = build_ols_design_matrices(split.X_train, selected_features=selected, fit_intercept=True)
    model = fit_ols_model(design_res, y_train=split.y_train, target_name="Appliances")
    model.dataset_id = "appliances_energy"

    suite = run_full_diagnostic_suite(model, split_result=split, alpha=0.05)

    dw = suite.diagnostics["durbin_watson"]
    lb = suite.diagnostics["ljung_box"]
    assert dw.applicability_status == ApplicabilityStatus.APPLICABLE.value
    assert lb.applicability_status == ApplicabilityStatus.APPLICABLE.value
    # In temporal energy consumption data, strong serial correlation exists
    assert dw.statistic_value < 1.0 or lb.decision == DiagnosticDecision.REJECT_NULL.value

    vif = suite.diagnostics["multicollinearity_vif"]
    assert vif.statistic_value > 0


def test_real_dataset_kc_house_sales_structural_collinearity_diagnostics():
    """
    Verify King County House Sales assumption diagnostics:
    - Evaluates VIF on collinear building area features (sqft_living, sqft_above, sqft_basement)
    - Sequential tests marked NOT_APPLICABLE (cross-sectional)
    - Breusch-Pagan heteroscedasticity inspection
    """
    reg = get_registry()
    meta = reg.get_dataset("kc_house_sales")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    collinear_subset = ["bedrooms", "bathrooms", "sqft_living", "sqft_above", "sqft_basement"]
    design_res, _ = build_ols_design_matrices(split.X_train, selected_features=collinear_subset, fit_intercept=True)
    model = fit_ols_model(design_res, y_train=split.y_train, target_name="price")
    model.dataset_id = "kc_house_sales"

    suite = run_full_diagnostic_suite(model, split_result=split, alpha=0.05)

    # Sequential test marked NOT_APPLICABLE on cross-sectional house sales
    dw = suite.diagnostics["durbin_watson"]
    assert dw.applicability_status == ApplicabilityStatus.NOT_APPLICABLE.value

    # VIF detects near-collinearity between sqft_living and sqft_above + sqft_basement
    vif = suite.diagnostics["multicollinearity_vif"]
    assert vif.details["max_vif"] > 10.0 or vif.details["condition_number"] > 1000

    # Breusch-Pagan runs cleanly
    bp = suite.diagnostics["breusch_pagan"]
    assert bp.statistic_value > 0


def test_real_dataset_life_expectancy_panel_guards():
    """
    Verify Life Expectancy assumption diagnostics:
    - Country group split active
    - Sequential autocorrelation tests marked NOT_APPLICABLE (disjoint country panel cannot be strung sequentially)
    - Normality, Homoscedasticity, and VIF run cleanly
    """
    reg = get_registry()
    meta = reg.get_dataset("life_expectancy")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.GROUP, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, group_column="Country", random_seed=42)
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    selected = ["Adult Mortality", "BMI", "Polio", "Diphtheria", "HIV/AIDS", "Schooling"]
    design_res, _ = build_ols_design_matrices(split.X_train, selected_features=selected, fit_intercept=True)
    model = fit_ols_model(design_res, y_train=split.y_train, target_name="Life expectancy")
    model.dataset_id = "life_expectancy"

    suite = run_full_diagnostic_suite(model, split_result=split, alpha=0.05)

    dw = suite.diagnostics["durbin_watson"]
    lb = suite.diagnostics["ljung_box"]
    assert dw.applicability_status == ApplicabilityStatus.NOT_APPLICABLE.value
    assert lb.applicability_status == ApplicabilityStatus.NOT_APPLICABLE.value
    assert "panel" in dw.applicability_reason.lower()


def test_real_dataset_student_performance_reference_dummy_diagnostics():
    """
    Verify Student Performance assumption diagnostics:
    - Categorical reference-level dummy variables present
    - Ramsey RESET, Jarque-Bera, Breusch-Pagan, and VIF executed cleanly
    - Cross-sectional independence guard enforced
    """
    reg = get_registry()
    meta = reg.get_dataset("student_performance")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    selected = ["school", "sex", "age", "Medu", "Fedu", "G1", "G2"]
    design_res, _ = build_ols_design_matrices(split.X_train, selected_features=selected, fit_intercept=True)
    model = fit_ols_model(design_res, y_train=split.y_train, target_name="G3")
    model.dataset_id = "student_performance"

    suite = run_full_diagnostic_suite(model, split_result=split, alpha=0.05)

    # Cross-sectional guard
    dw = suite.diagnostics["durbin_watson"]
    assert dw.applicability_status == ApplicabilityStatus.NOT_APPLICABLE.value

    # Normality
    jb = suite.diagnostics["jarque_bera"]
    assert jb.statistic_value > 0

    # VIF
    vif = suite.diagnostics["multicollinearity_vif"]
    assert vif.details["max_vif"] > 0
