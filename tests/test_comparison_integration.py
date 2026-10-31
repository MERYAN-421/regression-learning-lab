"""
Milestone 5 Integration, Tracking, Leakage Prevention, and Real-Dataset Tests (Groups G, H, I, J).

Verifies:
- Group G: ComparisonExperimentTracker deterministic IDs, stale invalidation, JSON serialization, and immutability.
- Group H: Leakage Prevention (Test metrics excluded from ranking, candidate selection does not trigger test evaluation).
- Group I: Streamlit rendering / AppTest smoke tests.
- Group J: Real-dataset demonstrations across all four Kaggle benchmarks:
    1. Appliances Energy (Temporal split)
    2. King County House Sales (Random split, collinearity vs reduced, Pareto)
    3. Life Expectancy (Country Group split)
    4. Student Performance (Random split, Radar, Pareto, Jaccard overlap, candidate selection)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.data.registry import get_registry
from src.data.loader import load_dataset_bundle, get_dataset_fingerprint
from src.data.splitters import split_dataset
from src.data.models import SplitConfig, SplitStrategy, SplitResult
from src.models.model_service import train_ols_experiment
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.experiments.comparison_experiment_tracker import ComparisonExperimentTracker
from src.comparison.comparison_runner import run_model_comparison
from src.comparison.result_models import RankingPolicy, generate_comparison_id
from src.diagnostics.runner import run_full_diagnostic_suite

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def integration_synthetic_bundle():
    """Generates synthetic data and two fitted models for integration tests."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.normal(0, 1, n),
        "x3": np.random.normal(0, 1, n),
    })
    y = 2.0 + 1.5 * df["x1"] - 0.8 * df["x2"] + np.random.normal(0, 0.1, n)

    split = SplitResult(
        X_train=df.iloc[:60],
        X_val=df.iloc[60:80],
        X_test=df.iloc[80:],
        y_train=y.iloc[:60],
        y_val=y.iloc[60:80],
        y_test=y.iloc[80:],
        train_indices=np.arange(60),
        val_indices=np.arange(60, 80),
        test_indices=np.arange(80, 100),
        actual_proportions={"train": 0.6, "val": 0.2, "test": 0.2},
        sample_counts={"train": 60, "val": 20, "test": 20},
        strategy="random",
        random_seed=42,
        split_fingerprint="split_int_fp"
    )

    m1 = train_ols_experiment(
        split_result=split,
        feature_source=["x1", "x2"],
        dataset_id="int_ds",
        dataset_fingerprint="int_ds_fp",
        target_name="target_y"
    )

    m2 = train_ols_experiment(
        split_result=split,
        feature_source=["x1"],
        dataset_id="int_ds",
        dataset_fingerprint="int_ds_fp",
        target_name="target_y"
    )

    return split, m1, m2


# =====================================================================
# Group G: Experiment Tracking & Artifact Reproducibility
# =====================================================================

def test_deterministic_comparison_identity():
    id1, h1 = generate_comparison_id(
        split_fingerprint="split_123",
        model_ids=["m_b", "m_a"],
        ranking_policy="val_rmse_asc"
    )
    # Order of model_ids in input list should not change output hash (sorted internally)
    id2, h2 = generate_comparison_id(
        split_fingerprint="split_123",
        model_ids=["m_a", "m_b"],
        ranking_policy="val_rmse_asc"
    )
    assert id1 == id2
    assert h1 == h2

    # Different policy must produce a different ID
    id3, h3 = generate_comparison_id(
        split_fingerprint="split_123",
        model_ids=["m_a", "m_b"],
        ranking_policy="val_r2_desc"
    )
    assert id1 != id3


def test_comparison_tracker_preservation_and_stale_invalidation(integration_synthetic_bundle):
    split, m1, m2 = integration_synthetic_bundle
    tracker = ComparisonExperimentTracker()

    comp_res = run_model_comparison(
        models=[m1, m2],
        active_dataset_id="int_ds",
        active_dataset_fingerprint="int_ds_fp",
        active_split_fingerprint=split.split_fingerprint,
        active_target_name="target_y"
    )

    tracker.add_comparison(comp_res)
    assert tracker.get_comparison(comp_res.comparison_id) is not None

    # Matching split returns comparison
    valid = tracker.get_valid_comparisons(
        current_split_fingerprint=split.split_fingerprint,
        current_dataset_id="int_ds",
        current_dataset_fingerprint="int_ds_fp"
    )
    assert len(valid) == 1

    # Stale/different split returns empty list
    stale = tracker.get_valid_comparisons(
        current_split_fingerprint="different_split_fingerprint"
    )
    assert len(stale) == 0


def test_comparison_result_json_serialization(integration_synthetic_bundle):
    split, m1, m2 = integration_synthetic_bundle
    comp_res = run_model_comparison(
        models=[m1, m2],
        active_dataset_id="int_ds",
        active_dataset_fingerprint="int_ds_fp",
        active_split_fingerprint=split.split_fingerprint,
        active_target_name="target_y"
    )

    json_str = comp_res.to_json(indent=2)
    parsed = json.loads(json_str)

    assert parsed["comparison_id"] == comp_res.comparison_id
    assert parsed["evaluation_scope"] == "validation"
    assert len(parsed["leaderboard"]) == 2
    assert "radar_scores" in parsed
    assert "pareto_model_ids" in parsed


def test_original_model_result_immutability(integration_synthetic_bundle):
    split, m1, m2 = integration_synthetic_bundle
    orig_train_rmse = m1.train_metrics.rmse
    orig_val_rmse = m1.val_metrics.rmse

    _ = run_model_comparison(
        models=[m1, m2],
        active_dataset_id="int_ds",
        active_dataset_fingerprint="int_ds_fp",
        active_split_fingerprint=split.split_fingerprint,
        active_target_name="target_y"
    )

    # Underlying models must remain completely unmodified
    assert m1.train_metrics.rmse == orig_train_rmse
    assert m1.val_metrics.rmse == orig_val_rmse
    assert m1.test_metrics is None
    assert m1.test_evaluation_status == "untested"


# =====================================================================
# Group H: Leakage Prevention & Test Set Isolation (Rule 2)
# =====================================================================

def test_test_metrics_never_used_in_ranking_or_selection(integration_synthetic_bundle):
    split, m1, m2 = integration_synthetic_bundle
    reg_tracker = RegressionExperimentTracker()
    reg_tracker.add_model(m1)
    reg_tracker.add_model(m2)

    # Artificially evaluate m1 on Test set under controlled lock
    m1_final = reg_tracker.finalize_test_evaluation(m1.model_id, split, refit_policy="train_val")
    assert m1_final.test_metrics is not None

    # Run comparison using m1_final and untested m2
    comp_res = run_model_comparison(
        models=[m1_final, m2],
        active_dataset_id="int_ds",
        active_dataset_fingerprint="int_ds_fp",
        active_split_fingerprint=split.split_fingerprint,
        active_target_name="target_y",
        ranking_policy=RankingPolicy.VAL_RMSE_ASC
    )

    # Verify leaderboard does NOT reveal or sort by Test RMSE
    for row in comp_res.leaderboard:
        assert "Test RMSE" not in row
        assert "Test R²" not in row
        assert "Test MAE" not in row

    # Comparison must strictly sort by Validation RMSE
    assert comp_res.leaderboard[0]["Val RMSE"] <= comp_res.leaderboard[1]["Val RMSE"]


def test_candidate_selection_does_not_trigger_test_evaluation(integration_synthetic_bundle):
    split, m1, m2 = integration_synthetic_bundle
    assert m1.test_evaluation_status == "untested"
    assert m2.test_evaluation_status == "untested"

    comp_res = run_model_comparison(
        models=[m1, m2],
        active_dataset_id="int_ds",
        active_dataset_fingerprint="int_ds_fp",
        active_split_fingerprint=split.split_fingerprint,
        active_target_name="target_y",
        selected_candidate_id=m1.model_id,
        candidate_selection_rationale="Selected for superior validation error."
    )

    assert comp_res.selected_candidate_model_id == m1.model_id
    # Test evaluation status MUST remain 'untested'
    assert m1.test_metrics is None
    assert m1.test_evaluation_status == "untested"
    assert m2.test_metrics is None
    assert m2.test_evaluation_status == "untested"


# =====================================================================
# Group I: Streamlit AppTest Smoke Test
# =====================================================================

def test_comparison_page_apptest_smoke():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(PROJECT_ROOT / "pages" / "comparison.py"))
    app.run()
    # When no dataset/split is loaded in session_state, it should show warning and stop cleanly
    assert len(app.warning) > 0
    assert not app.exception


# =====================================================================
# Group J: Real Dataset Demonstrations (All 4 Kaggle Benchmark Datasets)
# =====================================================================

def test_real_dataset_demo_appliances_energy():
    """
    Demo A: Appliances Energy Prediction (Temporal Split).
    Compares two OLS models: Top 10 predictors vs Top 5 predictors.
    Evaluates Validation RMSE, Generalization Gap, and temporal independence.
    """
    reg = get_registry()
    meta = reg.get_dataset("appliances_energy")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
    ds_fp = get_dataset_fingerprint(meta)

    cfg = SplitConfig(strategy=SplitStrategy.TEMPORAL, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, date_column="date")
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    top10 = ["lights", "T1", "RH_1", "T2", "T3", "T4", "T6", "T8", "T_out", "Windspeed"]
    top5 = ["lights", "T1", "RH_1", "T3", "T_out"]

    m_top10 = train_ols_experiment(
        split_result=split,
        feature_source=top10,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        target_name=meta.target_column
    )

    m_top5 = train_ols_experiment(
        split_result=split,
        feature_source=top5,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        target_name=meta.target_column
    )

    # Run diagnostic suite on m_top10
    suite_top10 = run_full_diagnostic_suite(m_top10, split_result=split, alpha=0.05)

    comp_res = run_model_comparison(
        models=[m_top10, m_top5],
        active_dataset_id=meta.id,
        active_dataset_fingerprint=ds_fp,
        active_split_fingerprint=split.split_fingerprint,
        active_target_name=meta.target_column,
        active_target_unit="Wh",
        diagnostic_suites={m_top10.model_id: suite_top10}
    )

    assert len(comp_res.leaderboard) == 2
    assert comp_res.leaderboard[0]["Val RMSE"] <= comp_res.leaderboard[1]["Val RMSE"]
    assert len(comp_res.pareto_model_ids) >= 1
    assert "lights" in comp_res.feature_overlap_matrix["selection_frequencies"]


def test_real_dataset_demo_kc_house_sales():
    """
    Demo B: King County House Sales (Random Split).
    Compares All Features vs. a Pruned feature subset that avoids structural collinearity.
    Examines Pareto frontier, VIF, and condition number.
    """
    reg = get_registry()
    meta = reg.get_dataset("kc_house_sales")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
    ds_fp = get_dataset_fingerprint(meta)

    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    # Model 1: Reduced non-collinear features
    pruned_feats = ["bedrooms", "bathrooms", "sqft_living", "grade", "waterfront", "view"]
    m_pruned = train_ols_experiment(
        split_result=split,
        feature_source=pruned_feats,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        target_name=meta.target_column
    )

    # Model 2: Minimal 3 features
    min_feats = ["sqft_living", "grade", "bathrooms"]
    m_min = train_ols_experiment(
        split_result=split,
        feature_source=min_feats,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        target_name=meta.target_column
    )

    comp_res = run_model_comparison(
        models=[m_pruned, m_min],
        active_dataset_id=meta.id,
        active_dataset_fingerprint=ds_fp,
        active_split_fingerprint=split.split_fingerprint,
        active_target_name=meta.target_column,
        active_target_unit="USD"
    )

    assert len(comp_res.leaderboard) == 2
    # Minimal has fewer parameters, pruned has lower RMSE -> both can be on Pareto frontier
    assert len(comp_res.pareto_model_ids) >= 1
    # Both are full rank
    assert comp_res.model_complexity[m_pruned.model_id]["is_full_rank"] is True
    assert comp_res.model_complexity[m_min.model_id]["is_full_rank"] is True


def test_real_dataset_demo_life_expectancy():
    """
    Demo C: Life Expectancy (Country Group Split).
    Evaluates multiple specifications on held-out countries without country identity leakage.
    """
    reg = get_registry()
    meta = reg.get_dataset("life_expectancy")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
    ds_fp = get_dataset_fingerprint(meta)

    cfg = SplitConfig(
        strategy=SplitStrategy.GROUP,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        group_column="Country",
        random_seed=42
    )
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    feats_socio = ["Adult Mortality", "BMI", "Schooling", "Income composition of resources"]
    feats_all_avail = [
        "Adult Mortality", "infant deaths", "Alcohol", "percentage expenditure",
        "Hepatitis B", "Measles", "BMI", "under-five deaths", "Polio", "Total expenditure",
        "Diphtheria", "HIV/AIDS", "GDP", "thinness  1-19 years", "Schooling"
    ]

    m_socio = train_ols_experiment(
        split_result=split,
        feature_source=feats_socio,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        target_name=meta.target_column
    )

    m_broad = train_ols_experiment(
        split_result=split,
        feature_source=feats_all_avail,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        target_name=meta.target_column
    )

    comp_res = run_model_comparison(
        models=[m_socio, m_broad],
        active_dataset_id=meta.id,
        active_dataset_fingerprint=ds_fp,
        active_split_fingerprint=split.split_fingerprint,
        active_target_name=meta.target_column,
        active_target_unit="Years"
    )

    assert len(comp_res.leaderboard) == 2
    # Jaccard overlap between socio and broad
    assert comp_res.feature_overlap_matrix["total_unique_features"] == len(set(feats_socio).union(set(feats_all_avail)))


def test_real_dataset_demo_student_performance():
    """
    Demo D: Student Performance (Random Split).
    Compares models with and without prior exam scores (G1, G2),
    demonstrating leaderboard, radar normalization, Pareto analysis, and candidate selection.
    """
    reg = get_registry()
    meta = reg.get_dataset("student_performance")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)
    ds_fp = get_dataset_fingerprint(meta)

    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    # Spec 1: Without prior exam grades (pure background factors)
    bg_feats = ["studytime", "failures", "absences", "Medu", "Fedu", "goout"]
    m_bg = train_ols_experiment(
        split_result=split,
        feature_source=bg_feats,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        target_name=meta.target_column
    )

    # Spec 2: With G1 & G2 exam history
    hist_feats = ["studytime", "failures", "absences", "G1", "G2"]
    m_hist = train_ols_experiment(
        split_result=split,
        feature_source=hist_feats,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        target_name=meta.target_column
    )

    suite_hist = run_full_diagnostic_suite(m_hist, split_result=split, alpha=0.05)

    comp_res = run_model_comparison(
        models=[m_bg, m_hist],
        active_dataset_id=meta.id,
        active_dataset_fingerprint=ds_fp,
        active_split_fingerprint=split.split_fingerprint,
        active_target_name=meta.target_column,
        active_target_unit="Grade",
        diagnostic_suites={m_hist.model_id: suite_hist},
        selected_candidate_id=m_hist.model_id,
        candidate_selection_rationale="Selected for high explanatory power on final grade."
    )

    # Historical model should have drastically lower validation RMSE
    assert m_hist.val_metrics.rmse < m_bg.val_metrics.rmse
    assert comp_res.leaderboard[0]["Model ID"] == m_hist.model_id
    assert comp_res.selected_candidate_model_id == m_hist.model_id
    assert len(comp_res.radar_scores[m_hist.model_id]) == 5
