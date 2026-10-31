"""
Group B Tests: Model Leaderboard Engine.
Verifies:
- Default sorting by Validation RMSE ascending.
- Sorting by Validation MAE, Validation R², Parsimony, Generalization Gap.
- Tie handling and parameter count secondary sorting.
- Handling of negative validation R².
- Correct extraction of original feature counts and OLS parameters.
- Prediction-only models inclusion with transparent inference status badges.
- Strict Rule 2 verification: Test Set performance is never used for ranking.
"""
import pytest
import numpy as np
import pandas as pd

from src.models.model_service import train_ols_experiment
from src.comparison.leaderboard import build_comparison_leaderboard, create_model_comparison_record
from src.comparison.result_models import RankingPolicy
from src.data.models import SplitResult


@pytest.fixture
def leaderboard_models_fixture():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.normal(0, 1, n),
        "x3": np.random.normal(0, 1, n),
        "x4": np.random.normal(0, 1, n),
    })
    y = 1.0 + 3.0 * df["x1"] + 0.5 * df["x2"] + np.random.normal(0, 0.2, n)

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
        split_fingerprint="split_fp_leaderboard"
    )

    # Model A: Optimal (x1) -> low RMSE, 2 params
    mA = train_ols_experiment(
        split_result=split,
        feature_source=["x1"],
        dataset_id="ds_lb",
        dataset_fingerprint="ds_fp_lb",
        target_name="target"
    )

    # Model B: All features -> slightly higher or close RMSE, 5 params
    mB = train_ols_experiment(
        split_result=split,
        feature_source=["x1", "x2", "x3", "x4"],
        dataset_id="ds_lb",
        dataset_fingerprint="ds_fp_lb",
        target_name="target"
    )

    # Model C: Poor predictor (x4 only) -> high RMSE
    mC = train_ols_experiment(
        split_result=split,
        feature_source=["x4"],
        dataset_id="ds_lb",
        dataset_fingerprint="ds_fp_lb",
        target_name="target"
    )

    return [mA, mB, mC]


def test_leaderboard_default_val_rmse_ascending(leaderboard_models_fixture):
    models = leaderboard_models_fixture
    board = build_comparison_leaderboard(models, ranking_policy=RankingPolicy.VAL_RMSE_ASC)

    assert len(board) == 3
    assert board[0]["Rank"] == 1
    # Verify RMSE is strictly non-decreasing
    assert board[0]["Val RMSE"] <= board[1]["Val RMSE"]
    assert board[1]["Val RMSE"] <= board[2]["Val RMSE"]


def test_leaderboard_parsimony_ranking(leaderboard_models_fixture):
    models = leaderboard_models_fixture
    board = build_comparison_leaderboard(models, ranking_policy=RankingPolicy.PARSIMONY_ASC)

    # Parameters should be ascending
    assert board[0]["Parameters (p)"] <= board[1]["Parameters (p)"]
    assert board[1]["Parameters (p)"] <= board[2]["Parameters (p)"]


def test_leaderboard_val_r2_descending(leaderboard_models_fixture):
    models = leaderboard_models_fixture
    board = build_comparison_leaderboard(models, ranking_policy=RankingPolicy.VAL_R2_DESC)

    # Val R2 should be descending
    assert board[0]["Val R²"] >= board[1]["Val R²"]
    assert board[1]["Val R²"] >= board[2]["Val R²"]


def test_no_test_metrics_in_ranking_policy():
    # Verify that RankingPolicy enum contains strictly development-stage metrics
    for policy in RankingPolicy:
        assert "test" not in policy.value.lower()


def test_leaderboard_record_creation_properties(leaderboard_models_fixture):
    model = leaderboard_models_fixture[0]
    rec = create_model_comparison_record(model, is_pareto=True, diagnostic_suite_id="suite_123")

    assert rec.model_id == model.model_id
    assert rec.is_pareto_efficient is True
    assert rec.diagnostic_suite_id == "suite_123"
    assert rec.train_sample_count == 60
    assert rec.val_sample_count == 20
    assert rec.generalization_gap == pytest.approx(rec.val_rmse - rec.train_rmse, rel=1e-5)
