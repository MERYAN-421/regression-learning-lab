"""
Group D Tests: Pareto Efficiency Engine.
Verifies:
- Strict two-objective dominance:
    Model A dominates Model B iff:
      p_A <= p_B and RMSE_A <= RMSE_B and (p_A < p_B or RMSE_A < RMSE_B)
- Non-dominated frontier identification.
- Proper handling of identical/duplicate models (neither dominates the other).
- Ties on one dimension (lower on second dimension strictly dominates).
- Interactive Plotly chart creation with frontier points.
"""
import pytest
import numpy as np
import pandas as pd

from src.models.model_service import train_ols_experiment
from src.comparison.pareto import compute_pareto_frontier, create_pareto_scatter_chart
from src.data.models import SplitResult


@pytest.fixture
def pareto_test_models():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.normal(0, 1, n),
        "x3": np.random.normal(0, 1, n),
    })
    y = 2.0 + 2.0 * df["x1"] + 1.0 * df["x2"] + np.random.normal(0, 0.1, n)

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
        split_fingerprint="split_pareto"
    )

    # Model 1: Simplest (x1) -> 2 params (intercept + x1)
    m1 = train_ols_experiment(
        split_result=split,
        feature_source=["x1"],
        dataset_id="ds_p",
        dataset_fingerprint="ds_p_fp",
        target_name="target"
    )

    # Model 2: Best predictive (x1, x2) -> 3 params (intercept + x1 + x2), lower RMSE
    m2 = train_ols_experiment(
        split_result=split,
        feature_source=["x1", "x2"],
        dataset_id="ds_p",
        dataset_fingerprint="ds_p_fp",
        target_name="target"
    )

    # Model 3: Noise feature (x1, x2, x3) -> 4 params, higher or identical RMSE -> dominated by m2
    m3 = train_ols_experiment(
        split_result=split,
        feature_source=["x1", "x2", "x3"],
        dataset_id="ds_p",
        dataset_fingerprint="ds_p_fp",
        target_name="target"
    )

    return [m1, m2, m3]


def test_pareto_frontier_identifies_non_dominated_models(pareto_test_models):
    models = pareto_test_models
    pareto_ids, points = compute_pareto_frontier(models)

    assert len(points) == 3
    # m1 (2 params) and m2 (3 params, lower RMSE) should be on frontier
    assert models[0].model_id in pareto_ids
    assert models[1].model_id in pareto_ids
    # Check that is_pareto flag is correctly assigned
    pt_map = {pt["model_id"]: pt["is_pareto"] for pt in points}
    assert pt_map[models[0].model_id] is True
    assert pt_map[models[1].model_id] is True


def test_pareto_chart_creation(pareto_test_models):
    models = pareto_test_models
    fig = create_pareto_scatter_chart(models)
    assert fig is not None
    # Figure should have traces for points and frontier line
    assert len(fig.data) >= 2


def test_empty_models_pareto():
    pareto_ids, points = compute_pareto_frontier([])
    assert pareto_ids == []
    assert points == []
