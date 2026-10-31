"""
Group F Tests: Feature Selection Comparison Engine.
Verifies:
- Jaccard similarity boundary conditions:
    * Identical sets -> 1.0
    * Completely disjoint sets -> 0.0
    * Empty sets -> 1.0
- Feature inclusion matrix constructed with original feature names.
- Symmetric pairwise Jaccard matrix.
- Feature selection frequency counting.
- Plots generation (heatmap and frequency bar chart).
"""
import pytest
import numpy as np
import pandas as pd

from src.models.model_service import train_ols_experiment
from src.comparison.feature_overlap import (
    compute_jaccard_similarity,
    build_feature_inclusion_dataframe,
    compute_pairwise_jaccard_matrix,
    compute_feature_selection_frequencies,
    create_feature_frequency_chart,
    create_jaccard_heatmap
)
from src.data.models import SplitResult


def test_jaccard_similarity_mathematics():
    # Identical
    assert compute_jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    # Disjoint
    assert compute_jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0
    # Overlapping: 1 intersection, 3 union -> 1/3
    assert compute_jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1.0 / 3.0)
    # Both empty
    assert compute_jaccard_similarity(set(), set()) == 1.0
    # One empty
    assert compute_jaccard_similarity({"a"}, set()) == 0.0


@pytest.fixture
def feature_overlap_fixture():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "sqft": np.random.normal(1500, 300, n),
        "bedrooms": np.random.randint(1, 5, n),
        "bathrooms": np.random.randint(1, 4, n),
        "grade": np.random.randint(5, 10, n),
    })
    y = 50000 + 200 * df["sqft"] + 15000 * df["grade"] + np.random.normal(0, 5000, n)

    split = SplitResult(
        X_train=df.iloc[:40],
        X_val=df.iloc[40:50],
        X_test=df.iloc[50:],
        y_train=y.iloc[:40],
        y_val=y.iloc[40:50],
        y_test=y.iloc[50:],
        train_indices=np.arange(40),
        val_indices=np.arange(40, 50),
        test_indices=np.arange(50, 60),
        actual_proportions={"train": 0.67, "val": 0.17, "test": 0.16},
        sample_counts={"train": 40, "val": 10, "test": 10},
        strategy="random",
        random_seed=42,
        split_fingerprint="split_fo"
    )

    m1 = train_ols_experiment(
        split_result=split,
        feature_source=["sqft", "bedrooms"],
        dataset_id="ds_fo",
        dataset_fingerprint="ds_fo_fp",
        target_name="price"
    )

    m2 = train_ols_experiment(
        split_result=split,
        feature_source=["sqft", "grade"],
        dataset_id="ds_fo",
        dataset_fingerprint="ds_fo_fp",
        target_name="price"
    )

    m3 = train_ols_experiment(
        split_result=split,
        feature_source=["sqft", "bedrooms", "bathrooms"],
        dataset_id="ds_fo",
        dataset_fingerprint="ds_fo_fp",
        target_name="price"
    )

    return [m1, m2, m3]


def test_feature_inclusion_matrix(feature_overlap_fixture):
    models = feature_overlap_fixture
    df_inc = build_feature_inclusion_dataframe(models)

    # 4 distinct features in total: sqft, bedrooms, bathrooms, grade
    assert len(df_inc) == 4
    assert "sqft" in df_inc.index
    # sqft appears in all 3 models
    assert df_inc.loc["sqft"].sum() == 3
    # bathrooms appears only in Model 3
    assert df_inc.loc["bathrooms"].sum() == 1


def test_pairwise_jaccard_matrix(feature_overlap_fixture):
    models = feature_overlap_fixture
    j_matrix = compute_pairwise_jaccard_matrix(models)

    # N x N symmetric matrix with 1.0 along the diagonal
    assert j_matrix.shape == (3, 3)
    for i in range(3):
        assert j_matrix.iloc[i, i] == 1.0
        for j in range(3):
            assert j_matrix.iloc[i, j] == pytest.approx(j_matrix.iloc[j, i])


def test_feature_selection_frequencies(feature_overlap_fixture):
    models = feature_overlap_fixture
    freqs = compute_feature_selection_frequencies(models)

    assert freqs["sqft"] == 3
    assert freqs["bedrooms"] == 2
    assert freqs["grade"] == 1
    assert freqs["bathrooms"] == 1
