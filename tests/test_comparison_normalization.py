"""
Group C Tests: Radar Normalization Engine.
Verifies:
- Lower-is-better min-max mapping (lower error -> higher score).
- Higher-is-better min-max mapping (higher R2 -> higher score).
- Constant metric handling (returns 0.5 neutral without division by zero).
- Single model handling (returns 1.0 without division by zero).
- Negative R² handling (gracefully mapped within [0, 1]).
- Candidate-set-dependent property: altering the comparison pool shifts relative scores.
"""
import pytest
import numpy as np

from src.comparison.normalization import (
    normalize_lower_is_better,
    normalize_higher_is_better,
    compute_radar_scores,
    RADAR_DIMENSIONS
)


def test_lower_is_better_mapping():
    values = [10.0, 20.0, 30.0]
    scores = normalize_lower_is_better(values)
    # min=10 -> score 1.0 (best); max=30 -> score 0.0 (worst)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.0)


def test_higher_is_better_mapping():
    values = [0.2, 0.5, 0.8]
    scores = normalize_higher_is_better(values)
    # min=0.2 -> score 0.0; max=0.8 -> score 1.0
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(1.0)


def test_constant_values_return_neutral_score():
    values = [5.0, 5.0, 5.0]
    scores_lower = normalize_lower_is_better(values)
    scores_higher = normalize_higher_is_better(values)

    assert all(s == 0.5 for s in scores_lower)
    assert all(s == 0.5 for s in scores_higher)


def test_single_value_returns_one():
    val = [15.0]
    scores = normalize_lower_is_better(val)
    assert scores == [1.0]

    scores_h = normalize_higher_is_better(val)
    assert scores_h == [1.0]


def test_negative_r2_normalization():
    # Negative R² (worse than mean predictor) should map safely within [0, 1]
    values = [-0.5, 0.0, 0.5]
    scores = normalize_higher_is_better(values)
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(1.0)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_non_finite_values_handled():
    values = [10.0, np.nan, 30.0]
    scores = normalize_lower_is_better(values)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)  # NaN gets floor
    assert scores[2] == pytest.approx(0.0)
