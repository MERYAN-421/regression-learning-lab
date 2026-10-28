"""
Filter-based feature selection methods:
1. Baseline: All eligible original features.
2. Pearson Correlation: Absolute correlation with target.
3. F-test: Univariate linear regression F-score (f_regression).
4. Mutual Information: Non-linear dependency score (mutual_info_regression).

Implements group-aware aggregation so that categorical features with multiple One-Hot dummies
are evaluated and selected as cohesive original feature groups.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression, mutual_info_regression
from scipy import stats

from src.features.result_models import OriginalFeatureMapping


def select_all_features_baseline(
    mappings: List[OriginalFeatureMapping]
) -> List[OriginalFeatureMapping]:
    """Baseline method: selects all eligible original features."""
    updated = []
    for idx, m in enumerate(mappings):
        copy_m = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=True,
            rank=idx + 1,
            score=1.0,
            details={"selection_reason": "Baseline: All features retained"}
        )
        updated.append(copy_m)
    return updated


def select_by_pearson_correlation(
    X_encoded: np.ndarray,
    y: np.ndarray,
    mappings: List[OriginalFeatureMapping],
    k: int
) -> List[OriginalFeatureMapping]:
    """
    Rank and select top-K original features using absolute Pearson correlation with Target.
    For categorical features, aggregates the maximum absolute correlation among its One-Hot dummy columns.
    """
    # Compute correlation of each encoded column with y
    n_features = X_encoded.shape[1]
    encoded_corr = np.zeros(n_features)

    for j in range(n_features):
        col_vals = X_encoded[:, j]
        if np.std(col_vals) > 0 and np.std(y) > 0:
            r, _ = stats.pearsonr(col_vals, y)
            encoded_corr[j] = abs(r) if not np.isnan(r) else 0.0
        else:
            encoded_corr[j] = 0.0

    # Aggregate to original feature level
    scored_mappings = []
    for m in mappings:
        if m.encoded_column_indices:
            group_scores = encoded_corr[m.encoded_column_indices]
            # Aggregation rule: maximum dummy correlation represents the feature's strongest linear component
            group_score = float(np.max(group_scores))
        else:
            group_score = 0.0

        scored_mappings.append((m, group_score))

    # Rank descending by score
    scored_mappings.sort(key=lambda item: item[1], reverse=True)

    updated = []
    k_bounded = min(max(1, k), len(scored_mappings))

    for rank_idx, (m, sc) in enumerate(scored_mappings):
        is_sel = rank_idx < k_bounded
        copy_m = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=is_sel,
            rank=rank_idx + 1,
            score=sc,
            details={"metric": "Max |Pearson r| across encoded columns"}
        )
        updated.append(copy_m)

    # Re-order to match original mapping order
    name_to_m = {m.original_name: m for m in updated}
    return [name_to_m[m.original_name] for m in mappings]


def select_by_f_test(
    X_encoded: np.ndarray,
    y: np.ndarray,
    mappings: List[OriginalFeatureMapping],
    k: int
) -> List[OriginalFeatureMapping]:
    """
    Rank and select top-K original features using univariate linear regression F-scores.
    Aggregates one-hot dummy column F-scores using maximum F-statistic per group.
    """
    # Safe handling of constant columns in f_regression
    var = np.var(X_encoded, axis=0)
    non_constant_mask = var > 1e-8

    f_scores = np.zeros(X_encoded.shape[1])
    if np.any(non_constant_mask):
        sub_X = X_encoded[:, non_constant_mask]
        scores, _ = f_regression(sub_X, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        f_scores[non_constant_mask] = scores

    scored_mappings = []
    for m in mappings:
        if m.encoded_column_indices:
            group_f = f_scores[m.encoded_column_indices]
            group_score = float(np.max(group_f))
        else:
            group_score = 0.0
        scored_mappings.append((m, group_score))

    scored_mappings.sort(key=lambda item: item[1], reverse=True)

    updated = []
    k_bounded = min(max(1, k), len(scored_mappings))

    for rank_idx, (m, sc) in enumerate(scored_mappings):
        is_sel = rank_idx < k_bounded
        copy_m = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=is_sel,
            rank=rank_idx + 1,
            score=sc,
            details={"metric": "Max univariate F-score across encoded columns"}
        )
        updated.append(copy_m)

    name_to_m = {m.original_name: m for m in updated}
    return [name_to_m[m.original_name] for m in mappings]


def select_by_mutual_information(
    X_encoded: np.ndarray,
    y: np.ndarray,
    mappings: List[OriginalFeatureMapping],
    k: int,
    random_seed: int = 42
) -> List[OriginalFeatureMapping]:
    """
    Rank and select top-K original features using Mutual Information regression.
    Captures non-linear dependencies. Aggregates dummy MI scores to original groups.
    """
    # Identify discrete features based on categorical mapping
    discrete_mask = np.zeros(X_encoded.shape[1], dtype=bool)
    for m in mappings:
        if m.feature_type == "categorical":
            discrete_mask[m.encoded_column_indices] = True

    mi_scores = mutual_info_regression(
        X_encoded, y,
        discrete_features=discrete_mask,
        random_state=random_seed
    )
    mi_scores = np.nan_to_num(mi_scores, nan=0.0)

    scored_mappings = []
    for m in mappings:
        if m.encoded_column_indices:
            # Aggregation: sum of mutual information across dummy components
            # (or max; sum reflects total group mutual information)
            group_score = float(np.sum(mi_scores[m.encoded_column_indices]))
        else:
            group_score = 0.0
        scored_mappings.append((m, group_score))

    scored_mappings.sort(key=lambda item: item[1], reverse=True)

    updated = []
    k_bounded = min(max(1, k), len(scored_mappings))

    for rank_idx, (m, sc) in enumerate(scored_mappings):
        is_sel = rank_idx < k_bounded
        copy_m = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=is_sel,
            rank=rank_idx + 1,
            score=sc,
            details={"metric": "Group Mutual Information (Sum of dummy MI)"}
        )
        updated.append(copy_m)

    name_to_m = {m.original_name: m for m in updated}
    return [name_to_m[m.original_name] for m in mappings]
