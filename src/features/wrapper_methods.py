"""
Wrapper-based feature selection methods:
1. Group-aware Recursive Feature Elimination (Group RFE)
2. Group-aware Forward Sequential Feature Selection (Forward SFS)
3. Group-aware Backward Sequential Feature Selection (Backward SFS)

Operates strictly at the original-feature-group level, ensuring that dummy one-hot columns
belonging to the same categorical predictor are eliminated or added together as a cohesive unit.
"""
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from src.features.result_models import OriginalFeatureMapping


def select_by_group_rfe(
    X_encoded: np.ndarray,
    y: np.ndarray,
    mappings: List[OriginalFeatureMapping],
    k: int
) -> List[OriginalFeatureMapping]:
    """
    Group-aware Recursive Feature Elimination (Group RFE) with LinearRegression.
    Eliminates one original feature group per step based on group coefficient L2-norm.
    """
    k_target = min(max(1, k), len(mappings))
    active_orig_names = [m.original_name for m in mappings]
    mapping_dict = {m.original_name: m for m in mappings}

    # Elimination history: store (name, elimination_step, score)
    eliminated: List[Tuple[str, float]] = []

    while len(active_orig_names) > k_target:
        # Collect active encoded column indices
        active_indices = []
        for name in active_orig_names:
            active_indices.extend(mapping_dict[name].encoded_column_indices)
        active_indices = sorted(list(set(active_indices)))

        sub_X = X_encoded[:, active_indices]
        reg = LinearRegression().fit(sub_X, y)
        coefs = reg.coef_

        # Map coefs back to active indices
        idx_to_coef = {idx: coefs[col_i] for col_i, idx in enumerate(active_indices)}

        # Compute group score: L2 norm of coefficients
        group_scores: Dict[str, float] = {}
        for name in active_orig_names:
            m = mapping_dict[name]
            g_coefs = [idx_to_coef[idx] for idx in m.encoded_column_indices if idx in idx_to_coef]
            g_norm = float(np.sqrt(np.sum(np.square(g_coefs)))) if g_coefs else 0.0
            group_scores[name] = g_norm

        # Find original feature group with weakest coefficient magnitude
        worst_feature = min(active_orig_names, key=lambda name: group_scores[name])
        eliminated.append((worst_feature, group_scores[worst_feature]))
        active_orig_names.remove(worst_feature)

    # For remaining features, get their final score
    active_indices = []
    for name in active_orig_names:
        active_indices.extend(mapping_dict[name].encoded_column_indices)
    active_indices = sorted(list(set(active_indices)))

    if active_indices:
        sub_X = X_encoded[:, active_indices]
        reg = LinearRegression().fit(sub_X, y)
        coefs = reg.coef_
        idx_to_coef = {idx: coefs[col_i] for col_i, idx in enumerate(active_indices)}
        final_scores: Dict[str, float] = {}
        for name in active_orig_names:
            m = mapping_dict[name]
            g_coefs = [idx_to_coef[idx] for idx in m.encoded_column_indices if idx in idx_to_coef]
            final_scores[name] = float(np.sqrt(np.sum(np.square(g_coefs)))) if g_coefs else 0.0
    else:
        final_scores = {name: 1.0 for name in active_orig_names}

    # Rank active features 1..k_target sorted descending by score
    sorted_active = sorted(active_orig_names, key=lambda name: final_scores[name], reverse=True)

    updated_dict = {}
    for rank_idx, name in enumerate(sorted_active):
        m = mapping_dict[name]
        updated_dict[name] = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=True,
            rank=rank_idx + 1,
            score=final_scores[name],
            details={"metric": "Group RFE coefficient L2 norm"}
        )

    # Assign ranks for eliminated features (from k_target+1 to total)
    # Most recently eliminated has rank k_target+1
    for rank_offset, (name, sc) in enumerate(reversed(eliminated)):
        m = mapping_dict[name]
        updated_dict[name] = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=False,
            rank=k_target + 1 + rank_offset,
            score=sc,
            details={"metric": "Group RFE elimination score"}
        )

    return [updated_dict[m.original_name] for m in mappings]


def _evaluate_feature_subset_cv(
    X_encoded: np.ndarray,
    y: np.ndarray,
    active_indices: List[int],
    cv_folds: int = 3
) -> float:
    """Internal helper: computes mean CV RMSE for a subset of encoded column indices."""
    if not active_indices:
        return 1e9

    n_samples = len(y)
    fold_size = n_samples // cv_folds
    rmses = []

    sub_X = X_encoded[:, active_indices]

    for f in range(cv_folds):
        val_idx = np.arange(f * fold_size, min((f + 1) * fold_size, n_samples))
        tr_idx = np.setdiff1d(np.arange(n_samples), val_idx)

        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue

        model = LinearRegression().fit(sub_X[tr_idx], y[tr_idx])
        y_pred = model.predict(sub_X[val_idx])
        rmse = float(root_mean_squared_error(y[val_idx], y_pred))
        rmses.append(rmse)

    return float(np.mean(rmses)) if rmses else 1e9


def select_by_group_forward_sfs(
    X_encoded: np.ndarray,
    y: np.ndarray,
    mappings: List[OriginalFeatureMapping],
    k: int,
    cv_folds: int = 3
) -> List[OriginalFeatureMapping]:
    """
    Group-aware Forward Sequential Feature Selection (Forward SFS).
    Greedily adds one original feature group per step to minimize internal CV RMSE.
    """
    k_target = min(max(1, k), len(mappings))
    mapping_dict = {m.original_name: m for m in mappings}
    available_names = [m.original_name for m in mappings]
    selected_names: List[str] = []
    step_scores: Dict[str, float] = {}

    while len(selected_names) < k_target and available_names:
        best_candidate = None
        best_candidate_score = 1e12

        for cand in available_names:
            trial_names = selected_names + [cand]
            trial_indices = []
            for name in trial_names:
                trial_indices.extend(mapping_dict[name].encoded_column_indices)
            trial_indices = sorted(list(set(trial_indices)))

            score = _evaluate_feature_subset_cv(X_encoded, y, trial_indices, cv_folds=cv_folds)
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate = cand

        if best_candidate is not None:
            selected_names.append(best_candidate)
            available_names.remove(best_candidate)
            step_scores[best_candidate] = best_candidate_score
        else:
            break

    # Build updated mappings
    updated_dict = {}
    for rank_idx, name in enumerate(selected_names):
        m = mapping_dict[name]
        updated_dict[name] = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=True,
            rank=rank_idx + 1,
            score=step_scores.get(name, 0.0),
            details={"metric": "Forward SFS Step CV RMSE"}
        )

    for rank_offset, name in enumerate(available_names):
        m = mapping_dict[name]
        updated_dict[name] = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=False,
            rank=k_target + 1 + rank_offset,
            score=1e9,
            details={"metric": "Forward SFS Unselected"}
        )

    return [updated_dict[m.original_name] for m in mappings]


def select_by_group_backward_sfs(
    X_encoded: np.ndarray,
    y: np.ndarray,
    mappings: List[OriginalFeatureMapping],
    k: int,
    cv_folds: int = 3
) -> List[OriginalFeatureMapping]:
    """
    Group-aware Backward Sequential Feature Selection (Backward SFS).
    Iteratively removes the original feature group that produces the lowest CV RMSE upon removal.
    """
    k_target = min(max(1, k), len(mappings))
    mapping_dict = {m.original_name: m for m in mappings}
    active_names = [m.original_name for m in mappings]
    eliminated: List[Tuple[str, float]] = []

    while len(active_names) > k_target:
        best_to_remove = None
        best_remaining_score = 1e12

        for cand in active_names:
            trial_names = [n for n in active_names if n != cand]
            trial_indices = []
            for name in trial_names:
                trial_indices.extend(mapping_dict[name].encoded_column_indices)
            trial_indices = sorted(list(set(trial_indices)))

            score = _evaluate_feature_subset_cv(X_encoded, y, trial_indices, cv_folds=cv_folds)
            if score < best_remaining_score:
                best_remaining_score = score
                best_to_remove = cand

        if best_to_remove is not None:
            eliminated.append((best_to_remove, best_remaining_score))
            active_names.remove(best_to_remove)
        else:
            break

    # Build updated mappings
    updated_dict = {}
    for rank_idx, name in enumerate(active_names):
        m = mapping_dict[name]
        updated_dict[name] = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=True,
            rank=rank_idx + 1,
            score=1.0,
            details={"metric": "Backward SFS Retained Feature"}
        )

    for rank_offset, (name, sc) in enumerate(reversed(eliminated)):
        m = mapping_dict[name]
        updated_dict[name] = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=False,
            rank=k_target + 1 + rank_offset,
            score=sc,
            details={"metric": "Backward SFS Step CV RMSE after removal"}
        )

    return [updated_dict[m.original_name] for m in mappings]
