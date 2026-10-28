"""
Wrapper-based feature selection methods:
1. Group-aware Recursive Feature Elimination (Group RFE)
2. Group-aware Forward Sequential Feature Selection (Forward SFS)
3. Group-aware Backward Sequential Feature Selection (Backward SFS)

Operates strictly at the original-feature-group level, ensuring that dummy one-hot columns
belonging to the same categorical predictor are eliminated or added together as a cohesive unit.
SFS internal evaluation enforces strict inner-fold isolation: fitting imputers, encoders,
and scalers exclusively on inner fold-training data using dataset-appropriate CV strategies
(TimeSeriesSplit, GroupKFold, KFold).
"""
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from src.features.result_models import OriginalFeatureMapping
from src.features.preprocessing import fit_train_preprocessor, transform_features


def select_by_group_rfe(
    X_input: Union[pd.DataFrame, np.ndarray],
    y_input: Union[pd.Series, np.ndarray],
    mappings: List[OriginalFeatureMapping],
    k: int
) -> List[OriginalFeatureMapping]:
    """
    Group-aware Recursive Feature Elimination (Group RFE) with LinearRegression.
    Eliminates one original feature group per step based on group coefficient L2-norm:
        ||beta_g||_2 = sqrt(sum_{j in group} beta_j^2)
    """
    k_target = min(max(1, k), len(mappings))
    active_orig_names = [m.original_name for m in mappings]
    mapping_dict = {m.original_name: m for m in mappings}

    if isinstance(X_input, pd.DataFrame):
        prep, _, _ = fit_train_preprocessor(X_input, scale_numeric=True)
        X_encoded = prep.transform(X_input)
        y = y_input.values if isinstance(y_input, pd.Series) else y_input
    else:
        X_encoded = X_input
        y = y_input

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


def _evaluate_candidate_subset_inner_cv(
    subset_orig_names: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    inner_splits: List[Tuple[np.ndarray, np.ndarray]]
) -> float:
    """
    Evaluates candidate original-feature subset with strict inner-fold isolation.
    Fits imputer, encoder, and scaler strictly on each inner fold-training slice.
    """
    if not subset_orig_names:
        return 1e9

    fold_rmses: List[float] = []
    for tr_idx, val_idx in inner_splits:
        sub_X_tr = X_train.iloc[tr_idx][subset_orig_names]
        sub_y_tr = y_train.iloc[tr_idx].values
        sub_X_val = X_train.iloc[val_idx][subset_orig_names]
        sub_y_val = y_train.iloc[val_idx].values

        # Strict fold isolation: fit preprocessor only on inner fold train
        inner_prep, _, _ = fit_train_preprocessor(sub_X_tr, scale_numeric=True)
        X_tr_enc = transform_features(inner_prep, sub_X_tr).values
        X_val_enc = transform_features(inner_prep, sub_X_val).values

        model = LinearRegression().fit(X_tr_enc, sub_y_tr)
        y_pred = model.predict(X_val_enc)
        rmse = float(root_mean_squared_error(sub_y_val, y_pred))
        fold_rmses.append(rmse)

    return float(np.mean(fold_rmses)) if fold_rmses else 1e9


def select_by_group_forward_sfs(
    X_input: Union[pd.DataFrame, np.ndarray],
    y_input: Union[pd.Series, np.ndarray],
    mappings: List[OriginalFeatureMapping],
    k: int,
    cv_strategy: str = "random",
    cv_folds: int = 3,
    random_seed: int = 42,
    groups_train: Optional[pd.Series] = None
) -> List[OriginalFeatureMapping]:
    """
    Group-aware Forward Sequential Feature Selection (Forward SFS).
    Greedily adds one original feature group per step to minimize inner CV RMSE.
    Enforces fold-isolated preprocessing and dataset-appropriate cross-validation.
    """
    k_target = min(max(1, k), len(mappings))
    mapping_dict = {m.original_name: m for m in mappings}
    available_names = [m.original_name for m in mappings]
    selected_names: List[str] = []
    step_scores: Dict[str, float] = {}

    # Setup DataFrame and CV splits
    if isinstance(X_input, pd.DataFrame):
        X_df = X_input
        y_s = y_input if isinstance(y_input, pd.Series) else pd.Series(y_input, index=X_input.index)
    else:
        # Array fallback (for synthetic unit tests)
        col_names = [m.original_name for m in mappings]
        X_df = pd.DataFrame(X_input, columns=col_names)
        y_s = pd.Series(y_input)

    # Reject missing group labels for group CV strategy
    if cv_strategy == "group":
        if groups_train is None or len(groups_train) != len(X_df):
            raise ValueError(
                f"Group cross-validation strategy requires aligned group labels (e.g. Country). "
                f"Expected {len(X_df)} items, got {len(groups_train) if groups_train is not None else 'None'}."
            )

    # Import cv splitter
    from src.features.evaluation import get_dataset_cv_splitter
    splitter = get_dataset_cv_splitter(cv_strategy, cv_folds=cv_folds, random_seed=random_seed)
    if cv_strategy == "group":
        inner_splits = list(splitter.split(X_df, y_s, groups=groups_train))
    else:
        inner_splits = list(splitter.split(X_df, y_s))

    while len(selected_names) < k_target and available_names:
        best_candidate = None
        best_candidate_score = float("inf")

        for cand in available_names:
            trial_names = selected_names + [cand]
            score = _evaluate_candidate_subset_inner_cv(trial_names, X_df, y_s, inner_splits)

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
    X_input: Union[pd.DataFrame, np.ndarray],
    y_input: Union[pd.Series, np.ndarray],
    mappings: List[OriginalFeatureMapping],
    k: int,
    cv_strategy: str = "random",
    cv_folds: int = 3,
    random_seed: int = 42,
    groups_train: Optional[pd.Series] = None
) -> List[OriginalFeatureMapping]:
    """
    Group-aware Backward Sequential Feature Selection (Backward SFS).
    Iteratively removes the original feature group that produces the lowest inner CV RMSE upon removal.
    Enforces fold-isolated preprocessing and dataset-appropriate cross-validation.
    """
    k_target = min(max(1, k), len(mappings))
    mapping_dict = {m.original_name: m for m in mappings}
    active_names = [m.original_name for m in mappings]
    eliminated: List[Tuple[str, float]] = []

    if isinstance(X_input, pd.DataFrame):
        X_df = X_input
        y_s = y_input if isinstance(y_input, pd.Series) else pd.Series(y_input, index=X_input.index)
    else:
        col_names = [m.original_name for m in mappings]
        X_df = pd.DataFrame(X_input, columns=col_names)
        y_s = pd.Series(y_input)

    if cv_strategy == "group":
        if groups_train is None or len(groups_train) != len(X_df):
            raise ValueError(
                f"Group cross-validation strategy requires aligned group labels (e.g. Country). "
                f"Expected {len(X_df)} items, got {len(groups_train) if groups_train is not None else 'None'}."
            )

    from src.features.evaluation import get_dataset_cv_splitter
    splitter = get_dataset_cv_splitter(cv_strategy, cv_folds=cv_folds, random_seed=random_seed)
    if cv_strategy == "group":
        inner_splits = list(splitter.split(X_df, y_s, groups=groups_train))
    else:
        inner_splits = list(splitter.split(X_df, y_s))

    while len(active_names) > k_target:
        best_to_remove = None
        best_remaining_score = float("inf")

        for cand in active_names:
            trial_names = [n for n in active_names if n != cand]
            score = _evaluate_candidate_subset_inner_cv(trial_names, X_df, y_s, inner_splits)

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
