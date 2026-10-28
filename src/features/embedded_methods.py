"""
Embedded feature selection methods:
Lasso (L1 Regularization) Selection with Dataset-Appropriate CV Alpha Selection
and Group-Aware Post-Hoc Aggregation.

Methodology:
1. When alpha is not manually provided, select alpha using strict inner fold-isolated
   cross-validation matching the dataset's splitting strategy (TimeSeriesSplit for temporal,
   GroupKFold for grouped panel, KFold for cross-sectional). Missing group labels are strictly rejected.
2. Fit Lasso on the full training partition with the chosen alpha.
3. Compute original feature group importance via the Euclidean L2 norm of its coefficients:
       S_g = sqrt(sum_{j in group} beta_j^2)
4. Distinguish natural non-zero sparsity from the user's requested K, and provide top-K
   selection as an educational post-hoc ranking.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error

from src.features.result_models import OriginalFeatureMapping
from src.features.preprocessing import fit_train_preprocessor, transform_features


def select_by_lasso(
    X_input: Union[pd.DataFrame, np.ndarray],
    y_input: Union[pd.Series, np.ndarray],
    mappings: List[OriginalFeatureMapping],
    k: int,
    alpha: Optional[float] = None,
    cv_strategy: str = "random",
    cv_folds: int = 5,
    random_seed: int = 42,
    groups_train: Optional[pd.Series] = None
) -> Tuple[List[OriginalFeatureMapping], Dict[str, Any]]:
    """
    Fit Lasso regression with dataset-appropriate cross-validation and fold-specific preprocessing.
    Aggregates coefficients into original feature groups.
    
    Parameters:
        X_input: DataFrame of candidate predictors, or preprocessed array.
        y_input: Target series or array.
        mappings: OriginalFeatureMapping definitions.
        k: Requested target number of original features (used for ranking).
        alpha: Optional fixed regularization penalty; if None, selected via dataset-appropriate CV.
        cv_strategy: "random", "temporal", or "group".
        cv_folds: Number of inner cross-validation folds.
        random_seed: Random state for deterministic reproducibility.
        groups_train: Aligned group identifiers (required when cv_strategy='group').
        
    Returns:
        (updated_mappings, diagnostics_dict)
    """
    if isinstance(X_input, pd.DataFrame):
        X_df = X_input
        y_s = y_input if isinstance(y_input, pd.Series) else pd.Series(y_input, index=X_input.index)
        # Full training set preprocessor fit
        preproc, _, _ = fit_train_preprocessor(X_df, scale_numeric=True)
        X_encoded = transform_features(preproc, X_df).values
        y_arr = y_s.values
    else:
        # Pre-encoded array fallback
        X_encoded = X_input
        y_arr = y_input.values if isinstance(y_input, pd.Series) else y_input
        col_names = [m.original_name for m in mappings]
        X_df = pd.DataFrame(X_input, columns=col_names)
        y_s = pd.Series(y_arr)

    # 1. Hyperparameter Selection via Dataset-Appropriate CV (when alpha is None)
    if alpha is None:
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

        # Build fold-isolated preprocessed datasets for inner CV
        fold_datasets = []
        for tr_idx, val_idx in inner_splits:
            f_tr_X = X_df.iloc[tr_idx]
            f_val_X = X_df.iloc[val_idx]

            fold_prep, _, _ = fit_train_preprocessor(f_tr_X, scale_numeric=True)
            f_X_tr_enc = transform_features(fold_prep, f_tr_X).values
            f_X_val_enc = transform_features(fold_prep, f_val_X).values
            fold_datasets.append((f_X_tr_enc, y_s.iloc[tr_idx].values, f_X_val_enc, y_s.iloc[val_idx].values))

        # Evaluate candidate alphas grid
        candidate_alphas = np.logspace(-4, 1, 15)
        best_alpha = float(candidate_alphas[0])
        best_rmse = float("inf")

        for cand_a in candidate_alphas:
            fold_rmses = []
            for tr_X_enc, tr_y, val_X_enc, val_y in fold_datasets:
                lasso_cv_model = Lasso(alpha=cand_a, random_state=random_seed, max_iter=2000)
                lasso_cv_model.fit(tr_X_enc, tr_y)
                preds = lasso_cv_model.predict(val_X_enc)
                fold_rmses.append(float(root_mean_squared_error(val_y, preds)))

            mean_cv_rmse = float(np.mean(fold_rmses))
            if mean_cv_rmse < best_rmse:
                best_rmse = mean_cv_rmse
                best_alpha = float(cand_a)

        effective_alpha = best_alpha
    else:
        effective_alpha = float(alpha)

    # 2. Fit Lasso on the full training set
    lasso = Lasso(alpha=effective_alpha, random_state=random_seed, max_iter=2000).fit(X_encoded, y_arr)
    coefs = lasso.coef_

    # 3. Aggregate coefficients into original feature groups
    scored_mappings = []
    natural_nonzero_count = 0

    for m in mappings:
        if m.encoded_column_indices:
            group_coefs = coefs[m.encoded_column_indices]
            group_norm = float(np.sqrt(np.sum(np.square(group_coefs))))
            is_nonzero = bool(group_norm > 1e-5)
            if is_nonzero:
                natural_nonzero_count += 1
        else:
            group_norm = 0.0
            is_nonzero = False

        scored_mappings.append((m, group_norm, is_nonzero))

    # Rank original feature groups descending by group norm
    scored_mappings.sort(key=lambda item: item[1], reverse=True)

    k_target = min(max(1, k), len(scored_mappings))
    updated = []

    for rank_idx, (m, g_norm, is_nonzero) in enumerate(scored_mappings):
        is_sel = rank_idx < k_target
        copy_m = OriginalFeatureMapping(
            original_name=m.original_name,
            feature_type=m.feature_type,
            encoded_column_names=list(m.encoded_column_names),
            encoded_column_indices=list(m.encoded_column_indices),
            is_selected=is_sel,
            rank=rank_idx + 1,
            score=g_norm,
            details={
                "metric": "Lasso Group Coefficient L2 Norm",
                "effective_alpha": effective_alpha,
                "natural_nonzero": is_nonzero
            }
        )
        updated.append(copy_m)

    diagnostics = {
        "effective_alpha": effective_alpha,
        "best_alpha": effective_alpha,
        "natural_nonzero_groups": natural_nonzero_count,
        "natural_nonzero_original_features": natural_nonzero_count,
        "requested_k": k,
        "cv_strategy": cv_strategy,
        "note": (
            f"Lasso selected {natural_nonzero_count} original feature groups naturally at alpha = {effective_alpha:.4e} "
            f"({cv_strategy} CV). The top {k_target} groups were retained based on group coefficient magnitude."
        )
    }

    name_to_m = {m.original_name: m for m in updated}
    return [name_to_m[m.original_name] for m in mappings], diagnostics
