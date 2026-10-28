"""
Embedded feature selection methods:
Lasso (L1 Regularization) Selection with Group-Aware Post-Hoc Aggregation.

Note on Methodology:
Standard Lasso induces sparsity on individual encoded columns according to regularization
parameter alpha, rather than selecting an exact fixed number of original feature groups.
This module evaluates group importance via the L2 norm of the group's coefficients:
    S_g = sqrt(sum_{j in group} beta_j^2)
Natural non-zero sparsity is detected, and top-K ranking is provided as an educational
post-hoc approximation.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso

from src.features.result_models import OriginalFeatureMapping


def select_by_lasso(
    X_encoded: np.ndarray,
    y: np.ndarray,
    mappings: List[OriginalFeatureMapping],
    k: int,
    alpha: Optional[float] = None,
    random_seed: int = 42
) -> Tuple[List[OriginalFeatureMapping], Dict[str, Any]]:
    """
    Fit Lasso regression on encoded features and aggregate coefficients into original groups.
    
    Parameters:
        X_encoded: Preprocessed numeric array.
        y: Target array.
        mappings: OriginalFeatureMapping definitions.
        k: Requested target number of original features (used for ranking).
        alpha: Optional fixed regularization penalty; if None, determined by 5-fold LassoCV.
        random_seed: Random state for reproducibility.
        
    Returns:
        (updated_mappings, diagnostics_dict)
    """
    if alpha is None:
        lasso = LassoCV(cv=5, random_state=random_seed, max_iter=2000).fit(X_encoded, y)
        effective_alpha = float(lasso.alpha_)
    else:
        lasso = Lasso(alpha=alpha, random_state=random_seed, max_iter=2000).fit(X_encoded, y)
        effective_alpha = float(alpha)

    coefs = lasso.coef_

    # Aggregate coefficients into original feature groups
    scored_mappings = []
    natural_nonzero_count = 0

    for m in mappings:
        if m.encoded_column_indices:
            group_coefs = coefs[m.encoded_column_indices]
            # Group score: L2 norm of coefficients
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
        "natural_nonzero_groups": natural_nonzero_count,
        "requested_k": k,
        "note": (
            f"Lasso selected {natural_nonzero_count} original feature groups naturally at alpha = {effective_alpha:.4e}. "
            f"The top {k_target} groups were retained based on group coefficient magnitude."
        )
    }

    name_to_m = {m.original_name: m for m in updated}
    return [name_to_m[m.original_name] for m in mappings], diagnostics
