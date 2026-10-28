"""
Unified Feature Selection API: manages and dispatches all 8 selection algorithms.
Methods:
1. Baseline (all features)
2. Pearson Correlation (Filter)
3. F-test (Filter)
4. Mutual Information (Filter)
5. Recursive Feature Elimination (Wrapper)
6. Forward Sequential Feature Selection (Wrapper)
7. Backward Sequential Feature Selection (Wrapper)
8. Lasso-based Selection (Embedded)
"""
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.result_models import FeatureSelectionResult, OriginalFeatureMapping
from src.features.preprocessing import fit_train_preprocessor, transform_features
from src.features.feature_mapping import (
    build_feature_mappings_from_transformer,
    get_encoded_indices_for_original_features,
    get_encoded_names_for_original_features
)
from src.features.filter_methods import (
    select_all_features_baseline,
    select_by_pearson_correlation,
    select_by_f_test,
    select_by_mutual_information
)
from src.features.wrapper_methods import (
    select_by_group_rfe,
    select_by_group_forward_sfs,
    select_by_group_backward_sfs
)
from src.features.embedded_methods import select_by_lasso
from src.features.evaluation import (
    run_isolated_cross_validation,
    evaluate_feature_subset_on_validation
)

FEATURE_SELECTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "id": "baseline",
        "name": "All Features",
        "category": "Baseline",
        "requires_k": False,
        "description": "Baseline reference retaining all available original predictor variables without screening.",
        "complexity": "O(1)"
    },
    "pearson": {
        "id": "pearson",
        "name": "Pearson Correlation",
        "category": "Filter",
        "requires_k": True,
        "description": "Ranks original features by absolute linear correlation with target. Groups dummy columns by maximum level magnitude.",
        "complexity": "O(N * P)"
    },
    "f_test": {
        "id": "f_test",
        "name": "F-test (f_regression)",
        "category": "Filter",
        "requires_k": True,
        "description": "Evaluates univariate linear regression F-statistic. Ranks dummy groups by maximum F-score.",
        "complexity": "O(N * P)"
    },
    "mutual_information": {
        "id": "mutual_information",
        "name": "Mutual Information",
        "category": "Filter",
        "requires_k": True,
        "description": "Measures linear and non-linear dependencies between predictors and target based on entropy estimation.",
        "complexity": "O(N * P * log N)"
    },
    "rfe": {
        "id": "rfe",
        "name": "Recursive Feature Elimination (RFE)",
        "category": "Wrapper",
        "requires_k": True,
        "description": "Iteratively fits LinearRegression and eliminates the weakest original feature group based on coefficient L2 norm.",
        "complexity": "O((P - K) * N * P^2)"
    },
    "forward_sfs": {
        "id": "forward_sfs",
        "name": "Forward Sequential Feature Selection",
        "category": "Wrapper",
        "requires_k": True,
        "description": "Starts empty and greedily adds the original feature group that produces the lowest CV RMSE at each step.",
        "complexity": "O(K * P * N * P_sub^2)"
    },
    "backward_sfs": {
        "id": "backward_sfs",
        "name": "Backward Sequential Feature Selection",
        "category": "Wrapper",
        "requires_k": True,
        "description": "Starts with all features and iteratively removes the original feature group whose removal yields the best CV RMSE.",
        "complexity": "O((P - K) * P * N * P_sub^2)"
    },
    "lasso": {
        "id": "lasso",
        "name": "Lasso-based Selection",
        "category": "Embedded",
        "requires_k": True,
        "description": "Applies L1 regularization to shrink coefficients. Evaluates original feature groups by coefficient L2 norm.",
        "complexity": "O(N * P * iterations)"
    }
}


def compute_experiment_identity(
    dataset_fingerprint: str,
    split_fingerprint: str,
    method_id: str,
    k: int,
    cv_strategy: str,
    cv_folds: int,
    random_seed: int,
    alpha: Optional[float] = None
) -> str:
    """
    Computes a deterministic, collision-free SHA-256 hash identity of the entire
    experiment configuration.
    """
    config_dict = {
        "dataset_fingerprint": str(dataset_fingerprint),
        "split_fingerprint": str(split_fingerprint),
        "method_id": str(method_id),
        "k": int(k),
        "cv_strategy": str(cv_strategy),
        "cv_folds": int(cv_folds),
        "random_seed": int(random_seed),
        "alpha": float(alpha) if alpha is not None else None,
    }
    raw_json = json.dumps(config_dict, sort_keys=True)
    hash_hex = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:12]
    return f"exp_{method_id}_k{k}_cv{cv_folds}_{hash_hex}"


def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method_id: str,
    k: int,
    cv_strategy: str,
    cv_folds: int = 5,
    random_seed: int = 42,
    dataset_id: str = "dataset",
    dataset_fingerprint: str = "",
    split_fingerprint: str = "",
    groups_train: Optional[pd.Series] = None,
    alpha: Optional[float] = None
) -> FeatureSelectionResult:
    """
    Execute end-to-end feature selection pipeline, cross-validation, and validation set scoring.
    """
    if method_id not in FEATURE_SELECTION_REGISTRY:
        raise ValueError(f"Unknown selection method '{method_id}'. Available: {list(FEATURE_SELECTION_REGISTRY.keys())}")

    meta_info = FEATURE_SELECTION_REGISTRY[method_id]
    total_original_features = len(X_train.columns)
    if meta_info["requires_k"]:
        if k < 1 or k > total_original_features:
            raise ValueError(f"Requested K ({k}) must be between 1 and {total_original_features} for dataset predictors.")
        k_bounded = k
    else:
        k_bounded = total_original_features

    t_start = time.perf_counter()
    warnings = []
    notes = ""

    # 1. Fit preprocessor on X_train only (Leakage-Safe)
    preprocessor, _, _ = fit_train_preprocessor(X_train, scale_numeric=True)
    mappings, encoded_names = build_feature_mappings_from_transformer(preprocessor, X_train)
    X_train_enc = preprocessor.transform(X_train)

    # 2. Execute selection algorithm on full Train
    if method_id == "baseline":
        selected_mappings = select_all_features_baseline(mappings)
    elif method_id == "pearson":
        selected_mappings = select_by_pearson_correlation(X_train_enc, y_train.values, mappings, k=k_bounded)
    elif method_id == "f_test":
        selected_mappings = select_by_f_test(X_train_enc, y_train.values, mappings, k=k_bounded)
    elif method_id == "mutual_information":
        selected_mappings = select_by_mutual_information(X_train_enc, y_train.values, mappings, k=k_bounded, random_seed=random_seed)
    elif method_id == "rfe":
        selected_mappings = select_by_group_rfe(X_train_enc, y_train.values, mappings, k=k_bounded)
    elif method_id == "forward_sfs":
        selected_mappings = select_by_group_forward_sfs(
            X_train, y_train, mappings, k=k_bounded,
            cv_strategy=cv_strategy, cv_folds=min(3, cv_folds),
            random_seed=random_seed, groups_train=groups_train
        )
    elif method_id == "backward_sfs":
        selected_mappings = select_by_group_backward_sfs(
            X_train, y_train, mappings, k=k_bounded,
            cv_strategy=cv_strategy, cv_folds=min(3, cv_folds),
            random_seed=random_seed, groups_train=groups_train
        )
    elif method_id == "lasso":
        selected_mappings, lasso_diag = select_by_lasso(
            X_train, y_train, mappings, k=k_bounded, alpha=alpha,
            cv_strategy=cv_strategy, cv_folds=min(5, cv_folds),
            random_seed=random_seed, groups_train=groups_train
        )
        notes = lasso_diag["note"]
        if lasso_diag["natural_nonzero_groups"] != k_bounded:
            warnings.append(
                f"Lasso sparsity naturally produced {lasso_diag['natural_nonzero_groups']} non-zero groups. "
                f"Selected top {k_bounded} groups based on coefficient L2 norm."
            )
    else:
        raise ValueError(f"Method '{method_id}' not implemented")

    selected_orig_features = [m.original_name for m in selected_mappings if m.is_selected]
    selected_enc_names = get_encoded_names_for_original_features(selected_mappings, selected_orig_features)

    # 3. Cross-Validation evaluation with fold isolation
    def cv_selector(f_X_enc, f_y, f_maps, f_X_train=None, f_y_series=None, f_groups=None):
        if method_id == "baseline":
            return select_all_features_baseline(f_maps)
        elif method_id == "pearson":
            return select_by_pearson_correlation(f_X_enc, f_y, f_maps, k=k_bounded)
        elif method_id == "f_test":
            return select_by_f_test(f_X_enc, f_y, f_maps, k=k_bounded)
        elif method_id == "mutual_information":
            return select_by_mutual_information(f_X_enc, f_y, f_maps, k=k_bounded, random_seed=random_seed)
        elif method_id == "rfe":
            return select_by_group_rfe(f_X_enc, f_y, f_maps, k=k_bounded)
        elif method_id == "forward_sfs":
            tr_df = f_X_train if f_X_train is not None else pd.DataFrame(f_X_enc, columns=[m.original_name for m in f_maps])
            tr_y = f_y_series if f_y_series is not None else pd.Series(f_y)
            return select_by_group_forward_sfs(
                tr_df, tr_y, f_maps, k=k_bounded,
                cv_strategy=cv_strategy, cv_folds=min(3, cv_folds),
                random_seed=random_seed, groups_train=f_groups
            )
        elif method_id == "backward_sfs":
            tr_df = f_X_train if f_X_train is not None else pd.DataFrame(f_X_enc, columns=[m.original_name for m in f_maps])
            tr_y = f_y_series if f_y_series is not None else pd.Series(f_y)
            return select_by_group_backward_sfs(
                tr_df, tr_y, f_maps, k=k_bounded,
                cv_strategy=cv_strategy, cv_folds=min(3, cv_folds),
                random_seed=random_seed, groups_train=f_groups
            )
        elif method_id == "lasso":
            tr_df = f_X_train if f_X_train is not None else pd.DataFrame(f_X_enc, columns=[m.original_name for m in f_maps])
            tr_y = f_y_series if f_y_series is not None else pd.Series(f_y)
            res_maps, _ = select_by_lasso(
                tr_df, tr_y, f_maps, k=k_bounded, alpha=alpha,
                cv_strategy=cv_strategy, cv_folds=min(5, cv_folds),
                random_seed=random_seed, groups_train=f_groups
            )
            return res_maps
        return select_all_features_baseline(f_maps)

    cv_mean, cv_std = run_isolated_cross_validation(
        X_train=X_train,
        y_train=y_train,
        selector_fn=cv_selector,
        cv_strategy=cv_strategy,
        cv_folds=cv_folds,
        random_seed=random_seed,
        groups=groups_train
    )

    # 4. Out-of-sample Validation partition evaluation
    val_mse, val_rmse, val_r2 = evaluate_feature_subset_on_validation(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        selected_original_features=selected_orig_features
    )

    t_elapsed = time.perf_counter() - t_start

    # Compute collision-free experiment identity hash
    exp_id = compute_experiment_identity(
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_fingerprint,
        method_id=method_id,
        k=k_bounded,
        cv_strategy=cv_strategy,
        cv_folds=cv_folds,
        random_seed=random_seed,
        alpha=alpha
    )

    total_orig = len(X_train.columns)
    sel_orig = len(selected_orig_features)
    full_enc = len(encoded_names)
    sel_enc = len(selected_enc_names)

    return FeatureSelectionResult(
        experiment_id=exp_id,
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_fingerprint,
        method_id=method_id,
        method_name=meta_info["name"],
        category=meta_info["category"],
        requested_k=k_bounded,
        selected_original_features=selected_orig_features,
        selected_original_count=sel_orig,
        total_original_count=total_orig,
        all_original_features=list(X_train.columns),
        encoded_feature_names=encoded_names,
        selected_encoded_feature_names=selected_enc_names,
        full_encoded_dimension=full_enc,
        selected_encoded_dimension=sel_enc,
        encoded_input_dimension=sel_enc,
        feature_mappings=selected_mappings,
        cv_rmse_mean=cv_mean,
        cv_rmse_std=cv_std,
        val_mse=val_mse,
        val_rmse=val_rmse,
        val_r2=val_r2,
        execution_time_seconds=t_elapsed,
        cv_strategy_used=cv_strategy,
        cv_folds=cv_folds,
        random_seed=random_seed,
        warnings=warnings,
        notes=notes
    )
