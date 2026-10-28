"""
Cross-validation and model subset evaluation module for Feature Selection.
Implements strict fold isolation to prevent data leakage:
- TimeSeriesSplit for time-series datasets (Appliances Energy)
- GroupKFold for clustered/panel datasets (Life Expectancy)
- KFold for cross-sectional datasets (King County, Student Performance)

Fits imputers, encoders, scalers, selectors, and estimators strictly on fold-train.
Evaluates on fold-validation and untouched Validation partitions using a temporary LinearRegression estimator.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, GroupKFold

from src.features.preprocessing import fit_train_preprocessor, transform_features
from src.features.feature_mapping import (
    build_feature_mappings_from_transformer,
    get_encoded_indices_for_original_features
)
from src.features.result_models import OriginalFeatureMapping


def get_dataset_cv_splitter(
    strategy: str,
    cv_folds: int = 5,
    random_seed: int = 42
):
    """
    Instantiate appropriate scikit-learn cross-validation splitter.
    """
    if strategy == "temporal":
        return TimeSeriesSplit(n_splits=cv_folds)
    elif strategy == "group":
        return GroupKFold(n_splits=cv_folds)
    else:
        return KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)


def evaluate_feature_subset_on_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    selected_original_features: List[str]
) -> Tuple[float, float, float]:
    """
    Fit preprocessor strictly on X_train, select requested original features,
    fit temporary LinearRegression, and evaluate on untouched X_val.
    
    Returns:
        (val_mse, val_rmse, val_r2)
    """
    if not selected_original_features:
        # Trivial prediction of train mean
        y_pred = np.full_like(y_val, fill_value=float(y_train.mean()))
        mse = float(mean_squared_error(y_val, y_pred))
        rmse = float(root_mean_squared_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))
        return mse, rmse, r2

    preprocessor, _, _ = fit_train_preprocessor(X_train, scale_numeric=True)
    mappings, _ = build_feature_mappings_from_transformer(preprocessor, X_train)
    selected_indices = get_encoded_indices_for_original_features(mappings, selected_original_features)

    X_tr_enc = preprocessor.transform(X_train)
    X_val_enc = preprocessor.transform(X_val)

    if not selected_indices:
        y_pred = np.full_like(y_val, fill_value=float(y_train.mean()))
    else:
        reg = LinearRegression().fit(X_tr_enc[:, selected_indices], y_train.values)
        y_pred = reg.predict(X_val_enc[:, selected_indices])

    mse = float(mean_squared_error(y_val.values, y_pred))
    rmse = float(root_mean_squared_error(y_val.values, y_pred))
    r2 = float(r2_score(y_val.values, y_pred))
    return mse, rmse, r2


def run_isolated_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selector_fn: Callable[[np.ndarray, np.ndarray, List[OriginalFeatureMapping]], List[OriginalFeatureMapping]],
    cv_strategy: str,
    cv_folds: int = 5,
    random_seed: int = 42,
    groups: Optional[pd.Series] = None
) -> Tuple[float, float]:
    """
    Execute strictly isolated cross-validation on Training Data.
    Within each fold:
      1. Preprocessor is fitted only on fold-train.
      2. Feature selection is fitted only on fold-train.
      3. LinearRegression estimator is fitted only on fold-train.
      4. Evaluated on fold-validation.
      
    Returns:
        (cv_rmse_mean, cv_rmse_std)
    """
    cv = get_dataset_cv_splitter(cv_strategy, cv_folds=cv_folds, random_seed=random_seed)
    n_samples = len(X_train)

    if cv_strategy == "group":
        if groups is None or len(groups) != n_samples:
            # Fallback if groups not provided
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
            splits = cv.split(X_train, y_train)
        else:
            splits = cv.split(X_train, y_train, groups=groups)
    else:
        splits = cv.split(X_train, y_train)

    fold_rmses: List[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        f_X_train = X_train.iloc[tr_idx]
        f_y_train = y_train.iloc[tr_idx].values
        f_X_val = X_train.iloc[val_idx]
        f_y_val = y_train.iloc[val_idx].values

        # 1. Fit preprocessor on fold-train only
        preproc, _, _ = fit_train_preprocessor(f_X_train, scale_numeric=True)
        f_mappings, _ = build_feature_mappings_from_transformer(preproc, f_X_train)

        f_X_tr_enc = preproc.transform(f_X_train)
        f_X_val_enc = preproc.transform(f_X_val)

        # 2. Run feature selector on fold-train only
        selected_mappings = selector_fn(f_X_tr_enc, f_y_train, f_mappings)
        selected_orig_names = [m.original_name for m in selected_mappings if m.is_selected]
        selected_indices = get_encoded_indices_for_original_features(f_mappings, selected_orig_names)

        if not selected_indices:
            y_pred = np.full_like(f_y_val, fill_value=float(np.mean(f_y_train)))
        else:
            # 3. Fit estimator on fold-train
            model = LinearRegression().fit(f_X_tr_enc[:, selected_indices], f_y_train)
            # 4. Predict on fold-val
            y_pred = model.predict(f_X_val_enc[:, selected_indices])

        rmse = float(root_mean_squared_error(f_y_val, y_pred))
        fold_rmses.append(rmse)

    cv_mean = float(np.mean(fold_rmses)) if fold_rmses else 0.0
    cv_std = float(np.std(fold_rmses)) if fold_rmses else 0.0
    return cv_mean, cv_std
