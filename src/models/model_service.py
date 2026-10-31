"""
High-level service interface for training Ordinary Least Squares (OLS) models.
Connects SplitResult, FeatureSelectionResult handoffs, design matrix building, and statsmodels fitting.
"""
from typing import List, Optional, Union
import pandas as pd

from src.data.models import SplitResult
from src.features.result_models import FeatureSelectionResult
from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model
from src.models.model_result import OLSModelResult


def train_ols_experiment(
    split_result: SplitResult,
    feature_source: Union[str, FeatureSelectionResult] = "all_features",
    target_name: str = "target",
    fit_intercept: bool = True,
    scale_numeric: bool = False,
    dataset_id: str = "dataset",
    dataset_fingerprint: str = ""
) -> OLSModelResult:
    """
    Orchestrates the complete OLS training workflow:
    1. Resolves selected original features from 'all_features' or FeatureSelectionResult.
    2. Validates split fingerprint compatibility.
    3. Fits median imputer, reference-level categorical encoder, and optional scaler strictly on X_train.
    4. Evaluates matrix rank, condition number, and linear dependencies.
    5. Estimates OLS coefficients via statsmodels.
    6. Computes Train and Validation metrics.
    7. Formulates LaTeX equations and disclaimers.
    """
    if feature_source == "all_features":
        selected_original_features = split_result.X_train.columns.tolist()
        source_id = "all_features"
        source_label = "All Features (Baseline)"
        is_post_selection = False
    elif isinstance(feature_source, FeatureSelectionResult):
        # Validate that the feature selection experiment matches the active data split
        if feature_source.split_fingerprint != split_result.split_fingerprint:
            raise ValueError(
                f"Split fingerprint mismatch! Feature selection experiment was executed on "
                f"split '{feature_source.split_fingerprint}', but active dataset split is '{split_result.split_fingerprint}'. "
                "Re-run feature selection on the active split to preserve methodological integrity."
            )
        selected_original_features = list(feature_source.selected_original_features)
        source_id = feature_source.experiment_id
        source_label = f"{feature_source.method_name} (K={feature_source.selected_original_count})"
        is_post_selection = True
    else:
        raise TypeError(f"Unsupported feature_source type: {type(feature_source)}")

    # Build OLS design matrices for Train, Val, and Test
    design_result, builder = build_ols_design_matrices(
        X_train=split_result.X_train,
        selected_features=selected_original_features,
        X_val=split_result.X_val,
        X_test=split_result.X_test,
        fit_intercept=fit_intercept,
        scale_numeric=scale_numeric,
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_result.split_fingerprint
    )

    # Fit statsmodels OLS model and calculate metrics
    model_result = fit_ols_model(
        design_result=design_result,
        y_train=split_result.y_train,
        y_val=split_result.y_val,
        target_name=target_name,
        feature_source_id=source_id,
        feature_source_label=source_label,
        builder=builder,
        is_post_selection=is_post_selection
    )

    return model_result
