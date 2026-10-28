"""
Feature Selection Laboratory module for Regression Learning Lab.
Provides train-only preprocessing, group-aware feature mapping, filter, wrapper,
and embedded selection algorithms, and leakage-free cross-validation evaluation.
"""
from src.features.result_models import (
    OriginalFeatureMapping,
    FeatureSelectionResult,
)
from src.features.preprocessing import (
    build_column_preprocessor,
    fit_train_preprocessor,
    transform_features,
)
from src.features.feature_mapping import (
    build_feature_mappings_from_transformer,
    get_encoded_indices_for_original_features,
    get_encoded_names_for_original_features,
)
from src.features.filter_methods import (
    select_all_features_baseline,
    select_by_pearson_correlation,
    select_by_f_test,
    select_by_mutual_information,
)
from src.features.wrapper_methods import (
    select_by_group_rfe,
    select_by_group_forward_sfs,
    select_by_group_backward_sfs,
)
from src.features.embedded_methods import (
    select_by_lasso,
)
from src.features.evaluation import (
    get_dataset_cv_splitter,
    evaluate_feature_subset_on_validation,
    run_isolated_cross_validation,
)
from src.features.selectors import (
    FEATURE_SELECTION_REGISTRY,
    run_feature_selection,
)

__all__ = [
    "OriginalFeatureMapping",
    "FeatureSelectionResult",
    "build_column_preprocessor",
    "fit_train_preprocessor",
    "transform_features",
    "build_feature_mappings_from_transformer",
    "get_encoded_indices_for_original_features",
    "get_encoded_names_for_original_features",
    "select_all_features_baseline",
    "select_by_pearson_correlation",
    "select_by_f_test",
    "select_by_mutual_information",
    "select_by_group_rfe",
    "select_by_group_forward_sfs",
    "select_by_group_backward_sfs",
    "select_by_lasso",
    "get_dataset_cv_splitter",
    "evaluate_feature_subset_on_validation",
    "run_isolated_cross_validation",
    "FEATURE_SELECTION_REGISTRY",
    "run_feature_selection",
]
