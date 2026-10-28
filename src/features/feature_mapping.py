"""
Feature mapping module: bridges original dataset predictors to encoded model columns.
Uses structural introspection of fitted ColumnTransformer, Pipeline, and OneHotEncoder
components rather than fragile substring parsing.
Guarantees correct grouping of dummy columns even with overlapping feature names,
names with prefixes/suffixes, and categories containing underscores.
"""
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.features.result_models import OriginalFeatureMapping


def build_feature_mappings_from_transformer(
    preprocessor: ColumnTransformer,
    original_df: pd.DataFrame
) -> Tuple[List[OriginalFeatureMapping], List[str]]:
    """
    Constructs OriginalFeatureMapping instances structurally from a fitted ColumnTransformer.
    
    Parameters:
        preprocessor: Fitted ColumnTransformer instance.
        original_df: DataFrame of original candidate predictors.
        
    Returns:
        (mappings, all_encoded_names): List of feature mappings and list of all encoded column names.
    """
    all_encoded_names = list(preprocessor.get_feature_names_out())
    mappings_by_name: Dict[str, OriginalFeatureMapping] = {}

    # 1. Map numeric features structurally
    if hasattr(preprocessor, "output_indices_") and "num" in preprocessor.output_indices_:
        num_slice = preprocessor.output_indices_["num"]
        num_cols = None
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                num_cols = cols
                break

        if num_cols and num_slice.start is not None and num_slice.stop is not None:
            start = num_slice.start
            for i, col in enumerate(num_cols):
                idx = start + i
                enc_name = all_encoded_names[idx]
                mappings_by_name[col] = OriginalFeatureMapping(
                    original_name=col,
                    feature_type="numeric",
                    encoded_column_names=[enc_name],
                    encoded_column_indices=[idx],
                    is_selected=False,
                    rank=0,
                    score=0.0
                )

    # 2. Map categorical features structurally via encoder.categories_
    if hasattr(preprocessor, "output_indices_") and "cat" in preprocessor.output_indices_:
        cat_slice = preprocessor.output_indices_["cat"]
        cat_pipe = preprocessor.named_transformers_.get("cat")
        cat_cols = None
        for name, trans, cols in preprocessor.transformers_:
            if name == "cat":
                cat_cols = cols
                break

        if cat_cols and cat_slice.start is not None and cat_slice.stop is not None and cat_pipe is not None:
            # Retrieve the fitted OneHotEncoder step
            encoder = None
            if hasattr(cat_pipe, "named_steps") and "encoder" in cat_pipe.named_steps:
                encoder = cat_pipe.named_steps["encoder"]
            elif hasattr(cat_pipe, "categories_"):
                encoder = cat_pipe

            if encoder is not None and hasattr(encoder, "categories_"):
                curr_idx = cat_slice.start
                for i, col in enumerate(cat_cols):
                    n_categories = len(encoder.categories_[i])
                    col_indices = list(range(curr_idx, curr_idx + n_categories))
                    col_names = [all_encoded_names[j] for j in col_indices]
                    curr_idx += n_categories

                    mappings_by_name[col] = OriginalFeatureMapping(
                        original_name=col,
                        feature_type="categorical",
                        encoded_column_names=col_names,
                        encoded_column_indices=col_indices,
                        is_selected=False,
                        rank=0,
                        score=0.0
                    )

    # 3. Fallback for any unmapped columns (e.g. custom or single transformer)
    for orig_col in original_df.columns:
        if orig_col not in mappings_by_name:
            is_num = pd.api.types.is_numeric_dtype(original_df[orig_col])
            # Strict prefix pattern
            prefix_num = f"num__{orig_col}"
            prefix_cat = f"cat__{orig_col}_"
            matched_indices = []
            matched_names = []
            for idx, enc_name in enumerate(all_encoded_names):
                if enc_name == prefix_num or enc_name.startswith(prefix_cat):
                    matched_indices.append(idx)
                    matched_names.append(enc_name)

            mappings_by_name[orig_col] = OriginalFeatureMapping(
                original_name=orig_col,
                feature_type="numeric" if is_num else "categorical",
                encoded_column_names=matched_names,
                encoded_column_indices=matched_indices,
                is_selected=False,
                rank=0,
                score=0.0
            )

    # Preserve exact column ordering from original_df
    ordered_mappings = [mappings_by_name[col] for col in original_df.columns if col in mappings_by_name]
    return ordered_mappings, all_encoded_names


def get_encoded_indices_for_original_features(
    mappings: List[OriginalFeatureMapping],
    selected_original_names: List[str]
) -> List[int]:
    """
    Given a list of selected original feature names, retrieve all corresponding
    encoded column indices in sorted order.
    """
    selected_set = set(selected_original_names)
    indices = []
    for m in mappings:
        if m.original_name in selected_set:
            indices.extend(m.encoded_column_indices)
    return sorted(list(set(indices)))


def get_encoded_names_for_original_features(
    mappings: List[OriginalFeatureMapping],
    selected_original_names: List[str]
) -> List[str]:
    """
    Given a list of selected original feature names, retrieve all corresponding
    encoded column names.
    """
    selected_set = set(selected_original_names)
    names = []
    for m in mappings:
        if m.original_name in selected_set:
            names.extend(m.encoded_column_names)
    return names
