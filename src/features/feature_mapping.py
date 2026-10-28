"""
Feature mapping module: bridges original dataset predictors to encoded model columns.
Ensures that a categorical feature with multiple One-Hot dummies is correctly tracked,
scored, and selected as ONE original feature entity.
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
    Constructs OriginalFeatureMapping instances from a fitted ColumnTransformer.
    
    Parameters:
        preprocessor: Fitted ColumnTransformer instance.
        original_df: DataFrame of original candidate predictors.
        
    Returns:
        (mappings, all_encoded_names): List of feature mappings and list of all encoded column names.
    """
    encoded_names = list(preprocessor.get_feature_names_out())
    name_to_index = {name: idx for idx, name in enumerate(encoded_names)}

    mappings: List[OriginalFeatureMapping] = []

    numeric_cols = original_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in original_df.columns if c not in numeric_cols]

    for orig_col in original_df.columns:
        is_num = orig_col in numeric_cols
        feat_type = "numeric" if is_num else "categorical"

        # Search for corresponding encoded names
        # ColumnTransformer prefixes names like 'num__col' or 'cat__col_val'
        matching_encoded = []
        matching_indices = []

        if is_num:
            # Pattern 'num__<col>'
            target_prefix = f"num__{orig_col}"
            for enc_name in encoded_names:
                if enc_name == target_prefix or enc_name.endswith(f"__{orig_col}") or enc_name == orig_col:
                    matching_encoded.append(enc_name)
                    matching_indices.append(name_to_index[enc_name])
        else:
            # Pattern 'cat__<col>_<val>'
            target_prefix = f"cat__{orig_col}_"
            for enc_name in encoded_names:
                if enc_name.startswith(target_prefix) or f"__{orig_col}_" in enc_name:
                    matching_encoded.append(enc_name)
                    matching_indices.append(name_to_index[enc_name])

        # Fallback if prefix convention differs
        if not matching_encoded:
            for enc_name in encoded_names:
                if orig_col in enc_name:
                    matching_encoded.append(enc_name)
                    matching_indices.append(name_to_index[enc_name])

        mappings.append(OriginalFeatureMapping(
            original_name=orig_col,
            feature_type=feat_type,
            encoded_column_names=matching_encoded,
            encoded_column_indices=matching_indices,
            is_selected=False,
            rank=0,
            score=0.0
        ))

    return mappings, encoded_names


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
