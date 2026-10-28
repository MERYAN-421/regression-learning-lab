"""
Training-only preprocessing pipelines for tabular regression datasets.
Builds scikit-learn ColumnTransformer with median imputation for numeric features,
mode imputation and one-hot encoding for categorical features, and optional scaling.
"""
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_column_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Construct an unfitted scikit-learn ColumnTransformer.
    
    Parameters:
        numeric_features: List of continuous/numeric predictor column names.
        categorical_features: List of categorical predictor column names.
        scale_numeric: Whether to apply StandardScaler to numeric predictors.
    """
    transformers = []

    if numeric_features:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(num_steps)
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True
    )
    return preprocessor


def fit_train_preprocessor(
    X_train: pd.DataFrame,
    scale_numeric: bool = True
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Fit preprocessor strictly on training data partition.
    Guarantees no data leakage from validation or test partitions.
    
    Returns:
        (fitted_preprocessor, numeric_features, categorical_features)
    """
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    preprocessor = build_column_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric
    )
    preprocessor.fit(X_train)
    return preprocessor, numeric_features, categorical_features


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform predictor DataFrame using a previously fitted ColumnTransformer.
    Returns a pandas DataFrame with descriptive encoded column headers.
    """
    arr = preprocessor.transform(X)
    col_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(arr, columns=col_names, index=X.index)
