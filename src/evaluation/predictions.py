"""
Prediction utilities for fitted OLS regression models.
"""
from typing import Union, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.models.model_result import OLSModelResult


def predict_ols(
    model_result: "OLSModelResult",
    X_raw: pd.DataFrame
) -> pd.Series:
    """
    Transforms raw input features using the model's fitted design matrix builder
    and generates predicted target values from the fitted statsmodels OLS estimator.
    """
    if model_result.builder is None:
        raise ValueError("ModelResult does not contain a fitted OLSDesignMatrixBuilder.")

    if model_result.fitted_model is None:
        raise ValueError("ModelResult does not contain a fitted statsmodels estimator.")

    # Transform raw features into OLS design matrix
    X_design = model_result.builder.transform(X_raw)

    # Predict using statsmodels
    preds_arr = model_result.fitted_model.predict(X_design.values)
    return pd.Series(preds_arr, index=X_raw.index, name=f"pred_{model_result.model_id}")
