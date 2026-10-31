"""
Regression modeling package for Regression Learning Lab.
"""
from src.models.design_matrix import (
    OLSDesignMatrixBuilder,
    DesignMatrixMetadata,
    DesignMatrixResult,
    build_ols_design_matrices,
)
from src.models.model_result import OLSModelResult
from src.models.ols_model import (
    fit_ols_model,
    generate_regression_equation_latex,
    generate_regression_equation_text,
    build_coefficient_table,
)
from src.models.model_service import train_ols_experiment

__all__ = [
    "OLSDesignMatrixBuilder",
    "DesignMatrixMetadata",
    "DesignMatrixResult",
    "build_ols_design_matrices",
    "OLSModelResult",
    "fit_ols_model",
    "generate_regression_equation_latex",
    "generate_regression_equation_text",
    "build_coefficient_table",
    "train_ols_experiment",
]
