"""
Module E: Multicollinearity Diagnostics.
Provides Variance Inflation Factor (VIF) on non-intercept regressors,
matrix rank verification, and condition number analysis.
"""
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from src.models.model_result import OLSModelResult
from src.diagnostics.result_models import (
    DiagnosticResult,
    DiagnosticType,
    DiagnosticDecision,
    ApplicabilityStatus,
    generate_diagnostic_id
)
from src.diagnostics.interpretation import format_vif_interpretation


def run_multicollinearity_diagnostics(
    model_result: OLSModelResult,
    vif_threshold_moderate: float = 5.0,
    vif_threshold_severe: float = 10.0
) -> DiagnosticResult:
    """
    Computes Variance Inflation Factor (VIF) for all non-intercept regressors,
    integrating design matrix rank, condition number, and linear dependency identification.
    
    Formula: VIF_j = 1 / (1 - R_j^2), where R_j^2 is obtained from regressing X_j on all other regressors.
    Note: An auxiliary constant is included in auxiliary regressions so that standard centered VIFs are obtained.
    """
    start_time = time.time()
    config = {
        "vif_threshold_moderate": vif_threshold_moderate,
        "vif_threshold_severe": vif_threshold_severe
    }
    diag_id, repro_hash = generate_diagnostic_id(model_result.model_id, "multicollinearity_vif", config)

    meta = model_result.design_matrix_metadata
    matrix_rank = meta.matrix_rank
    num_params = meta.num_parameters
    cond_num = meta.condition_number
    collinear_cols = meta.collinear_columns

    # Obtain the design matrix from the fitted model
    fitted_sm = model_result.fitted_model
    exog_df = pd.DataFrame(fitted_sm.model.exog, columns=meta.encoded_feature_names)

    # Exclude constant / intercept column from VIF reporting
    non_const_cols = [c for c in exog_df.columns if c != "const"]

    vif_records = []
    warnings_list = list(meta.warnings)

    if not non_const_cols:
        # Intercept-only model
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.MULTICOLLINEARITY.value,
            method_id="multicollinearity_vif",
            method_name="Multicollinearity & VIF Analysis",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Regressors are completely orthogonal (VIF_j = 1.0)",
            alternative_hypothesis="H1: Regressors share linear information (VIF_j > 1.0)",
            statistic_name="Max VIF",
            statistic_value=1.0,
            p_value=None,
            alpha=None,
            decision=DiagnosticDecision.NOT_APPLICABLE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason="Intercept-only model contains no slope predictors; VIF is not applicable.",
            assumptions_prerequisites=["At least one non-intercept predictor"],
            interpretation="Model contains only an intercept. Multicollinearity diagnostics are not applicable.",
            limitations="",
            recommendations=[],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    # Ensure matrix has a constant for auxiliary VIF regression
    if "const" in exog_df.columns:
        exog_for_vif = exog_df.copy()
    else:
        # Add auxiliary constant to ensure standard centered VIFs
        exog_for_vif = sm.add_constant(exog_df)

    exog_mat = exog_for_vif.values

    # Calculate VIF for each non-constant column
    for col_name in non_const_cols:
        col_idx = exog_for_vif.columns.get_loc(col_name)

        # Map back to original feature group
        orig_group = meta.encoded_to_original_mapping.get(col_name, col_name) if hasattr(meta, "encoded_to_original_mapping") else col_name
        if orig_group == col_name:
            for orig_name, enc_list in meta.original_to_encoded_mapping.items():
                if col_name in enc_list:
                    orig_group = orig_name
                    break

        try:
            # If column is an exact linear combination, VIF may be inf or raise LinAlgError
            if col_name in collinear_cols:
                vif_val = float("inf")
                risk = "Exact Linear Dependency (Rank Deficient)"
            else:
                val = variance_inflation_factor(exog_mat, col_idx)
                vif_val = float(val) if np.isfinite(val) else float("inf")

                if vif_val > vif_threshold_severe:
                    risk = "High Multicollinearity (Severe)"
                elif vif_val >= vif_threshold_moderate:
                    risk = "Moderate Multicollinearity"
                else:
                    risk = "Low Multicollinearity (Stable)"

        except Exception:
            vif_val = float("inf")
            risk = "Singular Matrix / Exact Collinearity"

        vif_records.append({
            "feature": col_name,
            "original_group": orig_group,
            "vif": vif_val,
            "risk_level": risk,
            "is_categorical_dummy": col_name != orig_group
        })

    # Sort records descending by VIF
    vif_records.sort(key=lambda r: (0 if np.isinf(r["vif"]) else 1, -r["vif"] if np.isfinite(r["vif"]) else 0))

    interp_data = format_vif_interpretation(
        vif_records=vif_records,
        condition_number=cond_num,
        matrix_rank=matrix_rank,
        num_parameters=num_params
    )

    max_vif = interp_data["max_vif"]

    details = {
        "vif_table": vif_records,
        "max_vif": max_vif,
        "condition_number": cond_num,
        "matrix_rank": matrix_rank,
        "num_parameters": num_params,
        "is_full_rank": meta.is_full_rank,
        "collinear_columns": collinear_cols,
        "high_vif_columns": interp_data["high_vif_columns"],
        "moderate_vif_columns": interp_data["moderate_vif_columns"]
    }

    return DiagnosticResult(
        diagnostic_id=diag_id,
        model_id=model_result.model_id,
        dataset_id=model_result.dataset_id,
        dataset_fingerprint=model_result.dataset_fingerprint,
        split_fingerprint=model_result.split_fingerprint,
        diagnostic_type=DiagnosticType.MULTICOLLINEARITY.value,
        method_id="multicollinearity_vif",
        method_name="Multicollinearity & VIF Analysis",
        is_valid_for_inference=model_result.is_valid_for_inference,
        null_hypothesis="H0: Predictors are orthogonal (VIF_j = 1.0; no collinearity inflation)",
        alternative_hypothesis="H1: Predictors share linear redundancy (VIF_j > 1.0)",
        statistic_name="Max VIF",
        statistic_value=max_vif,
        p_value=None,  # VIF is a descriptive diagnostic metric
        alpha=None,
        decision=DiagnosticDecision.DESCRIPTIVE_ONLY.value,
        applicability_status=ApplicabilityStatus.APPLICABLE.value,
        assumptions_prerequisites=[
            "Non-intercept regressors present",
            "Auxiliary regression with intercept computed"
        ],
        details=details,
        interpretation=interp_data["interpretation"],
        limitations=interp_data["limitations"],
        recommendations=interp_data["recommendations"],
        warnings=warnings_list,
        runtime_seconds=time.time() - start_time,
        configuration=config,
        reproducibility_hash=repro_hash
    )
