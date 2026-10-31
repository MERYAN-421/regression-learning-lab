"""
Module A: Linearity and Functional-Form Diagnostics.
Provides the Ramsey RESET hypothesis test and Residuals vs. Fitted diagnostic evaluations.
"""
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import linear_reset

from src.models.model_result import OLSModelResult
from src.diagnostics.result_models import (
    DiagnosticResult,
    DiagnosticType,
    DiagnosticDecision,
    ApplicabilityStatus,
    generate_diagnostic_id
)
from src.diagnostics.interpretation import format_reset_interpretation


def run_ramsey_reset_test(
    model_result: OLSModelResult,
    powers: List[int] = [2, 3],
    alpha: float = 0.05
) -> DiagnosticResult:
    """
    Executes the Ramsey RESET test for functional-form specification.
    
    H0: Model specification is correct (coefficients of higher-order fitted values are jointly zero).
    H1: Model is misspecified (higher-order powers contribute significant explanatory power).
    
    Parameters:
        model_result: Fitted OLSModelResult from Milestone 3.
        powers: List of positive integer powers of fitted values to include in auxiliary regression (default [2, 3]).
        alpha: Significance level (default 0.05).
    """
    start_time = time.time()
    config = {"powers": sorted(powers), "alpha": alpha}
    diag_id, repro_hash = generate_diagnostic_id(model_result.model_id, "ramsey_reset", config)

    # Check rank and inference validity
    if not model_result.is_valid_for_inference:
        interp_data = {
            "decision": DiagnosticDecision.NOT_APPLICABLE.value,
            "interpretation": (
                "The Ramsey RESET test cannot be performed because the active OLS model is rank-deficient "
                "or operates in prediction-only mode. Parameter identification and F-distributions are unidentifiable."
            ),
            "limitations": "Requires full column rank in both original and auxiliary design matrices.",
            "recommendations": ["Remove collinear predictors to achieve full matrix rank before running RESET."]
        }
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.LINEARITY.value,
            method_id="ramsey_reset",
            method_name="Ramsey RESET Test",
            is_valid_for_inference=False,
            null_hypothesis="H0: Additional powers of fitted values have zero coefficients (no functional-form misspecification)",
            alternative_hypothesis="H1: At least one power of fitted values has a non-zero coefficient (functional-form misspecification)",
            statistic_name="F-statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NOT_APPLICABLE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason="Design matrix is rank-deficient; conventional F-distribution is undefined.",
            assumptions_prerequisites=["Full column rank design matrix", "Sufficient residual degrees of freedom"],
            interpretation=interp_data["interpretation"],
            limitations=interp_data["limitations"],
            recommendations=interp_data["recommendations"],
            warnings=["Rank deficiency prevents valid Ramsey RESET calculation."],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    # Validate degrees of freedom
    n_obs = len(model_result.train_predictions)
    k_params = model_result.design_matrix_metadata.num_parameters
    max_power = max(powers) if powers else 3
    num_aux_terms = len(powers)

    if n_obs <= k_params + num_aux_terms:
        interp_data = {
            "decision": DiagnosticDecision.INSUFFICIENT_DATA.value,
            "interpretation": (
                f"Insufficient sample size for Ramsey RESET: n={n_obs} observations vs. "
                f"{k_params + num_aux_terms} required parameters in auxiliary regression."
            ),
            "limitations": "Auxiliary regression requires df_resid > 0.",
            "recommendations": ["Reduce feature count or acquire more training data."]
        }
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.LINEARITY.value,
            method_id="ramsey_reset",
            method_name="Ramsey RESET Test",
            is_valid_for_inference=True,
            null_hypothesis="H0: Additional powers of fitted values have zero coefficients (no functional-form misspecification)",
            alternative_hypothesis="H1: At least one power of fitted values has a non-zero coefficient (functional-form misspecification)",
            statistic_name="F-statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.INSUFFICIENT_DATA.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"Insufficient sample size: n={n_obs} <= parameters={k_params + num_aux_terms}.",
            interpretation=interp_data["interpretation"],
            limitations=interp_data["limitations"],
            recommendations=interp_data["recommendations"],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    try:
        fitted_sm = model_result.fitted_model
        # statsmodels linear_reset: power can be an integer specifying highest power
        reset_res = linear_reset(
            fitted_sm,
            power=max_power,
            test_type="fitted",
            use_f=True
        )

        f_stat = float(reset_res.fvalue) if hasattr(reset_res, "fvalue") else float(reset_res.statistic)
        p_val = float(reset_res.pvalue)

        # Get degrees of freedom if available
        df_num = getattr(reset_res, "df_num", num_aux_terms)
        df_denom = getattr(reset_res, "df_denom", n_obs - k_params - num_aux_terms)

        interp_data = format_reset_interpretation(
            statistic_value=f_stat,
            p_value=p_val,
            alpha=alpha,
            powers=powers,
            is_valid_for_inference=True
        )

        details = {
            "f_statistic": f_stat,
            "p_value": p_val,
            "df_numerator": float(df_num),
            "df_denominator": float(df_denom),
            "tested_powers": powers,
            "sample_size": n_obs
        }

        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.LINEARITY.value,
            method_id="ramsey_reset",
            method_name="Ramsey RESET Test",
            is_valid_for_inference=True,
            null_hypothesis="H0: Additional powers of fitted values have zero coefficients (no functional-form misspecification)",
            alternative_hypothesis="H1: At least one power of fitted values has a non-zero coefficient (functional-form misspecification)",
            statistic_name="F-statistic",
            statistic_value=f_stat,
            p_value=p_val,
            alpha=alpha,
            decision=interp_data["decision"],
            applicability_status=ApplicabilityStatus.APPLICABLE.value,
            assumptions_prerequisites=[
                "OLS model has full rank",
                "Auxiliary model can be estimated without collinearity",
                "Residual degrees of freedom > auxiliary parameters"
            ],
            details=details,
            interpretation=interp_data["interpretation"],
            limitations=interp_data["limitations"],
            recommendations=interp_data["recommendations"],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    except Exception as e:
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.LINEARITY.value,
            method_id="ramsey_reset",
            method_name="Ramsey RESET Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Additional powers of fitted values have zero coefficients (no functional-form misspecification)",
            alternative_hypothesis="H1: At least one power of fitted values has a non-zero coefficient (functional-form misspecification)",
            statistic_name="F-statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NUMERICAL_FAILURE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"statsmodels linear_reset failed: {e}",
            interpretation=f"Calculation error during Ramsey RESET test execution: {e}",
            limitations="Numerical instability or auxiliary collinearity occurred during auxiliary regression estimation.",
            recommendations=["Check for severe collinearity among regressors or scale features."],
            warnings=[f"RESET test execution failed: {e}"],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )
