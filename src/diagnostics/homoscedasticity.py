"""
Module C: Homoscedasticity Diagnostics.
Provides the Breusch-Pagan test, White's test, and Scale-Location data computation.
"""
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

from src.models.model_result import OLSModelResult
from src.diagnostics.result_models import (
    DiagnosticResult,
    DiagnosticType,
    DiagnosticDecision,
    ApplicabilityStatus,
    generate_diagnostic_id
)
from src.diagnostics.interpretation import format_breusch_pagan_interpretation


def run_breusch_pagan_test(
    model_result: OLSModelResult,
    alpha: float = 0.05
) -> DiagnosticResult:
    """
    Executes the Breusch-Pagan test for heteroscedasticity.
    
    H0: Error variance is constant across regressor levels (homoscedasticity: Var(e_i | X) = sigma^2).
    H1: Error variance is an explicit linear function of regressors (heteroscedasticity).
    
    Primary decision drives off the Lagrange Multiplier (LM) score statistic, distributed as Chi-Square(p).
    """
    start_time = time.time()
    config = {"alpha": alpha, "method": "breusch_pagan"}
    diag_id, repro_hash = generate_diagnostic_id(model_result.model_id, "breusch_pagan", config)

    # Check rank and inference validity
    if not model_result.is_valid_for_inference:
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.HOMOSCEDASTICITY.value,
            method_id="breusch_pagan",
            method_name="Breusch-Pagan Test",
            is_valid_for_inference=False,
            null_hypothesis="H0: Error variance is constant across all regressor values (Homoscedasticity)",
            alternative_hypothesis="H1: Error variance depends linearly on regressors (Heteroscedasticity)",
            statistic_name="LM statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NOT_APPLICABLE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason="Design matrix is rank-deficient; auxiliary variance regression cannot be identified.",
            assumptions_prerequisites=["Full column rank design matrix"],
            interpretation="The Breusch-Pagan test cannot be conducted on a rank-deficient or prediction-only model.",
            limitations="Auxiliary regression requires full rank.",
            recommendations=["Eliminate collinear features before running heteroscedasticity tests."],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    fitted_sm = model_result.fitted_model
    residuals = fitted_sm.resid
    exog = fitted_sm.model.exog

    # Validate degrees of freedom: n > k
    n_obs, k_cols = exog.shape
    if n_obs <= k_cols:
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.HOMOSCEDASTICITY.value,
            method_id="breusch_pagan",
            method_name="Breusch-Pagan Test",
            is_valid_for_inference=True,
            null_hypothesis="H0: Error variance is constant across all regressor values (Homoscedasticity)",
            alternative_hypothesis="H1: Error variance depends linearly on regressors (Heteroscedasticity)",
            statistic_name="LM statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.INSUFFICIENT_DATA.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"Insufficient sample size: n={n_obs} <= parameters={k_cols}.",
            interpretation=f"Cannot run Breusch-Pagan test with n={n_obs} <= k={k_cols}.",
            limitations="Auxiliary variance regression requires positive residual degrees of freedom.",
            recommendations=["Acquire more observations or reduce parameter count."],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    try:
        lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(residuals, exog)

        interp_data = format_breusch_pagan_interpretation(
            lm_stat=float(lm_stat),
            lm_pval=float(lm_pval),
            f_stat=float(f_stat),
            f_pval=float(f_pval),
            alpha=alpha,
            is_valid_for_inference=True
        )

        details = {
            "lm_statistic": float(lm_stat),
            "lm_pvalue": float(lm_pval),
            "f_statistic": float(f_stat),
            "f_pvalue": float(f_pval),
            "primary_decision_rule": "Lagrange Multiplier (LM) score test p-value compared to alpha",
            "sample_size": n_obs,
            "df_auxiliary": k_cols - 1
        }

        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.HOMOSCEDASTICITY.value,
            method_id="breusch_pagan",
            method_name="Breusch-Pagan Test",
            is_valid_for_inference=True,
            null_hypothesis="H0: Error variance is constant across all regressor values (Homoscedasticity)",
            alternative_hypothesis="H1: Error variance depends linearly on regressors (Heteroscedasticity)",
            statistic_name="LM statistic",
            statistic_value=float(lm_stat),
            p_value=float(lm_pval),
            alpha=alpha,
            decision=interp_data["decision"],
            applicability_status=ApplicabilityStatus.APPLICABLE.value,
            assumptions_prerequisites=[
                "OLS model has full column rank",
                "Auxiliary variance regression contains regressors",
                "Error terms are independent"
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
            diagnostic_type=DiagnosticType.HOMOSCEDASTICITY.value,
            method_id="breusch_pagan",
            method_name="Breusch-Pagan Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Error variance is constant across all regressor values (Homoscedasticity)",
            alternative_hypothesis="H1: Error variance depends linearly on regressors (Heteroscedasticity)",
            statistic_name="LM statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NUMERICAL_FAILURE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"Calculation error: {e}",
            interpretation=f"Error executing Breusch-Pagan test: {e}",
            limitations="",
            recommendations=[],
            warnings=[str(e)],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )


def run_white_test(
    model_result: OLSModelResult,
    alpha: float = 0.05
) -> DiagnosticResult:
    """
    Executes White's general test for heteroscedasticity.
    Includes squares and cross-products of regressors in the auxiliary regression.
    Guards against high-dimensional explosions when parameter count is large.
    """
    start_time = time.time()
    config = {"alpha": alpha, "method": "white"}
    diag_id, repro_hash = generate_diagnostic_id(model_result.model_id, "white_test", config)

    if not model_result.is_valid_for_inference:
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.HOMOSCEDASTICITY.value,
            method_id="white_test",
            method_name="White's Test",
            is_valid_for_inference=False,
            null_hypothesis="H0: Error variance is constant (Homoscedasticity)",
            alternative_hypothesis="H1: Error variance depends on regressors, squares, or cross-products",
            statistic_name="LM statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NOT_APPLICABLE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason="Design matrix is rank-deficient.",
            assumptions_prerequisites=["Full column rank"],
            interpretation="White's test requires a full-rank identifiable model.",
            limitations="",
            recommendations=[],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    fitted_sm = model_result.fitted_model
    residuals = fitted_sm.resid
    exog = fitted_sm.model.exog
    n_obs, k_cols = exog.shape

    # Number of terms in White's auxiliary regression: approx k*(k+1)/2
    num_white_terms = (k_cols * (k_cols + 1)) // 2
    if num_white_terms >= n_obs - 5 or k_cols > 25:
        # Auxiliary regression is overparameterized
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.HOMOSCEDASTICITY.value,
            method_id="white_test",
            method_name="White's Test",
            is_valid_for_inference=True,
            null_hypothesis="H0: Error variance is constant (Homoscedasticity)",
            alternative_hypothesis="H1: Error variance depends on regressors, squares, or cross-products",
            statistic_name="LM statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NOT_APPLICABLE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=(
                f"White's test auxiliary regression requires ~{num_white_terms} parameters, "
                f"which exceeds or approaches sample size n={n_obs}. Computational degrees of freedom exhausted."
            ),
            assumptions_prerequisites=["Sample size substantially exceeds k*(k+1)/2 terms"],
            interpretation=(
                f"White's test was bypassed due to high dimensionality: {k_cols} regressors generate "
                f"~{num_white_terms} quadratic and interaction terms for n={n_obs} observations. "
                "The Breusch-Pagan test should be utilized instead for linear variance diagnostics."
            ),
            limitations="White's test loses degrees of freedom quadratically as feature count increases.",
            recommendations=["Use the Breusch-Pagan test for linear variance dependency or reduce predictor dimensions."],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    try:
        lm_stat, lm_pval, f_stat, f_pval = het_white(residuals, exog)
        p_val = float(lm_pval)
        dec = DiagnosticDecision.REJECT_NULL.value if p_val < alpha else DiagnosticDecision.FAIL_TO_REJECT_NULL.value

        interp = (
            f"White's general test {'rejects' if dec == DiagnosticDecision.REJECT_NULL.value else 'fails to reject'} H0 "
            f"(LM = {lm_stat:.4f}, p = {p_val:.4f}). "
            f"{'Evidence of general heteroscedasticity or non-linear variance structure.' if dec == DiagnosticDecision.REJECT_NULL.value else 'No sufficient evidence of general heteroscedasticity.'}"
        )

        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.HOMOSCEDASTICITY.value,
            method_id="white_test",
            method_name="White's Test",
            is_valid_for_inference=True,
            null_hypothesis="H0: Error variance is constant (Homoscedasticity)",
            alternative_hypothesis="H1: Error variance depends on regressors, squares, or cross-products",
            statistic_name="LM statistic",
            statistic_value=float(lm_stat),
            p_value=p_val,
            alpha=alpha,
            decision=dec,
            applicability_status=ApplicabilityStatus.APPLICABLE.value,
            assumptions_prerequisites=["Sufficient sample size for quadratic auxiliary expansion"],
            details={"lm_statistic": float(lm_stat), "lm_pvalue": p_val, "f_statistic": float(f_stat), "f_pvalue": float(f_pval)},
            interpretation=interp,
            limitations="Sensitive to specification error and requires large sample sizes.",
            recommendations=["Compare with Breusch-Pagan and Scale-Location plots."],
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
            diagnostic_type=DiagnosticType.HOMOSCEDASTICITY.value,
            method_id="white_test",
            method_name="White's Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Error variance is constant (Homoscedasticity)",
            alternative_hypothesis="H1: Error variance depends on regressors, squares, or cross-products",
            statistic_name="LM statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NUMERICAL_FAILURE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"White's test execution error: {e}",
            interpretation=f"Error executing White's test: {e}",
            limitations="",
            recommendations=[],
            warnings=[str(e)],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )


def compute_scale_location_data(
    fitted_values: np.ndarray,
    residuals: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Computes coordinates for the Scale-Location plot:
    X-axis: Fitted values (y_hat)
    Y-axis: Square root of absolute standardized residuals: sqrt(|e_i / s|)
    """
    s_std = np.std(residuals, ddof=1)
    if s_std < 1e-12:
        std_resid = np.zeros_like(residuals)
    else:
        std_resid = residuals / s_std

    sqrt_abs_std_resid = np.sqrt(np.abs(std_resid))

    return {
        "fitted_values": fitted_values,
        "residuals": residuals,
        "standardized_residuals": std_resid,
        "sqrt_abs_std_residuals": sqrt_abs_std_resid
    }
