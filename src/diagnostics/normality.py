"""
Module D: Normality of Residuals Diagnostics.
Provides the Jarque-Bera test, optional Shapiro-Wilk test, and Q-Q / Histogram calculation helpers.
Educational principle: Normality is NOT required for the Gauss-Markov theorem (BLUE);
it is only needed for exact finite-sample Student-t and Fisher-F hypothesis testing.
"""
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import jarque_bera

from src.models.model_result import OLSModelResult
from src.diagnostics.result_models import (
    DiagnosticResult,
    DiagnosticType,
    DiagnosticDecision,
    ApplicabilityStatus,
    generate_diagnostic_id
)
from src.diagnostics.interpretation import format_jarque_bera_interpretation


def run_jarque_bera_test(
    model_result: OLSModelResult,
    alpha: float = 0.05
) -> DiagnosticResult:
    """
    Executes the Jarque-Bera goodness-of-fit test for normality of residuals based on sample skewness and kurtosis.
    
    H0: Error distribution is normal (skewness = 0, excess kurtosis = 0).
    H1: Error distribution is non-normal (skewness != 0 or excess kurtosis != 0).
    """
    start_time = time.time()
    config = {"alpha": alpha, "method": "jarque_bera"}
    diag_id, repro_hash = generate_diagnostic_id(model_result.model_id, "jarque_bera", config)

    residuals = model_result.fitted_model.resid.values
    n_obs = len(residuals)

    if n_obs < 8:
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.NORMALITY.value,
            method_id="jarque_bera",
            method_name="Jarque-Bera Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Residuals are drawn from a normal distribution",
            alternative_hypothesis="H1: Residuals are drawn from a non-normal distribution",
            statistic_name="JB statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.INSUFFICIENT_DATA.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"Insufficient sample size: n={n_obs} < 8 observations.",
            assumptions_prerequisites=["n >= 8 observations"],
            interpretation="Jarque-Bera requires at least 8 observations to compute third and fourth moments reliably.",
            limitations="",
            recommendations=["Increase training sample size."],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    try:
        jb_res = jarque_bera(residuals)
        jb_stat = float(jb_res[0])
        p_val = float(jb_res[1])
        skew = float(jb_res[2])
        kurt = float(jb_res[3])

        interp_data = format_jarque_bera_interpretation(
            jb_stat=jb_stat,
            p_value=p_val,
            skew=skew,
            kurtosis=kurt,
            alpha=alpha,
            sample_size=n_obs,
            is_valid_for_inference=model_result.is_valid_for_inference
        )

        details = {
            "jb_statistic": jb_stat,
            "p_value": p_val,
            "sample_skewness": skew,
            "sample_kurtosis": kurt,
            "excess_kurtosis": kurt - 3.0,
            "sample_size": n_obs
        }

        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.NORMALITY.value,
            method_id="jarque_bera",
            method_name="Jarque-Bera Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Residuals are drawn from a normal distribution (skewness = 0, kurtosis = 3)",
            alternative_hypothesis="H1: Residuals are drawn from a non-normal distribution (skewness != 0 or kurtosis != 3)",
            statistic_name="JB statistic",
            statistic_value=jb_stat,
            p_value=p_val,
            alpha=alpha,
            decision=interp_data["decision"],
            applicability_status=ApplicabilityStatus.APPLICABLE.value,
            assumptions_prerequisites=[
                "Residuals represent errors",
                "Asymptotic approximation holds for moderate sample size"
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
            diagnostic_type=DiagnosticType.NORMALITY.value,
            method_id="jarque_bera",
            method_name="Jarque-Bera Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Residuals are drawn from a normal distribution",
            alternative_hypothesis="H1: Residuals are drawn from a non-normal distribution",
            statistic_name="JB statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NUMERICAL_FAILURE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"Jarque-Bera calculation error: {e}",
            interpretation=f"Error running Jarque-Bera test: {e}",
            limitations="",
            recommendations=[],
            warnings=[str(e)],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )


def run_shapiro_wilk_test(
    model_result: OLSModelResult,
    alpha: float = 0.05
) -> DiagnosticResult:
    """
    Executes the Shapiro-Wilk test for normality.
    Appropriate for small-to-moderate sample sizes (n <= 5000).
    """
    start_time = time.time()
    config = {"alpha": alpha, "method": "shapiro_wilk"}
    diag_id, repro_hash = generate_diagnostic_id(model_result.model_id, "shapiro_wilk", config)

    residuals = model_result.fitted_model.resid.values
    n_obs = len(residuals)

    if n_obs > 5000:
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.NORMALITY.value,
            method_id="shapiro_wilk",
            method_name="Shapiro-Wilk Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Residuals are drawn from a normal distribution",
            alternative_hypothesis="H1: Residuals are drawn from a non-normal distribution",
            statistic_name="W statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NOT_APPLICABLE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"Sample size n={n_obs:,} exceeds the standard Shapiro-Wilk limit (n <= 5,000).",
            assumptions_prerequisites=["n <= 5000"],
            interpretation="For large samples, the Jarque-Bera test and Q-Q plots provide appropriate normality diagnostics.",
            limitations="Shapiro-Wilk is overpowered on large datasets and rejects normality on trivial deviations.",
            recommendations=["Use Jarque-Bera test and inspect Normal Q-Q plot for large samples."],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    try:
        w_stat, p_val = stats.shapiro(residuals)
        w_stat = float(w_stat)
        p_val = float(p_val)

        dec = DiagnosticDecision.REJECT_NULL.value if p_val < alpha else DiagnosticDecision.FAIL_TO_REJECT_NULL.value
        interp = (
            f"The Shapiro-Wilk test {'rejects' if dec == DiagnosticDecision.REJECT_NULL.value else 'fails to reject'} H0 "
            f"(W = {w_stat:.4f}, p = {p_val:.4f}). "
            f"{'Residuals depart significantly from normality.' if dec == DiagnosticDecision.REJECT_NULL.value else 'Residuals are consistent with a normal distribution.'} "
            "Note: Rejection does not invalidate BLUE property of OLS estimators."
        )

        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.NORMALITY.value,
            method_id="shapiro_wilk",
            method_name="Shapiro-Wilk Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Residuals are drawn from a normal distribution",
            alternative_hypothesis="H1: Residuals are drawn from a non-normal distribution",
            statistic_name="W statistic",
            statistic_value=w_stat,
            p_value=p_val,
            alpha=alpha,
            decision=dec,
            applicability_status=ApplicabilityStatus.APPLICABLE.value,
            assumptions_prerequisites=["Continuous univariate sample", "Sample size <= 5,000"],
            details={"w_statistic": w_stat, "p_value": p_val, "sample_size": n_obs},
            interpretation=interp,
            limitations="Overpowered in moderate-to-large samples; visual inspection of Q-Q plot is recommended.",
            recommendations=["Cross-reference with Jarque-Bera and Q-Q Plot."],
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
            diagnostic_type=DiagnosticType.NORMALITY.value,
            method_id="shapiro_wilk",
            method_name="Shapiro-Wilk Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Residuals are drawn from a normal distribution",
            alternative_hypothesis="H1: Residuals are drawn from a non-normal distribution",
            statistic_name="W statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NUMERICAL_FAILURE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"Shapiro-Wilk error: {e}",
            interpretation=f"Error executing Shapiro-Wilk test: {e}",
            limitations="",
            recommendations=[],
            warnings=[str(e)],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )


def compute_qq_plot_data(residuals: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes theoretical quantiles and sample ordered quantiles for Normal Q-Q plotting.
    Also computes the reference line parameters (slope and intercept).
    """
    s_std = np.std(residuals, ddof=1)
    std_residuals = residuals / s_std if s_std > 1e-12 else residuals

    # Use scipy probplot
    (osm, osr), (slope, intercept, r) = stats.probplot(std_residuals, dist="norm")

    return {
        "theoretical_quantiles": osm,
        "sample_quantiles": osr,
        "line_slope": slope,
        "line_intercept": intercept,
        "correlation_r": r,
        "sample_size": len(residuals)
    }
