"""
Module B: Independence and Autocorrelation Diagnostics.
Provides Durbin-Watson statistic, Ljung-Box portmanteau test, and Autocorrelation Function (ACF).
Strictly enforces dataset-specific applicability:
- Temporal datasets (Appliances Energy): Sequential tests are valid and preserve chronological ordering.
- Grouped panel datasets (Life Expectancy): Global sequential tests are marked NOT_APPLICABLE.
- Cross-sectional datasets (KC House, Student Performance): Sequential tests are marked NOT_APPLICABLE or exploratory.
"""
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from src.data.models import SplitResult
from src.models.model_result import OLSModelResult
from src.diagnostics.result_models import (
    DiagnosticResult,
    DiagnosticType,
    DiagnosticDecision,
    ApplicabilityStatus,
    generate_diagnostic_id
)
from src.diagnostics.interpretation import (
    format_durbin_watson_interpretation,
    format_ljung_box_interpretation
)


def _check_independence_applicability(
    dataset_id: str,
    split_result: Optional[SplitResult]
) -> Tuple[ApplicabilityStatus, str]:
    """
    Evaluates whether sequential autocorrelation tests are statistically valid
    for the active dataset and split strategy.
    """
    if split_result is not None and getattr(split_result, "strategy", "") == "temporal":
        return (
            ApplicabilityStatus.APPLICABLE,
            "Temporal split active: Observations are chronologically sequenced in the data-generating process."
        )

    if dataset_id == "appliances_energy":
        return (
            ApplicabilityStatus.APPLICABLE,
            "Appliances Energy is an intrinsically chronological time series recorded at 10-minute intervals."
        )
    elif dataset_id == "life_expectancy":
        return (
            ApplicabilityStatus.NOT_APPLICABLE,
            "Life Expectancy is a panel dataset grouped by Country. Arbitrarily concatenating disjoint "
            "countries into a single continuous sequential series is statistically invalid."
        )
    elif dataset_id in ("kc_house_sales", "student_performance"):
        return (
            ApplicabilityStatus.NOT_APPLICABLE,
            f"{dataset_id.replace('_', ' ').title()} is a cross-sectional dataset. "
            "Row order does not represent a physical time dimension; sequential autocorrelation tests are not applicable."
        )
    else:
        # Generic heuristic
        if split_result is not None and getattr(split_result, "strategy", "") in ("random", "group"):
            return (
                ApplicabilityStatus.EXPLORATORY,
                f"Split strategy is '{split_result.strategy}'. Row index does not imply chronological sequencing."
            )
        return (
            ApplicabilityStatus.APPLICABLE,
            "Dataset sequencing assumed applicable."
        )


def run_durbin_watson_test(
    model_result: OLSModelResult,
    split_result: Optional[SplitResult] = None
) -> DiagnosticResult:
    """
    Computes the Durbin-Watson statistic for first-order (lag 1) residual autocorrelation.
    Note: DW is a descriptive diagnostic metric; it does not produce a conventional exact p-value.
    """
    start_time = time.time()
    config = {"method": "durbin_watson"}
    diag_id, repro_hash = generate_diagnostic_id(model_result.model_id, "durbin_watson", config)

    # Determine applicability
    app_status, app_reason = _check_independence_applicability(
        dataset_id=model_result.dataset_id,
        split_result=split_result
    )

    y_train = split_result.y_train if split_result is not None else None
    if y_train is not None and len(y_train) == len(model_result.train_predictions):
        residuals = y_train.values - model_result.train_predictions.values
    else:
        # Fallback to statsmodels resid
        residuals = model_result.fitted_model.resid.values

    # Check for non-applicability
    if app_status == ApplicabilityStatus.NOT_APPLICABLE:
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.INDEPENDENCE.value,
            method_id="durbin_watson",
            method_name="Durbin-Watson Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Residuals have zero first-order autocorrelation (rho = 0)",
            alternative_hypothesis="H1: Residuals exhibit first-order autocorrelation (rho != 0)",
            statistic_name="DW",
            statistic_value=float("nan"),
            p_value=None,
            alpha=None,
            decision=DiagnosticDecision.NOT_APPLICABLE.value,
            applicability_status=app_status.value,
            applicability_reason=app_reason,
            assumptions_prerequisites=["Chronological time series ordering", "Strictly exogenous regressors"],
            interpretation=f"Durbin-Watson test is not applicable for {model_result.dataset_id}: {app_reason}",
            limitations="Sequential autocorrelation tests require an empirical time axis.",
            recommendations=["Do not interpret sequential lag diagnostics on cross-sectional or multi-panel data."],
            warnings=[app_reason],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    try:
        dw_val = float(durbin_watson(residuals))
        interp_data = format_durbin_watson_interpretation(dw_val)

        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.INDEPENDENCE.value,
            method_id="durbin_watson",
            method_name="Durbin-Watson Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Residuals have zero first-order autocorrelation (rho = 0)",
            alternative_hypothesis="H1: Residuals exhibit first-order autocorrelation (rho != 0)",
            statistic_name="DW",
            statistic_value=dw_val,
            p_value=None,  # DW does not have an automated exact p-value
            alpha=None,
            decision=DiagnosticDecision.DESCRIPTIVE_ONLY.value,
            applicability_status=app_status.value,
            applicability_reason=app_reason,
            assumptions_prerequisites=[
                "Model contains an intercept",
                "Observations are strictly ordered in time",
                "No lagged dependent variables among regressors"
            ],
            details={"dw_statistic": dw_val, "sample_size": len(residuals)},
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
            diagnostic_type=DiagnosticType.INDEPENDENCE.value,
            method_id="durbin_watson",
            method_name="Durbin-Watson Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Residuals have zero first-order autocorrelation (rho = 0)",
            alternative_hypothesis="H1: Residuals exhibit first-order autocorrelation (rho != 0)",
            statistic_name="DW",
            statistic_value=float("nan"),
            p_value=None,
            alpha=None,
            decision=DiagnosticDecision.NUMERICAL_FAILURE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"Calculation failed: {e}",
            interpretation=f"Error computing Durbin-Watson statistic: {e}",
            limitations="",
            recommendations=[],
            warnings=[str(e)],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )


def run_ljung_box_test(
    model_result: OLSModelResult,
    split_result: Optional[SplitResult] = None,
    max_lag: Optional[int] = None,
    alpha: float = 0.05
) -> DiagnosticResult:
    """
    Executes the Ljung-Box portmanteau test for joint residual autocorrelation across multiple lags.
    
    H0: Autocorrelations up to lag m are jointly zero (errors are white noise).
    H1: At least one autocorrelation up to lag m is non-zero.
    
    Lag rule: If max_lag is None, uses default rule min(10, max(1, n // 10)).
    """
    start_time = time.time()

    # Determine applicability
    app_status, app_reason = _check_independence_applicability(
        dataset_id=model_result.dataset_id,
        split_result=split_result
    )

    y_train = split_result.y_train if split_result is not None else None
    if y_train is not None and len(y_train) == len(model_result.train_predictions):
        residuals = y_train.values - model_result.train_predictions.values
    else:
        residuals = model_result.fitted_model.resid.values

    n_obs = len(residuals)

    # Resolve lag configuration according to documented statistical rules
    if max_lag is None or max_lag <= 0:
        resolved_lag = min(10, max(1, n_obs // 10))
    else:
        resolved_lag = min(max_lag, max(1, n_obs - 2))

    config = {"max_lag": resolved_lag, "alpha": alpha}
    diag_id, repro_hash = generate_diagnostic_id(model_result.model_id, "ljung_box", config)

    # Check for non-applicability
    if app_status == ApplicabilityStatus.NOT_APPLICABLE:
        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.INDEPENDENCE.value,
            method_id="ljung_box",
            method_name="Ljung-Box Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis=f"H0: Residual autocorrelations up to lag {resolved_lag} are jointly zero",
            alternative_hypothesis=f"H1: At least one residual autocorrelation up to lag {resolved_lag} is non-zero",
            statistic_name="Q-statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NOT_APPLICABLE.value,
            applicability_status=app_status.value,
            applicability_reason=app_reason,
            assumptions_prerequisites=["Chronological ordering", "Stationary residual sequence"],
            interpretation=f"Ljung-Box test is not applicable for {model_result.dataset_id}: {app_reason}",
            limitations="Portmanteau autocorrelation tests require a genuine sequential data-generating process.",
            recommendations=["Do not run portmanteau tests on cross-sectional data."],
            warnings=[app_reason],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )

    try:
        # acorr_ljungbox: auto_lag or lags
        # Degrees of freedom adjustment: in OLS residuals, df should account for fitted slope parameters
        model_df = model_result.df_model
        # Call acorr_ljungbox
        lb_df = acorr_ljungbox(
            residuals,
            lags=[resolved_lag],
            return_df=True
        )

        q_stat = float(lb_df["lb_stat"].iloc[-1])
        p_val = float(lb_df["lb_pvalue"].iloc[-1])

        # Also collect multi-lag profile for visualization/inspection
        full_lb_df = acorr_ljungbox(
            residuals,
            lags=list(range(1, resolved_lag + 1)),
            return_df=True
        )
        lags_table = []
        for lag_idx, row in full_lb_df.iterrows():
            lags_table.append({
                "lag": int(lag_idx),
                "q_stat": float(row["lb_stat"]),
                "p_value": float(row["lb_pvalue"])
            })

        interp_data = format_ljung_box_interpretation(
            lb_stat=q_stat,
            p_value=p_val,
            alpha=alpha,
            max_lag=resolved_lag,
            is_valid_for_inference=model_result.is_valid_for_inference
        )

        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.INDEPENDENCE.value,
            method_id="ljung_box",
            method_name="Ljung-Box Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis=f"H0: Residual autocorrelations up to lag {resolved_lag} are jointly zero",
            alternative_hypothesis=f"H1: At least one residual autocorrelation up to lag {resolved_lag} is non-zero",
            statistic_name="Q-statistic",
            statistic_value=q_stat,
            p_value=p_val,
            alpha=alpha,
            decision=interp_data["decision"],
            applicability_status=app_status.value,
            applicability_reason=app_reason,
            assumptions_prerequisites=[
                "Observations are chronologically ordered",
                "Sufficient sample size relative to tested lags"
            ],
            details={
                "max_lag": resolved_lag,
                "q_statistic": q_stat,
                "p_value": p_val,
                "lags_profile": lags_table,
                "sample_size": n_obs
            },
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
            diagnostic_type=DiagnosticType.INDEPENDENCE.value,
            method_id="ljung_box",
            method_name="Ljung-Box Test",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis=f"H0: Residual autocorrelations up to lag {resolved_lag} are jointly zero",
            alternative_hypothesis=f"H1: At least one residual autocorrelation up to lag {resolved_lag} is non-zero",
            statistic_name="Q-statistic",
            statistic_value=float("nan"),
            p_value=None,
            alpha=alpha,
            decision=DiagnosticDecision.NUMERICAL_FAILURE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            applicability_reason=f"Ljung-Box calculation error: {e}",
            interpretation=f"Error running Ljung-Box test: {e}",
            limitations="",
            recommendations=[],
            warnings=[str(e)],
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )


def compute_residual_acf(
    residuals: np.ndarray,
    nlags: int = 20
) -> Dict[str, Any]:
    """
    Computes sample Autocorrelation Function (ACF) values and approximate Bartlett 95% confidence bands.
    """
    n = len(residuals)
    actual_lags = min(nlags, max(1, n - 2))
    acf_vals, confint = acf(residuals, nlags=actual_lags, alpha=0.05)

    # 95% Bartlett band: approx +/- 1.96 / sqrt(n)
    bartlett_band = 1.96 / np.sqrt(n)

    return {
        "lags": list(range(len(acf_vals))),
        "acf_values": acf_vals.tolist(),
        "confint_lower": confint[:, 0].tolist(),
        "confint_upper": confint[:, 1].tolist(),
        "bartlett_threshold": float(bartlett_band),
        "sample_size": n
    }
