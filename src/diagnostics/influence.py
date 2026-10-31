"""
Module F (Extension): Influence and Outlier Diagnostics.
Calculates Cook's Distance, Hat Matrix Leverage, and Studentized Residuals.
"""
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.model_result import OLSModelResult
from src.diagnostics.result_models import (
    DiagnosticResult,
    DiagnosticType,
    DiagnosticDecision,
    ApplicabilityStatus,
    generate_diagnostic_id
)


def run_influence_diagnostics(
    model_result: OLSModelResult
) -> DiagnosticResult:
    """
    Computes Cook's distance, leverage values (diagonal of hat matrix),
    and internally studentized residuals.
    """
    start_time = time.time()
    config = {"method": "influence_analysis"}
    diag_id, repro_hash = generate_diagnostic_id(model_result.model_id, "influence_analysis", config)

    fitted_sm = model_result.fitted_model
    n_obs = len(model_result.train_predictions)
    k_params = model_result.design_matrix_metadata.num_parameters

    # Heuristic cutoffs
    cooks_threshold = 4.0 / n_obs if n_obs > 0 else 1.0
    leverage_threshold = 2.0 * k_params / n_obs if n_obs > 0 else 0.5

    try:
        influence = fitted_sm.get_influence()
        cooks_d, _ = influence.cooks_distance
        hat_diag = influence.hat_matrix_diag
        stud_resid = influence.resid_studentized_internal

        high_cooks_count = int(np.sum(cooks_d > cooks_threshold))
        high_leverage_count = int(np.sum(hat_diag > leverage_threshold))

        max_cooks = float(np.max(cooks_d)) if len(cooks_d) > 0 else 0.0
        max_leverage = float(np.max(hat_diag)) if len(hat_diag) > 0 else 0.0

        interp = (
            f"Influence inspection: Maximum Cook's Distance is {max_cooks:.4f} (heuristic threshold: {cooks_threshold:.4f}); "
            f"{high_cooks_count} observation(s) exceed 4/n. Maximum Leverage is {max_leverage:.4f} (threshold: {leverage_threshold:.4f}); "
            f"{high_leverage_count} observation(s) exhibit high leverage in regressor space. "
            "Note: Points exceeding heuristic thresholds should be examined for domain context rather than automatically discarded."
        )

        details = {
            "cooks_distance": cooks_d,
            "hat_matrix_diag": hat_diag,
            "studentized_residuals": stud_resid,
            "cooks_threshold": cooks_threshold,
            "leverage_threshold": leverage_threshold,
            "high_cooks_count": high_cooks_count,
            "high_leverage_count": high_leverage_count,
            "max_cooks": max_cooks,
            "max_leverage": max_leverage
        }

        return DiagnosticResult(
            diagnostic_id=diag_id,
            model_id=model_result.model_id,
            dataset_id=model_result.dataset_id,
            dataset_fingerprint=model_result.dataset_fingerprint,
            split_fingerprint=model_result.split_fingerprint,
            diagnostic_type=DiagnosticType.INFLUENCE.value,
            method_id="influence_analysis",
            method_name="Influence & Leverage Analysis",
            is_valid_for_inference=model_result.is_valid_for_inference,
            null_hypothesis="H0: Observations have uniform moderate influence on parameter estimates",
            alternative_hypothesis="H1: Specific observations exert disproportionate leverage or influence",
            statistic_name="Max Cook's D",
            statistic_value=max_cooks,
            p_value=None,
            alpha=None,
            decision=DiagnosticDecision.DESCRIPTIVE_ONLY.value,
            applicability_status=ApplicabilityStatus.APPLICABLE.value,
            assumptions_prerequisites=["Full-rank fitted OLS model"],
            details=details,
            interpretation=interp,
            limitations="Cutoff thresholds (4/n and 2p/n) are heuristics and depend on sample size.",
            recommendations=[
                "Inspect high Cook's Distance points to determine if they represent measurement anomalies.",
                "Never delete points solely to inflate R-squared."
            ],
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
            diagnostic_type=DiagnosticType.INFLUENCE.value,
            method_id="influence_analysis",
            method_name="Influence & Leverage Analysis",
            is_valid_for_inference=model_result.is_valid_for_inference,
            statistic_name="Max Cook's D",
            statistic_value=float("nan"),
            decision=DiagnosticDecision.NUMERICAL_FAILURE.value,
            applicability_status=ApplicabilityStatus.NOT_APPLICABLE.value,
            interpretation=f"Error computing influence diagnostics: {e}",
            runtime_seconds=time.time() - start_time,
            configuration=config,
            reproducibility_hash=repro_hash
        )
