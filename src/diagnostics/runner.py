"""
Diagnostic Suite Runner and Orchestration Engine.
Validates model compatibility, manages session caching, and executes comprehensive
assumption testing across all five econometric diagnostic modules.
"""
from datetime import datetime
import hashlib
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.data.models import SplitResult
from src.models.model_result import OLSModelResult
from src.diagnostics.result_models import (
    DiagnosticResult,
    DiagnosticSuiteResult,
    DiagnosticType,
    ApplicabilityStatus
)
from src.diagnostics.linearity import run_ramsey_reset_test
from src.diagnostics.independence import run_durbin_watson_test, run_ljung_box_test
from src.diagnostics.homoscedasticity import run_breusch_pagan_test, run_white_test
from src.diagnostics.normality import run_jarque_bera_test, run_shapiro_wilk_test
from src.diagnostics.multicollinearity import run_multicollinearity_diagnostics
from src.diagnostics.influence import run_influence_diagnostics


def validate_active_model_for_diagnostics(
    model: Optional[OLSModelResult],
    current_dataset_id: str,
    current_dataset_fingerprint: str = "",
    current_split_fingerprint: str = ""
) -> Tuple[bool, str]:
    """
    Validates whether the active model is compatible with the current dataset and split state.
    Blocks diagnostics execution if stale, mismatched, or incomplete.
    """
    if model is None:
        return (
            False,
            "No active OLS model selected. Please navigate to the Regression Laboratory to fit a model first."
        )

    # Validate dataset ID
    if current_dataset_id and model.dataset_id and model.dataset_id != current_dataset_id:
        return (
            False,
            f"Dataset mismatch: Active model '{model.model_id}' was trained on '{model.dataset_id}', "
            f"but current active dataset is '{current_dataset_id}'. Please refit in Regression Laboratory."
        )

    # Validate dataset fingerprint if available
    if current_dataset_fingerprint and model.dataset_fingerprint and model.dataset_fingerprint != current_dataset_fingerprint:
        return (
            False,
            f"Dataset fingerprint mismatch: Model belongs to dataset version '{model.dataset_fingerprint[:8]}', "
            f"but active dataset fingerprint is '{current_dataset_fingerprint[:8]}'. Model is stale."
        )

    # Validate split fingerprint if available
    if current_split_fingerprint and model.split_fingerprint and model.split_fingerprint != current_split_fingerprint:
        return (
            False,
            f"Split fingerprint mismatch: Model was trained on split '{model.split_fingerprint[:8]}', "
            f"but active split fingerprint is '{current_split_fingerprint[:8]}'. Re-train on current split."
        )

    # Validate fitted model object
    if model.fitted_model is None:
        return (
            False,
            f"Model '{model.model_id}' lacks a fitted statsmodels reference required for residual diagnostics."
        )

    # Validate training predictions
    if model.train_predictions is None or len(model.train_predictions) == 0:
        return (
            False,
            f"Model '{model.model_id}' lacks training residual/prediction vectors."
        )

    return True, "Model is valid and aligned with active dataset split."


def run_full_diagnostic_suite(
    model_result: OLSModelResult,
    split_result: Optional[SplitResult] = None,
    alpha: float = 0.05,
    include_white: bool = False,
    include_shapiro: bool = False,
    include_influence: bool = True
) -> DiagnosticSuiteResult:
    """
    Executes the complete battery of econometric assumption diagnostics for a fitted OLS model.
    Evaluates:
    1. Linearity: Ramsey RESET test
    2. Independence: Durbin-Watson statistic, Ljung-Box test
    3. Homoscedasticity: Breusch-Pagan test (and optional White test)
    4. Normality: Jarque-Bera test (and optional Shapiro-Wilk test)
    5. Multicollinearity: VIF on all non-intercept regressors, Condition Number, Rank
    6. Influence: Cook's Distance and Leverage (optional extension)
    """
    raw_hash_tokens = [
        model_result.model_id,
        str(alpha),
        str(include_white),
        str(include_shapiro),
        str(include_influence)
    ]
    suite_id = "suite_" + hashlib.sha256("||".join(raw_hash_tokens).encode("utf-8")).hexdigest()[:12]

    suite = DiagnosticSuiteResult(
        suite_id=suite_id,
        model_id=model_result.model_id,
        dataset_id=model_result.dataset_id,
        dataset_fingerprint=model_result.dataset_fingerprint,
        split_fingerprint=model_result.split_fingerprint,
        alpha=alpha,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # 1. Linearity: Ramsey RESET
    res_reset = run_ramsey_reset_test(model_result, powers=[2, 3], alpha=alpha)
    suite.add_result(res_reset)

    # 2. Independence: Durbin-Watson & Ljung-Box
    res_dw = run_durbin_watson_test(model_result, split_result)
    suite.add_result(res_dw)

    res_lb = run_ljung_box_test(model_result, split_result, alpha=alpha)
    suite.add_result(res_lb)

    # 3. Homoscedasticity: Breusch-Pagan (& optional White)
    res_bp = run_breusch_pagan_test(model_result, alpha=alpha)
    suite.add_result(res_bp)

    if include_white:
        res_white = run_white_test(model_result, alpha=alpha)
        suite.add_result(res_white)

    # 4. Normality: Jarque-Bera (& optional Shapiro-Wilk)
    res_jb = run_jarque_bera_test(model_result, alpha=alpha)
    suite.add_result(res_jb)

    if include_shapiro:
        res_sw = run_shapiro_wilk_test(model_result, alpha=alpha)
        suite.add_result(res_sw)

    # 5. Multicollinearity: VIF & Rank
    res_vif = run_multicollinearity_diagnostics(model_result)
    suite.add_result(res_vif)

    # 6. Influence & Outliers
    if include_influence:
        res_inf = run_influence_diagnostics(model_result)
        suite.add_result(res_inf)

    # Global educational notice
    if not model_result.is_valid_for_inference:
        suite.warnings.append(
            "⚠️ Model was trained in prediction-only mode due to design matrix rank deficiency. "
            "Formal hypothesis tests (RESET, Breusch-Pagan, Ljung-Box) are descriptive or unidentifiable."
        )

    return suite
