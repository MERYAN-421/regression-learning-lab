"""
Statistical Diagnostic Comparison Matrix Engine.
Aggregates and formats Milestone 4 DiagnosticSuiteResult objects side-by-side across candidate models.
Enforces Rule 3: Never converts statistical diagnostics into an artificial total score or assumption count.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.model_result import OLSModelResult
from src.diagnostics.result_models import DiagnosticSuiteResult, DiagnosticResult


# Educational disclaimers attached to diagnostic comparison
DIAGNOSTIC_COMPARISON_NOTES = (
    "ℹ️ **Methodological Ground Rules for Diagnostic Comparison**:\n"
    "1. **No Artificial Scoring (Rule 3)**: Assumption tests have distinct null hypotheses, asymptotic properties, and sample dependencies. "
    "They are never summed or converted into a 'pass percentage' or composite score.\n"
    "2. **p-Value Interpretation**: Higher p-values do not signify 'better' models. A p-value >= alpha simply indicates insufficient evidence to reject H0.\n"
    "3. **Gauss-Markov vs. Normality**: Normality of residuals is not required for OLS to be BLUE; it only justifies exact finite-sample t/F inference.\n"
    "4. **Exogeneity Unprovability**: In-sample residuals orthogonal to X (X'e = 0) cannot prove strict exogeneity (E[ε|X] = 0)."
)


def validate_suite_compatibility(
    suite: DiagnosticSuiteResult,
    model: OLSModelResult
) -> Tuple[bool, List[str]]:
    """
    Strictly verifies that a DiagnosticSuiteResult belongs to the candidate model
    and matches its dataset ID, dataset fingerprint, split fingerprint, and data scope.
    Never silently accepts misaligned or substituted diagnostic suites.
    """
    issues = []
    if suite.model_id != model.model_id:
        issues.append(f"Model ID mismatch: Suite '{suite.suite_id}' belongs to '{suite.model_id}', not '{model.model_id}'.")
    if suite.dataset_id != model.dataset_id:
        issues.append(f"Dataset ID mismatch: Suite '{suite.suite_id}' dataset '{suite.dataset_id}' != model '{model.dataset_id}'.")
    if suite.dataset_fingerprint != model.dataset_fingerprint:
        issues.append(f"Dataset fingerprint mismatch: Suite '{suite.suite_id}' ({suite.dataset_fingerprint[:8]}) != model ({model.dataset_fingerprint[:8]}).")
    if suite.split_fingerprint != model.split_fingerprint:
        issues.append(f"Split fingerprint mismatch: Suite '{suite.suite_id}' ({suite.split_fingerprint[:8]}) != model ({model.split_fingerprint[:8]}).")

    # Scope check: Diagnostics must evaluate training residuals
    for diag_key, diag_res in suite.diagnostics.items():
        if getattr(diag_res, "data_scope", "train") != "train":
            issues.append(f"Invalid data scope '{diag_res.data_scope}' for diagnostic '{diag_key}'. Must evaluate 'train'.")

    return (len(issues) == 0, issues)


def check_diagnostic_configuration_differences(
    diagnostic_suites: Optional[Dict[str, DiagnosticSuiteResult]]
) -> List[str]:
    """
    Inspects diagnostic suites across compared models to detect and flag
    mismatched hypothesis test configurations (differing alpha, RESET powers, or Ljung-Box lags).
    Ensures mismatched tests are visibly identified and not conflated as equivalent statistical evidence.
    """
    if not diagnostic_suites:
        return []

    suites = [s for s in diagnostic_suites.values() if s is not None]
    if len(suites) <= 1:
        return []

    notices = []

    # 1. Check alpha thresholds
    alphas = {round(s.alpha, 4) for s in suites}
    if len(alphas) > 1:
        notices.append(
            f"Significance Level Mismatch: Diagnostic suites evaluate different alpha thresholds "
            f"({sorted(list(alphas))}). Hypothesis test rejection decisions use unequal evidentiary standards."
        )

    # 2. Check Ramsey RESET powers
    reset_powers = set()
    for s in suites:
        diag = s.diagnostics.get("ramsey_reset")
        if diag and diag.configuration:
            powers = diag.configuration.get("powers", [2, 3])
            reset_powers.add(tuple(sorted(powers)))
    if len(reset_powers) > 1:
        notices.append(
            f"Ramsey RESET Specification Mismatch: Candidate models evaluate different polynomial power sets "
            f"({[list(p) for p in sorted(list(reset_powers))]}). Test statistics evaluate different functional alternative hypotheses."
        )

    # 3. Check Ljung-Box lags
    lb_lags = set()
    for s in suites:
        diag = s.diagnostics.get("ljung_box")
        if diag and diag.configuration:
            lags = diag.configuration.get("lags", 10)
            lb_lags.add(lags)
    if len(lb_lags) > 1:
        notices.append(
            f"Ljung-Box Lag Mismatch: Candidate models evaluate different autocorrelation lag orders "
            f"({sorted(list(lb_lags))}). Q-statistics and degrees of freedom are not directly comparable."
        )

    return notices


def build_diagnostic_comparison_matrix(
    models: List[OLSModelResult],
    diagnostic_suites: Optional[Dict[str, DiagnosticSuiteResult]] = None
) -> List[Dict[str, Any]]:
    """
    Constructs a comparative diagnostic matrix across candidate models.
    Each row details a model's assumption tests, test statistics, p-values, configurations, and formal decisions.
    
    Strictly enforces:
    1. Validation of suite identity, dataset fingerprint, and split fingerprint against each candidate model.
    2. Explicit marking of 'Not Run' or 'Incompatible Suite' rather than treating missing suites as favorable.
    3. Disclosure of alpha and diagnostic-specific parameters (e.g. RESET powers, Ljung-Box lags).
    4. Visible identification of differing test configurations across candidate models.
    5. Rule 3: Zero composite scoring, pass counts, or p-value ranking.
    """
    if not models:
        return []

    suites_dict = diagnostic_suites or {}
    config_diff_notices = check_diagnostic_configuration_differences(suites_dict)
    matrix_rows: List[Dict[str, Any]] = []

    for m in models:
        m_id = m.model_id
        suite = suites_dict.get(m_id)

        # Baseline info
        row: Dict[str, Any] = {
            "Model ID": m_id,
            "Feature Source": m.feature_source_label,
            "Parameters (p)": m.design_matrix_metadata.num_parameters,
            "Inference Status": "Full Rank" if m.is_valid_for_inference else "Prediction Only",
            "Suite ID": suite.suite_id if suite else "Not Run",
            "Suite Alpha": f"{suite.alpha:.2f}" if suite else "N/A",
            "Suite Compatibility": "Valid" if suite else "Not Available",
            "Config Status": "⚠️ Differing Test Configs" if (config_diff_notices and suite) else "Consistent"
        }

        # Methods to inspect
        methods_map = {
            "ramsey_reset": ("RESET", "F-statistic"),
            "durbin_watson": ("DW", "DW"),
            "ljung_box": ("Ljung-Box", "Q-statistic"),
            "breusch_pagan": ("Breusch-Pagan", "LM statistic"),
            "jarque_bera": ("Jarque-Bera", "JB statistic"),
            "multicollinearity_vif": ("Max VIF", "Max VIF")
        }

        if suite is None:
            for key, (label, _) in methods_map.items():
                row[f"{label} Decision"] = "Not Run"
                row[f"{label} Stat / p-val"] = "N/A"
                row[f"{label} Config"] = "N/A"
        else:
            is_compat, compat_issues = validate_suite_compatibility(suite, m)
            if not is_compat:
                row["Suite Compatibility"] = f"Incompatible: {'; '.join(compat_issues)}"
                for key, (label, _) in methods_map.items():
                    row[f"{label} Decision"] = "Incompatible Suite"
                    row[f"{label} Stat / p-val"] = "N/A"
                    row[f"{label} Config"] = "N/A"
            else:
                for key, (label, stat_name) in methods_map.items():
                    diag: Optional[DiagnosticResult] = suite.diagnostics.get(key)
                    if diag is None:
                        row[f"{label} Decision"] = "Not Run"
                        row[f"{label} Stat / p-val"] = "N/A"
                        row[f"{label} Config"] = "N/A"
                    elif diag.decision == "not_applicable":
                        row[f"{label} Decision"] = "Not Applicable"
                        row[f"{label} Stat / p-val"] = "N/A"
                        row[f"{label} Config"] = str(diag.configuration) if diag.configuration else "N/A"
                    else:
                        # Format statistic and p-value
                        stat_val = diag.statistic_value
                        is_nan = (stat_val is None or (isinstance(stat_val, float) and np.isnan(stat_val)))
                        stat_str = f"{stat_val:.2f}" if not is_nan else "N/A"

                        pval_str = f"p={diag.p_value:.4f}" if diag.p_value is not None else ""
                        combined_stat = f"{stat_str} ({pval_str})" if pval_str else stat_str

                        row[f"{label} Decision"] = diag.decision.replace("_", " ").title()
                        row[f"{label} Stat / p-val"] = combined_stat
                        # Diagnostic specific configuration details
                        cfg = diag.configuration or {}
                        if key == "ramsey_reset":
                            row[f"{label} Config"] = f"powers={cfg.get('powers', [2, 3])}"
                        elif key == "ljung_box":
                            row[f"{label} Config"] = f"lags={cfg.get('lags', 10)}"
                        else:
                            row[f"{label} Config"] = f"alpha={suite.alpha:.2f}"

        # Condition Number
        row["Condition Number"] = f"{m.design_matrix_metadata.condition_number:.1f}"

        matrix_rows.append(row)

    return matrix_rows
