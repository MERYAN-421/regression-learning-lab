"""
Statistical Diagnostic Comparison Matrix Engine.
Aggregates and formats Milestone 4 DiagnosticSuiteResult objects side-by-side across candidate models.
Enforces Rule 3: Never converts statistical diagnostics into an artificial total score or assumption count.
"""
from typing import Any, Dict, List, Optional
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


def build_diagnostic_comparison_matrix(
    models: List[OLSModelResult],
    diagnostic_suites: Optional[Dict[str, DiagnosticSuiteResult]] = None
) -> List[Dict[str, Any]]:
    """
    Constructs a comparative diagnostic matrix across candidate models.
    Each row details a model's assumption tests, test statistics, p-values, and formal decisions.
    
    If a diagnostic suite was not run for a model, marks all entries as 'Not Run'
    rather than treating missing tests as favorable.
    """
    if not models:
        return []

    suites_dict = diagnostic_suites or {}
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
            "Suite Alpha": f"{suite.alpha:.2f}" if suite else "N/A"
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
        else:
            for key, (label, stat_name) in methods_map.items():
                diag: Optional[DiagnosticResult] = suite.diagnostics.get(key)
                if diag is None:
                    row[f"{label} Decision"] = "Not Run"
                    row[f"{label} Stat / p-val"] = "N/A"
                elif diag.decision == "not_applicable":
                    row[f"{label} Decision"] = "Not Applicable"
                    row[f"{label} Stat / p-val"] = "N/A"
                else:
                    # Format statistic and p-value
                    stat_val = diag.statistic_value
                    is_nan = (stat_val is None or (isinstance(stat_val, float) and np.isnan(stat_val)))
                    stat_str = f"{stat_val:.2f}" if not is_nan else "N/A"

                    pval_str = f"p={diag.p_value:.4f}" if diag.p_value is not None else ""
                    combined_stat = f"{stat_str} ({pval_str})" if pval_str else stat_str

                    row[f"{label} Decision"] = diag.decision.replace("_", " ").title()
                    row[f"{label} Stat / p-val"] = combined_stat

        # Condition Number
        row["Condition Number"] = f"{m.design_matrix_metadata.condition_number:.1f}"

        matrix_rows.append(row)

    return matrix_rows
