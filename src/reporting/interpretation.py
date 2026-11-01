"""
Statistical Interpretation Engine for Academic Reports.
Implements rigorous, rule-based econometric interpretation standards.
Prevents ungrounded causal claims, false acceptance of null hypotheses,
and invalid inference from prediction-only models.
"""
from typing import Any, Dict, List, Optional
import numpy as np

from src.diagnostics.result_models import DiagnosticDecision, ApplicabilityStatus, DiagnosticType


def interpret_hypothesis_test(
    p_value: Optional[float],
    alpha: float = 0.05,
    null_hypothesis: str = "H0",
    alternative_hypothesis: str = "H1",
    test_name: str = "Test"
) -> Dict[str, Any]:
    """
    Rigorously interprets a statistical hypothesis test following scientific conventions.
    Never asserts that H0 is proven true.
    """
    if p_value is None or (isinstance(p_value, float) and (np.isnan(p_value) or np.isinf(p_value))):
        return {
            "decision": "Not Run / Descriptive Only",
            "p_value_display": "N/A",
            "alpha": alpha,
            "statement": f"{test_name} does not provide a formal sampling distribution p-value.",
            "reject_null": False,
            "caution": "Do not treat this descriptive statistic as a formal significance test."
        }

    pval_str = "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}"

    if p_value < alpha:
        statement = (
            f"Reject the null hypothesis ({null_hypothesis}) at significance level α = {alpha} "
            f"(p = {pval_str} < {alpha}). The sample provides sufficient statistical evidence "
            f"in favor of the alternative hypothesis ({alternative_hypothesis})."
        )
        reject = True
        decision_label = "Reject H0"
    else:
        statement = (
            f"Fail to reject the null hypothesis ({null_hypothesis}) at significance level α = {alpha} "
            f"(p = {pval_str} ≥ {alpha}). The sample data do not provide sufficient evidence "
            f"to reject {null_hypothesis}. Note: This failure to reject does NOT prove H0 is true."
        )
        reject = False
        decision_label = "Fail to Reject H0"

    return {
        "decision": decision_label,
        "p_value_display": pval_str,
        "p_value_raw": p_value,
        "alpha": alpha,
        "statement": statement,
        "reject_null": reject,
        "caution": "Absence of evidence against H0 is not mathematical or scientific proof of H0."
    }


def get_diagnostic_interpretation_guidance(
    diag_type: str,
    method_id: str,
    decision: str,
    p_value: Optional[float],
    statistic_value: float,
    alpha: float = 0.05
) -> Dict[str, str]:
    """
    Provides standard econometric interpretations and caveats for each diagnostic category.
    """
    method_clean = method_id.lower()

    if decision == DiagnosticDecision.NOT_APPLICABLE.value:
        return {
            "finding": "Test Not Applicable for this Data Structure.",
            "econometric_meaning": "The mathematical assumptions of this test (e.g. strict chronological ordering) are violated by the dataset structure.",
            "recommended_action": "Omit sequential diagnostic from inferential conclusions."
        }

    if decision == DiagnosticDecision.NUMERICAL_FAILURE.value:
        return {
            "finding": "Numerical Estimation Failure.",
            "econometric_meaning": "Hessian inversion, singular matrix, or zero residual variance prevented test statistic computation.",
            "recommended_action": "Inspect design matrix rank and multicollinearity."
        }

    # Linearity (Ramsey RESET)
    if "reset" in method_clean:
        if decision == DiagnosticDecision.REJECT_NULL.value:
            return {
                "finding": "Evidence of Model Functional Form Misspecification (RESET p < α).",
                "econometric_meaning": "Nonlinear powers of fitted values are statistically significant. The linear conditional mean specification E[Y|X] is likely inadequate.",
                "recommended_action": "Consider polynomial terms, interaction terms, or nonlinear transformations."
            }
        else:
            return {
                "finding": "No Detected Functional Form Misspecification via Polynomial Powers.",
                "econometric_meaning": "Fitted powers do not have statistically significant explanatory power. (Does not prove linear specification is globally correct).",
                "recommended_action": "Proceed with linear specification while monitoring out-of-sample residuals."
            }

    # Homoscedasticity (Breusch-Pagan, White)
    if "breusch_pagan" in method_clean or "white" in method_clean:
        if decision == DiagnosticDecision.REJECT_NULL.value:
            return {
                "finding": "Heteroscedasticity Detected (p < α).",
                "econometric_meaning": "Residual variance Var(ε|X) depends on regressors. OLS coefficient estimates remain unbiased and consistent, but standard errors and t/F tests are invalid.",
                "recommended_action": "Apply Heteroscedasticity-Consistent (HC1-HC3) robust standard errors or Weighted Least Squares (WLS)."
            }
        else:
            return {
                "finding": "Homoscedasticity Assumption Not Rejected (p ≥ α).",
                "econometric_meaning": "Insufficient sample evidence of variance dependency on regressors. Conventional OLS standard errors appear tenable.",
                "recommended_action": "Conventional OLS inference may be reported with standard caveats."
            }

    # Independence / Autocorrelation (Durbin-Watson, Ljung-Box)
    if "durbin_watson" in method_clean:
        stat = statistic_value
        if np.isnan(stat):
            return {"finding": "Durbin-Watson statistic unavailable.", "econometric_meaning": "N/A", "recommended_action": "N/A"}
        if stat < 1.5:
            finding = f"Positive First-Order Autocorrelation Indicated (DW = {stat:.3f} < 1.5)."
            meaning = "Residuals exhibit positive serial correlation. Conventional standard errors are biased downwards, inflating t-statistics."
            action = "Consider lag features, difference transformations, or Newey-West HAC standard errors."
        elif stat > 2.5:
            finding = f"Negative First-Order Autocorrelation Indicated (DW = {stat:.3f} > 2.5)."
            meaning = "Residuals exhibit negative serial correlation."
            action = "Inspect cyclical features or alternating pattern in ordering."
        else:
            finding = f"No Strong First-Order Autocorrelation Indicated (DW = {stat:.3f} ≈ 2.0)."
            meaning = "DW statistic is within acceptable range [1.5, 2.5] for uncorrelated successive residuals."
            action = "Serial independence assumption tenable for lag 1."
        return {"finding": finding, "econometric_meaning": meaning, "recommended_action": action}

    if "ljung_box" in method_clean:
        if decision == DiagnosticDecision.REJECT_NULL.value:
            return {
                "finding": "Higher-Order Autocorrelation Detected (Ljung-Box p < α).",
                "econometric_meaning": "Residuals exhibit significant autocorrelation across multiple lags.",
                "recommended_action": "Include autoregressive terms or use Newey-West standard errors."
            }
        else:
            return {
                "finding": "No Higher-Order Autocorrelation Detected (p ≥ α).",
                "econometric_meaning": "Residuals appear uncorrelated up to the tested lag order.",
                "recommended_action": "Residual independence assumption tenable across tested lags."
            }

    # Normality (Jarque-Bera, Shapiro-Wilk)
    if "jarque_bera" in method_clean or "shapiro" in method_clean:
        if decision == DiagnosticDecision.REJECT_NULL.value:
            return {
                "finding": "Non-Normal Residual Distribution (p < α).",
                "econometric_meaning": "Residuals deviate from Gaussian normality (skewness or excess kurtosis). Under Gauss-Markov, OLS remains Best Linear Unbiased Estimator (BLUE). However, exact finite-sample t and F tests rely on normality; for large samples, Central Limit Theorem ensures asymptotic validity.",
                "recommended_action": "Examine outliers and leverage; report asymptotic inference."
            }
        else:
            return {
                "finding": "Normality Assumption Not Rejected (p ≥ α).",
                "econometric_meaning": "Residual distribution does not significantly diverge from Gaussian.",
                "recommended_action": "Finite-sample inference (t and F distributions) is valid."
            }

    # Multicollinearity (VIF)
    if "vif" in method_clean or "multicollinearity" in method_clean:
        stat = statistic_value
        if stat > 10.0:
            return {
                "finding": f"Severe Multicollinearity Detected (Max VIF = {stat:.2f} > 10).",
                "econometric_meaning": "High correlation among predictors inflates variance of coefficient estimates, making individual coefficients sensitive to small data shifts. Note: High VIF does NOT invalidate predictive performance.",
                "recommended_action": "Remove redundant collinear predictors or apply shrinkage (Ridge regression)."
            }
        elif stat > 5.0:
            return {
                "finding": f"Moderate Multicollinearity Detected (Max VIF = {stat:.2f} > 5).",
                "econometric_meaning": "Some coefficient standard errors are inflated.",
                "recommended_action": "Monitor stability of correlated feature coefficients."
            }
        else:
            return {
                "finding": f"Low Multicollinearity (Max VIF = {stat:.2f} ≤ 5).",
                "econometric_meaning": "Predictor orthogonality is satisfactory. Standard errors are not severely inflated by linear dependency.",
                "recommended_action": "Coefficient estimates have stable individual interpretations."
            }

    return {
        "finding": f"Diagnostic executed ({decision.replace('_', ' ').title()}).",
        "econometric_meaning": "Standard assumption evaluation.",
        "recommended_action": "Review in context of overall model diagnostics."
    }


def get_core_methodological_caveats(is_valid_for_inference: bool, post_selection: bool) -> List[str]:
    """Returns mandatory econometric disclaimers tailored to the model's structure."""
    caveats = []
    # 1. Association vs Causation
    caveats.append(
        "**Association vs. Causation**: Regression coefficient estimates represent conditional associative "
        "relationships in observed data, not causal effects. Unobserved confounders, reverse causality, "
        "and measurement error preclude causal claims without experimental design or valid instrumental variables."
    )
    # 2. Exogeneity
    caveats.append(
        "**Exogeneity Assumption**: The core Gauss-Markov assumption E[ε|X] = 0 cannot be empirically proven "
        "from regression residuals alone, because residuals are mathematically constrained to be orthogonal "
        "to design columns by OLS construction (X'e = 0)."
    )
    # 3. Post-Selection Inference
    if post_selection:
        caveats.append(
            "**Post-Selection Inference**: Because predictor variables were selected through data-driven feature selection, "
            "conventional standard errors, t-statistics, and p-values are subject to selection bias (Freedman's paradox) "
            "and are typically overly optimistic."
        )
    # 4. Inference Validity
    if not is_valid_for_inference:
        caveats.append(
            "**Prediction-Only Status**: This model is rank-deficient or has singular variance-covariance estimation. "
            "Standard errors, confidence intervals, and p-values are mathematically invalid and must not be used "
            "for scientific hypothesis testing."
        )
    return caveats
