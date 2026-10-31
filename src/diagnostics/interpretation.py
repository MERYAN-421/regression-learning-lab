"""
Statistical interpretation engine for Regression Learning Lab.
Generates rigorous, educational statistical explanations following econometric principles:
1. Strict hypothesis testing semantics: 'fail to reject H0' is never described as proof of H0.
2. Differentiates Gauss-Markov assumptions (BLUE) from Normality (finite-sample t/F inference).
3. Explicitly caveats that Exogeneity E[e|X]=0 cannot be confirmed by residual diagnostics.
4. Describes consequences precisely (e.g. SE distortion vs. coefficient bias).
"""
from typing import Any, Dict, List, Optional, Tuple
from src.diagnostics.result_models import DiagnosticDecision, DiagnosticType


EXOGENEITY_EDUCATIONAL_NOTE = (
    "⚠️ **Fundamental Econometric Caveat: Exogeneity $E[\\epsilon|X] = 0$ Cannot Be Proven by Residuals**\n\n"
    "The strict exogeneity condition requires that error terms have conditional mean zero given all regressors. "
    "By mathematical construction, OLS residuals always sum to zero and are orthogonal to regressors in-sample "
    "($X^T e = 0$). Consequently, residual diagnostics (RESET, Breusch-Pagan, Jarque-Bera, Durbin-Watson) "
    "**cannot detect omitted confounding variables or simultaneous causality**. Favorable residual diagnostics "
    "must NEVER be interpreted as proof that the underlying regression model is exogenous or causal."
)

GAUSS_MARKOV_VS_NORMALITY_NOTE = (
    "💡 **Gauss-Markov Theorem vs. Error Normality**\n\n"
    "The Gauss-Markov theorem establishes that OLS estimators are **BLUE** (Best Linear Unbiased Estimator) "
    "under five structural conditions: Linearity, Strict Exogeneity, No Multicollinearity, Homoscedasticity, "
    "and No Autocorrelation. **Normality is NOT required for BLUE.** Normality of error terms is only needed "
    "to guarantee that conventional $t$ and $F$ test statistics follow exact finite-sample Student-t and Fisher-F "
    "distributions. In moderate-to-large samples ($n > 50$), the Central Limit Theorem ensures asymptotic "
    "validity of standard inference even when residuals deviate from normality."
)


def evaluate_hypothesis_decision(
    p_value: Optional[float],
    alpha: float,
    is_valid_for_inference: bool = True
) -> Tuple[DiagnosticDecision, str]:
    """
    Evaluates formal statistical hypothesis test decisions.
    Documented comparison rule:
    - If not valid for inference (e.g. rank-deficient): DESCRIPTIVE_ONLY or NOT_APPLICABLE.
    - If p_value is None or NaN: DESCRIPTIVE_ONLY or NUMERICAL_FAILURE.
    - If p < alpha: REJECT_NULL.
    - If p >= alpha: FAIL_TO_REJECT_NULL (never 'H0 proven').
    """
    if not is_valid_for_inference:
        return (
            DiagnosticDecision.DESCRIPTIVE_ONLY,
            "Statistical hypothesis testing is disabled because the design matrix is rank-deficient "
            "or parameter identification is compromised."
        )

    if p_value is None:
        return (
            DiagnosticDecision.DESCRIPTIVE_ONLY,
            "This diagnostic provides a descriptive statistic without an exact formal p-value."
        )

    if not isinstance(p_value, (int, float)) or (isinstance(p_value, float) and p_value != p_value):
        return (
            DiagnosticDecision.NUMERICAL_FAILURE,
            "p-value calculation encountered a numerical failure or non-finite result."
        )

    if p_value < alpha:
        return (
            DiagnosticDecision.REJECT_NULL,
            f"Reject H0 at the {alpha:.0%} significance level (p = {p_value:.4f} < {alpha:.2f}). "
            "Sufficient empirical evidence exists against the null hypothesis."
        )
    else:
        return (
            DiagnosticDecision.FAIL_TO_REJECT_NULL,
            f"Fail to reject H0 at the {alpha:.0%} significance level (p = {p_value:.4f} >= {alpha:.2f}). "
            "Insufficient empirical evidence to conclude that the null hypothesis is violated. "
            "(Note: This does NOT prove that the null hypothesis is true.)"
        )


def format_reset_interpretation(
    statistic_value: float,
    p_value: Optional[float],
    alpha: float,
    powers: List[int],
    is_valid_for_inference: bool
) -> Dict[str, Any]:
    """Interpretation for Ramsey RESET linearity/functional-form test."""
    decision, decision_reason = evaluate_hypothesis_decision(p_value, alpha, is_valid_for_inference)
    power_str = ", ".join(f"ŷ^{p}" for p in powers)

    if decision == DiagnosticDecision.REJECT_NULL:
        interp = (
            f"The Ramsey RESET test rejects H0 (F = {statistic_value:.4f}, p = {p_value:.4f} < {alpha:.2f}). "
            f"The auxiliary regression containing powers of fitted values ({power_str}) yields statistically "
            "significant explanatory power. This suggests potential functional-form misspecification, such as "
            "omitted non-linearities, missing interaction effects, or incorrect scale transformations."
        )
        recommendations = [
            "Inspect the Residuals vs. Fitted plot for distinct curvature or non-linear patterns.",
            "Consider adding polynomial terms (e.g. x^2) or interaction terms between key continuous regressors.",
            "Evaluate log or Box-Cox transformations of the target or skewed continuous features.",
            "Compare model specifications via out-of-sample Validation RMSE in the Model Evaluation tab."
        ]
    elif decision == DiagnosticDecision.FAIL_TO_REJECT_NULL:
        interp = (
            f"The Ramsey RESET test fails to reject H0 (F = {statistic_value:.4f}, p = {p_value:.4f} >= {alpha:.2f}). "
            f"Adding higher-order powers of fitted values ({power_str}) does not significantly improve explanatory power. "
            "There is no sufficient evidence of polynomial functional-form misspecification detectable by this test. "
            "Note: This does not prove the model is correctly specified, as other non-linearities or omitted variables may exist."
        )
        recommendations = [
            "Examine bivariate scatter plots in the EDA laboratory to verify domain-specific relationships.",
            "Confirm that expected interaction mechanisms are theoretically grounded."
        ]
    else:
        interp = decision_reason
        recommendations = ["Resolve design matrix rank deficiency or sample size limitations to conduct formal inference."]

    limitations = (
        "The Ramsey RESET test is a general diagnostic for functional form. While it tests if nonlinear combinations "
        "of fitted values explain the response, it cannot identify which specific variable is misspecified, nor can "
        "it distinguish missing non-linearities from omitted exogenous variables."
    )

    return {
        "decision": decision.value,
        "interpretation": interp,
        "limitations": limitations,
        "recommendations": recommendations
    }


def format_durbin_watson_interpretation(dw_stat: float) -> Dict[str, Any]:
    """Interpretation for Durbin-Watson first-order autocorrelation statistic."""
    if dw_stat < 1.4:
        interp = (
            f"Durbin-Watson statistic is {dw_stat:.4f} (< 1.5). This indicates potential positive first-order serial correlation "
            "(successive residuals tend to share the same sign). In time-series data, positive autocorrelation leads to "
            "artificially deflated standard errors and inflated t-statistics, yielding overly optimistic inferences."
        )
        recommendations = [
            "Examine the Residual ACF plot to observe the decay rate of serial correlation across lags.",
            "Consider incorporating lagged dependent variables or autoregressive terms in a dynamic model.",
            "Use Newey-West Heteroscedasticity and Autocorrelation Consistent (HAC) standard errors for robust inference."
        ]
    elif dw_stat > 2.6:
        interp = (
            f"Durbin-Watson statistic is {dw_stat:.4f} (> 2.5). This indicates potential negative first-order serial correlation "
            "(successive residuals tend to alternate signs). This causes conventional standard errors to be overstated."
        )
        recommendations = [
            "Check for overdifferencing or seasonal alternating oscillations.",
            "Inspect the residual time-series plot to identify alternating pattern mechanisms."
        ]
    else:
        interp = (
            f"Durbin-Watson statistic is {dw_stat:.4f} (centered around 2.0). Under the classical heuristic (1.5 - 2.5), "
            "there is limited evidence of first-order serial correlation. Note: DW only tests lag-1 autocorrelation "
            "and does not evaluate higher-order dependencies."
        )
        recommendations = [
            "Check the Ljung-Box test and Residual ACF plot to verify absence of higher-order seasonal autocorrelation."
        ]

    limitations = (
        "Durbin-Watson only assesses first-order (lag 1) autocorrelation. It assumes an intercept is included, "
        "regressors are strictly exogenous, and error terms are approximately normally distributed. It is invalid "
        "when lagged endogenous variables are included as regressors."
    )

    return {
        "decision": DiagnosticDecision.DESCRIPTIVE_ONLY.value,
        "interpretation": interp,
        "limitations": limitations,
        "recommendations": recommendations
    }


def format_ljung_box_interpretation(
    lb_stat: float,
    p_value: Optional[float],
    alpha: float,
    max_lag: int,
    is_valid_for_inference: bool
) -> Dict[str, Any]:
    """Interpretation for Ljung-Box portmanteau autocorrelation test."""
    decision, decision_reason = evaluate_hypothesis_decision(p_value, alpha, is_valid_for_inference)

    if decision == DiagnosticDecision.REJECT_NULL:
        interp = (
            f"The Ljung-Box test rejects H0 across {max_lag} lags (Q = {lb_stat:.4f}, p = {p_value:.4f} < {alpha:.2f}). "
            f"There is statistically significant cumulative autocorrelation up to lag {max_lag}. Residual errors are "
            "temporally correlated, violating the Gauss-Markov independence assumption."
        )
        recommendations = [
            "Inspect the Residual ACF plot to determine which specific lag orders exhibit significant spikes.",
            "Consider time-series autoregressive features, lag features, or seasonal differencing.",
            "Apply Newey-West HAC covariance estimators to compute valid standard errors."
        ]
    elif decision == DiagnosticDecision.FAIL_TO_REJECT_NULL:
        interp = (
            f"The Ljung-Box test fails to reject H0 across {max_lag} lags (Q = {lb_stat:.4f}, p = {p_value:.4f} >= {alpha:.2f}). "
            f"Residual autocorrelations up to lag {max_lag} are not jointly distinguishable from white noise. "
            "There is no sufficient evidence of sequential dependency across the tested lags."
        )
        recommendations = [
            "Verify that observation ordering accurately reflects the true temporal or data-generating sequence."
        ]
    else:
        interp = decision_reason
        recommendations = ["Ensure sufficient time-series observations are available for valid portmanteau testing."]

    limitations = (
        "Ljung-Box assumes observations are strictly sequential and ordered chronologically. "
        "Applying Ljung-Box to cross-sectional data or arbitrarily concatenated panel clusters (e.g. multi-country data) "
        "is statistically meaningless."
    )

    return {
        "decision": decision.value,
        "interpretation": interp,
        "limitations": limitations,
        "recommendations": recommendations
    }


def format_breusch_pagan_interpretation(
    lm_stat: float,
    lm_pval: Optional[float],
    f_stat: float,
    f_pval: Optional[float],
    alpha: float,
    is_valid_for_inference: bool
) -> Dict[str, Any]:
    """Interpretation for Breusch-Pagan homoscedasticity test."""
    decision, decision_reason = evaluate_hypothesis_decision(lm_pval, alpha, is_valid_for_inference)

    if decision == DiagnosticDecision.REJECT_NULL:
        interp = (
            f"The Breusch-Pagan test rejects H0 (Lagrange Multiplier LM = {lm_stat:.4f}, p = {lm_pval:.4f} < {alpha:.2f}; "
            f"F = {f_stat:.4f}, p = {f_pval:.4f}). Error variance is not constant across regressor levels. "
            "Under heteroscedasticity, OLS coefficient estimates remain unbiased and consistent (if exogeneity holds), "
            "but OLS is NO LONGER BLUE, and conventional standard errors, t-tests, and F-tests are mathematically invalid."
        )
        recommendations = [
            "Examine the Scale-Location plot to identify which regressors correlate with widening residual spread.",
            "Employ Huber-White heteroscedasticity-consistent robust standard errors (e.g. HC3) for hypothesis tests.",
            "Consider variance-stabilizing transformations (such as log or square root) on the response variable.",
            "Explore Weighted Least Squares (WLS) if the error variance structure is known or predictable."
        ]
    elif decision == DiagnosticDecision.FAIL_TO_REJECT_NULL:
        interp = (
            f"The Breusch-Pagan test fails to reject H0 (LM = {lm_stat:.4f}, p = {lm_pval:.4f} >= {alpha:.2f}; "
            f"F = {f_stat:.4f}, p = {f_pval:.4f}). There is no statistically significant evidence of linear heteroscedasticity "
            "associated with the model's regressors. Standard OLS variance estimates are empirically supported."
        )
        recommendations = [
            "Check the Scale-Location plot and Residuals vs. Fitted plot to verify no non-linear fan shapes exist."
        ]
    else:
        interp = decision_reason
        recommendations = ["Resolve rank deficiency or degree-of-freedom limitations before testing homoscedasticity."]

    limitations = (
        "Breusch-Pagan is sensitive to non-normality of error terms. If errors are heavy-tailed, Breusch-Pagan can "
        "produce inflated rejection rates. In addition, it primarily detects linear relationships between regressors "
        "and error variances; non-linear variance patterns may require White's test."
    )

    return {
        "decision": decision.value,
        "interpretation": interp,
        "limitations": limitations,
        "recommendations": recommendations
    }


def format_jarque_bera_interpretation(
    jb_stat: float,
    p_value: Optional[float],
    skew: float,
    kurtosis: float,
    alpha: float,
    sample_size: int,
    is_valid_for_inference: bool
) -> Dict[str, Any]:
    """Interpretation for Jarque-Bera residual normality test."""
    decision, decision_reason = evaluate_hypothesis_decision(p_value, alpha, is_valid_for_inference)
    excess_kurt = kurtosis - 3.0

    if decision == DiagnosticDecision.REJECT_NULL:
        skew_desc = "positively skewed (right tail)" if skew > 0.5 else ("negatively skewed (left tail)" if skew < -0.5 else "moderately symmetric")
        kurt_desc = "leptokurtic (heavy-tailed / outlier-prone)" if excess_kurt > 1.0 else ("platykurtic (light-tailed)" if excess_kurt < -1.0 else "mesokurtic")

        interp = (
            f"The Jarque-Bera test rejects H0 (JB = {jb_stat:.4f}, p = {p_value:.4f} < {alpha:.2f}). "
            f"Sample skewness is {skew:.3f} ({skew_desc}) and sample kurtosis is {kurtosis:.3f} ({kurt_desc}, excess = {excess_kurt:+.3f}). "
            "Residuals deviate significantly from Gaussian normality. "
            "IMPORTANT: Rejecting normality does NOT invalidate the Gauss-Markov theorem; OLS remains the Best Linear Unbiased Estimator (BLUE). "
            f"With n = {sample_size:,}, the Central Limit Theorem ensures that large-sample inferences remain asymptotically valid."
        )
        recommendations = [
            "Inspect the Normal Q-Q plot and residual histogram to locate whether skewness or extreme outliers cause the rejection.",
            "Consider log, Box-Cox, or quantile transformations if extreme positive skewness is present.",
            "Verify that outliers are genuine data observations rather than recording errors."
        ]
    elif decision == DiagnosticDecision.FAIL_TO_REJECT_NULL:
        interp = (
            f"The Jarque-Bera test fails to reject H0 (JB = {jb_stat:.4f}, p = {p_value:.4f} >= {alpha:.2f}). "
            f"Sample skewness ({skew:.3f}) and kurtosis ({kurtosis:.3f}) are statistically consistent with a Gaussian distribution. "
            "Exact finite-sample t and F distributions are well supported for inferential tests."
        )
        recommendations = [
            "Confirm alignment along the diagonal on the Normal Q-Q plot."
        ]
    else:
        interp = decision_reason
        recommendations = ["Ensure sufficient residual sample size for normality evaluation."]

    limitations = (
        "Jarque-Bera is an asymptotic test with power that grows rapidly with sample size. In very large samples "
        f"(n = {sample_size:,}), even trivial departures from Gaussianity will produce small p-values. "
        "Therefore, graphical inspection (Q-Q plot) is essential alongside formal tests."
    )

    return {
        "decision": decision.value,
        "interpretation": interp,
        "limitations": limitations,
        "recommendations": recommendations
    }


def format_vif_interpretation(
    vif_records: List[Dict[str, Any]],
    condition_number: float,
    matrix_rank: int,
    num_parameters: int
) -> Dict[str, Any]:
    """Interpretation for Variance Inflation Factor and matrix multicollinearity diagnostics."""
    max_vif = max([r["vif"] for r in vif_records], default=1.0)
    high_vif_cols = [r["feature"] for r in vif_records if r["vif"] > 10.0]
    moderate_vif_cols = [r["feature"] for r in vif_records if 5.0 <= r["vif"] <= 10.0]

    is_rank_deficient = matrix_rank < num_parameters

    if is_rank_deficient:
        interp = (
            f"CRITICAL: Exact multicollinearity (rank deficiency) detected! Matrix rank is {matrix_rank} for {num_parameters} columns. "
            "At least one column is an exact linear combination of others. OLS cannot uniquely invert (X^T X), "
            "rendering regression coefficients non-identifiable without regularization or pseudoinverse."
        )
        recommendations = [
            "Identify and remove redundant linear combinations (e.g. sqft_living = sqft_above + sqft_basement).",
            "Verify that categorical features drop a reference category to avoid the dummy variable trap."
        ]
    elif high_vif_cols:
        interp = (
            f"Severe multicollinearity detected. Regressors with VIF > 10: {', '.join(high_vif_cols)}. "
            f"Maximum VIF is {max_vif:.2f}, and design matrix condition number is {condition_number:.1f}. "
            "While multicollinearity does NOT bias OLS coefficient estimates, it dramatically inflates coefficient variances "
            "and standard errors, making individual t-tests insignificant despite high overall model R²."
        )
        recommendations = [
            "Consider removing one of the highly collinear regressors after consulting domain significance.",
            "Apply L2 Ridge regularization or Principal Component Analysis (PCA) to stabilize variance.",
            "Assess whether the collinear features represent sub-components of an aggregate concept."
        ]
    elif moderate_vif_cols:
        interp = (
            f"Moderate collinearity observed. Regressors with VIF between 5 and 10: {', '.join(moderate_vif_cols)}. "
            f"Maximum VIF is {max_vif:.2f}. Regressors share substantial mutual information, causing moderately wider confidence intervals."
        )
        recommendations = [
            "Monitor coefficient stability across validation folds.",
            "Evaluate whether feature grouping or dimensionality reduction improves test accuracy."
        ]
    else:
        interp = (
            f"Low multicollinearity. All non-intercept regressors have VIF < 5.0 (Maximum VIF = {max_vif:.2f}, "
            f"Condition Number = {condition_number:.1f}). Regressors provide largely independent linear information."
        )
        recommendations = [
            "Standard errors and coefficient t-tests are stable with respect to regressor collinearity."
        ]

    limitations = (
        "VIF is a diagnostic metric without a formal p-value. The cutoffs of 5.0 and 10.0 are widely used heuristics, "
        "not immutable statistical laws. For one-hot encoded categorical variables, per-dummy VIF values naturally "
        "correlate within the same category group and should be evaluated in context."
    )

    return {
        "decision": DiagnosticDecision.DESCRIPTIVE_ONLY.value,
        "interpretation": interp,
        "limitations": limitations,
        "recommendations": recommendations,
        "max_vif": max_vif,
        "high_vif_columns": high_vif_cols,
        "moderate_vif_columns": moderate_vif_cols
    }
