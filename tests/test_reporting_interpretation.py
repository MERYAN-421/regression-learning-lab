"""
Automated Test Suite — Milestone 6: Group D
Tests:
- Statistical interpretation rules for hypothesis tests (p < alpha, p = alpha, p > alpha)
- Never asserting H0 is proven true
- Handling of descriptive statistics without p-values (DW, VIF)
- Not Applicable and Numerical Failure diagnostic states
- Econometric caveats: Gauss-Markov BLUE vs Normality, exogeneity non-verifiability, post-selection bias
- Absence of misleading composite diagnostic scores
"""
import pytest
import numpy as np

from src.reporting.interpretation import (
    interpret_hypothesis_test,
    get_diagnostic_interpretation_guidance,
    get_core_methodological_caveats
)
from src.diagnostics.result_models import DiagnosticDecision


class TestStatisticalInterpretationEngine:
    """Group D: Interpretation rules and econometric caveats."""

    def test_p_less_than_alpha(self):
        res = interpret_hypothesis_test(p_value=0.012, alpha=0.05, null_hypothesis="Residuals are homoscedastic")
        assert res["decision"] == "Reject H0"
        assert res["reject_null"] is True
        assert "Reject the null hypothesis" in res["statement"]
        assert "p = 0.0120 < 0.05" in res["statement"]

    def test_p_equal_to_alpha(self):
        # At boundary p == alpha, standard convention fails to reject
        res = interpret_hypothesis_test(p_value=0.05, alpha=0.05, null_hypothesis="Residuals are normal")
        assert res["decision"] == "Fail to Reject H0"
        assert res["reject_null"] is False
        assert "Fail to reject the null hypothesis" in res["statement"]
        assert "NOT prove H0 is true" in res["statement"]

    def test_p_greater_than_alpha_never_proves_null(self):
        res = interpret_hypothesis_test(p_value=0.42, alpha=0.05, null_hypothesis="Linearity holds")
        assert res["decision"] == "Fail to Reject H0"
        assert res["reject_null"] is False
        assert "proven true" not in res["statement"].lower()
        assert "NOT prove H0 is true" in res["statement"]

    def test_no_pvalue_descriptive_only(self):
        res = interpret_hypothesis_test(p_value=None, test_name="Durbin-Watson")
        assert res["decision"] == "Not Run / Descriptive Only"
        assert res["p_value_display"] == "N/A"
        assert "does not provide a formal sampling distribution p-value" in res["statement"]

    def test_very_small_pvalue_rendering_convention(self):
        res = interpret_hypothesis_test(p_value=1e-8, alpha=0.05)
        assert res["p_value_display"] == "< 0.0001"
        assert "0.0000" not in res["p_value_display"]

    def test_not_applicable_diagnostic_guidance(self):
        guidance = get_diagnostic_interpretation_guidance(
            diag_type="independence",
            method_id="durbin_watson",
            decision=DiagnosticDecision.NOT_APPLICABLE.value,
            p_value=None,
            statistic_value=float("nan")
        )
        assert "Not Applicable" in guidance["finding"]
        assert "violated by the dataset structure" in guidance["econometric_meaning"]

    def test_numerical_failure_guidance(self):
        guidance = get_diagnostic_interpretation_guidance(
            diag_type="linearity",
            method_id="ramsey_reset",
            decision=DiagnosticDecision.NUMERICAL_FAILURE.value,
            p_value=None,
            statistic_value=float("nan")
        )
        assert "Numerical Estimation Failure" in guidance["finding"]

    def test_normality_vs_blue_caveat(self):
        guidance = get_diagnostic_interpretation_guidance(
            diag_type="normality",
            method_id="jarque_bera",
            decision=DiagnosticDecision.REJECT_NULL.value,
            p_value=0.001,
            statistic_value=18.4
        )
        meaning = guidance["econometric_meaning"]
        assert "Best Linear Unbiased Estimator (BLUE)" in meaning
        assert "exact finite-sample t and F tests rely on normality" in meaning

    def test_exogeneity_and_post_selection_caveats(self):
        caveats = get_core_methodological_caveats(is_valid_for_inference=True, post_selection=True)
        all_text = " ".join(caveats)
        assert "Exogeneity Assumption" in all_text
        assert "residuals are mathematically constrained to be orthogonal" in all_text
        assert "Post-Selection Inference" in all_text
        assert "Freedman's paradox" in all_text
