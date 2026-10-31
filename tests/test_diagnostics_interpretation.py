"""
Group F: Interpretation Engine and Econometric Semantics Tests.
Verifies:
- Strict p-value decision rules:
  - p < alpha -> REJECT_NULL
  - p == alpha -> FAIL_TO_REJECT_NULL
  - p > alpha -> FAIL_TO_REJECT_NULL
  - None / NaN -> DESCRIPTIVE_ONLY / NUMERICAL_FAILURE
- Prohibition against 'H0 proven' statements
- Exogeneity educational text guarantees
- Gauss-Markov vs. Normality distinction text guarantees
"""
import pytest
from src.diagnostics.result_models import DiagnosticDecision
from src.diagnostics.interpretation import (
    evaluate_hypothesis_decision,
    EXOGENEITY_EDUCATIONAL_NOTE,
    GAUSS_MARKOV_VS_NORMALITY_NOTE
)


def test_decision_boundary_p_less_than_alpha():
    """Verify p < alpha produces REJECT_NULL decision."""
    decision, text = evaluate_hypothesis_decision(p_value=0.049, alpha=0.05)
    assert decision == DiagnosticDecision.REJECT_NULL
    assert "Reject H0" in text


def test_decision_boundary_p_equal_to_alpha():
    """Verify p == alpha produces FAIL_TO_REJECT_NULL decision under strict inequality rule."""
    decision, text = evaluate_hypothesis_decision(p_value=0.050, alpha=0.05)
    assert decision == DiagnosticDecision.FAIL_TO_REJECT_NULL
    assert "Fail to reject H0" in text
    assert "NOT prove" in text or "not prove" in text


def test_decision_boundary_p_greater_than_alpha():
    """Verify p > alpha produces FAIL_TO_REJECT_NULL decision."""
    decision, text = evaluate_hypothesis_decision(p_value=0.15, alpha=0.05)
    assert decision == DiagnosticDecision.FAIL_TO_REJECT_NULL
    assert "Fail to reject H0" in text
    # Strict semantic prohibition: Never claim H0 is proven true!
    assert "proven" not in text.lower() or "not prove" in text.lower()


def test_decision_non_finite_and_none_pvalue():
    """Verify None and NaN p-values produce DESCRIPTIVE_ONLY or NUMERICAL_FAILURE."""
    decision_none, _ = evaluate_hypothesis_decision(p_value=None, alpha=0.05)
    assert decision_none == DiagnosticDecision.DESCRIPTIVE_ONLY

    decision_nan, _ = evaluate_hypothesis_decision(p_value=float("nan"), alpha=0.05)
    assert decision_nan == DiagnosticDecision.NUMERICAL_FAILURE


def test_decision_prediction_only_guard():
    """Verify that is_valid_for_inference=False disables hypothesis testing."""
    decision, text = evaluate_hypothesis_decision(p_value=0.001, alpha=0.05, is_valid_for_inference=False)
    assert decision == DiagnosticDecision.DESCRIPTIVE_ONLY
    assert "disabled" in text.lower() or "rank-deficient" in text.lower()


def test_exogeneity_and_gauss_markov_notes_integrity():
    """Verify that mandatory educational disclaimers are present and substantively complete."""
    # Exogeneity must explain E[e|X] = 0 cannot be proven by residual diagnostics
    assert "E[\\epsilon|X] = 0" in EXOGENEITY_EDUCATIONAL_NOTE
    assert "Cannot Be Proven" in EXOGENEITY_EDUCATIONAL_NOTE
    assert "NEVER be interpreted as proof" in EXOGENEITY_EDUCATIONAL_NOTE

    # Gauss-Markov note must explain normality is not required for BLUE
    assert "BLUE" in GAUSS_MARKOV_VS_NORMALITY_NOTE
    assert "NOT required for BLUE" in GAUSS_MARKOV_VS_NORMALITY_NOTE
    assert "Central Limit Theorem" in GAUSS_MARKOV_VS_NORMALITY_NOTE
