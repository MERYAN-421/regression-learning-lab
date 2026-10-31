"""
Educational Interpretation Engine for Experiment & Model Comparison Laboratory.
Generates pedagogical text explaining model comparisons, radar chart relativity,
Pareto efficiency limitations, and diagnostic non-additivity.
"""
from typing import Dict, List, Optional


RADAR_EDUCATIONAL_DISCLAIMER = (
    "💡 **Pedagogical Note on Radar Chart Normalization**:\n"
    "- **Relative Scoring**: All dimensions are min-max mapped to [0, 1] relative to the *currently compared candidate models*. "
    "A score of 1.0 means 'best among these candidates', NOT an absolute scientific optimum.\n"
    "- **Set Sensitivity**: Adding or removing a candidate model re-anchors the min and max, altering the normalized shapes.\n"
    "- **No Diagnostic p-Values**: Assumption test p-values are deliberately excluded from radar polygon areas to prevent "
    "statistically invalid composite scoring (Rule 3)."
)

PARETO_EDUCATIONAL_NOTE = (
    "💡 **Understanding Pareto Efficiency (Occam's Razor Trade-Off)**:\n"
    "- **Non-Dominated Candidates**: A model is on the Pareto Frontier (⭐) if no other model simultaneously achieves fewer "
    "parameters and lower Validation RMSE.\n"
    "- **No Automatic Best**: Pareto membership identifies the efficient boundary of the trade-off. Choosing among frontier models "
    "(e.g., maximum parsimony vs. maximum accuracy) is a modeling decision based on domain context, interpretability, and diagnostic health.\n"
    "- **Candidate-Set Bound**: A model deemed Pareto-efficient here might be dominated if additional specifications are tested."
)

GENERALIZATION_GAP_NOTE = (
    "💡 **Interpreting the Generalization Gap (Val RMSE - Train RMSE)**:\n"
    "- **Positive Gap**: Normal behavior in out-of-sample prediction. A large positive gap suggests potential variance/overfitting.\n"
    "- **Small Gap Warning**: A near-zero or small gap does *not* prove a good model. A model with 2 parameters that severely underfits "
    "will have high Train RMSE and high Val RMSE with a tiny gap.\n"
    "- **Negative Gap**: Occasionally, random validation partitions or smaller validation variance may produce Val RMSE < Train RMSE. "
    "This does not guarantee superior model quality."
)


DIAGNOSTIC_SCORING_WARNING = (
    "🚨 **Diagnostic Additivity & Scoring Warning (Rule 3)**:\n"
    "- Statistical assumption tests (Ramsey RESET, Breusch-Pagan, Jarque-Bera, Ljung-Box, etc.) evaluate distinct mathematical properties "
    "under specific null hypotheses.\n"
    "- They CANNOT be summed into an artificial 'Pass Count' or weighted 'Total Quality Score'.\n"
    "- High p-values indicate failure to reject $H_0$, not confirmation of model correctness or causality."
)


def format_candidate_selection_summary(
    selected_model_id: str,
    feature_source_label: str,
    parameters: int,
    val_rmse: float,
    val_r2: float,
    is_pareto: bool,
    rationale: str
) -> str:
    """Formats an educational audit trail entry for a chosen candidate model."""
    pareto_text = "Pareto-efficient (non-dominated trade-off)" if is_pareto else "dominated on the complexity-error frontier"
    return (
        f"**Selected Candidate Model**: `{selected_model_id}` ({feature_source_label})\n\n"
        f"- **Complexity**: {parameters} OLS parameters\n"
        f"- **Validation Performance**: RMSE = {val_rmse:.4f}, R² = {val_r2:.4f}\n"
        f"- **Frontier Status**: {pareto_text}\n"
        f"- **Modeler Rationale**: {rationale}\n\n"
        "> *Controlled Evaluation Notice*: Candidate selection records the development-stage preferred model. "
        "Test Set evaluation remains locked under controlled authorization."
    )


render_candidate_selection_summary = format_candidate_selection_summary
