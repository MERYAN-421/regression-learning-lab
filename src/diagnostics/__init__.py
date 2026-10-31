"""
Gauss-Markov statistical regression assumptions testing and diagnostic plots.
Milestone 4 Implementation:
- Linearity: Ramsey RESET test, Residuals vs Fitted
- Independence: Durbin-Watson, Ljung-Box, Residual ACF
- Homoscedasticity: Breusch-Pagan, White's test, Scale-Location
- Normality: Jarque-Bera, Shapiro-Wilk, Normal Q-Q, Histogram
- Multicollinearity: VIF, Condition Number, Rank
- Influence: Cook's Distance, Leverage
"""
from src.diagnostics.result_models import (
    DiagnosticType,
    DiagnosticDecision,
    ApplicabilityStatus,
    DiagnosticResult,
    DiagnosticSuiteResult
)
from src.diagnostics.linearity import run_ramsey_reset_test
from src.diagnostics.independence import (
    run_durbin_watson_test,
    run_ljung_box_test,
    compute_residual_acf
)
from src.diagnostics.homoscedasticity import (
    run_breusch_pagan_test,
    run_white_test,
    compute_scale_location_data
)
from src.diagnostics.normality import (
    run_jarque_bera_test,
    run_shapiro_wilk_test,
    compute_qq_plot_data
)
from src.diagnostics.multicollinearity import run_multicollinearity_diagnostics
from src.diagnostics.influence import run_influence_diagnostics
from src.diagnostics.runner import (
    validate_active_model_for_diagnostics,
    run_full_diagnostic_suite
)
from src.diagnostics.interpretation import (
    EXOGENEITY_EDUCATIONAL_NOTE,
    GAUSS_MARKOV_VS_NORMALITY_NOTE
)

__all__ = [
    "DiagnosticType",
    "DiagnosticDecision",
    "ApplicabilityStatus",
    "DiagnosticResult",
    "DiagnosticSuiteResult",
    "run_ramsey_reset_test",
    "run_durbin_watson_test",
    "run_ljung_box_test",
    "compute_residual_acf",
    "run_breusch_pagan_test",
    "run_white_test",
    "compute_scale_location_data",
    "run_jarque_bera_test",
    "run_shapiro_wilk_test",
    "compute_qq_plot_data",
    "run_multicollinearity_diagnostics",
    "run_influence_diagnostics",
    "validate_active_model_for_diagnostics",
    "run_full_diagnostic_suite",
    "EXOGENEITY_EDUCATIONAL_NOTE",
    "GAUSS_MARKOV_VS_NORMALITY_NOTE"
]
