"""
Milestone 5: Experiment & Model Comparison Laboratory.
Exposes compatibility validation, leaderboard ranking, radar normalization,
Pareto efficiency analysis, assumption diagnostic matrix, and feature overlap analysis.
"""
from src.comparison.result_models import (
    ExperimentComparisonResult,
    ModelComparisonRecord,
    CompatibilityResult,
    RankingPolicy,
    generate_comparison_id,
    safe_serialize_numeric,
)
from src.comparison.compatibility import validate_model_compatibility
from src.comparison.leaderboard import (
    build_comparison_leaderboard,
    create_model_comparison_record,
)
from src.comparison.normalization import (
    compute_radar_scores,
    normalize_lower_is_better,
    normalize_higher_is_better,
    RADAR_DIMENSIONS,
)
from src.comparison.radar import create_radar_chart
from src.comparison.pareto import (
    compute_pareto_frontier,
    create_pareto_scatter_chart,
)
from src.comparison.diagnostics_matrix import (
    build_diagnostic_comparison_matrix,
    DIAGNOSTIC_COMPARISON_NOTES,
)
from src.comparison.feature_overlap import (
    build_feature_inclusion_dataframe,
    compute_jaccard_similarity,
    compute_pairwise_jaccard_matrix,
    compute_feature_selection_frequencies,
    create_feature_frequency_chart,
    create_jaccard_heatmap,
)
from src.comparison.comparison_runner import run_model_comparison
from src.comparison.interpretation import (
    RADAR_EDUCATIONAL_DISCLAIMER,
    PARETO_EDUCATIONAL_NOTE,
    GENERALIZATION_GAP_NOTE,
    DIAGNOSTIC_SCORING_WARNING,
    render_candidate_selection_summary,
)

__all__ = [
    "ExperimentComparisonResult",
    "ModelComparisonRecord",
    "CompatibilityResult",
    "RankingPolicy",
    "generate_comparison_id",
    "safe_serialize_numeric",
    "validate_model_compatibility",
    "build_comparison_leaderboard",
    "create_model_comparison_record",
    "compute_radar_scores",
    "normalize_lower_is_better",
    "normalize_higher_is_better",
    "RADAR_DIMENSIONS",
    "create_radar_chart",
    "compute_pareto_frontier",
    "create_pareto_scatter_chart",
    "build_diagnostic_comparison_matrix",
    "DIAGNOSTIC_COMPARISON_NOTES",
    "build_feature_inclusion_dataframe",
    "compute_jaccard_similarity",
    "compute_pairwise_jaccard_matrix",
    "compute_feature_selection_frequencies",
    "create_feature_frequency_chart",
    "create_jaccard_heatmap",
    "run_model_comparison",
    "RADAR_EDUCATIONAL_DISCLAIMER",
    "PARETO_EDUCATIONAL_NOTE",
    "GENERALIZATION_GAP_NOTE",
    "DIAGNOSTIC_SCORING_WARNING",
    "render_candidate_selection_summary",
]
