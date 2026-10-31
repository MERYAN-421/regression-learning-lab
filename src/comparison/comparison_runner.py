"""
Orchestration Service for Experiment & Model Comparison Laboratory.
Coordinates compatibility validation, leaderboard ranking, radar normalization,
Pareto efficiency analysis, diagnostic matrix aggregation, and feature overlap analysis.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from src.models.model_result import OLSModelResult
from src.diagnostics.result_models import DiagnosticSuiteResult
from src.comparison.result_models import (
    ExperimentComparisonResult,
    RankingPolicy,
    generate_comparison_id,
    safe_serialize_numeric
)
from src.comparison.compatibility import validate_model_compatibility
from src.comparison.leaderboard import build_comparison_leaderboard
from src.comparison.normalization import compute_radar_scores, RADAR_DIMENSIONS
from src.comparison.pareto import compute_pareto_frontier
from src.comparison.diagnostics_matrix import build_diagnostic_comparison_matrix, DIAGNOSTIC_COMPARISON_NOTES
from src.comparison.feature_overlap import (
    build_feature_inclusion_dataframe,
    compute_pairwise_jaccard_matrix,
    compute_feature_selection_frequencies
)
from src.comparison.interpretation import (
    RADAR_EDUCATIONAL_DISCLAIMER,
    PARETO_EDUCATIONAL_NOTE,
    GENERALIZATION_GAP_NOTE
)


def run_model_comparison(
    models: List[OLSModelResult],
    active_dataset_id: str,
    active_dataset_fingerprint: str,
    active_split_fingerprint: str,
    active_target_name: str,
    active_target_unit: str = "",
    ranking_policy: RankingPolicy = RankingPolicy.VAL_RMSE_ASC,
    diagnostic_suites: Optional[Dict[str, DiagnosticSuiteResult]] = None,
    selected_candidate_id: Optional[str] = None,
    candidate_selection_rationale: Optional[str] = None
) -> ExperimentComparisonResult:
    """
    Executes a comprehensive, leakage-safe multi-model comparison across candidate OLS models.
    Enforces compatibility, validation-based rankings, Pareto efficiency, and original feature overlap.
    """
    # 1. Compatibility Check
    compat = validate_model_compatibility(
        models=models,
        active_dataset_id=active_dataset_id,
        active_dataset_fingerprint=active_dataset_fingerprint,
        active_split_fingerprint=active_split_fingerprint,
        active_target_name=active_target_name,
        active_target_unit=active_target_unit
    )
    if not compat.is_compatible:
        raise ValueError(f"Model comparison blocked by compatibility validation: {'; '.join(compat.reasons)}")

    sorted_model_ids = sorted([m.model_id for m in models])
    diag_suites = diagnostic_suites or {}
    diag_suite_ids = {m.model_id: diag_suites[m.model_id].suite_id for m in models if m.model_id in diag_suites}
    fs_ids = {m.model_id: m.feature_source_id for m in models}

    # 2. Pareto Efficiency
    pareto_model_ids, points_data = compute_pareto_frontier(models)

    # 3. Normalized Radar Scores
    radar_scores = compute_radar_scores(models)

    # 4. Diagnostic Matrix
    diag_matrix = build_diagnostic_comparison_matrix(models, diagnostic_suites=diag_suites)

    # 5. Feature Overlap
    inclusion_df = build_feature_inclusion_dataframe(models)
    jaccard_df = compute_pairwise_jaccard_matrix(models)
    freq_dict = compute_feature_selection_frequencies(models)

    feature_overlap_matrix = {
        "inclusion_table": inclusion_df.reset_index().to_dict(orient="records"),
        "jaccard_matrix": jaccard_df.to_dict(),
        "selection_frequencies": freq_dict,
        "total_unique_features": len(inclusion_df)
    }

    # 6. Leaderboard
    leaderboard = build_comparison_leaderboard(
        models=models,
        ranking_policy=ranking_policy,
        pareto_model_ids=pareto_model_ids,
        diagnostic_suite_map=diag_suite_ids
    )

    # 7. Model Complexity Summary
    model_complexity = {}
    for m in models:
        model_complexity[m.model_id] = {
            "original_k": len(m.selected_original_features),
            "parameters": m.design_matrix_metadata.num_parameters,
            "df_resid": m.df_resid,
            "condition_number": float(m.design_matrix_metadata.condition_number),
            "is_full_rank": m.design_matrix_metadata.is_full_rank
        }

    # 8. Deterministic Comparison ID
    extra_config = {
        "dimensions": RADAR_DIMENSIONS,
        "selected_candidate": selected_candidate_id,
        "diag_suites_count": len(diag_suite_ids)
    }
    comp_id, repro_hash = generate_comparison_id(
        split_fingerprint=active_split_fingerprint,
        model_ids=sorted_model_ids,
        ranking_policy=ranking_policy.value,
        extra_config=extra_config
    )

    warnings = list(compat.warnings)
    methodological_limitations = [
        "Default model ranking uses out-of-sample Validation RMSE, which avoids in-sample optimism but does not establish causal validity.",
        "Radar Chart dimensions are relative to the currently compared candidate set [0 = worst, 1 = best] and do not reflect absolute model quality.",
        "Pareto efficiency indicates non-dominated trade-offs between complexity and error; it does not automatically designate the optimal model.",
        "Statistical diagnostics assess mathematical regression assumptions individually and are not aggregated into a composite score."
    ]

    return ExperimentComparisonResult(
        comparison_id=comp_id,
        dataset_id=active_dataset_id,
        dataset_fingerprint=active_dataset_fingerprint,
        split_fingerprint=active_split_fingerprint,
        target_name=active_target_name,
        target_unit=active_target_unit,
        compared_model_ids=sorted_model_ids,
        referenced_feature_selection_ids=fs_ids,
        referenced_diagnostic_suite_ids=diag_suite_ids,
        evaluation_scope="validation",
        ranking_policy=ranking_policy.value,
        leaderboard=leaderboard,
        model_complexity=model_complexity,
        normalization_config={
            "method": "min_max",
            "dimensions": RADAR_DIMENSIONS
        },
        radar_scores=radar_scores,
        pareto_model_ids=pareto_model_ids,
        diagnostic_matrix=diag_matrix,
        feature_overlap_matrix=feature_overlap_matrix,
        selected_candidate_model_id=selected_candidate_id,
        candidate_selection_rationale=candidate_selection_rationale,
        warnings=warnings,
        methodological_limitations=methodological_limitations,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        configuration={
            "ranking_policy": ranking_policy.value,
            "compared_models_count": len(models),
            "pareto_models_count": len(pareto_model_ids)
        },
        reproducibility_hash=repro_hash
    )
