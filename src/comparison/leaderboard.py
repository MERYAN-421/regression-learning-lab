"""
Model Leaderboard Engine for Experiment & Model Comparison Laboratory.
Generates development-stage comparative leaderboards strictly ordered by Validation metrics.
Enforces Rule 2: Test Set performance is never used for candidate ranking or selection.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.model_result import OLSModelResult
from src.comparison.result_models import RankingPolicy, ModelComparisonRecord


def create_model_comparison_record(
    model: OLSModelResult,
    is_pareto: bool = False,
    diagnostic_suite_id: Optional[str] = None
) -> ModelComparisonRecord:
    """Extracts a structured development-stage comparison record from an OLSModelResult."""
    train_n = len(model.train_predictions) if model.train_predictions is not None else model.train_metrics.sample_count
    val_n = len(model.val_predictions) if model.val_predictions is not None else model.val_metrics.sample_count

    t_rmse = float(model.train_metrics.rmse)
    v_rmse = float(model.val_metrics.rmse)
    t_mae = float(model.train_metrics.mae)
    v_mae = float(model.val_metrics.mae)
    t_r2 = float(model.train_metrics.r2)
    v_r2 = float(model.val_metrics.r2)
    t_adj_r2 = float(model.train_metrics.adj_r2) if model.train_metrics.adj_r2 is not None else None
    gen_gap = v_rmse - t_rmse

    aic = float(model.aic) if np.isfinite(model.aic) else None
    bic = float(model.bic) if np.isfinite(model.bic) else None

    return ModelComparisonRecord(
        model_id=model.model_id,
        model_name=model.model_name,
        feature_source_id=model.feature_source_id,
        feature_source_label=model.feature_source_label,
        original_feature_count=len(model.selected_original_features),
        parameter_count=model.design_matrix_metadata.num_parameters,
        train_sample_count=train_n,
        val_sample_count=val_n,
        train_rmse=t_rmse,
        val_rmse=v_rmse,
        train_mae=t_mae,
        val_mae=v_mae,
        train_r2=t_r2,
        val_r2=v_r2,
        train_adj_r2=t_adj_r2,
        generalization_gap=gen_gap,
        aic=aic,
        bic=bic,
        is_valid_for_inference=model.is_valid_for_inference,
        is_pareto_efficient=is_pareto,
        diagnostic_suite_id=diagnostic_suite_id,
        test_evaluation_status=model.test_evaluation_status
    )


def build_comparison_leaderboard(
    models: List[OLSModelResult],
    ranking_policy: RankingPolicy = RankingPolicy.VAL_RMSE_ASC,
    pareto_model_ids: Optional[List[str]] = None,
    diagnostic_suite_map: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Constructs an interactive comparison leaderboard sorted by the chosen development-stage policy.
    
    Default ranking: Validation RMSE ascending (lower is better).
    Strictly excludes Test Set metrics from ranking, sorting, or recommendation (Rule 2).
    """
    if not models:
        return []

    pareto_set = set(pareto_model_ids or [])
    diag_map = diagnostic_suite_map or {}

    records: List[ModelComparisonRecord] = []
    for m in models:
        rec = create_model_comparison_record(
            model=m,
            is_pareto=(m.model_id in pareto_set),
            diagnostic_suite_id=diag_map.get(m.model_id)
        )
        records.append(rec)

    # Sort records by requested ranking policy
    if ranking_policy == RankingPolicy.VAL_RMSE_ASC:
        records.sort(key=lambda r: (r.val_rmse, r.parameter_count))
    elif ranking_policy == RankingPolicy.VAL_MAE_ASC:
        records.sort(key=lambda r: (r.val_mae, r.parameter_count))
    elif ranking_policy == RankingPolicy.VAL_R2_DESC:
        records.sort(key=lambda r: (-r.val_r2, r.parameter_count))
    elif ranking_policy == RankingPolicy.PARSIMONY_ASC:
        records.sort(key=lambda r: (r.parameter_count, r.val_rmse))
    elif ranking_policy == RankingPolicy.GEN_GAP_ASC:
        records.sort(key=lambda r: (abs(r.generalization_gap), r.val_rmse))
    else:
        records.sort(key=lambda r: r.val_rmse)

    # Build leaderboard display rows
    leaderboard_rows: List[Dict[str, Any]] = []
    for rank, r in enumerate(records, start=1):
        leaderboard_rows.append({
            "Rank": rank,
            "Model ID": r.model_id,
            "Feature Source": r.feature_source_label,
            "Original Features (K)": r.original_feature_count,
            "Parameters (p)": r.parameter_count,
            "Val RMSE": round(r.val_rmse, 4),
            "Train RMSE": round(r.train_rmse, 4),
            "Gen. Gap": round(r.generalization_gap, 4),
            "Val R²": round(r.val_r2, 4),
            "Train R²": round(r.train_r2, 4),
            "Train Adj R²": round(r.train_adj_r2, 4) if r.train_adj_r2 is not None else "N/A",
            "Val MAE": round(r.val_mae, 4),
            "Train MAE": round(r.train_mae, 4),
            "AIC": round(r.aic, 1) if r.aic is not None else "N/A",
            "BIC": round(r.bic, 1) if r.bic is not None else "N/A",
            "Inference Status": "✅ Full Rank" if r.is_valid_for_inference else "⚠️ Prediction Only",
            "Pareto Status": "⭐ Pareto Efficient" if r.is_pareto_efficient else "Dominated",
            "Diagnostic Suite": r.diagnostic_suite_id or "Not Run",
            "Test Status": r.test_evaluation_status
        })

    return leaderboard_rows
