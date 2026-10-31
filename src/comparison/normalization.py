"""
Multi-Metric Normalization Engine for Radar Charts and Multi-Objective Comparison.
Maps heterogeneous regression metrics onto a standardized [0, 1] scale where 1 is best.
Handles edge cases: single model, zero range, negative R², and constant dimensions.
"""
from typing import Any, Dict, List, Optional
import numpy as np
from src.models.model_result import OLSModelResult


# Standard 5 radar dimensions
RADAR_DIMENSIONS = [
    "Validation RMSE Score",
    "Validation MAE Score",
    "Validation R² Score",
    "Parsimony Score",
    "Generalization Gap Score"
]


def normalize_lower_is_better(values: List[float]) -> List[float]:
    """
    Min-Max transformation for lower-is-better metrics (RMSE, MAE, Parsimony, Abs Gen Gap):
    S_i = (max(x) - x_i) / (max(x) - min(x))
    Maps lowest value to 1.0 (best) and highest value to 0.0 (worst).
    If all values are identical or only 1 item: returns 0.5 (or 1.0 if single model).
    """
    if not values:
        return []
    if len(values) == 1:
        return [1.0]

    arr = np.array(values, dtype=float)
    # Replace non-finite with finite max
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return [0.5] * len(values)

    val_max = np.max(arr[finite_mask])
    val_min = np.min(arr[finite_mask])
    val_range = val_max - val_min

    if val_range <= 1e-12:
        return [0.5 if f else 0.0 for f in finite_mask]

    scores = np.where(finite_mask, (val_max - arr) / val_range, 0.0)
    scores = np.clip(scores, 0.0, 1.0)
    return [float(s) for s in scores]


def normalize_higher_is_better(values: List[float]) -> List[float]:
    """
    Min-Max transformation for higher-is-better metrics (Validation R²):
    S_i = (x_i - min(x)) / (max(x) - min(x))
    Maps highest value to 1.0 (best) and lowest value to 0.0 (worst).
    If all values are identical or only 1 item: returns 0.5 (or 1.0 if single model).
    """
    if not values:
        return []
    if len(values) == 1:
        return [1.0]

    arr = np.array(values, dtype=float)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return [0.5] * len(values)

    val_max = np.max(arr[finite_mask])
    val_min = np.min(arr[finite_mask])
    val_range = val_max - val_min

    if val_range <= 1e-12:
        return [0.5 if f else 0.0 for f in finite_mask]

    scores = np.where(finite_mask, (arr - val_min) / val_range, 0.0)
    scores = np.clip(scores, 0.0, 1.0)
    return [float(s) for s in scores]


def compute_radar_scores(
    models: List[OLSModelResult]
) -> Dict[str, Dict[str, float]]:
    """
    Computes normalized radar scores across all 5 standard dimensions for candidate models:
    1. Validation RMSE Score (lower is better)
    2. Validation MAE Score (lower is better)
    3. Validation R² Score (higher is better)
    4. Parsimony Score (fewer parameters is better)
    5. Generalization Gap Score (smaller |Val RMSE - Train RMSE| is better)
    
    Returns:
        Dict[model_id, Dict[dimension_name, normalized_score]]
    """
    if not models:
        return {}

    model_ids = [m.model_id for m in models]
    val_rmses = [float(m.val_metrics.rmse) for m in models]
    val_maes = [float(m.val_metrics.mae) for m in models]
    val_r2s = [float(m.val_metrics.r2) for m in models]
    parsimonies = [float(m.design_matrix_metadata.num_parameters) for m in models]
    gen_gaps = [abs(float(m.val_metrics.rmse) - float(m.train_metrics.rmse)) for m in models]

    rmse_scores = normalize_lower_is_better(val_rmses)
    mae_scores = normalize_lower_is_better(val_maes)
    r2_scores = normalize_higher_is_better(val_r2s)
    parsimony_scores = normalize_lower_is_better(parsimonies)
    gen_gap_scores = normalize_lower_is_better(gen_gaps)

    radar_dict: Dict[str, Dict[str, float]] = {}
    for i, mid in enumerate(model_ids):
        radar_dict[mid] = {
            "Validation RMSE Score": round(rmse_scores[i], 4),
            "Validation MAE Score": round(mae_scores[i], 4),
            "Validation R² Score": round(r2_scores[i], 4),
            "Parsimony Score": round(parsimony_scores[i], 4),
            "Generalization Gap Score": round(gen_gap_scores[i], 4),
        }

    return radar_dict
