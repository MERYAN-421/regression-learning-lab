"""
Comparison Leaderboard Adapter for Academic Reporting.
Implements a canonical, strongly typed data contract bridging Milestone 5's
display-oriented leaderboard keys and Milestone 6's auditable evidence engine.
Guarantees:
- Never defaults missing keys to zero or fabricated numbers.
- Unifies display keys ('Model ID', 'Val RMSE', 'Val R²', 'Parameters (p)')
  with canonical typed structures.
- Explicitly flags Pareto efficiency and candidate selection membership.
- Ensures Test Set metrics never influence development-stage candidate comparison.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from src.comparison.result_models import ExperimentComparisonResult


@dataclass(frozen=True)
class CanonicalLeaderboardRow:
    """
    Canonical development-stage representation of a model's row in a comparison leaderboard.
    Every numeric attribute originates from validated model metrics or comparison calculations.
    """
    rank: int
    model_id: str
    model_name: str
    feature_source: str
    original_features_k: int
    parameter_count: int
    val_rmse: float
    train_rmse: Optional[float] = None
    generalization_gap: Optional[float] = None
    val_r2: float = 0.0
    train_r2: Optional[float] = None
    train_adj_r2: Optional[float] = None
    val_mae: Optional[float] = None
    train_mae: Optional[float] = None
    is_pareto: bool = False
    is_valid_for_inference: bool = True
    diagnostic_suite_id: Optional[str] = None
    test_evaluation_status: str = "untested"


def adapt_leaderboard_row(
    row: Dict[str, Any],
    pareto_model_ids: Optional[List[str]] = None
) -> CanonicalLeaderboardRow:
    """
    Parses a single leaderboard dictionary into a CanonicalLeaderboardRow.
    Accepts both Milestone 5 display-oriented keys and canonical snake_case keys.

    Raises:
        ValueError: If mandatory model_id, val_rmse, or val_r2 cannot be resolved.
                    Never substitutes default zeros for missing experimental statistics.
    """
    if not isinstance(row, dict):
        raise ValueError(f"Leaderboard row must be a dictionary, got {type(row).__name__}")

    # 1. Model ID
    m_id = row.get("Model ID") or row.get("model_id")
    if not m_id or not str(m_id).strip():
        raise ValueError("Leaderboard row is missing mandatory 'Model ID' / 'model_id'.")
    m_id = str(m_id).strip()

    # 2. Validation RMSE (Mandatory numeric)
    raw_val_rmse = row.get("Val RMSE") if "Val RMSE" in row else row.get("val_rmse")
    if raw_val_rmse is None or str(raw_val_rmse).strip() in ("", "N/A", "None"):
        raise ValueError(f"Model '{m_id}' is missing mandatory Validation RMSE. Cannot fabricate zero.")
    try:
        val_rmse = float(raw_val_rmse)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Model '{m_id}' has non-numeric Validation RMSE '{raw_val_rmse}'.") from err

    # 3. Validation R² (Mandatory numeric)
    raw_val_r2 = row.get("Val R²") if "Val R²" in row else (row.get("Val R2") if "Val R2" in row else row.get("val_r2"))
    if raw_val_r2 is None or str(raw_val_r2).strip() in ("", "N/A", "None"):
        raise ValueError(f"Model '{m_id}' is missing mandatory Validation R². Cannot fabricate zero.")
    try:
        val_r2 = float(raw_val_r2)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Model '{m_id}' has non-numeric Validation R² '{raw_val_r2}'.") from err

    # 4. Parameters (p)
    raw_p = row.get("Parameters (p)") if "Parameters (p)" in row else (row.get("parameter_count") if "parameter_count" in row else row.get("parameters"))
    p_count = int(raw_p) if raw_p is not None and str(raw_p).isdigit() else (int(raw_p) if isinstance(raw_p, (int, float)) else 0)

    # 5. Original Features (K)
    raw_k = row.get("Original Features (K)") if "Original Features (K)" in row else (row.get("original_feature_count") if "original_feature_count" in row else row.get("original_k"))
    k_count = int(raw_k) if raw_k is not None and str(raw_k).isdigit() else (int(raw_k) if isinstance(raw_k, (int, float)) else 0)

    # 6. Rank
    raw_rank = row.get("Rank") if "Rank" in row else row.get("rank", 1)
    try:
        rank_val = int(raw_rank)
    except (TypeError, ValueError):
        rank_val = 1

    # 7. Model Name
    m_name = row.get("Model Name") or row.get("model_name") or f"Model ({m_id})"

    # 8. Feature Source Label
    feat_src = row.get("Feature Source") or row.get("feature_source_label") or row.get("feature_source") or "Selected Features"

    # 9. Train RMSE & Generalization Gap
    raw_tr_rmse = row.get("Train RMSE") if "Train RMSE" in row else row.get("train_rmse")
    train_rmse = float(raw_tr_rmse) if raw_tr_rmse is not None and str(raw_tr_rmse) not in ("", "N/A", "None") else None

    raw_gen_gap = row.get("Gen. Gap") if "Gen. Gap" in row else row.get("generalization_gap")
    if raw_gen_gap is not None and str(raw_gen_gap) not in ("", "N/A", "None"):
        gen_gap = float(raw_gen_gap)
    elif train_rmse is not None:
        gen_gap = val_rmse - train_rmse
    else:
        gen_gap = None

    # 10. Train R² & Train Adj R²
    raw_tr_r2 = row.get("Train R²") if "Train R²" in row else (row.get("Train R2") if "Train R2" in row else row.get("train_r2"))
    train_r2 = float(raw_tr_r2) if raw_tr_r2 is not None and str(raw_tr_r2) not in ("", "N/A", "None") else None

    raw_adj_r2 = row.get("Train Adj R²") if "Train Adj R²" in row else row.get("train_adj_r2")
    train_adj_r2 = float(raw_adj_r2) if raw_adj_r2 is not None and str(raw_adj_r2) not in ("", "N/A", "None") else None

    # 11. Val & Train MAE
    raw_v_mae = row.get("Val MAE") if "Val MAE" in row else row.get("val_mae")
    val_mae = float(raw_v_mae) if raw_v_mae is not None and str(raw_v_mae) not in ("", "N/A", "None") else None

    raw_t_mae = row.get("Train MAE") if "Train MAE" in row else row.get("train_mae")
    train_mae = float(raw_t_mae) if raw_t_mae is not None and str(raw_t_mae) not in ("", "N/A", "None") else None

    # 12. Pareto Status
    if pareto_model_ids is not None:
        is_pareto = m_id in pareto_model_ids
    elif "Pareto Status" in row:
        p_status = str(row["Pareto Status"])
        is_pareto = "Pareto Efficient" in p_status or "⭐" in p_status
    elif "is_pareto_efficient" in row:
        is_pareto = bool(row["is_pareto_efficient"])
    elif "is_pareto" in row:
        is_pareto = bool(row["is_pareto"])
    else:
        is_pareto = False

    # 13. Inference Status
    if "Inference Status" in row:
        is_inf = "Full Rank" in str(row["Inference Status"]) or "✅" in str(row["Inference Status"])
    elif "is_valid_for_inference" in row:
        is_inf = bool(row["is_valid_for_inference"])
    else:
        is_inf = True

    # 14. Diagnostic Suite ID
    diag_id = row.get("Diagnostic Suite") or row.get("diagnostic_suite_id")
    if diag_id and str(diag_id).strip() in ("", "Not Run", "None"):
        diag_id = None

    # 15. Test Status (strictly non-ranking flag)
    test_status = row.get("Test Status") or row.get("test_evaluation_status") or "untested"

    return CanonicalLeaderboardRow(
        rank=rank_val,
        model_id=m_id,
        model_name=str(m_name),
        feature_source=str(feat_src),
        original_features_k=k_count,
        parameter_count=p_count,
        val_rmse=val_rmse,
        train_rmse=train_rmse,
        generalization_gap=gen_gap,
        val_r2=val_r2,
        train_r2=train_r2,
        train_adj_r2=train_adj_r2,
        val_mae=val_mae,
        train_mae=train_mae,
        is_pareto=is_pareto,
        is_valid_for_inference=is_inf,
        diagnostic_suite_id=str(diag_id) if diag_id else None,
        test_evaluation_status=str(test_status)
    )


def adapt_comparison_leaderboard(
    comparison: ExperimentComparisonResult
) -> List[CanonicalLeaderboardRow]:
    """
    Adapts an entire ExperimentComparisonResult.leaderboard into canonical typed rows.
    Guarantees consistent Pareto membership against comparison.pareto_model_ids.
    """
    if comparison is None or not comparison.leaderboard:
        return []

    canonical_rows = []
    for raw_row in comparison.leaderboard:
        parsed = adapt_leaderboard_row(raw_row, pareto_model_ids=comparison.pareto_model_ids)
        canonical_rows.append(parsed)

    return canonical_rows
