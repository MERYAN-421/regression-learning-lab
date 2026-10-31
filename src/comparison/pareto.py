"""
Pareto Efficiency and Prediction-Complexity Trade-Off Analysis.
Implements multi-objective Pareto dominance (minimizing parameters and minimizing Validation RMSE).
Renders interactive trade-off scatter plots identifying non-dominated candidate models.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go

from src.models.model_result import OLSModelResult


def compute_pareto_frontier(
    models: List[OLSModelResult]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Computes Pareto-efficient models for the standard two-objective minimization problem:
    Objective 1: Model Complexity -> minimize OLS parameters (p)
    Objective 2: Predictive Error -> minimize Validation RMSE

    Definition of Dominance:
    Model A dominates Model B iff:
      p_A <= p_B AND RMSE_A <= RMSE_B
      AND (p_A < p_B OR RMSE_A < RMSE_B)

    A model is Pareto-efficient (non-dominated) iff no other candidate model dominates it.
    
    Returns:
        (pareto_model_ids, points_data_list)
    """
    if not models:
        return [], []

    points = []
    for m in models:
        p = int(m.design_matrix_metadata.num_parameters)
        rmse = float(m.val_metrics.rmse)
        k = len(m.selected_original_features)
        r2 = float(m.val_metrics.r2)
        points.append({
            "model_id": m.model_id,
            "model_name": m.model_name,
            "feature_source": m.feature_source_label,
            "parameters": p,
            "original_k": k,
            "val_rmse": rmse,
            "val_r2": r2,
            "train_rmse": float(m.train_metrics.rmse),
            "is_valid_inference": m.is_valid_for_inference
        })

    pareto_model_ids: List[str] = []

    # Check dominance for each candidate point
    for i, pt_a in enumerate(points):
        is_dominated = False
        for j, pt_b in enumerate(points):
            if i == j:
                continue
            # Check if pt_b dominates pt_a
            p_b, rmse_b = pt_b["parameters"], pt_b["val_rmse"]
            p_a, rmse_a = pt_a["parameters"], pt_a["val_rmse"]

            b_better_or_equal = (p_b <= p_a) and (rmse_b <= rmse_a)
            b_strictly_better = (p_b < p_a) or (rmse_b < rmse_a)

            if b_better_or_equal and b_strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_model_ids.append(pt_a["model_id"])

    # Attach is_pareto flag to each point record
    for pt in points:
        pt["is_pareto"] = (pt["model_id"] in pareto_model_ids)

    return pareto_model_ids, points


def create_pareto_scatter_chart(
    models: List[OLSModelResult],
    pareto_model_ids: Optional[List[str]] = None,
    selected_candidate_id: Optional[str] = None
) -> go.Figure:
    """
    Creates an interactive Plotly scatter plot visualizing the prediction-complexity trade-off.
    Plots OLS Parameter Count (X) vs. Validation RMSE (Y).
    Draws a step-wise connecting line along the Pareto Frontier.
    """
    fig = go.Figure()

    if not models:
        fig.update_layout(title="Prediction-Complexity Trade-Off: No Models Selected")
        return fig

    if pareto_model_ids is None:
        pareto_ids, points = compute_pareto_frontier(models)
    else:
        pareto_ids = pareto_model_ids
        _, points = compute_pareto_frontier(models)

    pareto_set = set(pareto_ids)

    pareto_points = [p for p in points if p["is_pareto"]]
    dominated_points = [p for p in points if not p["is_pareto"]]

    # Sort pareto points by parameter count ascending for clean frontier path
    pareto_points.sort(key=lambda x: (x["parameters"], x["val_rmse"]))

    # 1. Connect Pareto Frontier with a stepped line
    if len(pareto_points) > 1:
        fx = [p["parameters"] for p in pareto_points]
        fy = [p["val_rmse"] for p in pareto_points]
        fig.add_trace(go.Scatter(
            x=fx,
            y=fy,
            mode="lines",
            line=dict(color="#06D6A0", width=2, dash="dash"),
            name="Pareto Frontier (Non-Dominated)",
            hoverinfo="skip"
        ))

    # 2. Plot Dominated Models
    if dominated_points:
        dx = [p["parameters"] for p in dominated_points]
        dy = [p["val_rmse"] for p in dominated_points]
        d_hover = [
            f"<b>{p['feature_source']}</b> ({p['model_id']})<br>"
            f"Parameters (p): {p['parameters']}<br>"
            f"Original K: {p['original_k']}<br>"
            f"Val RMSE: {p['val_rmse']:.4f}<br>"
            f"Val R²: {p['val_r2']:.4f}<br>"
            f"Status: Dominated"
            for p in dominated_points
        ]
        fig.add_trace(go.Scatter(
            x=dx,
            y=dy,
            mode="markers+text",
            marker=dict(symbol="circle", size=11, color="#F24236", opacity=0.85, line=dict(width=1, color="#333")),
            text=[p["model_id"][:8] for p in dominated_points],
            textposition="top center",
            name="Dominated Candidates",
            hoverinfo="text",
            hovertext=d_hover
        ))

    # 3. Plot Pareto-Efficient Models
    if pareto_points:
        px = [p["parameters"] for p in pareto_points]
        py = [p["val_rmse"] for p in pareto_points]
        p_hover = [
            f"<b>⭐ {p['feature_source']}</b> ({p['model_id']})<br>"
            f"Parameters (p): {p['parameters']}<br>"
            f"Original K: {p['original_k']}<br>"
            f"Val RMSE: {p['val_rmse']:.4f}<br>"
            f"Val R²: {p['val_r2']:.4f}<br>"
            f"Status: ⭐ Pareto Efficient (Non-Dominated)"
            for p in pareto_points
        ]
        fig.add_trace(go.Scatter(
            x=px,
            y=py,
            mode="markers+text",
            marker=dict(symbol="star", size=15, color="#06D6A0", line=dict(width=1.5, color="#004D40")),
            text=[f"⭐ {p['model_id'][:8]}" for p in pareto_points],
            textposition="bottom right",
            name="⭐ Pareto Efficient Candidates",
            hoverinfo="text",
            hovertext=p_hover
        ))

    # 4. Highlight Selected Candidate Model if specified
    if selected_candidate_id:
        sel_match = [p for p in points if p["model_id"] == selected_candidate_id]
        if sel_match:
            s_pt = sel_match[0]
            fig.add_trace(go.Scatter(
                x=[s_pt["parameters"]],
                y=[s_pt["val_rmse"]],
                mode="markers",
                marker=dict(symbol="circle-open", size=22, color="#9D4EDD", line=dict(width=3)),
                name=f"Selected Candidate ({selected_candidate_id[:8]})",
                hoverinfo="text",
                hovertext=[f"Preferred Candidate: {selected_candidate_id}"]
            ))

    fig.update_layout(
        title=dict(
            text="Prediction-Complexity Trade-Off: Parameters vs. Validation RMSE",
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Model Complexity: OLS Parameter Count (p)",
            gridcolor="#EBEBEB",
            dtick=1 if len(models) < 10 else None
        ),
        yaxis=dict(
            title="Validation Error: Validation RMSE (lower is better)",
            gridcolor="#EBEBEB"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=40, t=60, b=80),
        height=520
    )

    return fig
