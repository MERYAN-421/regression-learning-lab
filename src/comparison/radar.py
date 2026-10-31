"""
Radar Chart Plotly Visualization Engine.
Renders multi-metric spider charts comparing candidate models across normalized performance dimensions.
Emphasizes that radar scores are relative to the active candidate pool and do not represent absolute truth.
"""
from typing import Dict, List, Optional
import plotly.graph_objects as go

from src.models.model_result import OLSModelResult
from src.comparison.normalization import RADAR_DIMENSIONS, compute_radar_scores


# Harmonious color palette for overlaid polygons
RADAR_COLORS = [
    "#2E86AB",  # Blue
    "#F24236",  # Coral / Red
    "#06D6A0",  # Emerald Green
    "#F77F00",  # Amber
    "#9D4EDD",  # Purple
    "#4361EE",  # Indigo
    "#E63946",  # Crimson
]


def create_radar_chart(
    models: List[OLSModelResult],
    radar_scores: Optional[Dict[str, Dict[str, float]]] = None,
    max_models: int = 5
) -> go.Figure:
    """
    Builds an interactive Plotly Radar Chart comparing up to `max_models` candidate models.
    
    Dimensions (clockwise):
    1. Validation RMSE Score (lower is better -> 1.0 best)
    2. Validation MAE Score (lower is better -> 1.0 best)
    3. Validation R² Score (higher is better -> 1.0 best)
    4. Parsimony Score (fewer parameters -> 1.0 best)
    5. Generalization Gap Score (smaller gap -> 1.0 best)
    """
    fig = go.Figure()

    if not models:
        fig.update_layout(
            title="Radar Chart: No Models Selected",
            xaxis={"visible": False},
            yaxis={"visible": False}
        )
        return fig

    scores_dict = radar_scores if radar_scores is not None else compute_radar_scores(models)
    display_models = models[:max_models]

    categories = list(RADAR_DIMENSIONS)
    # Close the polygon loop by repeating first category
    categories_closed = categories + [categories[0]]

    for idx, m in enumerate(display_models):
        m_id = m.model_id
        m_scores = scores_dict.get(m_id, {})
        color = RADAR_COLORS[idx % len(RADAR_COLORS)]

        r_values = [m_scores.get(dim, 0.5) for dim in categories]
        r_values_closed = r_values + [r_values[0]]

        # Raw values for rich hover tooltip
        hover_texts = [
            f"Val RMSE: {m.val_metrics.rmse:.4f} (Score: {m_scores.get('Validation RMSE Score', 0.5):.2f})",
            f"Val MAE: {m.val_metrics.mae:.4f} (Score: {m_scores.get('Validation MAE Score', 0.5):.2f})",
            f"Val R²: {m.val_metrics.r2:.4f} (Score: {m_scores.get('Validation R² Score', 0.5):.2f})",
            f"Parameters: {m.design_matrix_metadata.num_parameters} (Score: {m_scores.get('Parsimony Score', 0.5):.2f})",
            f"Gen Gap: {m.val_metrics.rmse - m.train_metrics.rmse:+.4f} (Score: {m_scores.get('Generalization Gap Score', 0.5):.2f})"
        ]
        hover_texts_closed = hover_texts + [hover_texts[0]]

        fig.add_trace(go.Scatterpolar(
            r=r_values_closed,
            theta=categories_closed,
            fill='toself',
            name=f"{m.feature_source_label} ({m_id})",
            line=dict(color=color, width=2),
            hoverinfo="text",
            hovertext=hover_texts_closed,
            opacity=0.75
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.05],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.2 (Poor)", "0.4", "0.6", "0.8", "1.0 (Best)"],
                gridcolor="#E0E0E0"
            ),
            angularaxis=dict(
                direction="clockwise",
                gridcolor="#E0E0E0"
            )
        ),
        title=dict(
            text="Multi-Metric Normalized Radar Comparison (Development Stage)",
            x=0.5,
            xanchor="center"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=60, t=60, b=80),
        height=520
    )

    return fig
