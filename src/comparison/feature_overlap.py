"""
Feature Selection Comparison Engine.
Analyzes original feature subsets across candidate OLS models:
- Feature inclusion matrix (original feature names)
- Pairwise Jaccard similarity
- Candidate feature selection frequency
"""
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.models.model_result import OLSModelResult


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Computes Jaccard Similarity index between two feature sets:
    J(A, B) = |A ∩ B| / |A ∪ B|
    
    Boundary rules:
    - If both sets are empty: returns 1.0 (identical empty sets).
    - If one set is empty and the other non-empty: returns 0.0.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 1.0
    return float(intersection / union)


def build_feature_inclusion_dataframe(models: List[OLSModelResult]) -> pd.DataFrame:
    """
    Constructs a binary inclusion DataFrame:
    Rows: All unique original feature names across all compared models.
    Columns: Model IDs (or feature source labels).
    Values: 1 (Selected) or 0 (Not Selected).
    """
    if not models:
        return pd.DataFrame()

    all_features: Set[str] = set()
    model_feature_map: Dict[str, Set[str]] = {}

    for m in models:
        feats = set(m.selected_original_features)
        model_feature_map[m.model_id] = feats
        all_features.update(feats)

    sorted_features = sorted(list(all_features))
    data: Dict[str, List[int]] = {"Feature": sorted_features}

    for m in models:
        col_name = f"{m.feature_source_label} ({m.model_id[:8]})"
        m_feats = model_feature_map[m.model_id]
        data[col_name] = [1 if f in m_feats else 0 for f in sorted_features]

    df = pd.DataFrame(data).set_index("Feature")
    return df


def compute_pairwise_jaccard_matrix(models: List[OLSModelResult]) -> pd.DataFrame:
    """Computes an N x N symmetric matrix of pairwise Jaccard similarities."""
    if not models:
        return pd.DataFrame()

    col_labels = [f"{m.feature_source_label} ({m.model_id[:8]})" for m in models]
    n = len(models)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        set_i = set(models[i].selected_original_features)
        for j in range(n):
            set_j = set(models[j].selected_original_features)
            matrix[i, j] = compute_jaccard_similarity(set_i, set_j)

    return pd.DataFrame(matrix, index=col_labels, columns=col_labels)


def compute_feature_selection_frequencies(models: List[OLSModelResult]) -> Dict[str, int]:
    """Counts how many candidate models included each original feature."""
    freq: Dict[str, int] = {}
    for m in models:
        for f in m.selected_original_features:
            freq[f] = freq.get(f, 0) + 1
    # Sort descending by frequency
    return dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))


def create_feature_frequency_chart(freq_dict: Dict[str, int], total_models: int) -> go.Figure:
    """Plots a horizontal bar chart of feature selection frequency across candidate models."""
    fig = go.Figure()
    if not freq_dict:
        fig.update_layout(title="Feature Selection Frequency: No Features")
        return fig

    features = list(freq_dict.keys())
    counts = list(freq_dict.values())
    pcts = [round((c / max(1, total_models)) * 100, 1) for c in counts]

    # Invert order for top-to-bottom bar chart
    features_inv = features[::-1]
    counts_inv = counts[::-1]
    pcts_inv = pcts[::-1]

    fig.add_trace(go.Bar(
        y=features_inv,
        x=counts_inv,
        orientation='h',
        marker=dict(
            color=counts_inv,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Count")
        ),
        text=[f"{c}/{total_models} ({p}%)" for c, p in zip(counts_inv, pcts_inv)],
        textposition="auto"
    ))

    fig.update_layout(
        title=dict(
            text="Original Feature Selection Frequency Across Candidate Models",
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title=f"Number of Models Selecting Feature (Out of {total_models})",
            range=[0, total_models + 0.5],
            dtick=1
        ),
        yaxis=dict(title="Original Feature"),
        margin=dict(l=140, r=40, t=60, b=60),
        height=max(400, len(features) * 22)
    )
    return fig


def create_jaccard_heatmap(jaccard_df: pd.DataFrame) -> go.Figure:
    """Plots an interactive heatmap of pairwise Jaccard similarities."""
    fig = go.Figure()
    if jaccard_df.empty:
        fig.update_layout(title="Jaccard Heatmap: Empty")
        return fig

    z_vals = jaccard_df.values
    labels = list(jaccard_df.columns)

    fig.add_trace(go.Heatmap(
        z=z_vals,
        x=labels,
        y=labels,
        colorscale="Blues",
        zmin=0.0,
        zmax=1.0,
        text=np.round(z_vals, 3),
        texttemplate="%{text}",
        colorbar=dict(title="Jaccard Index")
    ))

    fig.update_layout(
        title=dict(
            text="Pairwise Original Feature Set Similarity (Jaccard Index)",
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(tickangle=45),
        margin=dict(l=80, r=40, t=60, b=100),
        height=480
    )
    return fig
