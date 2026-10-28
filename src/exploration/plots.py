"""
Interactive Plotly visualizations for Exploratory Data Analysis.
Designed with academic-style aesthetics, tooltips, and performance sampling indicators.
"""
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


COLOR_PRIMARY = "#4F46E5"
COLOR_ACCENT = "#EC4899"
COLOR_NEUTRAL = "#64748B"


def plot_numeric_distribution(
    series: pd.Series,
    feature_name: Optional[str] = None,
    nbins: int = 40
) -> go.Figure:
    """Plot an interactive histogram with an integrated top marginal box plot."""
    name = feature_name or str(series.name)
    clean_s = series.dropna()

    fig = px.histogram(
        x=clean_s,
        nbins=nbins,
        marginal="box",
        title=f"Distribution of {name} (N = {len(clean_s):,})",
        labels={"x": name, "y": "Frequency"},
        color_discrete_sequence=[COLOR_PRIMARY]
    )

    fig.update_layout(
        template="plotly_white",
        bargap=0.05,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_title=name,
        yaxis_title="Count"
    )
    return fig


def plot_categorical_distribution(
    series: pd.Series,
    feature_name: Optional[str] = None,
    top_n: int = 15
) -> go.Figure:
    """Plot category frequency distribution bar chart."""
    name = feature_name or str(series.name)
    clean_s = series.dropna()
    counts = clean_s.value_counts().head(top_n).reset_index()
    counts.columns = ["Category", "Count"]

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        title=f"Frequency Breakdown of {name} (Top {len(counts)})",
        color="Count",
        color_continuous_scale="Purples"
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        coloraxis_showscale=False
    )
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Pairwise Feature Correlation Matrix"
) -> go.Figure:
    """Plot an interactive correlation heatmap with hover annotations."""
    if corr_matrix.empty:
        fig = go.Figure()
        fig.update_layout(title="No numeric features available for correlation heatmap")
        return fig

    z_vals = corr_matrix.values
    x_cols = list(corr_matrix.columns)
    y_cols = list(corr_matrix.index)

    # Format text labels rounded to 2 decimals
    text_vals = np.round(z_vals, 2).astype(str)
    text_vals[text_vals == "nan"] = ""

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=x_cols,
        y=y_cols,
        text=text_vals,
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        hoverongaps=False
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(tickangle=-45)
    )
    return fig


def plot_bivariate_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    add_trendline: bool = True,
    max_display_points: int = 3000
) -> Tuple[go.Figure, int, bool]:
    """
    Plot interactive scatter plot of numeric feature vs target or feature.
    Applies display-only sampling for large datasets to ensure browser responsiveness
    while clearly reporting the actual plotted sample count.
    
    Returns:
        (fig, sample_size, was_sampled)
    """
    valid_data = df[[x_col, y_col]].dropna()
    total_valid = len(valid_data)
    was_sampled = total_valid > max_display_points

    if was_sampled:
        plot_df = valid_data.sample(n=max_display_points, random_state=42)
    else:
        plot_df = valid_data

    sample_size = len(plot_df)
    sample_note = f" (Display Sample: {sample_size:,} / {total_valid:,} rows)" if was_sampled else f" (N = {sample_size:,})"

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        trendline="ols" if add_trendline and len(plot_df) >= 3 else None,
        title=f"{x_col} vs. {y_col}{sample_note}",
        labels={x_col: x_col, y_col: y_col},
        opacity=0.6,
        color_discrete_sequence=[COLOR_PRIMARY]
    )

    if add_trendline and len(plot_df) >= 3:
        # Style trendline distinctly
        fig.update_traces(
            line=dict(color=COLOR_ACCENT, width=2.5),
            selector=dict(mode="lines")
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig, sample_size, was_sampled


def plot_categorical_target_boxplot(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    max_display_points: int = 3000
) -> Tuple[go.Figure, int, bool]:
    """
    Plot box plot of target distribution grouped by categorical predictor categories.
    """
    valid_data = df[[cat_col, target_col]].dropna()
    total_valid = len(valid_data)
    was_sampled = total_valid > max_display_points

    if was_sampled:
        plot_df = valid_data.sample(n=max_display_points, random_state=42)
    else:
        plot_df = valid_data

    sample_size = len(plot_df)
    sample_note = f" (Sample: {sample_size:,} / {total_valid:,})" if was_sampled else f" (N = {sample_size:,})"

    fig = px.box(
        plot_df,
        x=cat_col,
        y=target_col,
        title=f"Target ({target_col}) Distribution across '{cat_col}' Categories{sample_note}",
        color=cat_col,
        points="outliers"
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(tickangle=-30)
    )
    return fig, sample_size, was_sampled
