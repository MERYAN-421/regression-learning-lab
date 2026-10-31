"""
Interactive Plotly visualizations for regression model evaluation.
Includes:
- Actual vs. Predicted scatter plot with 45-degree ideal identity line
- Residuals vs. Fitted scatter plot with zero-error reference line
- Residual distribution histogram with normal curve overlay
- Model comparison performance bar charts
"""
from typing import Optional, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_actual_vs_predicted(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    target_name: str = "Target",
    target_unit: str = "",
    split_name: str = "Validation",
    rmse: Optional[float] = None,
    r2: Optional[float] = None,
    max_scatter_points: int = 5000
) -> go.Figure:
    """
    Constructs an interactive Actual vs. Predicted scatter plot with the 45-degree identity line.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    # Subsample if dataset is exceptionally large for rendering efficiency
    if len(y_t) > max_scatter_points:
        idx = np.random.choice(len(y_t), size=max_scatter_points, replace=False)
        y_t_plot = y_t[idx]
        y_p_plot = y_p[idx]
        title_suffix = f" (Sampled {max_scatter_points:,} of {len(y_t):,} points)"
    else:
        y_t_plot = y_t
        y_p_plot = y_p
        title_suffix = ""

    df_plot = pd.DataFrame({
        "Actual": y_t_plot,
        "Predicted": y_p_plot,
        "Residual": y_t_plot - y_p_plot
    })

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=df_plot["Actual"],
        y=df_plot["Predicted"],
        mode="markers",
        marker=dict(
            color=np.abs(df_plot["Residual"]),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=f"|Residual| ({target_unit})" if target_unit else "|Residual|"),
            size=6,
            opacity=0.65
        ),
        name="Observations",
        hovertemplate=(
            f"<b>Actual</b>: %{{x:.2f}} {target_unit}<br>"
            f"<b>Predicted</b>: %{{y:.2f}} {target_unit}<br>"
            f"<b>Residual</b>: %{{customdata:.2f}} {target_unit}<extra></extra>"
        ),
        customdata=df_plot["Residual"]
    ))

    # Reference 45-degree line y = x
    min_val = min(float(np.min(y_t)), float(np.min(y_p)))
    max_val = max(float(np.max(y_t)), float(np.max(y_p)))
    margin = 0.05 * (max_val - min_val) if max_val > min_val else 1.0

    line_min = min_val - margin
    line_max = max_val + margin

    fig.add_trace(go.Scatter(
        x=[line_min, line_max],
        y=[line_min, line_max],
        mode="lines",
        line=dict(color="#EF4444", dash="dash", width=2.5),
        name="Ideal Fit (y = x)"
    ))

    unit_str = f" ({target_unit})" if target_unit else ""
    metrics_str = ""
    if rmse is not None and r2 is not None:
        metrics_str = f" | RMSE: {rmse:.3f}{unit_str}, R²: {r2:.4f}"

    fig.update_layout(
        title=f"Actual vs. Predicted ({split_name} Set){title_suffix}{metrics_str}",
        xaxis_title=f"Actual {target_name}{unit_str}",
        yaxis_title=f"Predicted {target_name}{unit_str}",
        template="plotly_white",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig


def plot_residuals_vs_fitted(
    y_pred: Union[pd.Series, np.ndarray],
    residuals: Union[pd.Series, np.ndarray],
    target_name: str = "Target",
    target_unit: str = "",
    split_name: str = "Validation",
    max_scatter_points: int = 5000
) -> go.Figure:
    """
    Constructs an interactive Residuals vs. Fitted plot to inspect error dispersion patterns.
    """
    y_p = np.asarray(y_pred, dtype=float).ravel()
    res = np.asarray(residuals, dtype=float).ravel()

    if len(y_p) > max_scatter_points:
        idx = np.random.choice(len(y_p), size=max_scatter_points, replace=False)
        y_p = y_p[idx]
        res = res[idx]
        title_suffix = f" (Sampled {max_scatter_points:,} points)"
    else:
        title_suffix = ""

    df_plot = pd.DataFrame({"Fitted": y_p, "Residual": res})

    fig = go.Figure()

    # Scatter of residuals
    fig.add_trace(go.Scatter(
        x=df_plot["Fitted"],
        y=df_plot["Residual"],
        mode="markers",
        marker=dict(
            color="#3B82F6",
            size=6,
            opacity=0.6,
            line=dict(width=0.5, color="#1D4ED8")
        ),
        name="Residuals",
        hovertemplate=(
            f"<b>Fitted Value</b>: %{{x:.2f}}<br>"
            f"<b>Residual</b>: %{{y:.2f}}<extra></extra>"
        )
    ))

    # Horizontal zero reference line
    min_x = float(np.min(y_p))
    max_x = float(np.max(y_p))
    fig.add_trace(go.Scatter(
        x=[min_x, max_x],
        y=[0.0, 0.0],
        mode="lines",
        line=dict(color="#EF4444", dash="dash", width=2),
        name="Zero Error Baseline"
    ))

    unit_str = f" ({target_unit})" if target_unit else ""
    fig.update_layout(
        title=f"Residuals vs. Fitted Values ({split_name} Set){title_suffix}",
        xaxis_title=f"Fitted (Predicted) {target_name}{unit_str}",
        yaxis_title=f"Residual (e = y - ŷ){unit_str}",
        template="plotly_white",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig


def plot_residuals_histogram(
    residuals: Union[pd.Series, np.ndarray],
    target_name: str = "Target",
    target_unit: str = "",
    split_name: str = "Validation",
    nbins: int = 40
) -> go.Figure:
    """
    Constructs a histogram of residuals with a superimposed normal distribution curve.
    """
    res = np.asarray(residuals, dtype=float).ravel()
    res = res[np.isfinite(res)]

    mean_res = float(np.mean(res))
    std_res = float(np.std(res))

    fig = go.Figure()

    # Histogram of residuals
    fig.add_trace(go.Histogram(
        x=res,
        nbinsx=nbins,
        histnorm="probability density",
        marker_color="#6366F1",
        opacity=0.75,
        name="Observed Residuals"
    ))

    # Superimposed theoretical normal density curve
    if std_res > 1e-12:
        x_norm = np.linspace(mean_res - 3.5 * std_res, mean_res + 3.5 * std_res, 200)
        y_norm = (1.0 / (std_res * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x_norm - mean_res) / std_res) ** 2)

        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode="lines",
            line=dict(color="#DC2626", width=2.5),
            name=f"Theoretical Normal (μ={mean_res:.2f}, σ={std_res:.2f})"
        ))

    # Vertical line at mean residual
    fig.add_vline(
        x=mean_res,
        line_width=1.5,
        line_dash="dash",
        line_color="#10B981",
        annotation_text=f"Mean: {mean_res:.2f}",
        annotation_position="top left"
    )

    unit_str = f" ({target_unit})" if target_unit else ""
    fig.update_layout(
        title=f"Residual Distribution ({split_name} Set)",
        xaxis_title=f"Residual (e = y - ŷ){unit_str}",
        yaxis_title="Probability Density",
        template="plotly_white",
        bargap=0.05,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig


def plot_model_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Constructs a grouped bar chart comparing Train RMSE and Validation RMSE across models.
    """
    if comparison_df.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_df["Model ID"],
        y=comparison_df["Train RMSE"],
        name="Train RMSE",
        marker_color="#93C5FD",
        text=comparison_df["Train RMSE"],
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        x=comparison_df["Model ID"],
        y=comparison_df["Val RMSE"],
        name="Validation RMSE",
        marker_color="#3B82F6",
        text=comparison_df["Val RMSE"],
        textposition="auto"
    ))

    fig.update_layout(
        title="Training vs. Validation RMSE across Evaluated Models (Lower is Better)",
        xaxis_title="Model Experiment",
        yaxis_title="Root Mean Squared Error (RMSE)",
        barmode="group",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig
