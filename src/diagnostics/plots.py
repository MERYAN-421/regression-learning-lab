"""
Interactive Plotly diagnostic visualization suite for the Statistical Assumption Diagnostics Laboratory.
Generates publication-quality econometric plots:
1. Residuals vs. Fitted with LOWESS smoothing curve
2. Residuals vs. Time / Sequence
3. Autocorrelation Function (ACF) with Bartlett 95% confidence bands
4. Scale-Location Plot (sqrt(|standardized residuals|) vs. fitted values)
5. Normal Q-Q Plot with theoretical reference line
6. Residual Distribution Histogram with overlaid normal density curve
7. Variance Inflation Factor (VIF) horizontal bar chart with 5.0 and 10.0 thresholds
8. Residuals vs. Leverage with Cook's distance contours
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess


PLOT_LAYOUT_THEME = dict(
    template="plotly_white",
    font=dict(family="Inter, Roboto, sans-serif", size=12),
    margin=dict(l=60, r=40, t=50, b=50),
    hovermode="closest"
)


def plot_residuals_vs_fitted_lowess(
    fitted_values: np.ndarray,
    residuals: np.ndarray,
    target_name: str = "Target",
    target_unit: str = "",
    max_plot_points: int = 5000
) -> go.Figure:
    """
    Renders the classic Residuals vs. Fitted diagnostic plot.
    Features:
    - Zero horizontal reference line (e = 0)
    - LOWESS non-parametric trend line to highlight non-linearities
    - Subsampling notice if n > max_plot_points (for UI responsiveness)
    """
    n_total = len(residuals)
    sampled = False
    if n_total > max_plot_points:
        idx = np.random.choice(n_total, max_plot_points, replace=False)
        f_plot = fitted_values[idx]
        r_plot = residuals[idx]
        sampled = True
    else:
        f_plot = fitted_values
        r_plot = residuals

    fig = go.Figure()

    # Scatter of points
    fig.add_trace(go.Scatter(
        x=f_plot,
        y=r_plot,
        mode="markers",
        name="Residuals",
        marker=dict(size=5, color="rgba(41, 128, 185, 0.5)", line=dict(width=0.5, color="#1b4f72")),
        hovertemplate="Fitted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>"
    ))

    # Horizontal zero line
    fig.add_hline(
        y=0,
        line=dict(color="#7f8c8d", width=1.5, dash="dash"),
        annotation_text="e = 0",
        annotation_position="bottom right"
    )

    # LOWESS smoother
    try:
        # Sort by fitted values for LOWESS
        sort_order = np.argsort(f_plot)
        f_sorted = f_plot[sort_order]
        r_sorted = r_plot[sort_order]
        smooth_res = lowess(r_sorted, f_sorted, frac=0.66)
        fig.add_trace(go.Scatter(
            x=smooth_res[:, 0],
            y=smooth_res[:, 1],
            mode="lines",
            name="LOWESS Smooth",
            line=dict(color="#e74c3c", width=2.5)
        ))
    except Exception:
        pass

    subtitle = f"Sample: {len(f_plot):,} / {n_total:,} points" if sampled else f"n = {n_total:,} observations"
    unit_str = f" ({target_unit})" if target_unit else ""

    fig.update_layout(
        **PLOT_LAYOUT_THEME,
        title=dict(
            text=f"<b>Residuals vs. Fitted Values</b><br><sup>{subtitle} | Detects non-linearities and curvature misspecification</sup>",
            x=0.02
        ),
        xaxis_title=f"Fitted Values ŷ{unit_str}",
        yaxis_title=f"Training Residuals e = y - ŷ{unit_str}",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)")
    )
    return fig


def plot_residuals_vs_time(
    residuals: np.ndarray,
    timestamps: Optional[pd.Series] = None,
    dataset_name: str = "Dataset",
    max_plot_points: int = 5000
) -> go.Figure:
    """
    Renders Residuals plotted along a sequential time axis or observation index.
    Crucial for temporal datasets like Appliances Energy.
    """
    n_total = len(residuals)
    x_axis = timestamps if timestamps is not None else np.arange(n_total)
    x_label = "Observation Timestamp" if timestamps is not None else "Observation Index (Row Order)"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=residuals,
        mode="lines+markers" if n_total < 200 else "lines",
        name="Residual Sequence",
        line=dict(color="#2980b9", width=1.0),
        marker=dict(size=3, color="#2980b9"),
        hovertemplate="%{x}<br>Residual: %{y:.3f}<extra></extra>"
    ))

    # Horizontal zero line
    fig.add_hline(
        y=0,
        line=dict(color="#7f8c8d", width=1.5, dash="dash"),
        annotation_text="e = 0"
    )

    fig.update_layout(
        **PLOT_LAYOUT_THEME,
        title=dict(
            text=f"<b>Residual Sequence Across Time / Order ({dataset_name})</b><br><sup>Detects persistent runs and seasonal cycles</sup>",
            x=0.02
        ),
        xaxis_title=x_label,
        yaxis_title="Training Residuals e_i",
        showlegend=False
    )
    return fig


def plot_residual_acf(
    acf_data: Dict[str, Any],
    max_lags: int = 20
) -> go.Figure:
    """
    Renders Autocorrelation Function (ACF) plot with Bartlett 95% confidence bands.
    """
    lags = acf_data["lags"][:max_lags + 1]
    acf_vals = acf_data["acf_values"][:max_lags + 1]
    bartlett_band = acf_data.get("bartlett_threshold", 1.96 / np.sqrt(acf_data.get("sample_size", 100)))

    fig = go.Figure()

    # Bartlett confidence band (upper & lower)
    fig.add_hline(
        y=bartlett_band,
        line=dict(color="#3498db", width=1.5, dash="dot"),
        annotation_text=f"+95% Band (+{bartlett_band:.3f})",
        annotation_position="top right"
    )
    fig.add_hline(
        y=-bartlett_band,
        line=dict(color="#3498db", width=1.5, dash="dot"),
        annotation_text=f"-95% Band (-{bartlett_band:.3f})",
        annotation_position="bottom right"
    )
    fig.add_hline(y=0, line=dict(color="#7f8c8d", width=1.0))

    # Vertical stems for autocorrelations
    for l, val in zip(lags[1:], acf_vals[1:]):
        fig.add_trace(go.Scatter(
            x=[l, l],
            y=[0, val],
            mode="lines",
            line=dict(color="#2c3e50" if abs(val) <= bartlett_band else "#e74c3c", width=2.5),
            hoverinfo="skip",
            showlegend=False
        ))

    # Marker heads
    colors = ["#e74c3c" if abs(v) > bartlett_band else "#2c3e50" for v in acf_vals[1:]]
    fig.add_trace(go.Scatter(
        x=lags[1:],
        y=acf_vals[1:],
        mode="markers",
        name="Autocorrelation",
        marker=dict(size=8, color=colors),
        hovertemplate="Lag %{x}: r = %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        **PLOT_LAYOUT_THEME,
        title=dict(
            text="<b>Residual Autocorrelation Function (ACF)</b><br><sup>Bartlett 95% confidence bands ±1.96/√n. Spikes exceeding bands violate independence.</sup>",
            x=0.02
        ),
        xaxis_title="Lag Order (k)",
        yaxis_title="Autocorrelation r_k",
        yaxis_range=[-1.05, 1.05],
        showlegend=False
    )
    return fig


def plot_scale_location(
    fitted_values: np.ndarray,
    residuals: np.ndarray,
    target_name: str = "Target",
    max_plot_points: int = 5000
) -> go.Figure:
    """
    Renders the Scale-Location (Spread-Location) diagnostic plot for homoscedasticity:
    X-axis: Fitted values (ŷ)
    Y-axis: sqrt(|Standardized Residuals|)
    A horizontal trend indicates constant error variance (homoscedasticity).
    An upward or downward trend indicates heteroscedasticity.
    """
    n_total = len(residuals)
    s_std = np.std(residuals, ddof=1)
    std_residuals = residuals / s_std if s_std > 1e-12 else residuals
    sqrt_std_resid = np.sqrt(np.abs(std_residuals))

    sampled = False
    if n_total > max_plot_points:
        idx = np.random.choice(n_total, max_plot_points, replace=False)
        f_plot = fitted_values[idx]
        y_plot = sqrt_std_resid[idx]
        sampled = True
    else:
        f_plot = fitted_values
        y_plot = sqrt_std_resid

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=f_plot,
        y=y_plot,
        mode="markers",
        name="Observations",
        marker=dict(size=5, color="rgba(142, 68, 173, 0.5)", line=dict(width=0.5, color="#5b2c6f")),
        hovertemplate="Fitted: %{x:.3f}<br>√|Std Resid|: %{y:.3f}<extra></extra>"
    ))

    # Add LOWESS trend
    try:
        sort_order = np.argsort(f_plot)
        smooth_res = lowess(y_plot[sort_order], f_plot[sort_order], frac=0.66)
        fig.add_trace(go.Scatter(
            x=smooth_res[:, 0],
            y=smooth_res[:, 1],
            mode="lines",
            name="Trend Line",
            line=dict(color="#e74c3c", width=2.5)
        ))
    except Exception:
        pass

    subtitle = f"Sample: {len(f_plot):,} / {n_total:,} observations" if sampled else f"n = {n_total:,} observations"

    fig.update_layout(
        **PLOT_LAYOUT_THEME,
        title=dict(
            text=f"<b>Scale-Location Plot</b><br><sup>{subtitle} | Horizontal red trend line supports homoscedasticity; tilt indicates heteroscedasticity</sup>",
            x=0.02
        ),
        xaxis_title="Fitted Values ŷ",
        yaxis_title="√|Standardized Residuals|",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)")
    )
    return fig


def plot_normal_qq(
    residuals: np.ndarray,
    max_plot_points: int = 5000
) -> go.Figure:
    """
    Renders Normal Q-Q (Quantile-Quantile) Plot comparing standardized residuals
    against theoretical Gaussian quantiles.
    """
    n_total = len(residuals)
    s_std = np.std(residuals, ddof=1)
    std_residuals = residuals / s_std if s_std > 1e-12 else residuals

    (osm, osr), (slope, intercept, r) = stats.probplot(std_residuals, dist="norm")

    fig = go.Figure()

    # Reference diagonal line
    min_x = float(np.min(osm))
    max_x = float(np.max(osm))
    line_x = np.linspace(min_x, max_x, 100)
    line_y = intercept + slope * line_x

    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="Theoretical Normal Line",
        line=dict(color="#e74c3c", width=2.0, dash="dash")
    ))

    # Sample quantiles
    fig.add_trace(go.Scatter(
        x=osm,
        y=osr,
        mode="markers",
        name="Residual Quantiles",
        marker=dict(size=5, color="rgba(41, 128, 185, 0.6)", line=dict(width=0.5, color="#1b4f72")),
        hovertemplate="Theoretical Quantile: %{x:.3f}<br>Sample Quantile: %{y:.3f}<extra></extra>"
    ))

    fig.update_layout(
        **PLOT_LAYOUT_THEME,
        title=dict(
            text=f"<b>Normal Q-Q Plot (n = {n_total:,}, r = {r:.4f})</b><br><sup>Points along the diagonal indicate normality. Departures at ends signal skewness or heavy tails.</sup>",
            x=0.02
        ),
        xaxis_title="Theoretical Standard Normal Quantiles",
        yaxis_title="Ordered Standardized Residuals",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)")
    )
    return fig


def plot_residuals_histogram_kde(
    residuals: np.ndarray,
    target_unit: str = ""
) -> go.Figure:
    """
    Renders residual distribution histogram with an overlaid theoretical Gaussian density curve.
    """
    n_total = len(residuals)
    mean_val = float(np.mean(residuals))
    std_val = float(np.std(residuals, ddof=1))
    skew_val = float(stats.skew(residuals))
    kurt_val = float(stats.kurtosis(residuals))

    fig = go.Figure()

    # Histogram normalized to density
    fig.add_trace(go.Histogram(
        x=residuals,
        histnorm="probability density",
        name="Residual Density",
        nbinsx=40,
        marker=dict(color="rgba(52, 152, 219, 0.6)", line=dict(color="#2980b9", width=1.0)),
        hovertemplate="Residual: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>"
    ))

    # Theoretical Normal PDF overlay
    if std_val > 1e-12:
        x_grid = np.linspace(mean_val - 4 * std_val, mean_val + 4 * std_val, 200)
        norm_pdf = stats.norm.pdf(x_grid, loc=mean_val, scale=std_val)
        fig.add_trace(go.Scatter(
            x=x_grid,
            y=norm_pdf,
            mode="lines",
            name="Normal Distribution N(μ, σ²)",
            line=dict(color="#e74c3c", width=2.5)
        ))

    unit_str = f" ({target_unit})" if target_unit else ""

    fig.update_layout(
        **PLOT_LAYOUT_THEME,
        title=dict(
            text=f"<b>Residual Distribution Histogram (n = {n_total:,})</b><br><sup>Mean: {mean_val:.3e} | Std: {std_val:.3f} | Skewness: {skew_val:.3f} | Excess Kurtosis: {kurt_val:+.3f}</sup>",
            x=0.02
        ),
        xaxis_title=f"Residual Value e_i{unit_str}",
        yaxis_title="Probability Density",
        showlegend=True,
        legend=dict(x=0.70, y=0.98, bgcolor="rgba(255,255,255,0.7)")
    )
    return fig


def plot_vif_horizontal_bar(
    vif_records: List[Dict[str, Any]]
) -> go.Figure:
    """
    Renders a horizontal bar chart displaying Variance Inflation Factor (VIF) for all regressors.
    Features:
    - Vertical guideline at VIF = 5.0 (Moderate Collinearity Threshold)
    - Vertical guideline at VIF = 10.0 (Severe Collinearity Threshold)
    - Color-coded bars: Green (< 5), Orange (5-10), Red (> 10)
    """
    if not vif_records:
        fig = go.Figure()
        fig.update_layout(title="No regressors available for VIF calculation.")
        return fig

    # Reverse for top-to-bottom bar order
    records_rev = list(reversed(vif_records))
    features = [r["feature"] for r in records_rev]
    # Cap infinite values for visualization
    capped_vif = [min(r["vif"], 100.0) if np.isfinite(r["vif"]) else 100.0 for r in records_rev]
    raw_vif = [r["vif"] for r in records_rev]

    colors = []
    for val in raw_vif:
        if val > 10.0 or np.isinf(val):
            colors.append("#e74c3c")  # Red: Severe
        elif val >= 5.0:
            colors.append("#f39c12")  # Orange: Moderate
        else:
            colors.append("#27ae60")  # Green: Stable

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=features,
        x=capped_vif,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#2c3e50", width=0.5)),
        customdata=[f"inf (exact collinear)" if np.isinf(v) else f"{v:.2f}" for v in raw_vif],
        hovertemplate="Feature: %{y}<br>VIF: %{customdata}<extra></extra>"
    ))

    # Threshold guidelines
    fig.add_vline(
        x=5.0,
        line=dict(color="#f39c12", width=1.5, dash="dash"),
        annotation_text="VIF = 5 (Moderate)",
        annotation_position="top right"
    )
    fig.add_vline(
        x=10.0,
        line=dict(color="#e74c3c", width=1.5, dash="dash"),
        annotation_text="VIF = 10 (Severe)",
        annotation_position="top right"
    )

    max_x = max(12.0, max(capped_vif) * 1.1)

    fig.update_layout(
        **PLOT_LAYOUT_THEME,
        title=dict(
            text="<b>Variance Inflation Factor (VIF) by Predictor</b><br><sup>VIF > 5 indicates moderate collinearity; VIF > 10 indicates severe variance inflation</sup>",
            x=0.02
        ),
        xaxis_title="Variance Inflation Factor (VIF)",
        yaxis_title="Encoded Regressor",
        xaxis_range=[0, max_x],
        height=max(350, len(features) * 28 + 120),
        showlegend=False
    )
    return fig
