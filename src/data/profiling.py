"""
Dataset profiling module: computes descriptive statistics and interactive exploratory plots.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data.models import DatasetMetadata, DatasetProfile


def compute_dataset_profile(
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    metadata: DatasetMetadata
) -> DatasetProfile:
    """Compute comprehensive descriptive summary of the dataset."""
    total_rows = len(raw_df)
    total_columns = len(raw_df.columns)
    usable_feature_count = len(feature_df.columns)

    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in feature_df.columns if c not in numeric_features]

    missing_total = int(raw_df.isnull().sum().sum())
    missing_pct_total = (missing_total / (total_rows * total_columns)) * 100.0 if total_rows > 0 else 0.0
    duplicate_rows = int(raw_df.duplicated().sum())

    # Target statistics (handle numeric / bounded targets)
    target_clean = target_series.dropna()
    is_target_numeric = pd.api.types.is_numeric_dtype(target_clean)

    target_stats: Dict[str, Any] = {
        "name": metadata.target_column,
        "unit": metadata.target_unit,
        "data_type": str(target_series.dtype),
        "total_count": len(target_series),
        "valid_count": len(target_clean),
        "missing_count": int(target_series.isnull().sum()),
        "missing_pct": (target_series.isnull().sum() / total_rows) * 100.0 if total_rows > 0 else 0.0,
    }

    if is_target_numeric and len(target_clean) > 0:
        target_stats.update({
            "min": float(target_clean.min()),
            "max": float(target_clean.max()),
            "mean": float(target_clean.mean()),
            "median": float(target_clean.median()),
            "std": float(target_clean.std()),
            "q25": float(target_clean.quantile(0.25)),
            "q75": float(target_clean.quantile(0.75)),
        })
    else:
        target_stats.update({
            "min": None, "max": None, "mean": None, "median": None, "std": None,
            "q25": None, "q75": None
        })

    # Missing values per column
    col_nulls = raw_df.isnull().sum()
    col_missing_summary = {col: int(cnt) for col, cnt in col_nulls.items() if cnt > 0}

    # Column dtypes
    column_dtypes = {col: str(dtype) for col, dtype in raw_df.dtypes.items()}

    return DatasetProfile(
        total_rows=total_rows,
        total_columns=total_columns,
        usable_feature_count=usable_feature_count,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        missing_count_total=missing_total,
        missing_percentage_total=missing_pct_total,
        duplicate_rows=duplicate_rows,
        target_stats=target_stats,
        column_missing_summary=col_missing_summary,
        column_dtypes=column_dtypes
    )


def create_target_histogram(
    target_series: pd.Series,
    metadata: DatasetMetadata,
    max_display_points: int = 15000
) -> go.Figure:
    """Create interactive Plotly histogram of the continuous response variable."""
    clean_target = target_series.dropna()
    n_total = len(clean_target)
    is_sampled = n_total > max_display_points

    if is_sampled:
        display_data = clean_target.sample(n=max_display_points, random_state=42)
        subtitle_note = f" (Display sampled: {max_display_points:,} of {n_total:,} rows)"
    else:
        display_data = clean_target
        subtitle_note = ""

    title = f"Distribution of Target Variable: {metadata.target_column}{subtitle_note}"

    fig = px.histogram(
        display_data,
        x=metadata.target_column,
        nbins=50,
        marginal="box",
        color_discrete_sequence=["#4F46E5"],
        labels={metadata.target_column: f"{metadata.target_column} ({metadata.target_unit})"}
    )

    fig.update_layout(
        title=title,
        xaxis_title=f"{metadata.target_column} [{metadata.target_unit}]",
        yaxis_title="Observation Count",
        template="plotly_white",
        bargap=0.05,
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def create_missing_values_bar(raw_df: pd.DataFrame) -> Optional[go.Figure]:
    """Create interactive bar chart of missing value percentages for columns with nulls."""
    null_counts = raw_df.isnull().sum()
    null_cols = null_counts[null_counts > 0].sort_values(ascending=True)

    if len(null_cols) == 0:
        return None

    null_pct = (null_cols / len(raw_df)) * 100.0
    plot_df = pd.DataFrame({
        "Feature": null_cols.index,
        "MissingCount": null_cols.values,
        "MissingPercent": null_pct.values
    })

    fig = px.bar(
        plot_df,
        x="MissingPercent",
        y="Feature",
        orientation="h",
        text=plot_df["MissingPercent"].apply(lambda v: f"{v:.1f}%"),
        color="MissingPercent",
        color_continuous_scale="Reds",
        labels={"MissingPercent": "Missing Values (%)", "Feature": "Column"}
    )

    fig.update_layout(
        title="Missing Values by Feature (% of Total Observations)",
        xaxis_title="Missingness Percentage (%)",
        yaxis_title="Feature Name",
        template="plotly_white",
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=50, b=40),
        height=max(280, len(null_cols) * 26)
    )
    return fig


def create_feature_breakdown_chart(profile: DatasetProfile) -> go.Figure:
    """Create donut chart illustrating the composition of model predictors."""
    n_num = len(profile.numeric_features)
    n_cat = len(profile.categorical_features)

    labels = ["Numeric Predictors", "Categorical Predictors"]
    values = [n_num, n_cat]
    colors = ["#4F46E5", "#06B6D4"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+value+percent",
        insidetextorientation="horizontal"
    )])

    fig.update_layout(
        title=f"Usable Predictor Types ({profile.usable_feature_count} Total)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
        showlegend=False
    )
    return fig
