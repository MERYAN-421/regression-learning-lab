"""
Descriptive statistics and univariate distribution computations for EDA.
Supports numeric features, categorical attributes, target variables, and IQR outlier detection.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class NumericDistributionStats:
    feature_name: str
    count: int
    missing_count: int
    missing_pct: float
    mean: float
    std: float
    median: float
    min: float
    max: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float
    outlier_count_iqr: int
    outlier_pct_iqr: float
    is_constant: bool = False


@dataclass
class CategoricalDistributionStats:
    feature_name: str
    count: int
    missing_count: int
    missing_pct: float
    cardinality: int
    mode: Any
    mode_frequency: int
    mode_pct: float
    frequency_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def compute_numeric_distribution(series: pd.Series) -> NumericDistributionStats:
    """
    Compute comprehensive univariate statistics for a continuous or numeric feature.
    Includes mean, median, dispersion, skewness, kurtosis, and IQR outlier flags.
    """
    clean_s = series.dropna()
    total_count = len(series)
    valid_count = len(clean_s)
    missing_count = total_count - valid_count
    missing_pct = (missing_count / total_count * 100.0) if total_count > 0 else 0.0

    if valid_count == 0:
        return NumericDistributionStats(
            feature_name=str(series.name),
            count=0,
            missing_count=missing_count,
            missing_pct=missing_pct,
            mean=0.0,
            std=0.0,
            median=0.0,
            min=0.0,
            max=0.0,
            q25=0.0,
            q75=0.0,
            iqr=0.0,
            skewness=0.0,
            kurtosis=0.0,
            outlier_count_iqr=0,
            outlier_pct_iqr=0.0,
            is_constant=True
        )

    mean_val = float(clean_s.mean())
    std_val = float(clean_s.std(ddof=1)) if valid_count > 1 else 0.0
    median_val = float(clean_s.median())
    min_val = float(clean_s.min())
    max_val = float(clean_s.max())
    q25 = float(clean_s.quantile(0.25))
    q75 = float(clean_s.quantile(0.75))
    iqr_val = q75 - q25

    # Constant feature detection
    is_constant = bool(clean_s.nunique() <= 1)

    # Skewness and Kurtosis (Fisher's definition: normal = 0)
    skew_val = float(stats.skew(clean_s, bias=False)) if valid_count > 2 and not is_constant else 0.0
    kurt_val = float(stats.kurtosis(clean_s, bias=False)) if valid_count > 3 and not is_constant else 0.0

    # IQR outlier audit (educational descriptive flag; observations are NOT deleted)
    if iqr_val > 0:
        lower_bound = q25 - 1.5 * iqr_val
        upper_bound = q75 + 1.5 * iqr_val
        outlier_mask = (clean_s < lower_bound) | (clean_s > upper_bound)
        outlier_count = int(outlier_mask.sum())
    else:
        outlier_count = 0
    outlier_pct = (outlier_count / valid_count * 100.0) if valid_count > 0 else 0.0

    return NumericDistributionStats(
        feature_name=str(series.name),
        count=valid_count,
        missing_count=missing_count,
        missing_pct=missing_pct,
        mean=mean_val,
        std=std_val,
        median=median_val,
        min=min_val,
        max=max_val,
        q25=q25,
        q75=q75,
        iqr=iqr_val,
        skewness=skew_val,
        kurtosis=kurt_val,
        outlier_count_iqr=outlier_count,
        outlier_pct_iqr=outlier_pct,
        is_constant=is_constant
    )


def compute_categorical_distribution(series: pd.Series, top_n: int = 15) -> CategoricalDistributionStats:
    """
    Compute frequency distribution and cardinality for a categorical or string feature.
    """
    clean_s = series.dropna()
    total_count = len(series)
    valid_count = len(clean_s)
    missing_count = total_count - valid_count
    missing_pct = (missing_count / total_count * 100.0) if total_count > 0 else 0.0

    if valid_count == 0:
        return CategoricalDistributionStats(
            feature_name=str(series.name),
            count=0,
            missing_count=missing_count,
            missing_pct=missing_pct,
            cardinality=0,
            mode=None,
            mode_frequency=0,
            mode_pct=0.0,
            frequency_table=pd.DataFrame()
        )

    val_counts = clean_s.value_counts()
    cardinality = len(val_counts)
    mode_val = val_counts.index[0]
    mode_freq = int(val_counts.iloc[0])
    mode_pct = (mode_freq / valid_count * 100.0) if valid_count > 0 else 0.0

    freq_df = val_counts.head(top_n).reset_index()
    freq_df.columns = ["Category", "Count"]
    freq_df["Percentage"] = (freq_df["Count"] / valid_count * 100.0).round(2)

    return CategoricalDistributionStats(
        feature_name=str(series.name),
        count=valid_count,
        missing_count=missing_count,
        missing_pct=missing_pct,
        cardinality=cardinality,
        mode=mode_val,
        mode_frequency=mode_freq,
        mode_pct=mode_pct,
        frequency_table=freq_df
    )
