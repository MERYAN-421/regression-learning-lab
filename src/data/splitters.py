"""
Data splitting engine: implements Random, Temporal, and Group partitioning strategies.
Prevents data leakage and ensures deterministic, non-overlapping dataset splits.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.data.models import DatasetMetadata, SplitConfig, SplitResult, SplitStrategy
from src.data.validation import validate_dataset_schema


class DataSplittingError(Exception):
    """Raised when data splitting fails validation or execution."""
    pass


def split_dataset(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    metadata: DatasetMetadata,
    config: SplitConfig,
    raw_df: Optional[pd.DataFrame] = None
) -> SplitResult:
    """
    Execute deterministic data splitting into Train, Validation, and Test partitions.
    Guarantees non-overlapping indices and zero data leakage.
    Blocks execution if critical schema errors are detected.
    """
    config.validate()
    warnings: List[str] = []

    if len(feature_df) == 0:
        raise DataSplittingError("Cannot split an empty dataset")

    # Enforce mandatory 14-39 usable predictor constraint
    if not (14 <= feature_df.shape[1] <= 39):
        raise DataSplittingError(
            f"Dataset splitting blocked: Usable feature count ({feature_df.shape[1]}) "
            f"violates the mandatory 14–39 requirement."
        )

    # Enforce schema validation if raw_df is available
    if raw_df is not None:
        report = validate_dataset_schema(raw_df, metadata)
        if report.has_blocking_errors:
            raise DataSplittingError(
                f"Dataset splitting blocked by critical schema validation errors: "
                f"{report.blocking_errors}"
            )

    # 1. Target Missingness Handling
    valid_mask = target_series.notnull()
    n_missing_target = int((~valid_mask).sum())
    if n_missing_target > 0:
        warnings.append(
            f"Explicitly excluded {n_missing_target} observations with missing Target values "
            f"('{metadata.target_column}') prior to partitioning."
        )

    eligible_indices = np.where(valid_mask)[0]
    n_eligible = len(eligible_indices)

    if n_eligible < 10:
        raise DataSplittingError(
            f"Insufficient valid observations ({n_eligible}) to perform a 3-way split"
        )

    strat = config.strategy

    if strat == SplitStrategy.RANDOM:
        train_idx, val_idx, test_idx, temp_bounds, grp_summary = _execute_random_split(
            n_eligible, config
        )
    elif strat == SplitStrategy.TEMPORAL:
        if raw_df is None or not config.date_column or config.date_column not in raw_df.columns:
            raise DataSplittingError(
                f"Temporal split requires valid date column '{config.date_column}' present in dataset"
            )
        train_idx, val_idx, test_idx, temp_bounds, grp_summary = _execute_temporal_split(
            raw_df.iloc[eligible_indices], config.date_column, config
        )
    elif strat == SplitStrategy.GROUP:
        if raw_df is None or not config.group_column or config.group_column not in raw_df.columns:
            raise DataSplittingError(
                f"Group split requires valid group column '{config.group_column}' present in dataset"
            )
        train_idx, val_idx, test_idx, temp_bounds, grp_summary = _execute_group_split(
            raw_df.iloc[eligible_indices], config.group_column, config
        )
    else:
        raise DataSplittingError(f"Unsupported split strategy: {strat}")

    # Map back to original row positions
    final_train_idx = eligible_indices[train_idx]
    final_val_idx = eligible_indices[val_idx]
    final_test_idx = eligible_indices[test_idx]

    # Verify zero overlap and complete partition of eligible rows
    set_train = set(final_train_idx)
    set_val = set(final_val_idx)
    set_test = set(final_test_idx)

    if (set_train & set_val) or (set_train & set_test) or (set_val & set_test):
        raise DataSplittingError("Internal split error: overlapping partition indices detected")

    if (len(final_train_idx) + len(final_val_idx) + len(final_test_idx)) != n_eligible:
        raise DataSplittingError("Internal split error: partitioned index count does not match eligible count")

    # Construct split dataframes
    X_train = feature_df.iloc[final_train_idx].copy()
    X_val = feature_df.iloc[final_val_idx].copy()
    X_test = feature_df.iloc[final_test_idx].copy()

    y_train = target_series.iloc[final_train_idx].copy()
    y_val = target_series.iloc[final_val_idx].copy()
    y_test = target_series.iloc[final_test_idx].copy()

    counts = {
        "train": len(X_train),
        "validation": len(X_val),
        "test": len(X_test),
        "total_eligible": n_eligible
    }

    actual_proportions = {
        "train": len(X_train) / n_eligible,
        "validation": len(X_val) / n_eligible,
        "test": len(X_test) / n_eligible
    }

    # Extract group labels if group column exists in raw_df
    grp_col = config.group_column or metadata.group_column
    groups_train = None
    groups_val = None
    groups_test = None
    if raw_df is not None and grp_col and grp_col in raw_df.columns:
        groups_train = raw_df[grp_col].iloc[final_train_idx].reset_index(drop=True)
        groups_val = raw_df[grp_col].iloc[final_val_idx].reset_index(drop=True)
        groups_test = raw_df[grp_col].iloc[final_test_idx].reset_index(drop=True)

    # Extract timestamp labels if date column exists in raw_df
    date_col = config.date_column or metadata.date_column
    timestamps_train = None
    timestamps_val = None
    timestamps_test = None
    if raw_df is not None and date_col and date_col in raw_df.columns:
        timestamps_train = raw_df[date_col].iloc[final_train_idx].reset_index(drop=True)
        timestamps_val = raw_df[date_col].iloc[final_val_idx].reset_index(drop=True)
        timestamps_test = raw_df[date_col].iloc[final_test_idx].reset_index(drop=True)

    # Compute deterministic split fingerprint
    import hashlib
    fp_raw = f"{metadata.id}_{strat.value}_{config.train_ratio}_{config.val_ratio}_{config.test_ratio}_{config.random_seed}_{len(final_train_idx)}"
    split_fp = hashlib.sha256(fp_raw.encode()).hexdigest()[:16]

    return SplitResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_indices=final_train_idx,
        val_indices=final_val_idx,
        test_indices=final_test_idx,
        actual_proportions=actual_proportions,
        sample_counts=counts,
        strategy=strat.value,
        random_seed=config.random_seed if strat != SplitStrategy.TEMPORAL else None,
        temporal_bounds=temp_bounds,
        group_summary=grp_summary,
        groups_train=groups_train,
        groups_val=groups_val,
        groups_test=groups_test,
        timestamps_train=timestamps_train,
        timestamps_val=timestamps_val,
        timestamps_test=timestamps_test,
        split_fingerprint=split_fp,
        warnings=warnings
    )


def _execute_random_split(
    n: int,
    config: SplitConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict], Optional[Dict]]:
    """Deterministic random index partitioning."""
    rng = np.random.RandomState(config.random_seed)
    shuffled = rng.permutation(n)

    n_train = int(np.floor(n * config.train_ratio))
    n_val = int(np.floor(n * config.val_ratio))
    # Ensure remaining goes to test
    n_train = max(1, n_train)
    n_val = max(1, n_val)

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train:n_train + n_val]
    test_idx = shuffled[n_train + n_val:]

    return train_idx, val_idx, test_idx, None, None


def _execute_temporal_split(
    eligible_df: pd.DataFrame,
    date_col: str,
    config: SplitConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Tuple[Any, Any]], Optional[Dict]]:
    """
    Chronological non-shuffled split: earliest 70% Train, next 15% Val, latest 15% Test.
    Prevents temporal data leakage into earlier training observations.
    """
    dates = pd.to_datetime(eligible_df[date_col])
    chronological_order = np.argsort(dates.values)
    n = len(chronological_order)

    n_train = int(np.floor(n * config.train_ratio))
    n_val = int(np.floor(n * config.val_ratio))

    train_idx = chronological_order[:n_train]
    val_idx = chronological_order[n_train:n_train + n_val]
    test_idx = chronological_order[n_train + n_val:]

    # Compute temporal boundaries
    train_dates = dates.iloc[train_idx]
    val_dates = dates.iloc[val_idx]
    test_dates = dates.iloc[test_idx]

    temp_bounds = {
        "train": (train_dates.min(), train_dates.max()),
        "validation": (val_dates.min(), val_dates.max()),
        "test": (test_dates.min(), test_dates.max())
    }

    return train_idx, val_idx, test_idx, temp_bounds, None


def _execute_group_split(
    eligible_df: pd.DataFrame,
    group_col: str,
    config: SplitConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict], Dict[str, Any]]:
    """
    Group-disjoint split: each unique group appears in exactly one partition.
    Evaluates out-of-group generalization (e.g. unseen countries).
    """
    groups = eligible_df[group_col].values
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    n_groups = len(unique_groups)

    if n_groups < 3:
        raise DataSplittingError(
            f"Group split requires at least 3 unique groups in '{group_col}' (found: {n_groups})"
        )

    # Deterministically shuffle groups
    rng = np.random.RandomState(config.random_seed)
    shuffled_group_indices = rng.permutation(n_groups)
    shuffled_groups = unique_groups[shuffled_group_indices]

    n_total_rows = len(eligible_df)
    target_train_rows = n_total_rows * config.train_ratio
    target_val_rows = n_total_rows * config.val_ratio

    train_groups = []
    val_groups = []
    test_groups = []

    current_train_count = 0
    current_val_count = 0

    group_size_map = dict(zip(unique_groups, group_counts))

    for g in shuffled_groups:
        size = group_size_map[g]
        if current_train_count + size <= target_train_rows or len(train_groups) == 0:
            train_groups.append(g)
            current_train_count += size
        elif current_val_count + size <= target_val_rows or len(val_groups) == 0:
            val_groups.append(g)
            current_val_count += size
        else:
            test_groups.append(g)

    # Edge case: ensure test has at least 1 group
    if len(test_groups) == 0:
        if len(val_groups) > 1:
            test_groups.append(val_groups.pop())
        elif len(train_groups) > 1:
            test_groups.append(train_groups.pop())

    # Verify disjointness
    set_tg = set(train_groups)
    set_vg = set(val_groups)
    set_teg = set(test_groups)
    if (set_tg & set_vg) or (set_tg & set_teg) or (set_vg & set_teg):
        raise DataSplittingError("Internal group split error: overlapping groups detected across partitions")

    # Map groups back to observation indices in eligible_df
    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups)
    test_mask = np.isin(groups, test_groups)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    grp_summary = {
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "test_group_count": len(test_groups),
        "total_groups": n_groups,
        "train_groups_sample": list(train_groups[:5]),
        "val_groups_sample": list(val_groups[:5]),
        "test_groups_sample": list(test_groups[:5])
    }

    return train_idx, val_idx, test_idx, None, grp_summary
