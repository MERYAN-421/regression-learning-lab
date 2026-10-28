"""
Dataset schema and feature count validator module.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import pandas as pd
from src.data.models import DatasetMetadata, SplitStrategy


class SchemaValidationError(Exception):
    """Raised when an actual dataset CSV fails required structural criteria."""
    pass


@dataclass
class ValidationReport:
    is_valid: bool
    total_columns: int
    usable_feature_count: int
    target_name: str
    target_null_count: int
    numeric_feature_count: int
    categorical_feature_count: int
    discrepancies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_dataset_schema(
    raw_df: pd.DataFrame,
    metadata: DatasetMetadata
) -> ValidationReport:
    """
    Validate the loaded DataFrame against system rules and registry metadata.
    Enforces the mandatory 14-39 usable predictor constraint.
    """
    discrepancies = []
    warnings = []

    # 1. Check duplicate column names
    col_counts = pd.Series(raw_df.columns).value_counts()
    duplicates = col_counts[col_counts > 1].index.tolist()
    if duplicates:
        discrepancies.append(f"Duplicate column headers detected: {duplicates}")

    # 2. Check Target column existence
    target_col = metadata.target_column.strip()
    if target_col not in raw_df.columns:
        raise SchemaValidationError(
            f"Target column '{target_col}' not found in dataset columns: {list(raw_df.columns)}"
        )

    # Check Target nulls
    target_nulls = int(raw_df[target_col].isnull().sum())
    if target_nulls > 0:
        warnings.append(
            f"Target column '{target_col}' contains {target_nulls} missing values ({target_nulls / len(raw_df):.2%}). "
            "These records must be excluded prior to model splitting."
        )

    # 3. Check strategy dependencies
    strat = SplitStrategy(metadata.recommended_split)
    if strat == SplitStrategy.TEMPORAL:
        if not metadata.date_column or metadata.date_column not in raw_df.columns:
            discrepancies.append(
                f"Temporal split declared, but date column '{metadata.date_column}' is missing from CSV"
            )
    elif strat == SplitStrategy.GROUP:
        if not metadata.group_column or metadata.group_column not in raw_df.columns:
            discrepancies.append(
                f"Group split declared, but group column '{metadata.group_column}' is missing from CSV"
            )

    # 4. Excluded columns existence check
    for excl in metadata.excluded_predictor_columns:
        if excl.strip() not in raw_df.columns:
            warnings.append(f"Declared excluded column '{excl}' was not found in raw CSV")

    # 5. Compute usable feature count
    total_cols = len(raw_df.columns)
    actual_excluded = {c.strip() for c in metadata.excluded_predictor_columns if c.strip() in raw_df.columns}
    non_predictors = actual_excluded | {target_col}
    feature_cols = [c for c in raw_df.columns if c not in non_predictors]
    usable_feature_count = len(feature_cols)

    # 6. Enforce mandatory 14-39 usable predictor constraint
    if not (14 <= usable_feature_count <= 39):
        discrepancies.append(
            f"Usable feature count ({usable_feature_count}) violates mandatory 14-39 rule "
            f"(Total: {total_cols}, Excluded: {len(actual_excluded)}, Target: 1)"
        )

    # Compare with metadata expectations
    if metadata.expected_usable_features > 0 and usable_feature_count != metadata.expected_usable_features:
        discrepancies.append(
            f"Usable feature count ({usable_feature_count}) differs from expected ({metadata.expected_usable_features})"
        )
    if metadata.expected_rows > 0 and abs(len(raw_df) - metadata.expected_rows) > 0:
        discrepancies.append(
            f"Row count ({len(raw_df)}) differs from expected ({metadata.expected_rows})"
        )

    # 7. Classify numeric and categorical features
    feature_df = raw_df[feature_cols]
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    is_valid = len(discrepancies) == 0

    return ValidationReport(
        is_valid=is_valid,
        total_columns=total_cols,
        usable_feature_count=usable_feature_count,
        target_name=target_col,
        target_null_count=target_nulls,
        numeric_feature_count=len(numeric_cols),
        categorical_feature_count=len(categorical_cols),
        discrepancies=discrepancies,
        warnings=warnings
    )
