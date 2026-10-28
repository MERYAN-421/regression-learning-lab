"""
Dataset schema and feature count validator module.
Enforces strict schema compliance and distinguishes blocking errors from non-blocking warnings.
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
    """
    Detailed audit report of a loaded dataset schema.
    Distinguishes critical blocking errors (which prevent splitting/modeling)
    from non-blocking warnings (data quality insights).
    """
    is_valid: bool
    total_columns: int
    usable_feature_count: int
    target_name: str
    target_null_count: int
    numeric_feature_count: int
    categorical_feature_count: int
    blocking_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def discrepancies(self) -> List[str]:
        """Backward-compatible alias for blocking_errors."""
        return self.blocking_errors

    @property
    def has_blocking_errors(self) -> bool:
        """Returns True if any critical schema criteria failed."""
        return len(self.blocking_errors) > 0


def validate_dataset_schema(
    raw_df: pd.DataFrame,
    metadata: DatasetMetadata,
    raise_on_blocking: bool = False
) -> ValidationReport:
    """
    Validate the loaded DataFrame against system rules and registry metadata.
    Enforces the mandatory 14-39 usable predictor constraint and structural integrity.
    
    Parameters:
        raw_df: Loaded Pandas DataFrame.
        metadata: DatasetMetadata object defining expected schema.
        raise_on_blocking: If True, raises SchemaValidationError if any blocking error occurs.
    """
    blocking_errors: List[str] = []
    warnings: List[str] = []

    # 1. Check for empty dataframe
    if raw_df is None or len(raw_df) == 0:
        blocking_errors.append("Dataset is empty (contains 0 observations).")
        report = ValidationReport(
            is_valid=False,
            total_columns=0,
            usable_feature_count=0,
            target_name=metadata.target_column,
            target_null_count=0,
            numeric_feature_count=0,
            categorical_feature_count=0,
            blocking_errors=blocking_errors,
            warnings=warnings
        )
        if raise_on_blocking:
            raise SchemaValidationError(blocking_errors[0])
        return report

    # 2. Check duplicate column names (Critical Blocking Error)
    col_counts = pd.Series(raw_df.columns).value_counts()
    duplicates = col_counts[col_counts > 1].index.tolist()
    if duplicates:
        blocking_errors.append(f"Duplicate column headers detected: {duplicates}")

    # 3. Check Target column existence (Critical Blocking Error)
    target_col = metadata.target_column.strip()
    if target_col not in raw_df.columns:
        blocking_errors.append(
            f"Target column '{target_col}' not found in dataset columns: {list(raw_df.columns)}"
        )

    # Check Target nulls (Non-blocking warning; rows will be excluded before splitting)
    target_nulls = 0
    if target_col in raw_df.columns:
        target_nulls = int(raw_df[target_col].isnull().sum())
        if target_nulls > 0:
            warnings.append(
                f"Target column '{target_col}' contains {target_nulls} missing values "
                f"({target_nulls / len(raw_df):.2%}). These records are excluded prior to model splitting."
            )

    # 4. Check strategy dependencies (Critical if strategy declared in metadata)
    strat = SplitStrategy(metadata.recommended_split)
    if strat == SplitStrategy.TEMPORAL:
        if not metadata.date_column or metadata.date_column not in raw_df.columns:
            blocking_errors.append(
                f"Temporal split declared, but date column '{metadata.date_column}' is missing from CSV"
            )
    elif strat == SplitStrategy.GROUP:
        if not metadata.group_column or metadata.group_column not in raw_df.columns:
            blocking_errors.append(
                f"Group split declared, but group column '{metadata.group_column}' is missing from CSV"
            )

    # 5. Excluded columns existence check (Non-blocking Warning)
    for excl in metadata.excluded_predictor_columns:
        if excl.strip() not in raw_df.columns:
            warnings.append(f"Declared excluded column '{excl}' was not found in raw CSV")

    # 6. Compute usable feature count
    total_cols = len(raw_df.columns)
    actual_excluded = {c.strip() for c in metadata.excluded_predictor_columns if c.strip() in raw_df.columns}
    non_predictors = actual_excluded | ({target_col} if target_col in raw_df.columns else set())
    feature_cols = [c for c in raw_df.columns if c not in non_predictors]
    usable_feature_count = len(feature_cols)

    # 7. Enforce mandatory 14-39 usable predictor constraint (Critical Blocking Error)
    if not (14 <= usable_feature_count <= 39):
        blocking_errors.append(
            f"Usable feature count ({usable_feature_count}) violates mandatory 14–39 rule. "
            f"(Total Raw: {total_cols}, Excluded: {len(actual_excluded)}, Target: 1)"
        )

    # 8. Compare with metadata expectations (Non-blocking Warnings)
    if metadata.expected_usable_features > 0 and usable_feature_count != metadata.expected_usable_features:
        warnings.append(
            f"Usable feature count ({usable_feature_count}) differs from expected metadata "
            f"({metadata.expected_usable_features})"
        )
    if metadata.expected_rows > 0 and abs(len(raw_df) - metadata.expected_rows) > 0:
        warnings.append(
            f"Row count ({len(raw_df)}) differs from expected metadata ({metadata.expected_rows})"
        )

    # 9. Classify numeric and categorical features
    feature_df = raw_df[feature_cols]
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Additional Data Quality Warnings (Non-blocking)
    # Check for constant features (safely skip if columns are duplicated)
    if not duplicates:
        for col in numeric_cols:
            col_series = feature_df[col]
            if isinstance(col_series, pd.Series) and col_series.nunique(dropna=True) <= 1:
                warnings.append(f"Feature '{col}' is constant (zero variance).")

    is_valid = len(blocking_errors) == 0

    if raise_on_blocking and not is_valid:
        raise SchemaValidationError(
            f"Dataset failed schema validation with {len(blocking_errors)} blocking errors:\n" +
            "\n".join([f"- {e}" for e in blocking_errors])
        )

    return ValidationReport(
        is_valid=is_valid,
        total_columns=total_cols,
        usable_feature_count=usable_feature_count,
        target_name=target_col,
        target_null_count=target_nulls,
        numeric_feature_count=len(numeric_cols),
        categorical_feature_count=len(categorical_cols),
        blocking_errors=blocking_errors,
        warnings=warnings
    )
