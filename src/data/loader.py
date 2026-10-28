"""
Dataset loader module: reads registered tabular datasets from local storage.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from src.utils import get_project_root
from src.data.models import DatasetMetadata


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected dataset file cannot be located on disk."""
    pass


class DatasetParsingError(Exception):
    """Raised when parsing fails on a tabular dataset."""
    pass


def resolve_dataset_path(metadata: DatasetMetadata) -> Path:
    """Resolve the absolute filesystem path for a dataset."""
    root = get_project_root()
    return root / metadata.local_path


def compute_file_fingerprint(file_path: Path) -> str:
    """
    Compute a deterministic file fingerprint based on file size and SHA-256 hash.
    Ensures that replacing a CSV file or updating its contents invalidates caches.
    """
    if not file_path.exists() or not file_path.is_file():
        return "missing"
    file_size = file_path.stat().st_size
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return f"{file_size}_{hasher.hexdigest()[:16]}"


def get_dataset_fingerprint(metadata: DatasetMetadata) -> str:
    """Get the file fingerprint for a registered dataset."""
    path = resolve_dataset_path(metadata)
    return compute_file_fingerprint(path)


def check_dataset_availability(metadata: DatasetMetadata) -> bool:
    """Check if the dataset file exists on disk."""
    path = resolve_dataset_path(metadata)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def load_raw_dataset(metadata: DatasetMetadata) -> pd.DataFrame:
    """
    Load the raw CSV dataset into a Pandas DataFrame.
    Normalizes column header whitespace while preserving column integrity.
    """
    file_path = resolve_dataset_path(metadata)
    if not file_path.exists():
        raise DatasetNotFoundError(
            f"Dataset '{metadata.name}' (ID: {metadata.id}) not found at '{metadata.local_path}'.\n"
            f"Acquisition instruction: Download from {metadata.kaggle_url} and place in 'data/raw/'."
        )

    try:
        df = pd.read_csv(
            file_path,
            sep=metadata.csv_separator,
            encoding=metadata.encoding
        )
    except Exception as e:
        raise DatasetParsingError(f"Failed to parse CSV file for '{metadata.id}': {str(e)}") from e

    # Clean leading/trailing whitespaces in column names (e.g. WHO Life Expectancy column padding)
    df.columns = [col.strip() for col in df.columns]

    # Date parsing if date column specified
    if metadata.date_column and metadata.date_column in df.columns:
        try:
            df[metadata.date_column] = pd.to_datetime(df[metadata.date_column])
        except Exception:
            # Leave as raw if datetime parsing fails
            pass

    return df


def split_features_and_target(
    df: pd.DataFrame,
    metadata: DatasetMetadata
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Separate the dataset into:
    1. feature_df: Only valid model predictor columns (14-39 count).
    2. target_series: The target response variable.
    3. metadata_df: Excluded ID, temporal, or group metadata columns.
    """
    target_name = metadata.target_column.strip()
    if target_name not in df.columns:
        raise KeyError(
            f"Target column '{target_name}' not found in dataset columns: {list(df.columns)}"
        )

    target_series = df[target_name].copy()

    # Identify excluded columns
    declared_excluded = {c.strip() for c in metadata.excluded_predictor_columns}
    actual_excluded = declared_excluded.intersection(set(df.columns))

    # All non-predictor columns: target + excluded
    non_predictor_cols = set(actual_excluded) | {target_name}

    # Features are remaining columns
    feature_cols = [c for c in df.columns if c not in non_predictor_cols]
    feature_df = df[feature_cols].copy()

    # Metadata dataframe containing excluded columns
    meta_cols = [c for c in df.columns if c in actual_excluded]
    metadata_df = df[meta_cols].copy()

    return feature_df, target_series, metadata_df


def load_dataset_bundle(metadata: DatasetMetadata):
    """
    High-level loader: returns (raw_df, feature_df, target_series, metadata_df).
    Ensures raw data is not mutated.
    """
    raw_df = load_raw_dataset(metadata)
    feature_df, target_series, metadata_df = split_features_and_target(raw_df, metadata)
    return raw_df, feature_df, target_series, metadata_df
