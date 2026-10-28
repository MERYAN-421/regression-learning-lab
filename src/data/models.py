"""
Data classes and structured types for Dataset Engine.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class SplitStrategy(str, Enum):
    RANDOM = "random"
    TEMPORAL = "temporal"
    GROUP = "group"


@dataclass
class DatasetMetadata:
    id: str
    name: str
    source_provider: str
    kaggle_url: str
    original_reference: str
    local_path: str
    csv_separator: str = ","
    encoding: str = "utf-8"
    target_column: str = ""
    target_unit: str = ""
    target_nature: str = ""
    date_column: Optional[str] = None
    group_column: Optional[str] = None
    identifier_columns: List[str] = field(default_factory=list)
    metadata_columns: List[str] = field(default_factory=list)
    excluded_predictor_columns: List[str] = field(default_factory=list)
    expected_rows: int = 0
    expected_original_columns: int = 0
    expected_usable_features: int = 0
    recommended_split: str = "random"
    default_split_ratios: List[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])
    default_random_seed: int = 42
    license_notes: str = ""
    domain: str = ""
    description: str = ""
    learning_objectives: List[str] = field(default_factory=list)
    status: str = "available"


@dataclass
class DatasetProfile:
    total_rows: int
    total_columns: int
    usable_feature_count: int
    numeric_features: List[str]
    categorical_features: List[str]
    missing_count_total: int
    missing_percentage_total: float
    duplicate_rows: int
    target_stats: Dict[str, Any]
    column_missing_summary: Dict[str, int]
    column_dtypes: Dict[str, str]


@dataclass
class SplitConfig:
    strategy: SplitStrategy
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    date_column: Optional[str] = None
    group_column: Optional[str] = None

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Split proportions must sum to 1.0 (current sum: {total:.3f})")
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("All split proportions must be strictly positive")


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    actual_proportions: Dict[str, float]
    sample_counts: Dict[str, int]
    strategy: str
    random_seed: Optional[int]
    temporal_bounds: Optional[Dict[str, Tuple[Any, Any]]] = None
    group_summary: Optional[Dict[str, List[Any]]] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class DatasetBundle:
    metadata: DatasetMetadata
    raw_df: pd.DataFrame
    feature_df: pd.DataFrame
    target_series: pd.Series
    metadata_df: pd.DataFrame
    profile: DatasetProfile
    split_result: Optional[SplitResult] = None
