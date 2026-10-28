"""
Dataset Library & Data Engine module for Regression Learning Lab (Milestone 1).
Provides dataset registry management, CSV loading, schema validation,
descriptive profiling, and reproducible train/validation/test splitting.
"""
from src.data.models import (
    DatasetMetadata,
    DatasetProfile,
    SplitConfig,
    SplitResult,
    DatasetBundle,
    SplitStrategy,
)
from src.data.registry import (
    DatasetRegistry,
    get_registry,
    RegistryValidationError,
)
from src.data.loader import (
    load_raw_dataset,
    load_dataset_bundle,
    split_features_and_target,
    check_dataset_availability,
    resolve_dataset_path,
    DatasetNotFoundError,
    DatasetParsingError,
)
from src.data.validation import (
    validate_dataset_schema,
    ValidationReport,
    SchemaValidationError,
)
from src.data.profiling import (
    compute_dataset_profile,
    create_target_histogram,
    create_missing_values_bar,
    create_feature_breakdown_chart,
)
from src.data.splitters import (
    split_dataset,
    DataSplittingError,
)
from src.data.download import (
    download_dataset,
    download_all,
)

__all__ = [
    "DatasetMetadata",
    "DatasetProfile",
    "SplitConfig",
    "SplitResult",
    "DatasetBundle",
    "SplitStrategy",
    "DatasetRegistry",
    "get_registry",
    "RegistryValidationError",
    "load_raw_dataset",
    "load_dataset_bundle",
    "split_features_and_target",
    "check_dataset_availability",
    "resolve_dataset_path",
    "DatasetNotFoundError",
    "DatasetParsingError",
    "validate_dataset_schema",
    "ValidationReport",
    "SchemaValidationError",
    "compute_dataset_profile",
    "create_target_histogram",
    "create_missing_values_bar",
    "create_feature_breakdown_chart",
    "split_dataset",
    "DataSplittingError",
    "download_dataset",
    "download_all",
]
