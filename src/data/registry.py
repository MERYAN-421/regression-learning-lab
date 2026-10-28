"""
Dataset Registry module: parses and validates declarative dataset configurations.
"""
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from src.utils import get_project_root
from src.data.models import DatasetMetadata, SplitStrategy


class RegistryValidationError(Exception):
    """Raised when datasets.yaml contains invalid configuration."""
    pass


class DatasetRegistry:
    def __init__(self, config_rel_path: str = "configs/datasets.yaml"):
        self.config_rel_path = config_rel_path
        self._metadata_dict: Dict[str, DatasetMetadata] = {}
        self.load_registry()

    def load_registry(self) -> None:
        """Load datasets configuration from YAML file."""
        if Path(self.config_rel_path).is_absolute():
            config_path = Path(self.config_rel_path)
        else:
            root = get_project_root()
            config_path = root / self.config_rel_path

        if not config_path.exists():
            raise FileNotFoundError(f"Dataset registry config not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if not isinstance(raw_data, dict):
            raise RegistryValidationError("Registry config root must be a dictionary")

        raw_datasets = raw_data.get("datasets", [])
        if not isinstance(raw_datasets, list):
            raise RegistryValidationError("'datasets' key in registry must be a list")

        metadata_dict = {}
        seen_ids = set()

        for d in raw_datasets:
            ds_id = d.get("id")
            if not ds_id:
                raise RegistryValidationError("Each dataset entry must have a non-empty 'id'")
            if ds_id in seen_ids:
                raise RegistryValidationError(f"Duplicate dataset ID found: '{ds_id}'")
            seen_ids.add(ds_id)

            metadata = DatasetMetadata(
                id=ds_id,
                name=d.get("name", ds_id),
                source_provider=d.get("source_provider", ""),
                kaggle_url=d.get("kaggle_url", ""),
                original_reference=d.get("original_reference", ""),
                local_path=d.get("local_path", f"data/raw/{ds_id}.csv"),
                csv_separator=d.get("csv_separator", ","),
                encoding=d.get("encoding", "utf-8"),
                target_column=d.get("target_column", ""),
                target_unit=d.get("target_unit", ""),
                target_nature=d.get("target_nature", ""),
                date_column=d.get("date_column"),
                group_column=d.get("group_column"),
                identifier_columns=d.get("identifier_columns", []),
                metadata_columns=d.get("metadata_columns", []),
                excluded_predictor_columns=d.get("excluded_predictor_columns", []),
                expected_rows=d.get("expected_rows", 0),
                expected_original_columns=d.get("expected_original_columns", 0),
                expected_usable_features=d.get("expected_usable_features", 0),
                recommended_split=d.get("recommended_split", "random"),
                default_split_ratios=d.get("default_split_ratios", [0.70, 0.15, 0.15]),
                default_random_seed=d.get("default_random_seed", 42),
                license_notes=d.get("license_notes", ""),
                domain=d.get("domain", ""),
                description=d.get("description", ""),
                learning_objectives=d.get("learning_objectives", []),
                status=d.get("status", "available")
            )
            metadata_dict[ds_id] = metadata

        self._metadata_dict = metadata_dict
        self.validate()

    def validate(self) -> None:
        """Validate all registered datasets against system requirements."""
        if not self._metadata_dict:
            raise RegistryValidationError("No datasets registered in config")

        for ds_id, meta in self._metadata_dict.items():
            # 1. Target column must be defined
            if not meta.target_column:
                raise RegistryValidationError(f"Dataset '{ds_id}' must define a target_column")

            # 2. Split strategy must be supported
            try:
                strat = SplitStrategy(meta.recommended_split)
            except ValueError:
                raise RegistryValidationError(
                    f"Dataset '{ds_id}' has unsupported split strategy: '{meta.recommended_split}'"
                )

            # 3. Strategy dependencies
            if strat == SplitStrategy.TEMPORAL and not meta.date_column:
                raise RegistryValidationError(
                    f"Temporal split for dataset '{ds_id}' requires a valid 'date_column'"
                )
            if strat == SplitStrategy.GROUP and not meta.group_column:
                raise RegistryValidationError(
                    f"Group split for dataset '{ds_id}' requires a valid 'group_column'"
                )

            # 4. Feature count bounds check on expected values
            if meta.expected_usable_features > 0:
                if not (14 <= meta.expected_usable_features <= 39):
                    raise RegistryValidationError(
                        f"Dataset '{ds_id}' expected usable feature count "
                        f"({meta.expected_usable_features}) violates the 14-39 constraint"
                    )

            # 5. Target not in excluded predictors
            if meta.target_column in meta.excluded_predictor_columns:
                raise RegistryValidationError(
                    f"Target column '{meta.target_column}' in '{ds_id}' cannot also be in excluded_predictor_columns"
                )

    def get_dataset(self, dataset_id: str) -> DatasetMetadata:
        """Retrieve metadata for a specific dataset ID."""
        if dataset_id not in self._metadata_dict:
            available = ", ".join(self._metadata_dict.keys())
            raise KeyError(f"Dataset '{dataset_id}' not found. Available datasets: [{available}]")
        return self._metadata_dict[dataset_id]

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return list of all registered dataset metadata objects."""
        return list(self._metadata_dict.values())

    def get_all_ids(self) -> List[str]:
        """Return list of all registered dataset IDs."""
        return list(self._metadata_dict.keys())


# Global singleton instance for easy import
_default_registry: Optional[DatasetRegistry] = None

def get_registry() -> DatasetRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = DatasetRegistry()
    return _default_registry
