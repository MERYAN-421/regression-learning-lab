"""
General utilities, configuration loaders, and plotting helpers.
"""
from pathlib import Path
import yaml

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent

def load_yaml_config(relative_path: str) -> dict:
    """Load a YAML configuration file relative to the project root."""
    root = get_project_root()
    config_file = root / relative_path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
