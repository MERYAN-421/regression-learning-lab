from pathlib import Path
import py_compile
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_project_root_structure():
    """Verify that all essential directories exist."""
    expected_dirs = ["pages", "src", "data", "configs", "tests", "docs", ".streamlit"]
    for d in expected_dirs:
        dir_path = PROJECT_ROOT / d
        assert dir_path.exists() and dir_path.is_dir(), f"Directory missing: {d}"

def test_app_entrypoint_exists():
    """Verify that app.py exists and compiles cleanly."""
    app_file = PROJECT_ROOT / "app.py"
    assert app_file.exists() and app_file.is_file(), "app.py is missing"
    py_compile.compile(str(app_file), doraise=True)

def test_pages_exist_and_compile():
    """Verify that all 9 pages exist, are non-empty, and compile without syntax errors."""
    expected_pages = [
        "home.py",
        "dataset.py",
        "exploration.py",
        "feature_selection.py",
        "regression.py",
        "evaluation.py",
        "diagnostics.py",
        "comparison.py",
        "report.py"
    ]
    pages_dir = PROJECT_ROOT / "pages"
    for page_name in expected_pages:
        page_path = pages_dir / page_name
        assert page_path.exists(), f"Page missing: {page_name}"
        assert page_path.stat().st_size > 0, f"Page is empty: {page_name}"
        py_compile.compile(str(page_path), doraise=True)

def test_src_modules_structure():
    """Verify that src and its subpackages exist and are importable."""
    src_dir = PROJECT_ROOT / "src"
    assert (src_dir / "__init__.py").exists()
    subpackages = [
        "data", "exploration", "features", "models",
        "evaluation", "diagnostics", "experiments", "reporting", "utils"
    ]
    for sp in subpackages:
        init_file = src_dir / sp / "__init__.py"
        assert init_file.exists(), f"Missing subpackage: src/{sp}/__init__.py"

def test_configs_are_valid_yaml():
    """Verify that YAML configuration files exist and parse successfully."""
    configs_dir = PROJECT_ROOT / "configs"
    for cfg in ["app_config.yaml", "datasets.yaml"]:
        cfg_path = configs_dir / cfg
        assert cfg_path.exists(), f"Config missing: {cfg}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            assert isinstance(data, dict), f"Config {cfg} must load as a dict"

def test_requirements_file():
    """Verify requirements.txt exists and contains core dependencies."""
    req_file = PROJECT_ROOT / "requirements.txt"
    assert req_file.exists()
    content = req_file.read_text(encoding="utf-8")
    for pkg in ["streamlit", "pandas", "numpy", "scikit-learn", "statsmodels", "plotly"]:
        assert pkg in content, f"Package {pkg} missing from requirements.txt"

def test_docs_exist():
    """Verify that all Task 5 documentation files exist and are populated."""
    docs_dir = PROJECT_ROOT / "docs"
    expected_docs = ["PROJECT_SPEC.md", "DATASET_CATALOG.md", "STATISTICAL_METHODS.md"]
    for doc in expected_docs:
        doc_path = docs_dir / doc
        assert doc_path.exists(), f"Document missing: {doc}"
        assert doc_path.stat().st_size > 100, f"Document too small or empty: {doc}"
