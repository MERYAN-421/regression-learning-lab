from pathlib import Path
import ast
import py_compile
import pytest
import yaml
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_project_root_structure():
    """Verify that all essential directories exist."""
    expected_dirs = ["pages", "src", "data", "configs", "tests", "docs", ".streamlit"]
    for d in expected_dirs:
        dir_path = PROJECT_ROOT / d
        assert dir_path.exists() and dir_path.is_dir(), f"Directory missing: {d}"

def test_license_exists():
    """Verify that LICENSE file exists and specifies MIT License."""
    license_file = PROJECT_ROOT / "LICENSE"
    assert license_file.exists(), "LICENSE file missing"
    content = license_file.read_text(encoding="utf-8")
    assert "MIT License" in content

def test_app_entrypoint_exists_and_compiles():
    """Verify that app.py exists and compiles cleanly."""
    app_file = PROJECT_ROOT / "app.py"
    assert app_file.exists() and app_file.is_file(), "app.py is missing"
    py_compile.compile(str(app_file), doraise=True)

def test_navigation_registration_and_groups():
    """
    Verify that app.py defines navigation across the 3 official categories,
    registers all 9 pages, and marks home.py as default.
    """
    app_content = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "st.navigation(" in app_content, "app.py must use st.navigation"
    assert "st.Page(" in app_content, "app.py must use st.Page"

    tree = ast.parse(app_content)
    pages_dict_found = False
    registered_files = []
    found_sections = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "pages" and isinstance(node.value, ast.Dict):
                    pages_dict_found = True
                    for k, v in zip(node.value.keys, node.value.values):
                        if isinstance(k, ast.Constant):
                            found_sections.append(k.value)
                        if isinstance(v, ast.List):
                            for elt in v.elts:
                                if isinstance(elt, ast.Call) and elt.args:
                                    first_arg = elt.args[0]
                                    if isinstance(first_arg, ast.Constant):
                                        registered_files.append(first_arg.value)

    assert pages_dict_found, "pages dict not found in app.py AST"
    assert set(found_sections) == {"Introduction", "Regression Workflow", "Results"}

    expected_pages = [
        "pages/home.py",
        "pages/dataset.py",
        "pages/exploration.py",
        "pages/feature_selection.py",
        "pages/regression.py",
        "pages/evaluation.py",
        "pages/diagnostics.py",
        "pages/comparison.py",
        "pages/report.py"
    ]
    for p in expected_pages:
        assert p in registered_files, f"Page {p} not registered in app.py navigation"
    assert len(registered_files) == 9, f"Expected exactly 9 registered pages, got {len(registered_files)}"

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

def test_under_development_badge_on_subpages():
    """Verify that all 8 subpages clearly state 'Under Development'."""
    subpages = [
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
    for page_name in subpages:
        content = (pages_dir / page_name).read_text(encoding="utf-8")
        assert "Under Development" in content, f"Subpage {page_name} must state 'Under Development'"
        assert "Scheduled for Milestone" in content, f"Subpage {page_name} must specify target Milestone"

@pytest.mark.parametrize("page_name", [
    "pages/home.py",
    "pages/dataset.py",
    "pages/exploration.py",
    "pages/feature_selection.py",
    "pages/regression.py",
    "pages/evaluation.py",
    "pages/diagnostics.py",
    "pages/comparison.py",
    "pages/report.py"
])
def test_page_renders_cleanly_via_apptest(page_name):
    """Verify that each page renders without throwing exceptions using Streamlit AppTest."""
    page_file = PROJECT_ROOT / page_name
    at = AppTest.from_file(str(page_file), default_timeout=10).run()
    assert len(at.exception) == 0, f"Page {page_name} threw exception: {at.exception}"

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

def test_dataset_catalog_structure_and_criteria():
    """Verify datasets.yaml adheres to natural 14-39 raw features criteria."""
    cfg_path = PROJECT_ROOT / "configs" / "datasets.yaml"
    assert cfg_path.exists()
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    criteria = data.get("criteria", {})
    assert criteria.get("min_raw_features") == 14
    assert criteria.get("max_raw_features") == 39

    candidates = data.get("verified_candidates", [])
    assert len(candidates) >= 4, "Expected at least 4 verified candidate datasets"
    for c in candidates:
        for field in ["id", "name", "source", "domain", "target_variable", "target_type", "raw_feature_count", "status"]:
            assert field in c, f"Candidate {c.get('id')} missing field {field}"
        assert 14 <= c["raw_feature_count"] <= 39, f"Feature count out of bounds for {c['id']}"
        assert c["status"] == "verified_candidate"

    assert "unverified_or_excluded" in data
    assert len(data["unverified_or_excluded"]) >= 2

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

def test_docs_exist_and_statistical_methods():
    """Verify documentation files and critical econometric clarifications."""
    docs_dir = PROJECT_ROOT / "docs"
    expected_docs = ["PROJECT_SPEC.md", "DATASET_CATALOG.md", "STATISTICAL_METHODS.md"]
    for doc in expected_docs:
        doc_path = docs_dir / doc
        assert doc_path.exists(), f"Document missing: {doc}"
        assert doc_path.stat().st_size > 100, f"Document too small or empty: {doc}"

    stat_doc = (docs_dir / "STATISTICAL_METHODS.md").read_text(encoding="utf-8")
    assert "Gauss" in stat_doc and "Diagnostic" in stat_doc
    assert "Normality is NOT" in stat_doc or "Normality is not" in stat_doc.lower()
    assert "Exogeneity" in stat_doc or "Zero Conditional Mean" in stat_doc
    assert "Residual Standard Deviation" in stat_doc
    assert "not evidence of absence" in stat_doc.lower() or "fail to reject" in stat_doc.lower()
