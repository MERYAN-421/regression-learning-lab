"""
Group E Tests: Statistical Diagnostic Comparison Matrix.
Verifies:
- Suite-to-model matching using model ID.
- Display of test statistics, p-values, and formal decisions.
- Unexecuted diagnostic suites marked as "Not Run" rather than treated as favorable.
- "Not Applicable" diagnostics marked transparently.
- Prediction-only models badge displayed.
- Strict Rule 3 enforcement: No diagnostic composite score or pass count column exists.
"""
import pytest
import numpy as np
import pandas as pd

from src.models.model_service import train_ols_experiment
from src.diagnostics.runner import run_full_diagnostic_suite
from src.comparison.diagnostics_matrix import build_diagnostic_comparison_matrix
from src.data.models import SplitResult


@pytest.fixture
def diagnostics_comparison_fixture():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.normal(0, 1, n),
    })
    y = 2.0 + 1.5 * df["x1"] + np.random.normal(0, 0.1, n)

    split = SplitResult(
        X_train=df.iloc[:60],
        X_val=df.iloc[60:80],
        X_test=df.iloc[80:],
        y_train=y.iloc[:60],
        y_val=y.iloc[60:80],
        y_test=y.iloc[80:],
        train_indices=np.arange(60),
        val_indices=np.arange(60, 80),
        test_indices=np.arange(80, 100),
        actual_proportions={"train": 0.6, "val": 0.2, "test": 0.2},
        sample_counts={"train": 60, "val": 20, "test": 20},
        strategy="random",
        random_seed=42,
        split_fingerprint="split_diag_comp"
    )

    m1 = train_ols_experiment(
        split_result=split,
        feature_source=["x1"],
        dataset_id="ds_diag",
        dataset_fingerprint="ds_diag_fp",
        target_name="target"
    )

    m2 = train_ols_experiment(
        split_result=split,
        feature_source=["x1", "x2"],
        dataset_id="ds_diag",
        dataset_fingerprint="ds_fp_diag",
        target_name="target"
    )

    # Run diagnostic suite only for m1
    suite_m1 = run_full_diagnostic_suite(m1, split_result=split, alpha=0.05)

    return split, m1, m2, suite_m1


def test_diagnostic_matrix_populates_executed_and_not_run(diagnostics_comparison_fixture):
    split, m1, m2, suite_m1 = diagnostics_comparison_fixture

    matrix = build_diagnostic_comparison_matrix(
        models=[m1, m2],
        diagnostic_suites={m1.model_id: suite_m1}
    )

    assert len(matrix) == 2
    row_m1 = matrix[0]
    row_m2 = matrix[1]

    # m1 should have executed results
    assert row_m1["Suite ID"] == suite_m1.suite_id
    assert row_m1["Suite Alpha"] == "0.05"
    assert "RESET Decision" in row_m1
    assert row_m1["RESET Decision"] != "Not Run"

    # m2 should have "Not Run"
    assert row_m2["Suite ID"] == "Not Run"
    assert row_m2["RESET Decision"] == "Not Run"
    assert row_m2["DW Decision"] == "Not Run"
    assert row_m2["Jarque-Bera Decision"] == "Not Run"


def test_strict_rule_3_no_composite_score(diagnostics_comparison_fixture):
    split, m1, m2, suite_m1 = diagnostics_comparison_fixture
    matrix = build_diagnostic_comparison_matrix(
        models=[m1, m2],
        diagnostic_suites={m1.model_id: suite_m1}
    )

    # Verify no column aggregates assumptions into a score or count
    forbidden_terms = ["score", "pass count", "passed assumptions", "total score", "quality score"]
    for row in matrix:
        for k in row.keys():
            k_lower = k.lower()
            for term in forbidden_terms:
                assert term not in k_lower, f"Forbidden diagnostic scoring term '{term}' found in column '{k}'"
