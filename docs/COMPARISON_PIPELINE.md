# Model Comparison Pipeline & Architecture (Milestone 5)

## 1. Pipeline Overview & Component Topology

The **Experiment & Model Comparison Laboratory** connects upstream modeling outputs into a reproducible, multi-objective evaluation pipeline. It prepares structured, auditable comparison artifacts for **Milestone 6 Academic Report Generation**.

```mermaid
graph TD
    A[Milestone 1: SplitResult] --> B[Milestone 3: RegressionExperimentTracker]
    A --> C[Milestone 4: DiagnosticSuiteResult]
    B --> D[src/comparison/compatibility.py]
    D --> E[src/comparison/comparison_runner.py]
    C --> E
    E --> F[Leaderboard Engine: Validation-Only Ranking]
    E --> G[Pareto Engine: Occam's Razor Frontier]
    E --> H[Radar Engine: [0, 1] Normalization]
    E --> I[Diagnostic Matrix: Side-by-Side Assessment]
    E --> J[Feature Overlap: Jaccard & Frequencies]
    F & G & H & I & J --> K[ExperimentComparisonResult]
    K --> L[src/experiments/comparison_experiment_tracker.py]
    L --> M[pages/comparison.py: Streamlit UI]
    K --> N[Milestone 6: Academic Report Handoff]
```

---

## 2. Directory Structure & Module Responsibilities

```
src/
├── comparison/
│   ├── __init__.py                  # Public exports
│   ├── result_models.py             # Typed dataclasses & JSON serializers
│   ├── compatibility.py             # Multi-model compatibility validator
│   ├── leaderboard.py               # Validation-only ranking engine
│   ├── normalization.py             # Safe min-max [0, 1] score normalizer
│   ├── radar.py                     # Plotly interactive spider chart
│   ├── pareto.py                    # Multi-objective Pareto frontier & scatter
│   ├── diagnostics_matrix.py        # Diagnostic aggregation without composite scoring
│   ├── feature_overlap.py           # Jaccard similarity & selection frequency
│   ├── interpretation.py            # Pedagogical text & audit trail formatting
│   └── comparison_runner.py         # Orchestrator service
├── experiments/
│   ├── regression_experiment_tracker.py  # (Milestone 3 Tracker - Reused)
│   └── comparison_experiment_tracker.py  # Lightweight comparison session tracker
pages/
└── comparison.py                    # Streamlit Comparison Laboratory UI
tests/
├── test_comparison_compatibility.py    # Group A: Compatibility tests
├── test_comparison_leaderboard.py      # Group B: Leaderboard & ranking tests
├── test_comparison_normalization.py    # Group C: Radar normalization tests
├── test_comparison_pareto.py           # Group D: Pareto efficiency tests
├── test_comparison_diagnostics.py      # Group E: Diagnostic matrix tests
├── test_comparison_feature_overlap.py  # Group F: Feature overlap & Jaccard tests
└── test_comparison_integration.py      # Groups G, H, I, J: Integration & real datasets
docs/
├── MODEL_COMPARISON_METHODS.md         # Statistical methodologies
└── COMPARISON_PIPELINE.md              # Pipeline architecture & Milestone 6 handoff
```

---

## 3. Data Contract: `ExperimentComparisonResult`

The comparison output is an immutable, JSON-serializable dataclass:

```python
@dataclass
class ExperimentComparisonResult:
    comparison_id: str                      # Deterministic ID: comp_{sha256[:12]}
    dataset_id: str                         # Dataset identifier
    dataset_fingerprint: str                # Dataset content hash
    split_fingerprint: str                  # Split partition hash
    target_name: str                        # Target variable name
    target_unit: str                        # Target physical unit (e.g. USD, Wh, Years)
    compared_model_ids: List[str]           # Sorted list of evaluated model IDs
    referenced_feature_selection_ids: Dict[str, Optional[str]]
    referenced_diagnostic_suite_ids: Dict[str, Optional[str]]
    evaluation_scope: str = "validation"    # Strictly development-stage validation
    ranking_policy: str                     # e.g. "val_rmse_asc", "parsimony_asc"
    leaderboard: List[Dict[str, Any]]       # Full leaderboard table
    model_complexity: Dict[str, Dict[str, Any]] # Parameters, rank, condition number
    normalization_config: Dict[str, Any]    # Normalization method & dimensions
    radar_scores: Dict[str, Dict[str, float]] # Model ID -> Dimension -> Score [0, 1]
    pareto_model_ids: List[str]             # List of non-dominated model IDs
    diagnostic_matrix: List[Dict[str, Any]] # Side-by-side diagnostic test outcomes
    feature_overlap_matrix: Dict[str, Any]  # Binary matrix, Jaccard, frequencies
    selected_candidate_model_id: Optional[str] # Chosen development candidate
    candidate_selection_rationale: Optional[str] # Modeler justification text
    warnings: List[str]                     # Non-blocking warnings
    methodological_limitations: List[str]   # Explicit scientific caveats
    timestamp: str                          # Creation timestamp (ISO format)
    configuration: Dict[str, Any]           # Full comparison hyperparameter dict
    reproducibility_hash: str               # Cryptographic audit hash
```

### 3.1 Numeric Serialization Robustness
All floating-point metrics, NumPy arrays, pandas DataFrames/Series, and non-finite quantities (`NaN`, `inf`) are safely converted using `safe_serialize_numeric()` into valid JSON primitives. Non-finite values are encoded as `None` (`null` in JSON) to comply with RFC 8259.

---

## 4. Execution Workflow

1. **Step 1: Session State Inspection**:
   Validates active dataset and partition. Retrieves all models from `RegressionExperimentTracker` matching `split_fingerprint` and `dataset_fingerprint`.
2. **Step 2: Candidate Selection & Compatibility Guard**:
   User selects 2–5 candidate models. `validate_model_compatibility()` verifies target identity, dataset hash, split hash, and sample counts.
3. **Step 3: Multi-Objective Trade-Off & Pareto Analysis**:
   `compute_pareto_frontier()` evaluates strict dominance ($p$ vs. $\text{RMSE}_{\text{val}}$).
4. **Step 4: Leaderboard Construction**:
   `build_comparison_leaderboard()` orders models according to the selected validation-based policy. Test metrics are strictly excluded.
5. **Step 5: Multi-Metric Normalization & Radar Plot**:
   `compute_radar_scores()` scales metrics onto $[0, 1]$. `create_radar_chart()` renders the interactive Plotly radar chart.
6. **Step 6: Diagnostic Matrix Aggregation**:
   `build_diagnostic_comparison_matrix()` extracts test statistics, $p$-values, and decisions from associated `DiagnosticSuiteResult` records.
7. **Step 7: Original Feature Overlap Analysis**:
   Builds binary inclusion table, pairwise Jaccard similarity heatmap, and candidate selection frequency chart.
8. **Step 8: Candidate Selection & Audit Trail**:
   Modeler selects preferred candidate, records justification, and commits comparison to `ComparisonExperimentTracker`.
9. **Step 9: Milestone 6 Handoff**:
   Exportable JSON payload enables Milestone 6 report generation scripts to reconstitute complete comparison tables and figures without re-running OLS estimation.

---

## 5. Invalidation Protocol for Stale Comparisons

When a user switches datasets or re-partitions data in Milestone 1:
- A new `split_fingerprint` is generated.
- `ComparisonExperimentTracker.get_valid_comparisons()` filters stored comparisons by matching `split_fingerprint` and `dataset_fingerprint`.
- Comparisons belonging to older or altered splits are automatically invalidated and omitted from the active view.

---

## 6. Zero-Leakage & Test Set Protection Protocol

1. **No Test Ranking**: `RankingPolicy` excludes Test RMSE, Test MAE, and Test $R^2$.
2. **No Test Triggering**: `run_model_comparison()` never invokes statsmodels fitting on test partitions or calls `finalize_test_evaluation()`.
3. **Immutability**: Underlying `OLSModelResult` instances remain completely unmodified by the comparison pipeline.
4. **Final Test Lock**: Final test evaluation remains exclusively under the controlled Milestone 3 Test Lock protocol.
