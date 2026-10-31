# Diagnostic Pipeline Architecture & Execution Workflow

This document describes the architectural flow, component lifecycle, isolation guarantees, and state transitions of the **Statistical Assumption Diagnostics Pipeline** (Milestone 4).

---

## 1. End-to-End Pipeline Workflow

The educational regression journey follows a strict unidirectional sequence:

```
[1. Dataset Library]
       │
       ▼ (DatasetBundle & SplitResult with dataset_fingerprint, split_fingerprint)
[2. EDA Laboratory]
       │
       ▼ (Feature correlations & bivariate patterns)
[3. Feature Selection Laboratory]
       │
       ▼ (FeatureSelectionResult with selected_original_features)
[4. Regression Modeling Laboratory]
       │
       ▼ (OLSModelResult fitted strictly on X_train via statsmodels)
[5. Model Evaluation Laboratory]
       │
       ▼ (Loss metrics MSE/RMSE/MAE/R², Actual vs. Predicted, Test Lock)
[6. Assumption Diagnostics Laboratory] (Milestone 4)
       │
       ▼ (DiagnosticResult & DiagnosticSuiteResult)
[7. Comparative Leaderboards & Reporting] (Milestones 5–6)
```

---

## 2. Component Architecture & Data Models

### 2.1 Inputs to the Diagnostic Engine
The diagnostics engine requires:
- **`OLSModelResult`** (from Milestone 3):
  - `model_id`: Unique deterministic model hash.
  - `fitted_model`: Intact `statsmodels.regression.linear_model.RegressionResultsWrapper` reference.
  - `train_predictions`: Vector of in-sample predictions $\hat{y}_{\text{train}}$.
  - `design_matrix_metadata`: Information on columns, matrix rank, collinear columns, and condition number.
  - `is_valid_for_inference`: Boolean indicating whether model has full column rank.
  - `dataset_fingerprint` & `split_fingerprint`: Provenance hashes for cache validation.
- **`SplitResult`**:
  - `X_train`, `y_train`: Training data used to compute in-sample residuals $e_i = y_i - \hat{y}_i$.
  - `timestamps_train`: Chronological series (used exclusively for temporal datasets like Appliances Energy).
  - `strategy`: Active splitting strategy (`random`, `temporal`, `group`).

### 2.2 Output Artifacts
- **`DiagnosticResult`**: Individual immutable test result.
- **`DiagnosticSuiteResult`**: Aggregate collection representing the complete assumption profile of the model at a specific significance level $\alpha$.

---

## 3. Data Leakage Isolation Guarantees

> [!IMPORTANT]
> **Strict Training-Only Data Scope**: Assumption diagnostics evaluate the statistical properties of the model's **in-sample training residuals** ($e_i = y_{\text{train}, i} - \hat{y}_{\text{train}, i}$).

1. **No Test Partition Access**: Diagnostic calculations never touch `X_test`, `y_test`, or test residuals.
2. **No Model Retraining**: Running diagnostics does not alter the coefficients, covariance matrix, or specification of the fitted model.
3. **No Iterative Specification Search**: Unlike feature selection, assumption diagnostics are descriptive and inferential evaluations, not an optimization loop.
4. **Controlled Refit Separation**: If a user previously finalized a model on the combined $(X_{\text{train}} \cup X_{\text{val}})$ sample, Milestone 4 explicitly diagnoses the development-stage model fitted on $X_{\text{train}}$ to preserve clear baseline comparisons.

---

## 4. Active Model Validation & Stale State Prevention

Before executing any diagnostic test, [`validate_active_model_for_diagnostics()`](file:///d:/碩一/down_temp/classification/src/diagnostics/runner.py) executes six verification checks:

1. **Model Existence**: Verifies that an active `OLSModelResult` is loaded in the tracker.
2. **Dataset Identity**: Ensures `model.dataset_id == current_meta.dataset_id`.
3. **Dataset Fingerprint**: Verifies that the dataset content hash matches `current_meta.dataset_fingerprint`.
4. **Split Fingerprint**: Verifies that the partition hash matches `split_res.split_fingerprint`.
5. **statsmodels Wrapper Availability**: Verifies `model.fitted_model` is accessible for residual retrieval.
6. **Residual Vector Alignment**: Verifies that `len(model.train_predictions) == len(split_res.y_train)`.

If any check fails, diagnostic execution is halted and the UI directs the user back to the appropriate prior laboratory.

---

## 5. Rank Deficiency & Prediction-Only Mode Handling

When an OLS model was trained with `allow_rank_deficient=True`:
- **Inference Status**: `is_valid_for_inference = False`.
- **Rank & Condition Number**: Diagnostics display matrix rank, condition number, and linearly dependent columns.
- **Descriptive Plots**: Graphical plots (Residuals vs. Fitted, Scale-Location, Histogram) remain available for exploratory inspection.
- **Formal Hypothesis Tests**: Ramsey RESET and Breusch-Pagan are marked `NOT_APPLICABLE` with explicit educational warnings explaining that non-invertible design matrices cannot form well-defined $F$ or $\chi^2$ reference distributions.

---

## 6. Handoff to Milestone 5 and Milestone 6

- **Milestone 5 (Model Comparison)**: Diagnostic summary tables from `DiagnosticSuiteResult.to_summary_table()` can be compared across multiple models side-by-side to assess which specification best satisfies Gauss-Markov criteria.
- **Milestone 6 (Report Generator)**: `DiagnosticSuiteResult` JSON exports provide structured tables, test statistics, and educational interpretations for automated academic report compilation.
