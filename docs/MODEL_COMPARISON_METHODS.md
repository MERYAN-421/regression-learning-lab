# Model Comparison & Multi-Objective Benchmarking Methods (Milestone 5)

## 1. Methodological Philosophy & Core Principles

The **Experiment & Model Comparison Laboratory** enables side-by-side benchmarking of Ordinary Least Squares (OLS) regression models developed across Milestones 1–4. The module adheres to three non-negotiable econometric and machine learning evaluation principles:

1. **Rule 1 — Tracker & Result Reuse**: Reuses previously fitted models (`RegressionExperimentTracker`), feature selection results (`FeatureSelectionExperimentTracker`), and assumption diagnostic suites (`DiagnosticSuiteResult`). It does not re-fit models or duplicate existing tracking storage.
2. **Rule 2 — Strict Test Set Isolation**: Development-stage model ranking, sorting, highlighting, and candidate selection are strictly governed by **Validation partition metrics**. Final Test RMSE, MAE, and $R^2$ are never revealed or used for candidate ranking, preserving the Milestone 3 controlled Final Test Lock.
3. **Rule 3 — Non-Additive Diagnostic Interpretation**: Statistical assumption tests (Ramsey RESET, Breusch-Pagan, Jarque-Bera, Durbin-Watson, Ljung-Box, VIF, Condition Number) evaluate distinct mathematical null hypotheses with differing asymptotic behaviors and sample-size sensitivities. They are **never aggregated into an artificial composite score, compliance percentage, or assumption pass count**.

---

## 2. Model Compatibility Engine

To guarantee fair, statistically sound model comparisons, candidate models must satisfy rigorous partition and target compatibility checks before entering the leaderboard.

### 2.1 Mandatory Compatibility Requirements
Models are verified against the active environment and against each other:
- **Dataset Identity**: Both `dataset_id` and content `dataset_fingerprint` must match identically.
- **Split Partition Identity**: The `split_fingerprint` must match identically, ensuring that training, validation, and test observations are perfectly aligned row-by-row.
- **Target Variable & Units**: The predicted target column must match identically. If target measurement units differ, an explicit warning is recorded.
- **Observation Counts**: Sample counts ($n_{\text{train}}$ and $n_{\text{val}}$) must be identical across all compared models.

### 2.2 Legitimate Comparison Dimensions
The comparison laboratory explicitly accommodates valid modeling variations:
- Feature selection algorithms (Pearson, ANOVA F-test, Mutual Information, RFE, Forward SFS, Backward SFS, Lasso).
- Requested feature count ($K$) and OLS parameter count ($p$).
- Numeric scaling (unscaled vs. StandardScaler).
- Rank-deficient vs. full-rank design matrices (transparently badged).

---

## 3. Validation-Based Model Leaderboard

The default model ranking policy is **Validation RMSE Ascending** (lower out-of-sample error is better).

### 3.1 Ranking Policies
Modelers can sort the leaderboard using development-stage criteria:
- `val_rmse_asc`: Validation Root Mean Squared Error (Ascending — Default)
- `val_mae_asc`: Validation Mean Absolute Error (Ascending)
- `val_r2_desc`: Validation Coefficient of Determination ($R^2$) (Descending)
- `parsimony_asc`: Total OLS parameter count ($p$) (Ascending)
- `gen_gap_asc`: Generalization Gap magnitude ($|\text{Val RMSE} - \text{Train RMSE}|$) (Ascending)

### 3.2 Metric Definitions & Educational Interpretations
- **Validation RMSE**: Out-of-sample root mean squared error calculated on the held-out validation partition:
  $$\text{RMSE}_{\text{val}} = \sqrt{\frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} (y_i - \hat{y}_i)^2}$$
- **Generalization Gap**:
  $$\text{Gap} = \text{RMSE}_{\text{val}} - \text{RMSE}_{\text{train}}$$
  *Pedagogical Warning*: A small or near-zero generalization gap does *not* imply a superior model. A severely underfitted model (e.g. constant mean or 1 predictor) will produce high training error and high validation error with a tiny gap.
- **Information Criteria (AIC & BIC)**:
  $$\text{AIC} = 2p - 2\ln(\hat{L}), \quad \text{BIC} = p \ln(n) - 2\ln(\hat{L})$$
  *Comparability Restriction*: AIC and BIC are only valid for relative comparison when candidate models are fitted on the *identical training observations* and use the *identical likelihood function and response variable*. They represent in-sample penalized likelihood, not out-of-sample prediction error.
- **Inference Status**:
  - `✅ Full Rank`: $(X^T X)$ is non-singular with strictly positive residual degrees of freedom ($\text{df}_{\text{resid}} > 0$). Standard errors, $t$-tests, and $F$-tests are statistically valid.
  - `⚠️ Prediction Only`: Design matrix is rank deficient or collinear. Individual parameter $p$-values are mathematically unidentifiable, but out-of-sample point predictions remain comparable.

---

## 4. Multi-Metric Radar Chart Normalization

The radar chart provides an interactive Plotly visualization mapping candidate models across five normalized dimensions:
1. **Validation RMSE Score** (lower is better $\rightarrow 1.0$)
2. **Validation MAE Score** (lower is better $\rightarrow 1.0$)
3. **Validation $R^2$ Score** (higher is better $\rightarrow 1.0$)
4. **Parsimony Score** (fewer parameters $\rightarrow 1.0$)
5. **Generalization Gap Score** (smaller magnitude $\rightarrow 1.0$)

### 4.1 Normalization Formulas
For lower-is-better metrics ($x$):
$$S_i = \frac{\max(x) - x_i}{\max(x) - \min(x)}$$

For higher-is-better metrics ($x$):
$$S_i = \frac{x_i - \min(x)}{\max(x) - \min(x)}$$

### 4.2 Boundary & Robustness Handling
- **Constant Values**: If all compared models have identical values ($\max(x) - \min(x) \le 10^{-12}$), all models receive a neutral score of $0.5$ to avoid division by zero.
- **Single Model**: If only one model is evaluated, it receives a score of $1.0$.
- **Negative $R^2$**: Mapped smoothly into $[0, 1]$ relative to the minimum observed in the candidate set.
- **Non-Finite (NaN / Inf)**: Assigned $0.0$ (floor) safely.

### 4.3 Interpretation Limits
- **Relative to Candidate Set**: Radar scores are strictly relative to the currently compared candidate models. Adding or removing a candidate rescales the min-max bounds.
- **No Area Equivalence**: Total polygon area on a radar chart is mathematically dependent on the arbitrary rotational order of axes and must never be interpreted as an absolute model quality score.
- **Exclusion of Diagnostics**: Assumption test $p$-values are excluded from radar charts to satisfy Rule 3.

---

## 5. Prediction-Complexity Trade-Off (Pareto Frontier)

The laboratory visualizes Occam's razor through an interactive two-objective optimization scatter plot:
- **Objective 1 (Complexity)**: Minimize OLS parameter count ($p = \text{rank}(X)$).
- **Objective 2 (Error)**: Minimize Validation RMSE ($\text{RMSE}_{\text{val}}$).

### 5.1 Pareto Dominance Definition
Model $A$ strictly dominates Model $B$ ($A \succ B$) if and only if:
$$p_A \le p_B \quad \text{AND} \quad \text{RMSE}_A \le \text{RMSE}_B$$
$$\text{AND} \quad (p_A < p_B \quad \text{OR} \quad \text{RMSE}_A < \text{RMSE}_B)$$

A candidate model is on the **Pareto Frontier** (⭐) if no other model in the candidate set dominates it.

### 5.2 Educational Caveats
1. **No Automatic Optimum**: Being on the Pareto Frontier does not mean a model is "the best"; it merely means it represents an efficient trade-off where lower error cannot be achieved without increasing parameter count.
2. **Context Dependency**: Choosing between a 3-parameter model with RMSE = 45.2 and a 12-parameter model with RMSE = 42.1 depends on interpretability requirements, diagnostic compliance, and domain costs.
3. **Candidate Bound**: Pareto efficiency is strictly bounded by the evaluated candidate set.

---

## 6. Statistical Diagnostic Comparison Matrix

Integrates Milestone 4 `DiagnosticSuiteResult` records to present assumption test outcomes side-by-side:
- **Linearity**: Ramsey RESET ($F$-statistic and $p$-value).
- **Independence**: Durbin-Watson statistic and Ljung-Box Portmanteau test ($Q$-statistic and $p$-value).
- **Homoscedasticity**: Breusch-Pagan Lagrange Multiplier test ($\text{LM}$ statistic and $p$-value).
- **Normality**: Jarque-Bera test ($\text{JB}$ statistic and $p$-value).
- **Multicollinearity**: Maximum Variance Inflation Factor ($\text{VIF}_{\max}$) and Design Matrix Condition Number ($\kappa(X)$).

### 6.1 Strict Enforcement of Rule 3
- Missing diagnostic suites are labeled as `"Not Run"` rather than treated as favorable.
- Inapplicable diagnostics (e.g. Durbin-Watson on cross-sectional random splits) are labeled `"Not Applicable"`.
- No composite quality score, pass count, or compliance percentage is computed.
- High $p$-values indicate failure to reject $H_0$, not confirmation that the regression specification is true or causal.

---

## 7. Original Feature Selection Overlap

Compares original feature subsets selected across candidate models:
1. **Binary Feature Inclusion Matrix**: Rows = Unique original features, Columns = Model IDs, Values = 1 (Selected) / 0 (Not Selected).
2. **Pairwise Jaccard Similarity**:
   $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
   Boundary cases: $J(\emptyset, \emptyset) = 1.0$, $J(A, \emptyset) = 0.0$.
3. **Candidate Selection Frequency**: Frequency count of how many models retained each original predictor.

---

## 8. Candidate Model Selection & Milestone 6 Handoff

The laboratory allows modelers to register a preferred candidate model:
- Modeler selects the preferred model ID.
- Modeler tags primary justification criteria (e.g. "Lowest Validation RMSE", "Parsimonious Occam's Razor", "Clean Diagnostics").
- Modeler enters an educational justification text.
- The selection is committed to `ComparisonExperimentTracker` and stored in `ExperimentComparisonResult`.
- **Zero Test Leakage**: Candidate selection does *not* trigger Test evaluation. Test evaluation remains safely locked under Milestone 3 Final Test Lock.
