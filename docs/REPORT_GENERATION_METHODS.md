# Academic Report Generation Laboratory: Econometric & Statistical Methods

## 1. Overview & Architectural Philosophy

The **Academic Report Generation Laboratory (Milestone 6)** transforms empirical regression models, statistical diagnostic results, and multi-model benchmark comparisons into formal, publication-ready academic reports.

In adherence to scientific integrity principles, the reporting laboratory operates strictly as a **read-only synthesis engine**. It enforces six foundational non-negotiable rules:
1. **Never Fabricate Evidence**: Every numerical claim (point estimates, standard errors, test statistics, $p$-values, confidence intervals, evaluation metrics) is derived deterministically from an explicit, verified experimental artifact.
2. **Reuse Existing Results & Trackers**: The laboratory reads directly from `RegressionExperimentTracker`, `ComparisonExperimentTracker`, `FeatureSelectionExperimentTracker`, and diagnostic suites without recreating data structures.
3. **No Automatic Retraining**: Opening or rendering a report will never trigger feature selection, model fitting, diagnostic execution, or leaderboard recomputation.
4. **Preserve Final Test Lock**: Development reports use strictly Training and Validation evidence. The Test Set remains sealed; Final Test metrics appear only in a dedicated, explicitly authorized appendix if an existing, compatible `FinalTestEvaluationResult` artifact exists.
5. **Defensible Statistical Interpretation**: Rule-based econometric interpretations prevent common empirical fallacies (e.g., claiming high $R^2$ proves causality, or $p \ge \alpha$ proves $H_0$).
6. **Verified Reproducibility Claims**: Exported standalone reproduction packages clearly demarcate the exact reproducibility boundaries (e.g., exact model-fitting replay vs. full pipeline).

---

## 2. Academic Report Templates

The laboratory provides three pre-configured, academically structured report templates that share a unified evidence validation and interpretation pipeline:

### Template A — Full Academic Report (9 Sections)
The primary comprehensive academic monograph for research publications, theses, and coursework:
1. **Executive Summary & Research Objectives**: Problem formulation, dataset identity, target variable, candidate model summary, validation performance, and key caveats.
2. **Dataset Description & Data Preparation**: Source provenance, partition sample counts ($N_{train}, N_{val}, N_{test}$), splitting strategy (Random, Temporal, or Group), and immutable dataset/split fingerprints.
3. **Exploratory Data Analysis & Feature Selection**: Pre-modeling insights, selection algorithms ($K$-best, Forward/Backward Stepwise, LASSO, Ridge, ElasticNet, Exhaustive), CV scores, and encoded feature dimensions.
4. **OLS Regression Methodology**: Mathematical specification ($y = X\beta + \varepsilon$), design matrix rank, column dimensions, intercept policy, numerical scaling, and categorical reference-level encoding.
5. **Parameter Estimates & Goodness-of-Fit**: Full coefficient table ($\hat{\beta}, \text{SE}, t, p, \text{95\% CI}$), Goodness-of-Fit ($R^2, R^2_{adj}, F$-stat, $p$-value), generalization gap ($RMSE_{val} - RMSE_{train}$), and prediction-only inference guards.
6. **Statistical Assumption Diagnostics**: Rigorous evaluation of the five classical Gauss-Markov and normal error assumptions:
   - Specification / Linearity (Ramsey RESET)
   - Disturbance Independence (Durbin-Watson, Ljung-Box)
   - Homoscedasticity (Breusch-Pagan, White test)
   - Residual Normality (Jarque-Bera, Shapiro-Wilk)
   - Multicollinearity (VIF, Condition Number)
7. **Multi-Model Comparison & Candidate Selection**: Validation-based leaderboard, Pareto frontier (accuracy vs. parsimony), feature overlap matrix, relative radar scores, and the user's recorded candidate rationale.
8. **Discussion, Limitations & Conclusion**: Econometric distinction between predictive association and causal inference, separated cleanly from student/author reflections.
9. **Reproducibility, References & Appendix**: Full cryptographic audit manifest, environment dependencies, execution instructions, and citations.

### Template B — Executive Summary
A streamlined brief tailored for research briefings and decision memos:
- Executive Summary & Research Question
- Dataset Description & Fingerprints
- Parameter Estimates & Out-of-Sample Validation Performance
- Benchmark Comparison & Pareto Efficiency
- Discussion, Key Practical Limitations & Evidence Provenance

### Template C — Statistical Diagnostics Report
A specialized econometric audit document focused on Gauss-Markov integrity and model identification:
- Model Specification & Design Matrix Rank
- Classical Assumption Diagnostics (Hypotheses, Test Statistics, $p$-values, Decisions)
- Visual Diagnostic Suite (Residuals vs. Fitted, Normal Q-Q, VIF Profile)
- Discussion of Violations, Exogeneity Caveats & Remedial Actions

---

## 3. Evidence Manifest & Provenance Architecture

To guarantee that no hallucinated or unverified numbers ever appear in published reports, the system compiles an **`EvidenceManifest`** prior to generating any textual or tabular report sections:

```
+------------------------------------+
| ResolvedReportSources              |
| - Model, Split, Metadata, Diag     |
+-----------------+------------------+
                  |
                  v
+-----------------+------------------+
| EvidenceManifestBuilder            |
| - Builds EvidenceRecord for each   |
|   claim (metric, stat, p-val, etc.)|
+-----------------+------------------+
                  |
                  v
+-----------------+------------------+
| ArtifactCompatibilityValidator     |
| - Validates fingerprints, IDs,     |
|   target units, partition alignment|
+-----------------+------------------+
                  |
                  v
+-----------------+------------------+
| ReadinessEngine                    |
| - READY / LIMITATIONS / BLOCKED    |
+-----------------+------------------+
                  |
                  v
+-----------------+------------------+
| AcademicSectionBuilder & Exporters |
+------------------------------------+
```

### Evidence Status Definitions
Each reported numerical claim is indexed as an `EvidenceRecord` with one of five explicit statuses:
- **`verified`**: The artifact exists, matches all dataset/split fingerprints, and contains a validated numerical observation.
- **`missing`**: The referenced field or optional artifact is unavailable. The report explicitly displays `Unavailable` or `Not Run`; no default is ever guessed.
- **`incompatible`**: The artifact belongs to a different dataset, split, model, or unit. Generation is blocked or the claim is segregated.
- **`not_applicable`**: The statistical test cannot legitimately be evaluated for this data structure (e.g., sequential autocorrelation tests on grouped cross-sectional panels).
- **`unverified`**: Provenance metadata is insufficient to verify authenticity.

---

## 4. Defensible Statistical Interpretation Rules

The reporting engine adheres strictly to modern econometric standards via `src/reporting/interpretation.py`:

1. **Hypothesis Test Decisions**:
   - $p < \alpha$: *"Reject $H_0$ at the specified significance level ($\alpha$)."*
   - $p \ge \alpha$: *"Fail to reject $H_0$ at the specified significance level ($\alpha$)."*
   - **Prohibited**: Claiming that $p \ge \alpha$ "proves $H_0$ is true" or confirms the absence of misspecification.
2. **Gauss-Markov vs. Normality**:
   - Error normality is **not required** for OLS estimators to be BLUE (Best Linear Unbiased Estimator). Normality is required only for exact finite-sample $t$ and $F$ inference.
3. **Exogeneity**:
   - Residual diagnostics test orthogonality between residuals and regressors ($\sum e_i X_i = 0$), which holds by mathematical construction of OLS. They **cannot test or prove strict exogeneity** ($E[\varepsilon|X] = 0$).
4. **Multicollinearity & VIF**:
   - High VIF (e.g., $> 10$) inflates standard errors and reduces inferential precision, but **does not bias OLS coefficients** and does not invalidate out-of-sample prediction accuracy.
5. **Post-Selection Inference**:
   - Models fitted after algorithmic feature selection (Forward, LASSO, etc.) suffer from selection-induced overfitting; conventional $p$-values and confidence intervals must be interpreted with documented caution.
6. **Prediction-Only vs. Inferential Models**:
   - For rank-deficient models ($\text{rank}(X) < p$), conventional coefficient standard errors, $t$-statistics, and $p$-values are statistically undefined. The report suppresses significance claims and labels parameters as *Prediction Only*.
7. **No Synthetic "Quality Scores"**:
   - Diagnostic tests are never tallied into an unscientific "Pass Count (e.g., 4/5)" or composite index. Each diagnostic addresses distinct mathematical assumptions.

---

## 5. Export Architecture & Reproducibility Package

The laboratory offers multi-format export via `ReportExportService`:
- **`report.md`**: Fully formatted GitHub-flavored Markdown document with LaTeX math expressions, clean ASCII tables, and complete evidence citations.
- **`report.html`**: Completely standalone, responsive HTML file embedding styled CSS typography, client-side MathJax equation formatting, interactive Plotly charts, and strict XSS HTML-escaping of author notes.
- **`manifest.json`**: Machine-readable JSON export of all `EvidenceRecord` entries, source artifact IDs, cryptographic hashes, and environment metadata.
- **`reproduction_package.zip`**: Self-contained archive containing:
  - `report.md`
  - `report.html`
  - `manifest.json`
  - `reproduce.py` (Syntactically valid Python replay script utilizing project APIs)
  - `requirements.txt`
  - `README_REPRODUCTION.md` (Explicit reproducibility limitations and execution guide)
