# Regression Learning Lab — Comprehensive User Guide

**Target Audience**: Students, Instructors, Applied Econometricians, and Data Science Practitioners  
**Application Entrypoint**: `app.py`  
**Primary Tutorial Dataset**: Student Performance (Mathematics) (`student_performance`)  

---

## 1. Introduction & Pedagogical Philosophy

Regression Learning Lab is an interactive educational platform designed to bridge the gap between theoretical econometrics and practical machine learning. It guides users through an end-to-end multiple linear regression workflow while rigorously enforcing:
- **Leakage Prevention**: Data preprocessing and cross-validation are strictly isolated to training folds.
- **Full-Rank Safeguards**: Predictor collinearity and dummy-variable traps are detected before OLS parameter estimation.
- **Statistical Assumption Audits**: Assumptions (linearity, homoscedasticity, normality, independence, multicollinearity) are checked using formal statistical tests with applicability guards.
- **Multi-Model Comparison**: Models are ranked on out-of-sample validation error, visualised via Radar profiles and Pareto efficiency frontiers.
- **Evidence-Manifest Provenance**: Every metric in generated academic reports is tied to cryptographic data and split fingerprints.
- **Final Test Lock**: The holdout Test set remains strictly sealed until the user explicitly authorizes final unbiased evaluation.

---

## 2. Nine-Page Step-by-Step Workflow Tutorial

This walkthrough uses the **Student Performance (Mathematics)** dataset (~395 observations) to illustrate a complete research cycle.

### Page 1: Home & Platform Overview
1. Launch the platform locally with `streamlit run app.py` or visit the public deployment URL.
2. The Home page outlines the 9-stage econometric pipeline, system architecture, and provides links to documentation.
3. Review the **Guided Learning Workflow** vs. **Free Exploration Mode** descriptions.

### Page 2: Dataset Library & Partitioning
1. In the sidebar, select **Dataset Library**.
2. From the dataset selection card, choose **Student Performance (Mathematics)**.
3. Review the metadata:
   - **Target**: `G3` (Final Mathematics Grade, 0–20 scale).
   - **Predictors**: 32 features (demographics, family background, study habits, school attendance).
4. Configure the Split Strategy:
   - Strategy: **Random Split** (suitable for cross-sectional independent student surveys).
   - Split Ratios: 70% Train, 15% Validation, 15% Test.
   - Random Seed: `42`.
5. Click **Create & Lock Partition**.
6. Note the computed **Dataset Fingerprint** and **Split Fingerprint**. These SHA-256 hashes guarantee reproducibility across the entire session.

### Page 3: Exploratory Data Analysis (EDA)
1. Navigate to **Data Exploration**.
2. **Data Partition Scope**: Ensure the scope selector is set to **Training Partition Only** (guaranteeing that validation and test distributions do not leak into exploratory decisions).
3. **Target Distribution**: Examine the histogram and KDE for `G3`. Note the concentration around 10–15 and the bimodal cluster at zero (students who missed the final exam).
4. **Predictor Correlations**: Review the Pearson correlation heatmap. Notice strong positive associations between `G1` (first period grade), `G2` (second period grade), and `G3`.
5. **Categorical Features**: Inspect box plots of `G3` across categories such as `school`, `sex`, `address`, and `higher` (aspiration for higher education).

### Page 4: Feature Selection Laboratory
1. Navigate to **Feature Selection**.
2. The application provides eight original-feature-aware selection methods:
   - *Baseline*: All 32 original features.
   - *Filter*: Pearson Correlation, F-test, Mutual Information.
   - *Wrapper*: Recursive Feature Elimination (RFE), Forward SFS, Backward SFS.
   - *Embedded*: Lasso (L1 Regularization).
3. **Train Model 1 (Baseline)**:
   - Select **All Features (Baseline)**.
   - Click **Run Feature Selection Experiment**.
   - Note the resulting experiment ID (`exp_baseline_...`).
4. **Train Model 2 (Parsimonious Filter)**:
   - Select **Pearson Correlation**.
   - Set Target Number of Original Features ($K$) = `5`.
   - CV Strategy: 5-Fold Cross-Validation.
   - Click **Run Feature Selection Experiment**.
   - Observe the selected original predictors (e.g., `G2`, `G1`, `failures`, `higher`, `age`).
   - The original-feature mapper automatically groups one-hot dummy levels together so categorical predictors are never partially fragmented.

### Page 5: Regression Modeling
1. Navigate to **Regression Model**.
2. **Select Feature Set**: Choose the experiment corresponding to the Pearson $K=5$ selection.
3. **Model Options**:
   - Fit Intercept: **Yes** (required for unbiased OLS residuals).
   - Standardize Numeric Predictors: **Yes** (aids coefficient interpretability).
   - Reference Category: Automatically drops the first categorical level to prevent the dummy variable trap.
4. Click **Fit Ordinary Least Squares Model**.
5. Inspect the parameter estimation table:
   - Unstandardized coefficients ($\beta_j$), standard errors, $t$-statistics, and $p$-values.
   - Note the condition number ($\kappa$). The platform verifies full rank ($rank(X) = p$); if rank-deficiency occurs, inference is disabled and prediction-only mode is engaged.

### Page 6: Model Evaluation & Final Test Lock
1. Navigate to **Model Evaluation**.
2. Select your trained OLS model.
3. Review out-of-sample metrics:
   - **Training Metrics**: $R^2$, Adjusted $R^2$, RMSE, MAE.
   - **Validation Metrics**: Validation RMSE, Validation $R^2$.
   - **Generalization Gap**: $|\text{Train RMSE} - \text{Val RMSE}|$.
4. Inspect evaluation plots: Residuals vs. Fitted Values, Actual vs. Predicted scatter plot.
5. **Final Test Lock**:
   - The holdout Test set is sealed with a padlock banner.
   - *Pedagogical Warning*: Never unlock the Test set while iterating on model specifications or tuning hyperparameters!
   - (Optional final step: once all candidate models are evaluated and frozen, check the authorization box to reveal final unbiased test performance).

### Page 7: Statistical Assumption Diagnostics
1. Navigate to **Assumption Diagnostics**.
2. Select the candidate OLS model.
3. The platform executes a battery of formal tests:
   - **Linearity**: Ramsey RESET test ($F$-statistic and $p$-value).
   - **Autocorrelation**: Durbin–Watson ($d$) and Ljung–Box ($Q$). *(Guarded: marked Not Applicable on random cross-sectional splits to prevent false interpretations!)*
   - **Homoscedasticity**: Breusch–Pagan test for constant error variance.
   - **Normality of Residuals**: Jarque–Bera test and Q-Q Plot.
   - **Multicollinearity**: Variance Inflation Factor (VIF) per predictor and matrix Condition Number.
   - **Influential Points**: Cook's Distance and leverage diagnostics.
4. Review the educational guidance cards for violations (e.g., heteroscedasticity-robust standard errors or non-linear polynomial expansions).

### Page 8: Experiment & Model Comparison
1. Navigate to **Experiment Comparison**.
2. Select both trained models: Model 1 (All Features) and Model 2 (Pearson $K=5$).
3. **Compatibility Check**: The engine confirms both models share identical dataset and split fingerprints.
4. **Leaderboard**: Models are ranked strictly by Validation RMSE.
5. **Radar Chart**: Inspect multi-dimensional trade-offs across validation fit, parsimony, stability, and diagnostic compliance.
6. **Pareto Frontier**: Identifies non-dominated models balancing predictive accuracy against model complexity ($K$).
7. **Candidate Selection**: Mark Model 2 as the preferred Candidate Model and document the pedagogical rationale.

### Page 9: Academic Report & Export
1. Navigate to **Learning Report**.
2. **Evidence Manifest**: The platform checks artifact provenance. All required evidence records (Dataset, Split, Model, Diagnostics, Comparison) must show status **VERIFIED**.
3. **Author Content**: Fill in the Student Reflection section (research question, practical significance, modeling limitations).
4. **Report Rendering**: Preview the comprehensive 9-section academic report.
5. **Export Formats**:
   - **Download Markdown (`.md`)**: Full formatted text suitable for publication.
   - **Download HTML (`.html`)**: Sanitized interactive report with embedded tables.
   - **Download Manifest (`.json`)**: Machine-readable cryptographic provenance audit trail.
   - **Download Reproduction Package (`.zip`)**: Standalone archive containing raw configurations, trained model metadata, and executable Python reproduction scripts.

---

## 3. Interpreting Statistical Results

### Association vs. Causation
Ordinary Least Squares measures conditional statistical expectations ($E[Y \mid X]$). A statistically significant coefficient (e.g., positive coefficient on `higher` [aspiration for higher education]) indicates a predictive association, not necessarily a causal intervention effect, due to potential unobserved confounders (e.g., student motivation, parental engagement).

### Multicollinearity & VIF
When predictors are correlated (e.g., `G1` and `G2`), standard errors inflate, making individual $t$-tests insignificant despite high overall $R^2$. Use feature selection or ridge regularization when VIF exceeds 5.0.

### Out-of-Sample Validation
Never judge models solely on training $R^2$, which monotonically increases as features are added. Always select candidate models based on minimum Validation RMSE or Pareto efficiency.
