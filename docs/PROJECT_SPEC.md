# Project Specification: Regression Learning Lab

## 1. Executive Summary & Vision
**Regression Learning Lab** is an interactive, web-based educational and diagnostic platform tailored for undergraduate and graduate students in Data Science, Statistics, and Machine Learning. The platform demystifies Multiple Linear Regression (MLR) by providing an exploratory, transparent, and mathematically rigorous environment.

Rather than treating regression as a pure predictive "black box", the lab emphasizes the statistical validity of parameter estimates, the Gauss-Markov theorem, diagnostic hypothesis testing, and the consequences of assumption violations using curated Kaggle regression datasets featuring 14 to 39 predictive variables.

---

## 2. Target Audience & Educational Goals
- **Target Audience**: Students taking Applied Statistics, Econometrics, or Machine Learning courses, as well as data practitioners seeking an intuitive understanding of regression diagnostics.
- **Key Pedagogical Outcomes**:
  1. Distinguish between correlation and true regression causation/conditioning.
  2. Master feature selection paradigms (Filter, Wrapper, Embedded) and understand selection stability.
  3. Formulate and interpret Ordinary Least Squares (OLS) models, coefficients, standard errors, and confidence intervals.
  4. Verify the **Five Classical Linear Regression Assumptions** using formal statistical tests and diagnostic residual plots.
  5. Identify influential observations and high-leverage data points (Cook's distance, leverage values).
  6. Synthesize empirical findings into reproducible academic laboratory reports.

---

## 3. System Architecture & Page Hierarchy
The application leverages Streamlit's official `st.navigation` framework, organized into three semantic clusters:

```text
Regression Learning Lab (app.py)
├── [Introduction]
│   └── Home (pages/home.py)
│       - Lab Overview & Scope
│       - Learning Objectives
│       - Guided Learning vs. Free Experiment Modes
│       - Workflow Flowchart & Module Guide
│       - Milestone Roadmap & Status
│
├── [Regression Workflow]
│   ├── 1. Dataset Library (pages/dataset.py)
│   │   - Curated Kaggle Datasets (14–39 features)
│   │   - Dataset Schema, Profiling & Metadata
│   │   - Train/Test Split Partitioning & Seed Control
│   │
│   ├── 2. Data Exploration (pages/exploration.py)
│   │   - Univariate Distributions (Histogram, KDE, Boxplot)
│   │   - Missing Value Detection & Imputation
│   │   - Correlation Heatmap (Pearson & Spearman)
│   │   - Target Skewness & Normalizing Transforms
│   │
│   ├── 3. Feature Selection (pages/feature_selection.py)
│   │   - Filter Methods (Pearson, ANOVA F-test, Mutual Information)
│   │   - Wrapper Methods (RFE, Forward SFS, Backward SFS)
│   │   - Embedded Methods (Lasso L1 Sparsity, ElasticNet)
│   │   - Multicollinearity Screening (Variance Inflation Factor - VIF)
│   │
│   ├── 4. Regression Model (pages/regression.py)
│   │   - Ordinary Least Squares (OLS) Model Fitting via statsmodels
│   │   - Econometric Parameter Table (Coefficients, SE, t-stats, p-values)
│   │   - Omnibus F-test & Global Significance
│   │   - Formatted LaTeX Regression Equation
│   │
│   ├── 5. Model Evaluation (pages/evaluation.py)
│   │   - Continuous Error Metrics (MSE, RMSE, MAE)
│   │   - Goodness-of-Fit (R², Adjusted R²)
│   │   - Actual vs. Predicted Diagnostics ($45^\circ$ line)
│   │   - Residual Distribution & Quantile Spread
│   │
│   └── 6. Assumption Diagnostics (pages/diagnostics.py)
│       - Assumption 1: Linearity (Ramsey RESET, Residuals vs. Fitted)
│       - Assumption 2: Independence (Durbin-Watson, Ljung-Box, ACF)
│       - Assumption 3: Homoscedasticity (Breusch-Pagan, White Test, Scale-Location)
│       - Assumption 4: Normality of Residuals (Jarque-Bera, Shapiro-Wilk, Q-Q Plot)
│       - Assumption 5: Multicollinearity (VIF Matrix, Matrix Condition Number)
│       - Influential Points (Cook's Distance, Hat Matrix Leverage)
│
└── [Results]
    ├── 7. Experiment Comparison (pages/comparison.py)
    │   - Multi-Model Performance Leaderboard
    │   - Bias-Variance Tradeoff Visualization (Train vs. Test Divergence)
    │   - Metric Radar Chart
    │   - Assumption Compliance Matrix
    │
    └── 8. Learning Report (pages/report.py)
        - Automated Synthesis of Findings
        - Executive Summary & Audit Log
        - Formatted Markdown / HTML / Python Reproduction Script Export
```

---

## 4. Technical Stack
- **Programming Language**: Python 3.11
- **User Interface Framework**: Streamlit (v1.35+) with official `st.navigation`
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning & Preprocessing**: scikit-learn
- **Statistical Modeling & Inference**: statsmodels, SciPy
- **Interactive Visualization**: Plotly, Matplotlib, Seaborn
- **Configuration & Persistence**: PyYAML
- **Automated Testing**: pytest

---

## 5. Milestone Roadmap
| Milestone | Phase | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Milestone 0** | **Project Foundation** | Environment setup, project architecture, 9-page skeleton, UI/UX design, documentation, testing, and Git/GitHub setup | ✅ **Current Phase** |
| **Milestone 1** | **Dataset & Preprocessing** | Kaggle regression dataset pipeline (14–39 features), data validation, caching, and partitioning | ⏳ Next Up |
| **Milestone 2** | **EDA & Feature Selection** | Univariate/bivariate visualizer, correlation heatmaps, VIF, Filter, Wrapper, and Lasso feature selection | ⏳ Scheduled |
| **Milestone 3** | **Regression & Evaluation** | statsmodels OLS integration, summary tables, LaTeX equation generator, MSE/RMSE/R² metrics | ⏳ Scheduled |
| **Milestone 4** | **Assumption Diagnostics** | Five Gauss-Markov diagnostic tests and visual plots, Cook's distance, leverage diagnostics | ⏳ Scheduled |
| **Milestone 5** | **Comparison & Reporting** | Experiment tracking, side-by-side radar charts, automated laboratory report generator | ⏳ Scheduled |
