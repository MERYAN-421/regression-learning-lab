# 📈 Regression Learning Lab

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Milestone-3%20Completed-success.svg)]()

> **Interactive Multiple Linear Regression & Statistical Diagnostics Platform**  
> An educational web platform designed for data science and statistics students to explore real-world regression modeling, feature selection, and the classical Gauss-Markov assumptions using curated Kaggle tabular datasets (14–39 features).

---

## 🌟 Key Highlights

- **Pedagogical Rigor**: Bridges the gap between predictive machine learning and econometric inference.
- **Two Learning Modalities**:
  - **Guided Learning**: Step-by-step sequential path through the 8-stage data science lifecycle.
  - **Free Experiment**: Open sandbox for custom feature subsets, regularization penalties, and hypothesis tests.
- **Natural 14–39 Usable Predictor Rule**: All benchmark datasets naturally possess 14 to 39 usable predictor variables without arbitrary column sub-setting.
- **Leakage-Free Splitting Engine**:
  - **Random Split**: Deterministic cross-sectional partitioning (Seed 42).
  - **Temporal Split**: Chronological ordering on sequential data (`appliances_energy`) to prevent forward-looking leakage.
  - **Group-Disjoint Split**: Partitioning by country clusters (`life_expectancy`) to evaluate generalization to unseen sovereign states.
- **Exploratory Data Analysis (EDA) Laboratory (Milestone 2)**:
  - Partition scope isolation (Training data default to prevent visual leakage).
  - Univariate distribution metrics (mean, std, median, skewness, kurtosis, IQR outlier flags).
  - Pearson & Spearman correlation heatmaps with constant feature exclusion.
  - Bivariate scatter plots with OLS trendlines and categorical target box plots.
- **Feature Selection Laboratory (Milestone 2)**:
  - 8 selection methods: Baseline, Pearson, F-test, Mutual Information, RFE, Forward SFS, Backward SFS, and Lasso.
  - **Structural Feature Mapping**: Introspects fitted transformer metadata, preventing prefix/substring collisions and handling underscores in category levels.
  - **Nested SFS Fold Isolation**: Imputer, encoder, and scaler fitted strictly per inner fold; dataset-appropriate splitting (`TimeSeriesSplit`, `GroupKFold`, `KFold`).
  - **Dataset-Appropriate Lasso**: Hyperparameter selection with fold-specific preprocessing, deterministic execution, and natural sparsity vs. exact-$K$ ranking distinction.
  - **Stable Experiment Identity**: 12-char SHA-256 hash incorporating all configuration parameters, ensuring distinct CV settings or alphas never collide.
- **Regression Modeling Laboratory (Milestone 3)**:
  - **OLS Design Matrix Contract**: Median imputation fitted on Train only, reference-category dummy encoding (`drop='first'`) to prevent dummy-variable trap, and single explicit constant.
  - **Matrix Rank & Near-Singularity Diagnostics**: Detects rank deficiency, collinear columns via QR decomposition with column pivoting, and flags high condition numbers (e.g. King County House Sales structural identity $\text{sqft\_living} \approx \text{sqft\_above} + \text{sqft\_basement}$).
  - **Econometric Parameter Estimation**: statsmodels OLS parameter estimation, standard errors, $t$-statistics, $p$-values, 95% confidence intervals, and $F$-tests.
  - **LaTeX Equation Rendering**: Interactive mathematical equation displays with actual fitted coefficients.
  - **Post-Selection Inference Caveat**: Transparently warns students when statistical inference follows data-driven feature selection.
- **Model Evaluation Laboratory (Milestone 3)**:
  - **Quantitative Metrics**: MSE, RMSE, MAE, $R^2$, and in-sample Adjusted $R^2$ (labeled distinctly from validation metrics).
  - **Generalization Gap**: Directly compares Training and Validation RMSE to quantify potential overfitting.
  - **Interactive Descriptive Visualizations**: Actual vs. Predicted scatter plot with $45^\circ$ line, Residuals vs. Fitted plot, and Residual Distribution histogram with normal curve overlay.
  - **Model Comparison Leaderboard**: Multi-model comparison across feature selection approaches under identical split partitions.
  - **Controlled Final Test Evaluation**: Strict test set isolation during development; explicit authorization workflow with documented Train+Val refit or Train-only policy.
- **Five Classical Gauss-Markov Diagnostics (Milestone 4)**:
  1. *Linearity* (Ramsey RESET & Residuals vs. Fitted)
  2. *Independence* (Durbin-Watson, Ljung-Box & ACF)
  3. *Homoscedasticity* (Breusch-Pagan, White's Test & Scale-Location)
  4. *Normality* (Jarque-Bera, Shapiro-Wilk & Normal Q-Q Plot)
  5. *Multicollinearity* (Variance Inflation Factor & Matrix Condition Number)

---

## 📂 Supported Benchmark Datasets

| Dataset | Domain | Samples | Raw Cols | Usable Predictors | Recommended Split | Strategy Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Appliances Energy** | Energy IoT | 19,735 | 29 | **27 features** | **Temporal** | Chronological 70/15/15 split on `date` prevents future data leakage. |
| **King County House Sales** | Real Estate | 21,613 | 21 | **18 features** | **Random** | Independent cross-sectional home sales with `id` and `date` preserved as metadata. |
| **Life Expectancy (WHO)** | Public Health | 2,938 | 22 | **20 features** | **Group (Country)** | Panel dataset partitioned disjointly across 193 nations; evaluates out-of-country fit. |
| **Student Performance** | Education | 395 | 33 | **32 features** | **Random** | Semicolon-delimited CSV with 17 categorical and 15 numeric attributes ($G3$ target). |

---

## 🏛️ Project Architecture

```text
regression-learning-lab/
│
├── app.py                     # Streamlit main entrypoint & navigation controller
├── requirements.txt           # Verified Python 3.11 dependencies
├── README.md                  # Project overview, installation, and roadmap
├── LICENSE                    # MIT Open Source License
├── .gitignore                 # Git ignore rules for caches, envs, and large data
│
├── .streamlit/
│   └── config.toml            # Streamlit theme (academic light) & server configuration
│
├── pages/                     # Multi-page application structure
│   ├── home.py                # Home: Objectives, learning modes & roadmap
│   ├── dataset.py             # Module 1: Functional Dataset Library & Splitting Engine
│   ├── exploration.py         # Module 2: Interactive EDA Laboratory
│   ├── feature_selection.py   # Module 3: Feature Selection Laboratory (8 algorithms)
│   ├── regression.py          # Module 4: Ordinary Least Squares (OLS) modeling
│   ├── evaluation.py          # Module 5: Metrics (MSE, RMSE, MAE, R², Adj R²)
│   ├── diagnostics.py         # Module 6: Five Gauss-Markov assumption tests
│   ├── comparison.py          # Module 7: Multi-model comparison & radar chart
│   └── report.py              # Module 8: Comprehensive lab report generator
│
├── src/                       # Modular Python packages
│   ├── __init__.py
│   ├── data/                  # Ingestion, schema validation & partitioning
│   ├── exploration/           # Descriptive stats, correlation & relationship visualizers
│   │   ├── distributions.py   # Univariate statistics, skewness, kurtosis & IQR outliers
│   │   ├── correlations.py    # Pearson & Spearman matrices with constant exclusion
│   │   ├── relationships.py   # Numeric & categorical bivariate relationship stats
│   │   └── plots.py           # Academic-style Plotly interactive visualizers
│   ├── features/              # Feature selection engine & ColumnTransformer
│   │   ├── preprocessing.py   # Train-only SimpleImputer, StandardScaler, OneHotEncoder
│   │   ├── feature_mapping.py # Group-aware original-to-encoded feature bridge
│   │   ├── filter_methods.py  # Baseline, Pearson, F-test, Mutual Information
│   │   ├── wrapper_methods.py # Group RFE, Forward SFS, Backward SFS
│   │   ├── embedded_methods.py# Lasso L1 regularization with group L2 scoring
│   │   ├── evaluation.py      # Isolated CV (TimeSeriesSplit, GroupKFold, KFold)
│   │   ├── selectors.py       # Unified dispatch facade & method registry
│   │   └── result_models.py   # FeatureSelectionResult & OriginalFeatureMapping
│   ├── models/                # Milestone 3 OLS Modeling Engine
│   │   ├── design_matrix.py   # OLS design matrix builder, reference encoding & rank diagnostics
│   │   ├── ols_model.py       # statsmodels OLS fitting & LaTeX equation formulations
│   │   ├── model_result.py    # OLSModelResult data contract
│   │   └── model_service.py   # High-level training orchestrator
│   ├── evaluation/            # Milestone 3 Evaluation & Plotting Engine
│   │   ├── regression_metrics.py # MSE, RMSE, MAE, R², Train Adj R²
│   │   ├── predictions.py     # OLS inference & out-of-sample prediction
│   │   ├── plots.py           # Actual vs Pred, Residuals vs Fitted & Distribution histogram
│   │   └── comparison.py      # Cross-model leaderboard builder
│   ├── experiments/           # Experiment tracker, stale state purger & regression tracker
│   └── utils/                 # Path helpers & YAML configuration loaders
│
├── configs/                   # Declarative system & dataset configurations
│   ├── app_config.yaml        # Application parameters & milestone definitions
│   └── datasets.yaml          # Single source of truth for dataset metadata
│
├── tests/                     # Automated test suites (118 tests passing)
│   ├── test_structure.py      # Milestone 0 architecture, navigation & doc tests
│   ├── test_dataset_engine.py # Milestone 1 Registry, Loader, Splitter & Leakage tests
│   ├── test_eda.py            # Milestone 2 EDA distribution, correlation & plot tests
│   ├── test_feature_selection.py # Milestone 2 Preprocessing, 8 selectors & CV tests
│   ├── test_integration_datasets.py # Milestone 2 Real-dataset verification tests
│   ├── test_ols_design_matrix.py # Milestone 3 Design matrix, reference encoding & rank tests
│   ├── test_regression_modeling.py # Milestone 3 OLS estimation, inference & LaTeX equation tests
│   ├── test_regression_evaluation.py # Milestone 3 MSE, RMSE, MAE, R² & Adj R² metric tests
│   └── test_regression_integration.py # Milestone 3 Real-dataset integration & controlled Test tests
│
└── docs/                      # Technical specifications & statistical references
    ├── PROJECT_SPEC.md        # Comprehensive system specification & milestone status
    ├── DATASET_CATALOG.md     # Verified CSV schemas, exact predictor counts & audits
    ├── DATA_PIPELINE.md       # Data Engine architecture, splitting logic & Mermaid flow
    ├── EDA_METHODS.md         # EDA formulas, skewness, kurtosis, Pearson vs Spearman
    ├── FEATURE_SELECTION_METHODS.md # Specifications for all 8 selection algorithms & empirical table
    ├── FEATURE_SELECTION_PIPELINE.md# Pipeline, structural mapping, fold isolation & dimension hierarchy
    ├── OLS_REGRESSION_METHODS.md # Milestone 3 OLS mathematics, reference encoding & rank conditions
    ├── MODEL_EVALUATION_METHODS.md # Milestone 3 Metrics, generalization gap & descriptive residual plots
    ├── MODELING_PIPELINE.md   # Milestone 3 Modeling architecture, data contracts & test isolation
    └── STATISTICAL_METHODS.md # Mathematical definitions, hypotheses & decision thresholds
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Conda Environment
```bash
conda create -n regression-lab python=3.11 -y
conda activate regression-lab
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Acquire Datasets (Optional for Unit Tests, Required for Real Modeling)
The platform includes an automated downloader that retrieves and cryptographically verifies the four benchmark datasets into `data/raw/`:
```bash
python -m src.data.download
```
*(Alternatively, missing datasets can be downloaded on-demand directly in the UI via `pages/dataset.py`)*

### 4. Run Automated Test Suite (92 Tests Passing)
The test suite is structured into offline Unit Tests and real-dataset Integration Tests:

- **Offline Unit Tests** (Runs 100% offline without external network or CSV downloads):
  ```bash
  pytest tests/test_structure.py tests/test_dataset_engine.py tests/test_eda.py tests/test_feature_selection.py -v
  ```

- **Real-Dataset Integration Workflow** (Verifies SHA-256 signatures, real schemas & splits):
  ```bash
  python tests/run_integration_tests.py
  ```

- **All 92 Tests Combined**:
  ```bash
  pytest -v
  ```

### 5. Launch the Streamlit Platform
```bash
python -m streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🗺️ Milestone Roadmap

| Milestone | Phase | Description | Status |
| :---: | :--- | :--- | :---: |
| **0** | **Project Foundation** | Architecture, multi-page routing, light UI theme, documentation, tests, Git/GitHub setup | ✅ **Completed** |
| **1** | **Dataset & Data Engine** | Working Dataset Library, schema validation, profiling, and Random/Temporal/Group splits | ✅ **Completed** |
| **2** | **EDA & Feature Selection** | Interactive EDA Lab, 8 group-aware selection methods, leakage-free CV, experiment comparison | ✅ **Completed** |
| **3** | **Regression & Evaluation** | statsmodels OLS fitting, econometric summary table, LaTeX equations, MSE/RMSE/R² | ⏳ Next Up |
| **4** | **Assumption Diagnostics** | Five Gauss-Markov diagnostic tests, residual plots, Cook's distance, leverage | ⏳ Scheduled |
| **5** | **Comparison & Reporting** | Side-by-side experiment comparison, radar plots, exportable lab reports | ⏳ Scheduled |

---

## 📖 Documentation Reference

- **[PROJECT_SPEC.md](docs/PROJECT_SPEC.md)**: Full system specification, interface contracts, and milestone roadmap.
- **[DATASET_CATALOG.md](docs/DATASET_CATALOG.md)**: Verified CSV schemas, exact predictor counts, target definitions, and empirical audit notes.
- **[DATA_PIPELINE.md](docs/DATA_PIPELINE.md)**: Data Engine technical architecture, Mermaid flowcharts, and splitting algorithms.
- **[EDA_METHODS.md](docs/EDA_METHODS.md)**: Univariate distributions, skewness, kurtosis, Pearson vs Spearman, and relationship diagnostics.
- **[FEATURE_SELECTION_METHODS.md](docs/FEATURE_SELECTION_METHODS.md)**: Formulations, complexities, and group aggregation rules for all 8 selection algorithms.
- **[FEATURE_SELECTION_PIPELINE.md](docs/FEATURE_SELECTION_PIPELINE.md)**: Train-only ColumnTransformer, group mapping, and leakage-free cross-validation.
- **[STATISTICAL_METHODS.md](docs/STATISTICAL_METHODS.md)**: Mathematical formulations, Gauss-Markov assumptions, hypotheses, and decision thresholds.

---

## 👥 Contributors & License
Released under the [MIT License](LICENSE). Educational platform for statistical regression and data science pedagogy.
