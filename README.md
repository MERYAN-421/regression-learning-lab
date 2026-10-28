# 📈 Regression Learning Lab

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Milestone-1%20Completed-success.svg)]()

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
- **Five Classical Gauss-Markov Diagnostics**:
  1. *Linearity* (Ramsey RESET & Residuals vs. Fitted)
  2. *Independence* (Durbin-Watson, Ljung-Box & ACF)
  3. *Homoscedasticity* (Breusch-Pagan, White's Test & Scale-Location)
  4. *Normality* (Jarque-Bera, Shapiro-Wilk & Normal Q-Q Plot)
  5. *Multicollinearity* (Variance Inflation Factor & Matrix Condition Number)

---

## 📂 Supported Benchmark Datasets (Milestone 1)

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
│   ├── exploration.py         # Module 2: EDA, distributions & correlation matrix
│   ├── feature_selection.py   # Module 3: Filter, Wrapper, and Lasso L1 selection
│   ├── regression.py          # Module 4: Ordinary Least Squares (OLS) modeling
│   ├── evaluation.py          # Module 5: Metrics (MSE, RMSE, MAE, R², Adj R²)
│   ├── diagnostics.py         # Module 6: Five Gauss-Markov assumption tests
│   ├── comparison.py          # Module 7: Multi-model comparison & radar chart
│   └── report.py              # Module 8: Comprehensive lab report generator
│
├── src/                       # Modular Python packages
│   ├── __init__.py
│   ├── data/                  # Ingestion, schema validation & partitioning
│   │   ├── __init__.py        # Public API exports
│   │   ├── models.py          # Dataclasses (DatasetMetadata, Profile, SplitResult)
│   │   ├── registry.py        # YAML configuration loader & validator
│   │   ├── loader.py          # CSV loader (encoding, delimiters, date parsing)
│   │   ├── validation.py      # Schema verification & 14-39 feature rule enforcer
│   │   ├── profiling.py       # Descriptive statistics & Plotly visualizers
│   │   ├── splitters.py       # Random, Temporal, and Group partitioning engine
│   │   └── download.py        # Automated reproducible dataset acquisition
│   ├── exploration/           # Descriptive statistics & correlation engine
│   ├── features/              # Feature selection & VIF reduction
│   ├── models/                # OLS & regularized regression estimators
│   ├── evaluation/            # Performance metrics & error visualizers
│   ├── diagnostics/           # Statistical tests & diagnostic plots
│   ├── experiments/           # Experiment registry & multi-model comparison
│   ├── reporting/             # Report synthesis & export formats
│   └── utils/                 # Path helpers & YAML configuration loaders
│
├── configs/                   # Declarative system & dataset configurations
│   ├── app_config.yaml        # Application parameters & milestone definitions
│   └── datasets.yaml          # Single source of truth for dataset metadata
│
├── data/                      # Local data directory
│   ├── README.md              # Data storage policies and layout
│   └── raw/                   # Cached benchmark CSV files (ignored by Git)
│
├── tests/                     # Automated test suites
│   ├── __init__.py
│   ├── test_structure.py      # Milestone 0 architecture, navigation & doc tests
│   └── test_dataset_engine.py # Milestone 1 Registry, Loader, Splitter & Leakage tests
│
└── docs/                      # Technical specifications & statistical references
    ├── PROJECT_SPEC.md        # Comprehensive system specification & milestone status
    ├── DATASET_CATALOG.md     # Verified CSV schemas, feature counts, and audit notes
    ├── DATA_PIPELINE.md       # Data Engine architecture, splitting logic & Mermaid flow
    └── STATISTICAL_METHODS.md # Mathematical definitions, hypotheses & decision rules
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

### 4. Run Automated Test Suite (50 Tests)
The test suite is structured into offline Unit Tests and real-dataset Integration Tests:

- **Offline Unit Tests** (Runs 100% offline without external network or CSV downloads):
  ```bash
  pytest tests/test_dataset_engine.py tests/test_structure.py -v
  ```

- **Real-Dataset Integration Workflow** (Verifies SHA-256 signatures, real schemas & splits):
  ```bash
  python tests/run_integration_tests.py
  ```

- **All Tests Combined**:
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
| **2** | **EDA & Feature Selection** | Univariate/bivariate analysis, correlation heatmaps, VIF, Filter, Wrapper, Lasso | ⏳ Next Up |
| **3** | **Regression & Evaluation** | statsmodels OLS fitting, econometric summary table, LaTeX equations, MSE/RMSE/R² | ⏳ Scheduled |
| **4** | **Assumption Diagnostics** | Five Gauss-Markov diagnostic tests, residual plots, Cook's distance, leverage | ⏳ Scheduled |
| **5** | **Comparison & Reporting** | Side-by-side experiment comparison, radar plots, exportable lab reports | ⏳ Scheduled |

---

## 📖 Documentation Reference

- **[DATA_PIPELINE.md](docs/DATA_PIPELINE.md)**: Data Engine technical architecture, Mermaid flowcharts, and splitting algorithms.
- **[DATASET_CATALOG.md](docs/DATASET_CATALOG.md)**: Verified CSV schemas, exact predictor counts, target definitions, and empirical audit notes.
- **[PROJECT_SPEC.md](docs/PROJECT_SPEC.md)**: Full system specification, interface contracts, and milestone roadmap.
- **[STATISTICAL_METHODS.md](docs/STATISTICAL_METHODS.md)**: Mathematical formulations, Gauss-Markov assumptions, hypotheses, and decision thresholds.

---

## 👥 Contributors & License
Released under the [MIT License](LICENSE). Educational platform for statistical regression and data science pedagogy.
