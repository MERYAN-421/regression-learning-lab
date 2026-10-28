# 📈 Regression Learning Lab

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Milestone-0%20Completed-success.svg)]()

> **Interactive Multiple Linear Regression & Statistical Diagnostics Platform**  
> An educational web platform designed for data science and statistics students to explore real-world regression modeling, feature selection, and the classical Gauss-Markov assumptions using curated Kaggle tabular datasets (14–39 features).

---

## 🌟 Key Highlights

- **Pedagogical Rigor**: Bridges the gap between predictive machine learning and econometric inference.
- **Two Learning Modalities**:
  - **Guided Learning**: Step-by-step sequential path through the 8-stage data science lifecycle.
  - **Free Experiment**: Open sandbox for custom feature subsets, regularization penalties, and hypothesis tests.
- **Real-World Kaggle Benchmarks**: Datasets featuring 14 to 39 predictors exhibiting authentic challenges (collinearity, skewness, heteroscedasticity).
- **Five Classical Gauss-Markov Diagnostics**:
  1. *Linearity* (Ramsey RESET & Residuals vs. Fitted)
  2. *Independence* (Durbin-Watson, Ljung-Box & ACF)
  3. *Homoscedasticity* (Breusch-Pagan, White's Test & Scale-Location)
  4. *Normality* (Jarque-Bera, Shapiro-Wilk & Normal Q-Q Plot)
  5. *Multicollinearity* (Variance Inflation Factor & Matrix Condition Number)
- **Influential Point Triage**: Cook's distance and Hat matrix leverage analysis.
- **Reproducible Academic Reports**: Automatic synthesis of empirical findings into exportable Markdown, HTML, and Python reproduction scripts.

---

## 🏛️ Project Architecture

```text
regression-learning-lab/
│
├── app.py                     # Streamlit main entrypoint & navigation controller
├── requirements.txt           # Verified Python 3.11 dependencies
├── README.md                  # Project overview, installation, and roadmap
├── .gitignore                 # Git ignore rules for caches, envs, and large data
│
├── .streamlit/
│   └── config.toml            # Streamlit theme & server configuration
│
├── pages/                     # Multi-page application structure
│   ├── home.py                # Home: Objectives, learning modes & roadmap
│   ├── dataset.py             # Module 1: Kaggle dataset library & partitioning
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
│   └── datasets.yaml          # Kaggle dataset catalog schema & definitions
│
├── data/                      # Local data directory (raw & processed cache)
│   └── README.md
│
├── tests/                     # Automated test suites
│   ├── __init__.py
│   └── test_structure.py      # Architecture, importability & schema tests
│
└── docs/                      # Technical specifications & statistical references
    ├── PROJECT_SPEC.md        # Comprehensive system specification
    ├── DATASET_CATALOG.md     # Kaggle dataset inclusion criteria & schema
    └── STATISTICAL_METHODS.md # Mathematical definitions, hypotheses & decision rules
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.11 installed. We strongly recommend creating a dedicated Conda environment:

```bash
conda create -n regression-lab python=3.11 -y
conda activate regression-lab
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Architecture Tests
```bash
pytest tests/
```

### 4. Launch the Streamlit Platform
```bash
python -m streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🗺️ Milestone Roadmap

| Milestone | Phase | Description | Status |
| :---: | :--- | :--- | :---: |
| **0** | **Project Foundation** | Architecture, multi-page routing, light UI theme, documentation, tests, Git/GitHub setup | ✅ **Completed** |
| **1** | **Dataset & Preprocessing** | Kaggle datasets (14–39 features), data ingestion, validation, and splitting | ⏳ Scheduled |
| **2** | **EDA & Feature Selection** | Univariate/bivariate analysis, correlation heatmaps, VIF, Filter, Wrapper, Lasso | ⏳ Scheduled |
| **3** | **Regression & Evaluation** | statsmodels OLS fitting, econometric summary table, LaTeX equations, MSE/RMSE/R² | ⏳ Scheduled |
| **4** | **Assumption Diagnostics** | Five Gauss-Markov diagnostic tests, residual plots, Cook's distance, leverage | ⏳ Scheduled |
| **5** | **Comparison & Reporting** | Side-by-side experiment comparison, radar plots, exportable lab reports | ⏳ Scheduled |

---

## 📖 Documentation Reference

- **[PROJECT_SPEC.md](docs/PROJECT_SPEC.md)**: Full architecture, page hierarchy, and component specifications.
- **[DATASET_CATALOG.md](docs/DATASET_CATALOG.md)**: Candidate Kaggle datasets, criteria (14–39 features), and schema.
- **[STATISTICAL_METHODS.md](docs/STATISTICAL_METHODS.md)**: Mathematical formulations, null hypotheses ($H_0$), test statistics, and decision thresholds.

---

## 👥 Contributors & Academic License
Designed for academic coursework and interactive statistical learning. Released under the MIT License.
