# Data Engine & Splitting Pipeline Specification

This document details the architectural design, data structures, and operational contracts of the **Data Engine** implemented in **Milestone 1** of the Regression Learning Lab.

---

## 1. Pipeline Architecture Overview

The Data Engine is designed as a modular, reusable backend service decoupled from Streamlit UI controls. It ensures deterministic data ingestion, strict schema compliance with the 14–39 usable predictor constraint, robust profiling, and leakage-free partitioning.

```mermaid
flowchart TD
    A[configs/datasets.yaml\nSingle Source of Truth] --> B[src.data.registry\nDatasetRegistry]
    B --> C[src.data.loader\nload_raw_dataset]
    C --> D[src.data.validation\nvalidate_dataset_schema]
    
    subgraph Separation & Profiling
        D --> E[split_features_and_target\nX_features, y_target, metadata_df]
        E --> F[src.data.profiling\ncompute_dataset_profile]
        F --> G[DatasetBundle\nIn-Memory Immutable Representation]
    end
    
    subgraph Data Splitting Engine
        G --> H{Split Strategy Selection}
        H -->|Cross-Sectional| I[Random Splitter\nDeterministic Seed 42]
        H -->|Time-Series| J[Temporal Splitter\nChronological Order]
        H -->|Panel / Clustered| K[Group Splitter\nCountry-Disjoint Partition]
        
        I --> L[SplitResult Contract\nX_train, X_val, X_test\ny_train, y_val, y_test]
        J --> L
        K --> L
    end
    
    L --> M[Streamlit Session State\nst.session_state['active_split_result']]
    M --> N[Downstream ML Modules\nMilestone 2: Feature Selection\nMilestone 3: OLS Regression]
```

---

## 2. Core Modules & Responsibilities

| Module | Primary Responsibility | Key Output / Classes |
| :--- | :--- | :--- |
| `src/data/models.py` | Typed dataclasses defining domain contracts | `DatasetMetadata`, `DatasetProfile`, `SplitConfig`, `SplitResult`, `DatasetBundle` |
| `src/data/registry.py` | Loads and validates declarative YAML configuration | `DatasetRegistry`, `RegistryValidationError` |
| `src/data/loader.py` | Resolves paths, parses CSVs (encodings/delimiters), sanitizes header whitespace | `load_raw_dataset()`, `split_features_and_target()` |
| `src/data/validation.py` | Enforces 14–39 feature rule, checks target validity and required strategy columns | `validate_dataset_schema()`, `ValidationReport` |
| `src/data/profiling.py` | Computes descriptive statistics and builds interactive Plotly charts | `compute_dataset_profile()`, `create_target_histogram()` |
| `src/data/splitters.py` | Executes deterministic Random, Temporal, and Group splitting | `split_dataset()`, `SplitResult` |
| `src/data/download.py` | Automated, reproducible dataset downloader from authoritative sources | `download_dataset()`, `download_all()` |

---

## 3. Predictor and Target Separation Formula

The engine calculates usable model predictors mathematically:

$$\text{Usable Feature Count} = \text{Total Raw Columns} - |\text{Excluded Columns}| - 1 (\text{Target})$$

- **Target Column ($y$)**: Strictly separated from the predictor matrix.
- **Excluded Metadata Columns**: Identifiers (e.g. `id`), timestamps used only for temporal ordering (e.g. `date`), and panel identifiers used only for group partitioning (e.g. `Country`).
- **Feature Matrix ($X$)**: Retains original names and raw types (both numeric and categorical) without premature one-hot encoding or normalization.

---

## 4. The Three Splitting Strategies

### 4.1 Random Splitting
- **Intended Use**: Independent, identically distributed (i.i.d.) cross-sectional observations (e.g., `kc_house_sales`, `student_performance`).
- **Mechanism**: Permutes row indices using deterministic `np.random.RandomState(seed)`.
- **Default Proportions**: $70\%$ Train, $15\%$ Validation, $15\%$ Test.
- **Guarantee**: Partitions are mutually exclusive and collectively exhaustive over all eligible rows.

### 4.2 Temporal Splitting
- **Intended Use**: Time-series or sequential sensor data (e.g., `appliances_energy`).
- **Mechanism**: Sorts observations strictly chronologically using the declared `date_column`.
  - Earliest $70\%$ $\to$ Train
  - Intermediate $15\%$ $\to$ Validation
  - Most recent $15\%$ $\to$ Test
- **Leakage Prevention**: Never shuffles observations. Strictly enforces $\max(\text{Date}_{\text{train}}) \le \min(\text{Date}_{\text{val}})$ and $\max(\text{Date}_{\text{val}}) \le \min(\text{Date}_{\text{test}})$.

### 4.3 Group-Disjoint Splitting
- **Intended Use**: Clustered, repeated measures, or panel data (e.g., `life_expectancy` across countries and years).
- **Mechanism**: Shuffles the unique group identifiers (193 countries) deterministically and allocates entire clusters to partitions to closely approximate $70/15/15$ row ratios.
- **Generalization Guarantee**: No country appears in more than one partition. The validation and test sets evaluate out-of-sample model generalization to previously unseen countries.

---

## 5. Strict Data Leakage Prevention Guarantees

1. **No Global Preprocessing**:
   - Feature scaling (`StandardScaler`, `MinMaxScaler`), categorical encoding (`OneHotEncoder`), missing value imputation, and feature selection are **intentionally not fitted** across the combined dataset in Milestone 1.
   - In subsequent milestones, all transformers will be fitted strictly on `X_train` and only applied (`transform()`) to `X_val` and `X_test`.
2. **Target Missingness Exclusion**:
   - Rows with missing target values (e.g., 10 missing rows in WHO Life Expectancy) are removed **before partitioning**, and the exclusion count is logged as an explicit warning.
3. **Predictor Missingness Retention**:
   - Rows with missing predictor values are **preserved** in the partitions so that students can evaluate training-set imputation strategies in Milestone 2.

---

## 6. Streamlit Session State Lifecycle

To ensure seamless navigation and prevent stale data contamination:
- `st.session_state["selected_dataset_id"]`: Tracks current active dataset.
- `st.session_state["active_dataset_bundle"]`: In-memory cached dataset bundle.
- `st.session_state["active_split_config"]`: Active split parameters.
- `st.session_state["active_split_result"]`: Active partitioned datasets.
- **Contamination Guard**: When a user selects a different dataset from the dropdown, all previous split results and downstream model states are automatically invalidated and purged from Session State.
