# Data Engine & Splitting Pipeline Specification

This document details the architectural design, data structures, operational contracts, and quality enforcement policies of the **Data Engine** in **Regression Learning Lab**.

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
    
    subgraph State & Invalidation
        L --> M[Streamlit Session State\nst.session_state['active_split_result']]
        M --> N[Configuration State Tracking\nDetects Outdated Settings]
        N --> O[Downstream ML Modules\nMilestone 2: Feature Selection\nMilestone 3: OLS Regression]
    end
```

---

## 2. Core Modules & Responsibilities

| Module | Primary Responsibility | Key Output / Classes |
| :--- | :--- | :--- |
| `src/data/models.py` | Typed dataclasses defining domain contracts | `DatasetMetadata`, `DatasetProfile`, `SplitConfig`, `SplitResult`, `DatasetBundle` |
| `src/data/registry.py` | Loads and validates declarative YAML configuration | `DatasetRegistry`, `RegistryValidationError` |
| `src/data/loader.py` | Resolves paths, parses CSVs, sanitizes headers, computes file fingerprints | `load_raw_dataset()`, `compute_file_fingerprint()` |
| `src/data/validation.py` | Distinguishes blocking errors from warnings, enforces 14–39 feature rule | `validate_dataset_schema()`, `ValidationReport` |
| `src/data/profiling.py` | Computes descriptive statistics and builds interactive Plotly charts | `compute_dataset_profile()`, `create_target_histogram()` |
| `src/data/splitters.py` | Executes deterministic Random, Temporal, and Group splitting | `split_dataset()`, `SplitResult` |
| `src/data/download.py` | Automated downloader with cryptographic SHA-256 verification | `download_dataset()`, `verify_dataset_integrity()` |

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

## 5. Strict Schema Validation & Blocking Policy

The system explicitly distinguishes **Critical Blocking Errors** from **Non-blocking Data Quality Warnings**:

### Critical Blocking Errors (Splitting Forbidden)
If any of these conditions occur, splitting is blocked at both the backend (`split_dataset()`) and frontend (Streamlit disabled button):
1. **Feature Count Violation**: Usable original features $< 14$ or $> 39$.
2. **Missing Target Column**: Declared target variable absent in dataset.
3. **Duplicate Column Headers**: Ambiguous duplicate column names.
4. **Missing Required Strategy Key**: Attempting temporal split without a date column or group split without a group column.
5. **Empty Dataset**: $0$ observations.

### Non-Blocking Warnings (Informational Alerts)
1. **Target Missing Values**: Rows with null targets are automatically dropped during partitioning with an explicit warning banner.
2. **Unused Excluded Columns**: A column marked for exclusion in metadata that was not present in the CSV.
3. **Data Quality Warnings**: High missingness (>50%) or constant (zero-variance) predictors.

---

## 6. File Fingerprinting & Cache Invalidation

Streamlit caching in `pages/dataset.py` is keyed by both Dataset ID and a deterministic **File Fingerprint**:
$$\text{Fingerprint} = \text{FileSize}_{\text{bytes}} \text{ \_ } \text{SHA256}_{[:16]}$$

- When a CSV file in `data/raw/` is replaced or updated, the fingerprint changes automatically.
- Streamlit cache misses immediately, reloading and re-validating the fresh dataset bundle.

---

## 7. Split Configuration State & Guidance

1. **Strategy Guidance**:
   - For time-series datasets (`appliances_energy`), selecting Random Split displays an educational warning explaining lookahead data leakage.
   - For panel datasets (`life_expectancy`), selecting Random Split displays an entity contamination warning.
   - Incompatible strategies (e.g. Temporal split without timestamp) are automatically filtered out.
2. **Active vs Outdated Split State**:
   - The UI displays an **Active Partition Configuration Card** indicating the exact strategy, proportions, and seed used to generate the active split.
   - If the user changes any parameter in the UI without re-executing, the partition status immediately transitions to **`⚠️ State: Outdated`** with a prompt to re-run.

---

## 8. Dual Testing Architecture

To guarantee 100% reproducible testing across CI and offline environments:

1. **Offline Unit Tests** (`tests/test_dataset_engine.py` & `tests/test_structure.py`):
   - Runs completely offline using in-memory synthetic fixtures.
   - Requires zero external downloads or network access.
   - Command:
     ```bash
     pytest tests/test_dataset_engine.py tests/test_structure.py -v
     ```
2. **Real-Dataset Integration Workflow** (`tests/test_integration_datasets.py`):
   - Locates or downloads the 4 authentic benchmark datasets.
   - Verifies cryptographic SHA-256 signatures.
   - Validates real schemas and executes real splits.
   - Command:
     ```bash
     python tests/run_integration_tests.py
     ```
