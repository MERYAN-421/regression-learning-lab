# Performance & Resource Stability Report

**Project**: Regression Learning Lab  
**Release Target**: `v1.0.0` (Milestone 7)  
**Execution Environment**: Python 3.11.16 on Windows 11 (64-bit)  
**Hardware Profiling**: `tracemalloc` memory tracing & high-precision monotonic timing (`time.time()`)  
**Data Artifact**: `docs/benchmarks_summary.json`  
**Evaluation Date**: 2026-11-01  

---

## 1. Executive Summary

This report establishes the empirical performance characteristics and resource consumption profiles of Regression Learning Lab across all four benchmark datasets. Workflows were profiled end-to-end through data ingestion, partitioning, feature selection, OLS model fitting, statistical assumption testing, multi-model comparison, academic report generation, and reproduction ZIP export.

### Key Empirical Findings
- **Sub-Second to 5-Second Latency**: Total workflow execution times range from **1.05s** (King County House Sales) to **5.34s** (Appliances Energy with TimeSeriesSplit CV).
- **Extremely Low Memory Footprint**: Peak memory consumption during full workflow execution never exceeded **32.5 MB** (maximum observed: 32.47 MB on Appliances Energy).
- **Fast Startup & Cold-Start**: Application startup and dataset bundle loading complete in under **210 ms** across all datasets.
- **Scalable Diagnostics**: High-dimensional diagnostic suites (Ramsey RESET, Breusch-Pagan, Jarque-Bera, Matrix Condition Number, Cook's distance) execute in under **200 ms** even on 20,000+ observations.

---

## 2. Four-Dataset Benchmark Summary

| Metric | Demo D: Student Performance | Demo A: Appliances Energy | Demo B: King County House Sales | Demo C: Life Expectancy |
| :--- | :--- | :--- | :--- | :--- |
| **Domain** | Education | Smart Home IoT | Real Estate Valuation | Global Public Health |
| **Observations ($N$)** | 395 | 19,735 | 21,613 | 2,938 |
| **Partition Strategy** | Random (70/15/15) | Temporal (Chronological) | Random (70/15/15) | Group (by Country) |
| **Dataset Fingerprint** | `56993_e47f9ee225e1ee6e` | `11979363_2820bf712ad0275c` | `2359557_fb9ce109d108656e` | `333442_872125dd1dd0f914` |
| **Split Fingerprint** | `ad6c1cc7ca9adeef` | `11f581f9d98e2b82` | `4d10d191107c478c` | `dc070a746ffedb12` |
| **Dataset Ingestion** | 0.064 s | 0.205 s | 0.092 s | 0.019 s |
| **Partitioning Duration** | 0.019 s | 0.124 s | 0.072 s | 0.068 s |
| **Feature Selection** | 1.493 s (Pearson k=5) | 0.689 s (TS Pearson k=10) | 0.579 s (Pearson k=5) | N/A (All Usable) |
| **OLS Parameter Fit** | 0.059 s | 0.075 s | 0.064 s | 0.166 s |
| **Assumption Diagnostics** | 0.017 s | 0.199 s | 0.118 s | 0.061 s |
| **Model Comparison** | 0.010 s | N/A | 0.006 s | N/A |
| **Report & ZIP Export** | 1.388 s | 4.046 s | 0.002 s | 1.255 s |
| **Total Workflow Time** | **3.373 s** | **5.341 s** | **1.047 s** | **1.571 s** |
| **Peak Memory Overhead** | **22.82 MB** | **32.47 MB** | **30.34 MB** | **4.24 MB** |
| **Reproduction ZIP Size** | 6,820 bytes | 19,059 bytes | 6,857 bytes | 18,809 bytes |

---

## 3. Workflow Phase Detailed Analysis

### Phase 1: Data Ingestion & Fingerprinting
- **Operation**: CSV reading, schema casting, identifier extraction, metadata isolation, and SHA-256 fingerprint generation.
- **Latency**: Under 0.21s for the largest dataset (Appliances Energy, 19,735 rows $\times$ 29 columns).
- **Optimization**: `@st.cache_data` caches parsed bundles immutably in memory after the initial read. Subsequent UI interactions incur zero disk I/O.

### Phase 2: Partitioning & Boundary Integrity
- **Random Split**: Uses deterministic NumPy permutation indexing (`0.019s` on Student Performance, `0.072s` on King County).
- **Temporal Split**: Uses pandas datetime indexing to enforce non-overlapping chronological intervals (`0.124s` on Appliances Energy).
- **Group Split**: Partitions disjoint sovereign nations without cluster leakage (`0.068s` on Life Expectancy).

### Phase 3: Leakage-Safe Feature Selection
- Preprocessors fit strictly on $X_{\text{train}}$.
- Pearson filter selection with 5-fold cross-validation completes in `0.58s` to `1.49s`.
- Original feature mapping groups categorical dummy columns together, avoiding dimensional fragmentation.

### Phase 4: OLS Parameter Estimation & Diagnostics
- statsmodels OLS parameter estimation executes in under `0.17s` across all datasets.
- QR decomposition with column pivoting immediately flags exact collinearity (e.g. duplicate random variable $rv2$ in Appliances Energy or $sqft\_living = sqft\_above + sqft\_basement$ in King County).
- Full diagnostic suites (Ramsey RESET, Durbin-Watson / Ljung-Box, Breusch-Pagan, Jarque-Bera, VIF, Cook's distance) execute in `0.017s` to `0.199s`.

### Phase 5: Academic Report Generation & Reproduction Artifacts
- The reporting engine resolves upstream evidence manifest records, renders 9 structured academic sections, generates SVG/Plotly figures, and compiles Markdown and sanitized HTML.
- Compiling the reproduction package into an in-memory ZIP archive (`io.BytesIO`) takes `0.002s` to `4.05s`, producing lightweight bundles between 6.8 KB and 19.1 KB.

---

## 4. Multi-Tenant Resource Stability & Community Cloud Limits

Streamlit Community Cloud allocates approximately 1 GB of RAM per application container.
- **Observed Peak**: 32.47 MB during the largest workflow, consuming **less than 3.5%** of available container memory.
- **Concurrency Headroom**: Even under concurrent multi-user execution across multiple independent sessions, the platform's bounded memory profile easily operates within standard cloud container quotas.
- **No Disk Leaks**: All reports and reproduction packages are generated in-memory without accumulating temporary files on the host filesystem.
