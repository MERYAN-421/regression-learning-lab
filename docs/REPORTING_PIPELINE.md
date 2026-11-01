# Academic Reporting Laboratory: Architecture & Implementation Pipeline

## 1. Architectural Overview

The **Academic Reporting Pipeline** connects the verified experimental stages of the Regression Learning Lab into an auditable research document synthesis framework.

```
+---------------------------------------------------------------------------------------------------------+
|                                    EXPERIMENTAL RESEARCH PIPELINE                                       |
|                                                                                                         |
|  [Dataset Registry] ---> [SplitResult] ---> [Feature Selection] ---> [OLS Regression]                  |
|                                                    |                         |                          |
|                                                    v                         v                          |
|                                            [Feature Tracker]        [Regression Tracker]                |
|                                                                              |                          |
|                                                                              v                          |
|                                         [Diagnostic Suite] <---> [Multi-Model Comparison]               |
+----------------------------------------------------+-------------------------+--------------------------+
                                                     |                         |
                                                     +------------+------------+
                                                                  |
                                                                  v
                                              +---------------------------------------+
                                              |       src/reporting/ Pipeline         |
                                              |                                       |
                                              |  1. ReportSourceResolver              |
                                              |  2. ArtifactCompatibilityValidator    |
                                              |  3. EvidenceManifestBuilder           |
                                              |  4. ReportReadinessEngine             |
                                              |  5. AcademicSectionBuilder            |
                                              |  6. ReportFigureRegistry              |
                                              |  7. ReportExportService               |
                                              +-------------------+-------------------+
                                                                  |
                                                                  v
                                              +---------------------------------------+
                                              |       Multi-Format Deliverables       |
                                              |                                       |
                                              | - report.md (Markdown)                |
                                              | - report.html (Interactive HTML)      |
                                              | - manifest.json (Auditable Records)   |
                                              | - reproduction_package.zip (Full Code)|
                                              +---------------------------------------+
```

---

## 2. Source Resolution & Artifact Compatibility

### 2.1 Resolution Workflow (`src/reporting/source_resolver.py`)
`ReportSourceResolver` accepts an optional `comparison_id` or `selected_model_id` and traverses the existing session trackers:
- **Comparison Resolution**:
  - Validates `ComparisonExperimentTracker` membership.
  - Resolves all referenced `OLSModelResult` instances.
  - Identifies the user's selected candidate model and recorded rationale.
- **Diagnostic Suite Resolution**:
  - Matches the model's exact `model_id` against available `DiagnosticSuiteResult` instances.
  - Blocks silent substitution of newer or mismatched diagnostic suites.
- **Data & Split Resolution**:
  - Retrieves active `DatasetMetadata` and `SplitResult`.
  - Verifies partition alignment and sample sizes.

### 2.2 Compatibility Enforcement (`src/reporting/compatibility.py`)
Prior to manifest construction, `ArtifactCompatibilityValidator` strictly rejects:
1. **Dataset ID & Fingerprint Mismatches**: Model was trained on a different dataset or modified data.
2. **Split Fingerprint Mismatches**: Partitions were resplit or random seed altered.
3. **Target Identity & Unit Mismatches**: Models with incompatible target variables or units (e.g., `USD` vs `Points`).
4. **Candidate Discrepancies**: Selected candidate model is not a member of the comparison experiment.
5. **Metric Inconsistencies**: Stale session state where comparison leaderboard metrics differ from model artifact metrics.

---

## 3. Tri-State Readiness Engine (`src/reporting/readiness.py`)

The pipeline evaluates readiness deterministically into three explicit states:

| Status | Conditions | Permitted Action | User Warning / Disclosure |
| :--- | :--- | :--- | :--- |
| **`READY`** | Primary model and comparison are valid; all referenced diagnostics and feature selection artifacts are verified. | Full Report Export & Preview. | None. Document represents full empirical rigor. |
| **`READY_WITH_LIMITATIONS`** | Core model and split are verified, but optional artifacts (e.g., diagnostic suite not run, feature selection tracker missing) are absent. | Export & Preview permitted. | Explicitly lists missing optional sections and analytical limitations. |
| **`BLOCKED`** | Critical integrity failure: missing model, fingerprint mismatch, target unit divergence, or unauthorized Test set access. | **Export Blocked.** Preview shows diagnostic blocker reasons only. | Displays blocker alert with exact reasons preventing publication. |

---

## 4. UI Architecture: Streamlit Report Laboratory (`pages/report.py`)

The Streamlit interface provides eight cohesive sections (A through H):

- **Section A — Report Source Selection**: Select between multi-model comparison experiments or standalone OLS models; choose between Template A (Full Academic), Template B (Executive Summary), or Template C (Statistical Diagnostics).
- **Section B — Evidence Readiness Dashboard**: Real-time audit metrics displaying verified evidence counts, missing optional fields, and readiness state badge.
- **Section C — Template Configuration**: Customize title, author name, research objective, section inclusion checkboxes, and the authorized Final Test Appendix toggle.
- **Section D — Report Preview**: Academic document preview rendering mathematical equations, structured tables, and evidence citations.
- **Section E — Publication Tables & Figures**: Selection and preview of diagnostic and comparative Plotly figures with caption metadata and provenance.
- **Section F — Author Discussion & Reflection**: Segregated author input fields for principal findings, practical significance, modeling limitations, and future remediation.
- **Section G — Reproducibility Audit**: Complete cryptographic and software environment audit summary.
- **Section H — Export & Report History**: Download buttons for Markdown, standalone HTML, Manifest JSON, and Reproduction ZIP package, alongside session report history.

---

## 5. Final Test Set Isolation Protocol

The reporting pipeline adheres strictly to **Rule 4 (Final Test Lock)**:
1. `run_academic_report()` and `AcademicSectionBuilder` **never** invoke `finalize_test_evaluation()` or directly access `SplitResult.y_test`.
2. Development reports present only Training and Validation metrics ($RMSE_{train}, RMSE_{val}, R^2_{val}, \text{Gen Gap}$).
3. Test metrics can appear **only** in the dedicated *Final Evaluation Appendix (Section 10)* when:
   - The candidate model has already been finalized via a preexisting, compatible `FinalTestEvaluationResult`.
   - The user explicitly enables the Final Test Appendix toggle in Template Configuration.
4. Test performance **never** influences candidate selection, Pareto efficiency, leaderboard ranking, or radar scores.
