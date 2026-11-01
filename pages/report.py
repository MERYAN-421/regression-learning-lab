"""
Academic Report Generation Laboratory for Ordinary Least Squares (OLS) Regression.
Milestone 6: Synthesizes verified regression experiments into structured, statistically accurate,
auditable, and exportable academic reports.

Core Non-Negotiable Rules Enforced:
- RULE 1: Never Fabricate Evidence — every numerical claim originates from a validated artifact.
- RULE 2: Reuse Existing Results and Trackers — no duplicate modeling or diagnostic engines.
- RULE 3: No Automatic Retraining — report generation is strictly read-only.
- RULE 4: Preserve Final Test Lock — never call finalize_test_evaluation(); test metrics strictly isolated.
- RULE 5: Defensible Statistical Interpretation — no causal claims from R², no H0 proof, no pass counts.
- RULE 6: Reproducibility Claims Must Be Verified — transparent disclosure of reproduction scope.
Downstream interactive features and tutorial enhancements are Under Development and Scheduled for Milestone 7.
"""
from datetime import datetime
import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.data.models import DatasetMetadata, SplitResult
from src.data.loader import get_dataset_fingerprint
from src.models.model_result import OLSModelResult
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.experiments.comparison_experiment_tracker import ComparisonExperimentTracker
from src.experiments.report_experiment_tracker import ReportExperimentTracker
from src.diagnostics.result_models import DiagnosticSuiteResult

from src.reporting.result_models import (
    AcademicReportResult,
    ReportTemplateType,
    ReadinessStatus,
    AuthorContent,
    SectionID
)
from src.reporting.source_resolver import ReportSourceResolver, ResolvedReportSources
from src.reporting.readiness import evaluate_report_readiness, ReportReadinessEvaluation
from src.reporting.figure_registry import ReportFigureRegistry
from src.reporting.export_service import ReportExportService
from src.reporting.report_runner import run_academic_report


# -----------------------------------------------------------------------------
# PAGE TITLE & ACADEMIC HEADER
# -----------------------------------------------------------------------------
st.title("📝 Academic Report Generation Laboratory")
st.subheader("Synthesis of Verified Experimental Findings, Statistical Audits & Reproducibility Packages")

st.caption(
    "ℹ️ **Milestone 6 Active**: Transform empirical regression models, assumption diagnostics, and "
    "multi-model comparisons into formal, exportable academic reports. Zero statistics are fabricated (Rule 1). "
    "Report generation is strictly read-only and never retrains models (Rule 3) or unlocks Test targets (Rule 4)."
)

st.divider()


# -----------------------------------------------------------------------------
# SESSION STATE & TRACKERS INITIALIZATION
# -----------------------------------------------------------------------------
split_res: Optional[SplitResult] = st.session_state.get("dataset_split_result", None)
current_meta: Optional[DatasetMetadata] = st.session_state.get("active_dataset_metadata", None)
regression_tracker: RegressionExperimentTracker = st.session_state.get(
    "regression_tracker", RegressionExperimentTracker()
)
if "comparison_tracker" not in st.session_state:
    st.session_state["comparison_tracker"] = ComparisonExperimentTracker()
comparison_tracker: ComparisonExperimentTracker = st.session_state["comparison_tracker"]

if "report_tracker" not in st.session_state:
    st.session_state["report_tracker"] = ReportExperimentTracker()
report_tracker: ReportExperimentTracker = st.session_state["report_tracker"]

raw_diagnostic_suites = st.session_state.get("diagnostic_suites", {})
fs_tracker = st.session_state.get("fs_tracker", None) or st.session_state.get("feature_selection_tracker", None)

if split_res is None or current_meta is None:
    st.warning("⚠️ **Dataset & Split Required**: Please load and partition a dataset in the **Dataset Library**.")
    st.stop()

active_ds_fp = st.session_state.get("active_dataset_fingerprint") or get_dataset_fingerprint(current_meta)
active_ds_id = getattr(current_meta, "id", "dataset")
active_ds_name = getattr(current_meta, "name", "Dataset")

# Retrieve valid models for active split
available_models: List[OLSModelResult] = regression_tracker.get_valid_models(
    current_split_fingerprint=split_res.split_fingerprint,
    current_dataset_fingerprint=active_ds_fp
)

# Retrieve valid comparisons for active split
available_comparisons = comparison_tracker.get_valid_comparisons(
    current_split_fingerprint=split_res.split_fingerprint,
    current_dataset_fingerprint=active_ds_fp
)

if not available_models:
    st.info(
        "ℹ️ **No Trained Models Found**: Fit at least one OLS regression model in the "
        "**Regression Laboratory** before generating academic reports."
    )
    st.stop()


# -----------------------------------------------------------------------------
# SECTION A: REPORT SOURCE SELECTION
# -----------------------------------------------------------------------------
st.header("A. Report Source Selection")

with st.container(border=True):
    col_ds1, col_ds2, col_ds3, col_ds4 = st.columns(4)
    col_ds1.metric("Active Dataset", active_ds_name)
    col_ds2.metric("Target Variable", current_meta.target_column)
    col_ds3.metric("Split Strategy", split_res.strategy.capitalize())
    col_ds4.metric("Valid Models", len(available_models))

    st.caption(
        f"**Dataset Fingerprint**: `{active_ds_fp[:12]}` | "
        f"**Split Fingerprint**: `{split_res.split_fingerprint[:12]}`"
    )

    st.markdown("#### Choose Reporting Workflow & Primary Subject")
    source_mode = st.radio(
        "Select Primary Report Source:",
        options=[
            "Completed Comparison Experiment (Recommended: Includes Leaderboard & Radar)",
            "Individual OLS Model Specification"
        ],
        index=0 if available_comparisons else 1,
        help="Full academic reports are typically built from multi-model comparison benchmarks, but single model reports are supported."
    )

    selected_comp_id: Optional[str] = None
    selected_model_id: Optional[str] = None

    if source_mode.startswith("Completed Comparison"):
        if not available_comparisons:
            st.warning("No comparison records available for this split. Please run a comparison in the **Comparison Laboratory** or select an Individual Model below.")
            source_mode = "Individual OLS Model Specification"
        else:
            comp_options = {c.comparison_id: f"{c.comparison_id} ({len(c.compared_model_ids)} models, Cand: {c.selected_candidate_model_id})" for c in available_comparisons}
            selected_comp_id = st.selectbox(
                "Select Comparison Experiment:",
                options=list(comp_options.keys()),
                format_func=lambda x: comp_options[x]
            )
            comp_obj = comparison_tracker.get_comparison(selected_comp_id)
            if comp_obj:
                # Candidate defaults to selected candidate in comparison, but user can inspect any compared model
                cand_idx = 0
                if comp_obj.selected_candidate_model_id in comp_obj.compared_model_ids:
                    cand_idx = comp_obj.compared_model_ids.index(comp_obj.selected_candidate_model_id)
                selected_model_id = st.selectbox(
                    "Primary Model Specification to Detail in Report:",
                    options=comp_obj.compared_model_ids,
                    index=cand_idx,
                    help="Default is the formally recorded candidate model from the comparison experiment."
                )

    if source_mode.startswith("Individual OLS Model"):
        model_options = {m.model_id: f"{m.model_name} ({m.model_id}) — Val RMSE: {m.val_metrics.rmse:.3f}" for m in available_models}
        selected_model_id = st.selectbox(
            "Select OLS Model Specification:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )

# Resolver execution
resolver = ReportSourceResolver(
    regression_tracker=regression_tracker,
    comparison_tracker=comparison_tracker,
    diagnostic_suites=raw_diagnostic_suites,
    feature_selection_tracker=fs_tracker,
    dataset_metadata=current_meta,
    split_result=split_res
)
resolved_sources = resolver.resolve(
    selected_model_id=selected_model_id,
    comparison_id=selected_comp_id
)

st.divider()


# -----------------------------------------------------------------------------
# SECTION B: EVIDENCE READINESS DASHBOARD
# -----------------------------------------------------------------------------
st.header("B. Evidence Readiness Dashboard")

# Preliminary readiness evaluation
prelim_readiness = evaluate_report_readiness(
    sources=resolved_sources,
    expected_dataset_id=active_ds_id,
    expected_dataset_fingerprint=active_ds_fp,
    expected_split_fingerprint=split_res.split_fingerprint
)

# Render Status Banner
status_col, v_col, m_col, w_col = st.columns([2, 1, 1, 1])

with status_col:
    if prelim_readiness.status == ReadinessStatus.READY:
        st.success("### Status: 🟢 READY FOR ACADEMIC GENERATION\nAll core experimental artifacts verified.")
    elif prelim_readiness.status == ReadinessStatus.READY_WITH_LIMITATIONS:
        st.warning("### Status: 🟡 READY WITH LIMITATIONS\nCore model verified, but optional diagnostics or artifacts are missing.")
    else:
        st.error("### Status: 🔴 GENERATION BLOCKED\nIntegrity requirements violated. Report cannot be certified.")

with v_col:
    st.metric("Primary Artifacts", "Verified" if resolved_sources.model else "Missing")
with m_col:
    st.metric("Blockers", len(prelim_readiness.blockers))
with w_col:
    st.metric("Limitations", len(prelim_readiness.limitations))

if prelim_readiness.blockers:
    with st.container(border=True):
        st.error("🚫 **Integrity Blockers Preventing Report Generation**:")
        for b in prelim_readiness.blockers:
            st.markdown(f"- {b}")

if prelim_readiness.limitations:
    with st.expander("ℹ️ **Disclosed Limitations & Missing Optional Material**", expanded=False):
        for lim in prelim_readiness.limitations:
            st.markdown(f"- {lim}")

if prelim_readiness.warnings:
    with st.expander("⚠️ **Methodological Warnings**", expanded=False):
        for warn in prelim_readiness.warnings:
            st.markdown(f"- {warn}")

st.divider()


# -----------------------------------------------------------------------------
# SECTION C: TEMPLATE CONFIGURATION
# -----------------------------------------------------------------------------
st.header("C. Template Configuration")

c_col1, c_col2 = st.columns(2)

with c_col1:
    template_choice = st.selectbox(
        "Academic Report Template:",
        options=[
            ("full_academic", "Template A — Full Academic Report (9 Sections)"),
            ("executive_summary", "Template B — Executive Summary (Condensed)"),
            ("statistical_diagnostics", "Template C — Statistical Diagnostics Report (Specialized)")
        ],
        format_func=lambda x: x[1]
    )[0]
    template_enum = ReportTemplateType(template_choice)

    rep_title = st.text_input(
        "Report Title:",
        value=f"Empirical OLS Analysis: {active_ds_name} Regression Modeling"
    )

with c_col2:
    author_name = st.text_input("Lead Author / Student Name:", value="Student Researcher")
    research_q = st.text_input(
        "Research Question / Objective:",
        value=f"How effectively do selected predictors estimate {current_meta.target_column} under Gauss-Markov assumptions?"
    )

# Section toggles based on template
st.markdown("##### Section Inclusion Controls")
sec_cols = st.columns(3)
sec_map = {
    SectionID.SEC_EXEC_SUMMARY.value: ("1. Exec Summary", True),
    SectionID.SEC_DATASET.value: ("2. Dataset & Split", True),
    SectionID.SEC_EDA_SELECTION.value: ("3. EDA & Selection", template_enum == ReportTemplateType.FULL_ACADEMIC),
    SectionID.SEC_OLS_METHODOLOGY.value: ("4. OLS Methodology", template_enum != ReportTemplateType.EXECUTIVE_SUMMARY),
    SectionID.SEC_COEFFICIENTS.value: ("5. Coefficients & Fit", True),
    SectionID.SEC_DIAGNOSTICS.value: ("6. Assumption Tests", template_enum != ReportTemplateType.EXECUTIVE_SUMMARY),
    SectionID.SEC_COMPARISON.value: ("7. Model Comparison", template_enum != ReportTemplateType.STATISTICAL_DIAGNOSTICS),
    SectionID.SEC_DISCUSSION.value: ("8. Discussion & Notes", True),
    SectionID.SEC_REPRODUCIBILITY.value: ("9. Reproducibility", True),
}

enabled_sections = []
for idx, (s_key, (s_label, s_default)) in enumerate(sec_map.items()):
    col_target = sec_cols[idx % 3]
    if col_target.checkbox(s_label, value=s_default, key=f"sec_toggle_{s_key}"):
        enabled_sections.append(s_key)

# Final Test Appendix Authorization
include_test_appendix = False
if resolved_sources.model and resolved_sources.model.test_evaluation_status != "untested":
    st.info("🔒 **Controlled Final Test Evaluation Available**: Model has a previously authorized Test evaluation artifact.")
    include_test_appendix = st.checkbox("Include Final Out-of-Sample Test Evaluation Appendix", value=False)
else:
    st.caption("🔒 *Final Test Appendix Unavailable*: Model has not completed out-of-sample Test evaluation. Test Lock strictly preserved (Rule 4).")

st.divider()


# -----------------------------------------------------------------------------
# SECTION F: AUTHOR DISCUSSION & REFLECTION (CONFIGURED BEFORE GENERATION)
# -----------------------------------------------------------------------------
st.header("F. Author Reflections & Discussion (Editable)")

st.caption(
    "Author reflections are clearly labeled and segregated from verified empirical evidence in all generated reports."
)

with st.expander("✍️ Author Analytical Reflections & Study Limitations", expanded=False):
    af_findings = st.text_area("Key Practical Findings:", placeholder="Describe the practical domain significance of the observed relationships...")
    af_limits = st.text_area("Study & Modeling Limitations:", placeholder="Document omitted variables, data collection biases, measurement errors...")
    af_future = st.text_area("Future Improvements & Next Steps:", placeholder="Suggest additional nonlinear features, robust standard errors, or instrumentation...")

author_obj = AuthorContent(
    author_name=author_name,
    research_question=research_q,
    main_findings=af_findings,
    modeling_limitations=af_limits,
    future_improvements=af_future
)

st.divider()


# -----------------------------------------------------------------------------
# REPORT GENERATION & PREVIEW ORCHESTRATION
# -----------------------------------------------------------------------------
if prelim_readiness.is_blocked:
    st.error("Cannot generate report while integrity blockers are active. Please address the errors above.")
    st.stop()

# Run report runner
current_report: AcademicReportResult = run_academic_report(
    sources=resolved_sources,
    template_type=template_enum,
    report_title=rep_title,
    author_content=author_obj,
    enabled_section_ids=enabled_sections,
    include_final_test_authorized=include_test_appendix,
    expected_dataset_id=active_ds_id,
    expected_dataset_fingerprint=active_ds_fp,
    expected_split_fingerprint=split_res.split_fingerprint
)

# Register report into Report Tracker
report_tracker.add_report(current_report)

# Build figure registry for preview and export
fig_registry = ReportFigureRegistry(sources=resolved_sources)
fig_registry.build_all_available_figures()

export_service = ReportExportService(
    report=current_report,
    sources=resolved_sources,
    figure_registry=fig_registry
)


# -----------------------------------------------------------------------------
# SECTION D: REPORT PREVIEW
# -----------------------------------------------------------------------------
st.header("D. Academic Report Preview")

with st.container(border=True):
    st.markdown(f"### {current_report.report_title}")
    st.caption(
        f"**Author**: {current_report.author_content.author_name} | "
        f"**Template**: {current_report.template_type.replace('_', ' ').title()} | "
        f"**Report ID**: `{current_report.report_id}` | "
        f"**Verified Claims**: {current_report.evidence_manifest.verified_count}"
    )

    # Render each section cleanly
    for sec in current_report.report_sections:
        with st.expander(f"📑 {sec.title}", expanded=(sec.order in (1, 5, 6))):
            for block in sec.content_blocks:
                st.markdown(block)

            for tbl in sec.tables:
                st.markdown(f"**{tbl.get('title', 'Table')}**")
                df_tbl = pd.DataFrame(tbl.get("rows", []))
                st.dataframe(df_tbl, use_container_width=True, hide_index=True)

            for fid in sec.figure_ids:
                fig_ref = fig_registry.get_reference(fid)
                fig_obj = fig_registry.get_figure(fid)
                if fig_ref:
                    st.markdown(f"**Figure: {fig_ref.title}**")
                    if fig_obj is not None:
                        st.plotly_chart(fig_obj, use_container_width=True)
                    else:
                        st.info(f"Figure unavailable: {fig_ref.caveats[0] if fig_ref.caveats else 'Data not available'}")
                    st.caption(f"**Caption**: {fig_ref.caption}")

st.divider()


# -----------------------------------------------------------------------------
# SECTION E: TABLES & FIGURES AUDIT
# -----------------------------------------------------------------------------
st.header("E. Visual Assets & Figures Provenance")

with st.container(border=True):
    fig_summary_rows = []
    for fref in current_report.figure_references:
        fig_summary_rows.append({
            "Figure ID": fref.figure_id,
            "Title": fref.title,
            "Data Scope": fref.data_scope.capitalize(),
            "Status": "✅ Rendered" if fref.is_available else "⚠️ Unavailable",
            "Provenance Reference": fref.provenance,
            "Methodological Caveat": fref.caveats[0] if fref.caveats else "None"
        })
    st.dataframe(pd.DataFrame(fig_summary_rows), use_container_width=True, hide_index=True)

st.divider()


# -----------------------------------------------------------------------------
# SECTION G: REPRODUCIBILITY AUDIT
# -----------------------------------------------------------------------------
st.header("G. Reproducibility & Provenance Audit")

with st.container(border=True):
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Verified Claims", current_report.evidence_manifest.verified_count)
    col_r2.metric("Missing Claims", current_report.evidence_manifest.missing_count)
    col_r3.metric("Reproducibility Scope", "Development Replay (Train+Val)")

    st.markdown(
        f"- **Dataset Fingerprint**: `{current_report.dataset_fingerprint}`\n"
        f"- **Split Fingerprint**: `{current_report.split_fingerprint}`\n"
        f"- **Model ID**: `{current_report.selected_model_id}`\n"
        f"- **Comparison ID**: `{current_report.comparison_id or 'None'}`\n"
        f"- **Reproducibility Hash**: `{current_report.reproducibility_hash}`"
    )

st.divider()


# -----------------------------------------------------------------------------
# SECTION H: EXPORT CONTROLS & REPORT HISTORY
# -----------------------------------------------------------------------------
st.header("H. Export Laboratory & Report Archive")

col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)

# 1. Markdown Export
with col_exp1:
    md_text = export_service.export_markdown()
    st.download_button(
        label="📄 Download Markdown (.md)",
        data=md_text,
        file_name=f"{current_report.report_id}.md",
        mime="text/markdown",
        use_container_width=True
    )

# 2. Standalone HTML Export
with col_exp2:
    html_text = export_service.export_html()
    st.download_button(
        label="🌐 Download Standalone HTML (.html)",
        data=html_text,
        file_name=f"{current_report.report_id}.html",
        mime="text/html",
        use_container_width=True
    )

# 3. Evidence Manifest JSON
with col_exp3:
    manifest_json = export_service.export_manifest_json()
    st.download_button(
        label="📊 Download Manifest (.json)",
        data=manifest_json,
        file_name=f"{current_report.report_id}_manifest.json",
        mime="application/json",
        use_container_width=True
    )

# 4. Reproduction Package ZIP
with col_exp4:
    zip_bytes = export_service.export_reproduction_zip()
    st.download_button(
        label="📦 Download Reproduction Package (.zip)",
        data=zip_bytes,
        file_name=f"{current_report.report_id}_reproducibility_package.zip",
        mime="application/zip",
        use_container_width=True
    )

# Report Archive History
st.subheader("Saved Report History")
all_saved = report_tracker.get_all_reports()
if all_saved:
    history_rows = []
    for r in all_saved:
        is_active = (r.split_fingerprint == split_res.split_fingerprint)
        history_rows.append({
            "Report ID": r.report_id,
            "Title": r.report_title,
            "Template": r.template_type.replace('_', ' ').title(),
            "Model ID": r.selected_model_id,
            "Status": "🟢 Active Split" if is_active else "⚠️ Stale Partition",
            "Verified Claims": r.evidence_manifest.verified_count if r.evidence_manifest else 0,
            "Timestamp": r.generated_timestamp
        })
    st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No prior reports stored in this session.")
