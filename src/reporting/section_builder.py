"""
Academic Report Section Builder.
Assembles structured, typed report sections with verifiable evidence linkages,
mathematical formulas, structured tables, and educational caveats.
"""
from typing import Any, Dict, List, Optional
import html
import numpy as np
import pandas as pd


def _escape_user_text(val: Optional[str]) -> str:
    if not val:
        return ""
    return html.escape(str(val))

from src.reporting.result_models import (
    ReportSection,
    SectionID,
    ReportTemplateType,
    AuthorContent,
    EvidenceManifest
)
from src.reporting.comparison_adapter import adapt_comparison_leaderboard
from src.reporting.source_resolver import ResolvedReportSources
from src.reporting.interpretation import (
    interpret_hypothesis_test,
    get_diagnostic_interpretation_guidance,
    get_core_methodological_caveats
)
from src.diagnostics.result_models import DiagnosticDecision


class AcademicSectionBuilder:
    """
    Constructs structured academic sections corresponding to Templates A, B, and C.
    """

    def __init__(
        self,
        sources: ResolvedReportSources,
        manifest: EvidenceManifest,
        author_content: Optional[AuthorContent] = None,
        include_final_test_authorized: bool = False
    ):
        self.sources = sources
        self.manifest = manifest
        self.author_content = author_content or AuthorContent()
        self.include_final_test = include_final_test_authorized

    def build_sections_for_template(
        self,
        template_type: ReportTemplateType = ReportTemplateType.FULL_ACADEMIC,
        section_ids_enabled: Optional[List[str]] = None
    ) -> List[ReportSection]:
        """
        Builds the ordered list of ReportSections according to template and user toggles.
        """
        sections: List[ReportSection] = []

        if template_type == ReportTemplateType.FULL_ACADEMIC:
            target_ids = [
                SectionID.SEC_EXEC_SUMMARY.value,
                SectionID.SEC_DATASET.value,
                SectionID.SEC_EDA_SELECTION.value,
                SectionID.SEC_OLS_METHODOLOGY.value,
                SectionID.SEC_COEFFICIENTS.value,
                SectionID.SEC_DIAGNOSTICS.value,
                SectionID.SEC_COMPARISON.value,
                SectionID.SEC_DISCUSSION.value,
                SectionID.SEC_REPRODUCIBILITY.value,
            ]
        elif template_type == ReportTemplateType.EXECUTIVE_SUMMARY:
            target_ids = [
                SectionID.SEC_EXEC_SUMMARY.value,
                SectionID.SEC_DATASET.value,
                SectionID.SEC_COEFFICIENTS.value,
                SectionID.SEC_COMPARISON.value,
                SectionID.SEC_DISCUSSION.value,
                SectionID.SEC_REPRODUCIBILITY.value,
            ]
        elif template_type == ReportTemplateType.STATISTICAL_DIAGNOSTICS:
            target_ids = [
                SectionID.SEC_OLS_METHODOLOGY.value,
                SectionID.SEC_COEFFICIENTS.value,
                SectionID.SEC_DIAGNOSTICS.value,
                SectionID.SEC_DISCUSSION.value,
                SectionID.SEC_REPRODUCIBILITY.value,
            ]
        else:
            target_ids = [SectionID.SEC_EXEC_SUMMARY.value]

        # Filter by user configuration if provided
        if section_ids_enabled is not None:
            target_ids = [sid for sid in target_ids if sid in section_ids_enabled]

        order = 1
        for sid in target_ids:
            sec = self._build_section_by_id(sid, order)
            if sec is not None:
                sections.append(sec)
                order += 1

        # Optional Appendix: Final Test Evaluation (only if authorized and requested)
        if self.include_final_test and self.sources.model and self.sources.model.test_evaluation_status != "untested":
            test_sec = self._build_final_test_appendix(order)
            if test_sec is not None:
                sections.append(test_sec)

        return sections

    def _build_section_by_id(self, section_id: str, order: int) -> Optional[ReportSection]:
        if section_id == SectionID.SEC_EXEC_SUMMARY.value:
            return self._build_sec_1_exec_summary(order)
        elif section_id == SectionID.SEC_DATASET.value:
            return self._build_sec_2_dataset(order)
        elif section_id == SectionID.SEC_EDA_SELECTION.value:
            return self._build_sec_3_eda_selection(order)
        elif section_id == SectionID.SEC_OLS_METHODOLOGY.value:
            return self._build_sec_4_ols_methodology(order)
        elif section_id == SectionID.SEC_COEFFICIENTS.value:
            return self._build_sec_5_coefficients(order)
        elif section_id == SectionID.SEC_DIAGNOSTICS.value:
            return self._build_sec_6_diagnostics(order)
        elif section_id == SectionID.SEC_COMPARISON.value:
            return self._build_sec_7_comparison(order)
        elif section_id == SectionID.SEC_DISCUSSION.value:
            return self._build_sec_8_discussion(order)
        elif section_id == SectionID.SEC_REPRODUCIBILITY.value:
            return self._build_sec_9_reproducibility(order)
        return None

    # -------------------------------------------------------------------------
    # Section 1: Executive Summary & Research Objectives
    # -------------------------------------------------------------------------
    def _build_sec_1_exec_summary(self, order: int) -> ReportSection:
        model = self.sources.model
        meta = self.sources.dataset_metadata
        comp = self.sources.comparison

        ds_name = getattr(meta, "name", getattr(meta, "dataset_name", model.dataset_id if model else "Dataset")) if meta else (model.dataset_id if model else "Dataset")
        target_name = model.target_name if model else "Target"
        target_unit = f" ({model.target_unit})" if model and model.target_unit else ""
        rq = _escape_user_text(self.author_content.research_question) if self.author_content.research_question else f"To model and predict {target_name}{target_unit} using multiple linear regression under Gauss-Markov assumptions."

        blocks = [
            f"**Dataset**: {ds_name} | **Target Variable**: `{target_name}`{target_unit}",
            f"**Research Question**: {rq}",
        ]

        if model:
            blocks.append(
                f"This report presents an empirical Ordinary Least Squares (OLS) regression analysis. "
                f"The evaluated specification, **{model.model_name}** (`{model.model_id}`), was estimated "
                f"using {model.design_matrix_metadata.num_parameters} design parameters across {len(model.selected_original_features)} "
                f"selected original predictors. On the strictly isolated validation partition, the model achieved a "
                f"Root Mean Squared Error (RMSE) of **{model.val_metrics.rmse:.3f}{target_unit}** and a coefficient of "
                f"determination ($R^2$) of **{model.val_metrics.r2:.4f}**."
            )

        if comp and comp.candidate_selection_rationale:
            blocks.append(
                f"**Selection Audit**: This model was identified as the preferred candidate among "
                f"{len(comp.compared_model_ids)} evaluated alternatives. *Candidate Rationale*: \"{comp.candidate_selection_rationale}\""
            )

        evidence_refs = ["dataset_name", "dataset_target_name", "model_name", "val_rmse", "val_r2"]
        if comp:
            evidence_refs.extend(["comp_candidate_id", "comp_candidate_rationale"])

        return ReportSection(
            section_id=SectionID.SEC_EXEC_SUMMARY.value,
            title="1. Executive Summary & Research Objectives",
            order=order,
            content_blocks=blocks,
            evidence_reference_ids=evidence_refs
        )

    # -------------------------------------------------------------------------
    # Section 2: Dataset Description & Data Preparation
    # -------------------------------------------------------------------------
    def _build_sec_2_dataset(self, order: int) -> ReportSection:
        meta = self.sources.dataset_metadata
        split = self.sources.split_result
        model = self.sources.model

        blocks = []
        tables = []
        warnings = []

        if meta:
            ds_name = getattr(meta, "name", getattr(meta, "dataset_name", model.dataset_id if model else "Dataset"))
            source_cit = getattr(meta, "original_reference", getattr(meta, "provenance_notes", "Verified Registry"))
            tot_rows = getattr(meta, "expected_rows", (len(split.X_train) + len(split.X_val) + len(split.X_test)) if split else 0)
            tot_cols = getattr(meta, "expected_original_columns", 0)
            excl_cols = getattr(meta, "excluded_predictor_columns", getattr(meta, "excluded_columns", []))

            blocks.append(
                f"The study utilizes the **{ds_name}** dataset ({source_cit}). "
                f"The dataset comprises {tot_rows:,} total observations and {tot_cols} raw columns."
            )
            if excl_cols:
                blocks.append(f"**Excluded Identifier/Metadata Columns**: {', '.join(excl_cols)}.")

        if split:
            strat_val = getattr(split, "strategy", getattr(split, "strategy_used", "random"))
            strat_note = ""
            if strat_val == "temporal":
                strat_note = " Chronological sequence was strictly preserved to prevent future-to-past temporal leakage."
            elif strat_val == "group":
                strat_note = " Group-level isolation was enforced so that no group (e.g., country or entity) appears simultaneously in Train and Validation."

            blocks.append(
                f"**Partitioning Strategy**: {strat_val.capitalize()} splitting.{strat_note}"
            )

            # Split summary table
            n_tr = len(split.X_train) if hasattr(split, "X_train") and split.X_train is not None else getattr(split, "n_train", 0)
            n_va = len(split.X_val) if hasattr(split, "X_val") and split.X_val is not None else getattr(split, "n_val", 0)
            n_te = len(split.X_test) if hasattr(split, "X_test") and split.X_test is not None else getattr(split, "n_test", 0)
            tot_s = n_tr + n_va + n_te if (n_tr + n_va + n_te) > 0 else 1

            split_data = [
                {"Partition": "Training", "Sample Count (n)": f"{n_tr:,}", "Percentage": f"{n_tr/tot_s*100:.1f}%"},
                {"Partition": "Validation", "Sample Count (n)": f"{n_va:,}", "Percentage": f"{n_va/tot_s*100:.1f}%"},
                {"Partition": "Test (Locked)", "Sample Count (n)": f"{n_te:,}", "Percentage": f"{n_te/tot_s*100:.1f}%"}
            ]
            tables.append({
                "table_id": "tbl_split_summary",
                "title": "Data Splitting & Sample Partitioning",
                "headers": ["Partition", "Sample Count (n)", "Percentage"],
                "rows": split_data
            })

        evidence_refs = ["dataset_name", "dataset_total_rows", "split_strategy", "split_train_rows", "split_val_rows", "split_test_rows"]

        return ReportSection(
            section_id=SectionID.SEC_DATASET.value,
            title="2. Dataset Description & Data Preparation",
            order=order,
            content_blocks=blocks,
            tables=tables,
            evidence_reference_ids=evidence_refs,
            warnings=warnings
        )

    # -------------------------------------------------------------------------
    # Section 3: EDA & Feature Selection Audit
    # -------------------------------------------------------------------------
    def _build_sec_3_eda_selection(self, order: int) -> ReportSection:
        model = self.sources.model
        blocks = []
        tables = []
        evidence_refs = []

        if model is None:
            return ReportSection(section_id=SectionID.SEC_EDA_SELECTION.value, title="3. EDA & Feature Selection", order=order)

        fs_id = model.feature_source_id
        if fs_id in self.sources.feature_selection_results:
            fs = self.sources.feature_selection_results[fs_id]
            blocks.append(
                f"Predictors were selected via the **{fs.method_name}** algorithm ({fs.category} category). "
                f"The algorithm evaluated {fs.total_original_count} candidate original features and selected "
                f"**{fs.selected_original_count}** features (Requested $K = {fs.requested_k}$). "
                f"After reference-level categorical encoding, these span **{fs.selected_encoded_dimension}** design columns."
            )
            blocks.append(
                f"**Cross-Validation Performance**: Under {fs.cv_strategy_used.replace('_', ' ')} cross-validation, "
                f"the selected subset yielded a mean CV RMSE of **{fs.cv_rmse_mean:.3f} ± {fs.cv_rmse_std:.3f}**. "
                f"On the held-out validation set, the subset achieved RMSE = **{fs.val_rmse:.3f}** and $R^2$ = **{fs.val_r2:.4f}**."
            )

            # Selected features table
            feat_rows = []
            for fmap in fs.feature_mappings:
                if fmap.is_selected:
                    feat_rows.append({
                        "Original Feature": fmap.original_name,
                        "Type": fmap.feature_type.capitalize(),
                        "Encoded Dim": str(fmap.encoded_dimension),
                        "Encoded Columns": ", ".join(fmap.encoded_column_names)
                    })
            if feat_rows:
                tables.append({
                    "table_id": "tbl_selected_features",
                    "title": f"Selected Feature Subset ({fs.method_name})",
                    "headers": ["Original Feature", "Type", "Encoded Dim", "Encoded Columns"],
                    "rows": feat_rows
                })
            evidence_refs.extend(["fs_method_name", "fs_requested_k", "fs_cv_rmse_mean", "fs_cv_rmse_std"])
        elif fs_id == "all_features":
            blocks.append(
                f"This specification represents an **All Features (Full Baseline)** regression model. "
                f"All {len(model.selected_original_features)} original candidate predictors were included without subset selection."
            )
            evidence_refs.append("fs_method_name")
        else:
            blocks.append(
                f"*Limitation*: Referenced feature selection experiment `{fs_id}` was not found in the active session registry. "
                "Feature selection CV metrics cannot be independently verified."
            )

        return ReportSection(
            section_id=SectionID.SEC_EDA_SELECTION.value,
            title="3. Exploratory Data Analysis & Feature Selection Audit",
            order=order,
            content_blocks=blocks,
            tables=tables,
            evidence_reference_ids=evidence_refs
        )

    # -------------------------------------------------------------------------
    # Section 4: OLS Regression Methodology
    # -------------------------------------------------------------------------
    def _build_sec_4_ols_methodology(self, order: int) -> ReportSection:
        model = self.sources.model
        if model is None:
            return ReportSection(section_id=SectionID.SEC_OLS_METHODOLOGY.value, title="4. OLS Methodology", order=order)

        blocks = []
        meta = model.design_matrix_metadata

        intercept_desc = "included (β₀)" if model.fit_intercept else "suppressed through origin"
        scale_desc = "standardized (z-score)" if model.scale_numeric else "unscaled (raw measurement units)"

        blocks.append(
            f"The Ordinary Least Squares (OLS) model was estimated with the conditional expectation function "
            f"$$\\mathbb{{E}}[Y \\mid X] = X\\beta$$. An intercept term was **{intercept_desc}**, and numeric regressors "
            f"were **{scale_desc}**. Categorical regressors were expanded using one-hot dummy coding with an explicit reference level."
        )

        n_obs = getattr(model.train_metrics, "sample_count", getattr(model.design_matrix_metadata, "sample_size", model.df_resid + meta.num_parameters))
        m_rank = getattr(meta, "matrix_rank", getattr(meta, "rank", meta.num_parameters))

        blocks.append(
            f"**Design Matrix Geometry**: The training design matrix contains $n = {n_obs:,}$ "
            f"observations and $p = {meta.num_parameters}$ estimated parameters. "
            f"Matrix rank is **{m_rank}** ({'Full Column Rank' if meta.is_full_rank else 'Rank Deficient'}), "
            f"yielding **{model.df_resid}** residual degrees of freedom ($df_{{resid}} = n - p$). "
            f"The 2-norm Condition Number is **{meta.condition_number:.2f}**."
        )

        blocks.append(f"**Regression Equation (Symbolic)**:\n$$\n{model.equation_general}\n$$")

        caveats = get_core_methodological_caveats(model.is_valid_for_inference, model.post_selection_inference_warning)
        blocks.extend(caveats)

        evidence_refs = [
            "model_num_parameters", "model_is_full_rank", "model_rank",
            "model_condition_number", "model_is_valid_for_inference", "model_equation_text"
        ]

        return ReportSection(
            section_id=SectionID.SEC_OLS_METHODOLOGY.value,
            title="4. OLS Regression Methodology & Model Specification",
            order=order,
            content_blocks=blocks,
            evidence_reference_ids=evidence_refs
        )

    # -------------------------------------------------------------------------
    # Section 5: Coefficients & Model Performance
    # -------------------------------------------------------------------------
    def _build_sec_5_coefficients(self, order: int) -> ReportSection:
        model = self.sources.model
        if model is None:
            return ReportSection(section_id=SectionID.SEC_COEFFICIENTS.value, title="5. Coefficients & Performance", order=order)

        blocks = []
        tables = []
        u = f" {model.target_unit}" if model.target_unit else ""

        # Performance Overview Table
        perf_rows = [
            {"Metric": "RMSE (Root Mean Squared Error)", "Training Set": f"{model.train_metrics.rmse:.3f}{u}", "Validation Set": f"{model.val_metrics.rmse:.3f}{u}", "Interpretation": "Prediction error standard deviation"},
            {"Metric": "MAE (Mean Absolute Error)", "Training Set": f"{model.train_metrics.mae:.3f}{u}", "Validation Set": f"{model.val_metrics.mae:.3f}{u}", "Interpretation": "Average absolute deviation"},
            {"Metric": "R² (Coefficient of Determination)", "Training Set": f"{model.train_metrics.r2:.4f}", "Validation Set": f"{model.val_metrics.r2:.4f}", "Interpretation": "Proportion of target variance explained"},
            {"Metric": "Adjusted R²", "Training Set": f"{model.train_metrics.adj_r2:.4f}" if model.train_metrics.adj_r2 is not None else "N/A", "Validation Set": "N/A (In-sample only)", "Interpretation": "Penalized for parameter count (Train only)"},
        ]
        tables.append({
            "table_id": "tbl_performance_summary",
            "title": "Model Goodness-of-Fit & Out-of-Sample Performance",
            "headers": ["Metric", "Training Set", "Validation Set", "Interpretation"],
            "rows": perf_rows
        })

        gen_gap = model.val_metrics.rmse - model.train_metrics.rmse
        blocks.append(
            f"**Generalization Gap**: The generalization gap ($RMSE_{{val}} - RMSE_{{train}}$) is **{gen_gap:.3f}{u}**. "
            f"The overall model F-statistic is **{model.f_statistic:.2f}** ($p = {model.f_pvalue:.4e}$), confirming that "
            f"the predictors collectively explain a statistically significant proportion of variance relative to an intercept-only null."
        )

        # Coefficients Table
        coef_rows = []
        if model.coef_table is not None and not model.coef_table.empty:
            for term, row in model.coef_table.iterrows():
                coef_val = row.get("coef")
                c_str = f"{coef_val:.4f}" if coef_val is not None else "N/A"

                if model.is_valid_for_inference:
                    se_val = row.get("std_err")
                    t_val = row.get("t_stat")
                    p_val = row.get("p_value")
                    ci_l = row.get("ci_lower")
                    ci_u = row.get("ci_upper")

                    se_str = f"{se_val:.4f}" if se_val is not None else "N/A"
                    t_str = f"{t_val:.3f}" if t_val is not None else "N/A"
                    p_str = "< 0.0001" if (p_val is not None and p_val < 0.0001) else (f"{p_val:.4f}" if p_val is not None else "N/A")
                    ci_str = f"[{ci_l:.3f}, {ci_u:.3f}]" if (ci_l is not None and ci_u is not None) else "N/A"
                else:
                    se_str = "N/A (Prediction Only)"
                    t_str = "N/A (Prediction Only)"
                    p_str = "N/A (Prediction Only)"
                    ci_str = "N/A (Prediction Only)"

                coef_rows.append({
                    "Regressor": str(term),
                    "Estimate (β̂)": c_str,
                    "Std. Error": se_str,
                    "t-statistic": t_str,
                    "p-value": p_str,
                    "95% Conf. Interval": ci_str
                })

            tables.append({
                "table_id": "tbl_coefficients",
                "title": f"Ordinary Least Squares Parameter Estimates ({'Inference Valid' if model.is_valid_for_inference else 'Prediction Only — Inference Blocked'})",
                "headers": ["Regressor", "Estimate (β̂)", "Std. Error", "t-statistic", "p-value", "95% Conf. Interval"],
                "rows": coef_rows
            })

        evidence_refs = [
            "train_rmse", "val_rmse", "train_mae", "val_mae", "train_r2", "val_r2",
            "train_adj_r2", "generalization_gap", "model_f_statistic", "model_f_pvalue"
        ]

        return ReportSection(
            section_id=SectionID.SEC_COEFFICIENTS.value,
            title="5. Parameter Estimates & Goodness-of-Fit",
            order=order,
            content_blocks=blocks,
            tables=tables,
            figure_ids=["fig_actual_vs_pred"],
            evidence_reference_ids=evidence_refs
        )

    # -------------------------------------------------------------------------
    # Section 6: Statistical Assumption Diagnostics
    # -------------------------------------------------------------------------
    def _build_sec_6_diagnostics(self, order: int) -> ReportSection:
        diag = self.sources.diagnostic_suite
        model = self.sources.model

        blocks = []
        tables = []
        evidence_refs = []

        if diag is None:
            blocks.append(
                "*Limitation*: No statistical assumption diagnostic suite has been executed for this model specification. "
                "Compliance with Gauss-Markov conditions (linearity, homoscedasticity, independence, normality, and non-collinearity) "
                "cannot be formally audited from this report."
            )
            return ReportSection(
                section_id=SectionID.SEC_DIAGNOSTICS.value,
                title="6. Statistical Assumption Diagnostics",
                order=order,
                content_blocks=blocks
            )

        blocks.append(
            f"The statistical validity of classical OLS inference rests on Gauss-Markov assumptions. "
            f"Formal diagnostic tests were conducted on model `{diag.model_id}` at significance level $\\alpha = {diag.alpha}$."
        )

        diag_rows = []
        for m_id, d_res in diag.diagnostics.items():
            clean_m = m_id.lower().replace("-", "_").replace(" ", "_")
            stat_str = f"{d_res.statistic_value:.4f}" if np.isfinite(d_res.statistic_value) else "N/A"
            pval_str = "< 0.0001" if (d_res.p_value is not None and d_res.p_value < 0.0001) else (f"{d_res.p_value:.4f}" if d_res.p_value is not None else "N/A")
            guidance = get_diagnostic_interpretation_guidance(
                d_res.diagnostic_type, m_id, d_res.decision, d_res.p_value, d_res.statistic_value, diag.alpha
            )

            diag_rows.append({
                "Category": d_res.diagnostic_type.capitalize(),
                "Diagnostic Test": d_res.method_name,
                "Statistic": f"{d_res.statistic_name} = {stat_str}" if stat_str != "N/A" else "N/A",
                "p-value": pval_str,
                "Decision": d_res.decision.replace("_", " ").title(),
                "Applicability": d_res.applicability_status.title(),
                "Finding": guidance["finding"]
            })
            evidence_refs.extend([f"diag_stat_{clean_m}", f"diag_pval_{clean_m}", f"diag_decision_{clean_m}"])

        tables.append({
            "table_id": "tbl_diagnostics_audit",
            "title": f"Gauss-Markov Diagnostic Test Battery (α = {diag.alpha})",
            "headers": ["Category", "Diagnostic Test", "Statistic", "p-value", "Decision", "Applicability", "Finding"],
            "rows": diag_rows
        })

        blocks.append(
            "**Note on Composite Scores**: This laboratory strictly avoids composite diagnostic pass counts or percentage scores. "
            "Individual diagnostic tests assess mathematically distinct assumptions, each carrying different econometric implications "
            "for estimator bias, consistency, and standard error efficiency."
        )

        return ReportSection(
            section_id=SectionID.SEC_DIAGNOSTICS.value,
            title="6. Statistical Assumption Diagnostics",
            order=order,
            content_blocks=blocks,
            tables=tables,
            figure_ids=["fig_residuals_vs_fitted", "fig_normal_qq", "fig_vif_bars"],
            evidence_reference_ids=evidence_refs
        )

    # -------------------------------------------------------------------------
    # Section 7: Model Comparison & Candidate Selection
    # -------------------------------------------------------------------------
    def _build_sec_7_comparison(self, order: int) -> ReportSection:
        comp = self.sources.comparison
        model = self.sources.model

        blocks = []
        tables = []
        evidence_refs = []

        if comp is None:
            blocks.append(
                "*Note*: This report evaluated an individual OLS model specification without a multi-model comparison record."
            )
            return ReportSection(
                section_id=SectionID.SEC_COMPARISON.value,
                title="7. Multi-Model Comparison & Candidate Selection",
                order=order,
                content_blocks=blocks
            )

        u = f" {comp.target_unit}" if comp.target_unit else ""
        blocks.append(
            f"A multi-model benchmarking experiment (`{comp.comparison_id}`) evaluated **{len(comp.compared_model_ids)}** "
            f"candidate OLS specifications on the validation partition under ranking policy `{comp.ranking_policy}`."
        )

        # Leaderboard table using canonical typed adapter
        target_cand_id = comp.selected_candidate_model_id or (self.sources.model.model_id if self.sources.model else None)
        canonical_rows = adapt_comparison_leaderboard(comp)
        lb_rows = []
        for row in canonical_rows:
            is_cand = (row.model_id == target_cand_id)
            cand_tag = " ⭐ (Candidate)" if is_cand else ""
            model_id_tag = f"{row.model_id} [Selected Candidate]" if is_cand else row.model_id
            val_rmse_str = f"{row.val_rmse:.3f}{u}"
            val_r2_str = f"{row.val_r2:.4f}"
            gen_gap_str = f"{row.generalization_gap:.3f}{u}" if row.generalization_gap is not None else "N/A"
            pareto_str = "Yes (Pareto Efficient)" if row.is_pareto else "No (Dominated)"

            lb_rows.append({
                "Rank": str(row.rank),
                "Model Name": f"{row.model_name}{cand_tag}",
                "Model ID": model_id_tag,
                "Feature Method": row.feature_source,
                "Params (p)": str(row.parameter_count),
                "Val RMSE": val_rmse_str,
                "Val R²": val_r2_str,
                "Gen. Gap": gen_gap_str,
                "Pareto Status": pareto_str
            })

        tables.append({
            "table_id": "tbl_comparison_leaderboard",
            "title": "Validation-Based Candidate Leaderboard (Strictly No Test Data)",
            "headers": ["Rank", "Model Name", "Model ID", "Feature Method", "Params (p)", "Val RMSE", "Val R²", "Gen. Gap", "Pareto Status"],
            "rows": lb_rows
        })

        if comp.selected_candidate_model_id:
            blocks.append(
                f"**Formally Selected Candidate**: `{comp.selected_candidate_model_id}`\n"
                f"**Recorded Rationale**: \"{comp.candidate_selection_rationale}\""
            )

        blocks.append(
            "**Pareto Frontier & Radar Analysis**: Multi-dimensional trade-offs were examined across accuracy, parsimony, "
            "and generalization stability. Models on the Pareto efficiency frontier represent non-dominated trade-offs between "
            "parameter complexity and validation loss. Radar scores represent min-max normalized rankings strictly relative "
            "to the evaluated set, and must not be used as absolute measures of statistical truth."
        )

        evidence_refs.extend(["comp_num_models", "comp_candidate_id", "comp_candidate_rationale", "comp_is_pareto"])

        return ReportSection(
            section_id=SectionID.SEC_COMPARISON.value,
            title="7. Multi-Model Comparison & Candidate Selection",
            order=order,
            content_blocks=blocks,
            tables=tables,
            figure_ids=["fig_pareto_chart", "fig_radar_chart", "fig_feature_overlap"],
            evidence_reference_ids=evidence_refs
        )

    # -------------------------------------------------------------------------
    # Section 8: Discussion, Limitations & Conclusion
    # -------------------------------------------------------------------------
    def _build_sec_8_discussion(self, order: int) -> ReportSection:
        ac = self.author_content

        blocks = [
            "### 8.1 Methodological & Statistical Synthesis",
            "This empirical regression investigation highlights the fundamental distinction between predictive modeling "
            "and structural causal inference. While Ordinary Least Squares provides minimum-variance unbiased linear predictions "
            "when Gauss-Markov assumptions hold, observational data regressions cannot prove the absence of unobserved confounding. "
            "Furthermore, high $R^2$ values indicate sample variance explained, not causal validity."
        ]

        # Explicit separation of student/author reflection
        blocks.append("### 8.2 Author Interpretation & Analytical Reflection")
        if ac.main_findings:
            blocks.append(f"**Principal Findings (Author)**: {_escape_user_text(ac.main_findings)}")
        if ac.practical_significance:
            blocks.append(f"**Practical Significance (Author)**: {_escape_user_text(ac.practical_significance)}")
        if ac.modeling_limitations:
            blocks.append(f"**Modeling Limitations (Author)**: {_escape_user_text(ac.modeling_limitations)}")
        if ac.statistical_limitations:
            blocks.append(f"**Statistical Limitations (Author)**: {_escape_user_text(ac.statistical_limitations)}")
        if ac.alternative_explanations:
            blocks.append(f"**Alternative Explanations (Author)**: {_escape_user_text(ac.alternative_explanations)}")
        if ac.future_improvements:
            blocks.append(f"**Future Research & Remediation (Author)**: {_escape_user_text(ac.future_improvements)}")

        if not any([ac.main_findings, ac.modeling_limitations, ac.future_improvements]):
            blocks.append(
                "*Note*: No specific author reflections were recorded during report configuration. "
                "Students and researchers are encouraged to synthesize findings using Section F of the Report Laboratory."
            )

        return ReportSection(
            section_id=SectionID.SEC_DISCUSSION.value,
            title="8. Discussion, Limitations & Conclusion",
            order=order,
            content_blocks=blocks,
            author_notes="Author reflections are segregated from verified statistical evidence."
        )

    # -------------------------------------------------------------------------
    # Section 9: Reproducibility, References & Appendix
    # -------------------------------------------------------------------------
    def _build_sec_9_reproducibility(self, order: int) -> ReportSection:
        model = self.sources.model
        split = self.sources.split_result
        meta = self.sources.dataset_metadata
        comp = self.sources.comparison

        blocks = [
            "### 9.1 Experimental Provenance & Fingerprints",
            f"- **Report Audit Timestamp**: `{self.manifest.audit_timestamp}`",
            f"- **Evidence Records Verified**: `{self.manifest.verified_count}` verified, `{self.manifest.missing_count}` missing, `{self.manifest.not_applicable_count}` N/A",
            f"- **Dataset Fingerprint**: `{model.dataset_fingerprint if model else 'N/A'}`",
            f"- **Split Fingerprint**: `{split.split_fingerprint if split else 'N/A'}`",
            f"- **Model ID**: `{model.model_id if model else 'N/A'}`",
        ]
        if comp:
            blocks.append(f"- **Comparison ID**: `{comp.comparison_id}`")

        blocks.append(
            "### 9.2 Reproducibility Protocol & Environment\n"
            "All empirical results in this report were generated via deterministic pipelines. "
            "To replicate these findings, use the accompanying standalone Python reproduction script `reproduce.py`. "
            "The reproduction pipeline requires Python 3.11 with `statsmodels`, `scikit-learn`, `numpy`, and `pandas`."
        )

        source_cit = getattr(meta, "original_reference", getattr(meta, "provenance_notes", getattr(meta, "source_citation", None)))
        if meta and source_cit:
            blocks.append(f"### 9.3 Dataset Citation\n{source_cit}")

        return ReportSection(
            section_id=SectionID.SEC_REPRODUCIBILITY.value,
            title="9. Reproducibility, References & Evidence Manifest",
            order=order,
            content_blocks=blocks
        )

    # -------------------------------------------------------------------------
    # Optional Appendix: Final Out-of-Sample Test Evaluation
    # -------------------------------------------------------------------------
    def _build_final_test_appendix(self, order: int) -> ReportSection:
        model = self.sources.model
        if model is None or model.test_evaluation_status == "untested" or model.test_metrics is None:
            return None

        u = f" {model.target_unit}" if model.target_unit else ""
        blocks = [
            "> [!IMPORTANT]\n"
            "> **Controlled Final Test Evaluation Disclosure**: This appendix documents a previously executed, "
            "authorized final out-of-sample evaluation. These Test Set metrics were strictly locked during "
            "all exploratory data analysis, feature selection, model fitting, and multi-model candidate ranking. "
            "Test metrics did not influence candidate model selection.",
            f"**Refit Policy**: `{model.test_refit_policy}` "
            f"({'Retrained on combined Train+Validation before Test' if model.test_refit_policy == 'train_val' else 'Evaluated original training-fitted model directly on Test'})."
        ]

        test_rows = [
            {"Metric": "Test RMSE", "Value": f"{model.test_metrics.rmse:.3f}{u}"},
            {"Metric": "Test MAE", "Value": f"{model.test_metrics.mae:.3f}{u}"},
            {"Metric": "Test R²", "Value": f"{model.test_metrics.r2:.4f}"},
            {"Metric": "Test Sample Size (n)", "Value": f"{getattr(model.test_metrics, 'sample_count', getattr(model.test_metrics, 'sample_size', 0)):,}"}
        ]

        tables = [{
            "table_id": "tbl_final_test_metrics",
            "title": f"Authorized Final Out-of-Sample Test Evaluation ({model.model_id})",
            "headers": ["Metric", "Value"],
            "rows": test_rows
        }]

        return ReportSection(
            section_id=SectionID.SEC_APPENDIX_FINAL_TEST.value,
            title="Appendix: Controlled Final Out-of-Sample Test Evaluation",
            order=order,
            content_blocks=blocks,
            tables=tables,
            evidence_reference_ids=["test_rmse", "test_r2", "test_refit_policy"]
        )
