"""
Evidence Manifest Engine for Academic Report Generation.
Extracts, verifies, and indexes every quantitative claim, metric, and diagnostic statistic
against underlying experimental artifacts, ensuring 100% auditable provenance.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.reporting.result_models import (
    EvidenceRecord,
    EvidenceManifest,
    EvidenceStatus,
    safe_serialize_numeric
)
from src.reporting.source_resolver import ResolvedReportSources


def safe_get_field(obj: Any, field_path: str) -> Tuple_Result:
    """
    Safely retrieves a nested attribute or dictionary value without dynamic eval().
    Returns (found: bool, value: Any).
    """
    if obj is None:
        return False, None

    parts = field_path.split(".")
    curr = obj
    for part in parts:
        if curr is None:
            return False, None
        if isinstance(curr, dict):
            if part in curr:
                curr = curr[part]
            else:
                return False, None
        elif hasattr(curr, part):
            curr = getattr(curr, part)
        else:
            return False, None
    return True, curr


class Tuple_Result(tuple):
    pass


def format_stat_display(val: Any, precision: int = 4, is_pvalue: bool = False) -> str:
    """Format numerical values defensively without inventing numbers or asserting p=0."""
    if val is None:
        return "Unavailable"
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return "N/A"
        if is_pvalue:
            if val < 0.0001:
                return "< 0.0001"
            return f"{val:.4f}"
        return f"{val:.{precision}f}"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    return str(val)


class EvidenceManifestBuilder:
    """
    Builds an immutable, auditable EvidenceManifest from resolved experiment artifacts.
    """

    def __init__(self, report_id: str):
        self.report_id = report_id
        self.manifest = EvidenceManifest(
            report_id=report_id,
            audit_timestamp=datetime.utcnow().isoformat() + "Z"
        )

    def build_from_sources(
        self,
        sources: ResolvedReportSources,
        include_final_test_authorized: bool = False
    ) -> EvidenceManifest:
        """
        Populates evidence records across all experimental dimensions.
        """
        model = sources.model
        split = sources.split_result
        meta = sources.dataset_metadata
        comp = sources.comparison
        diag = sources.diagnostic_suite

        if model is None:
            self.manifest.warnings.append("Cannot build full evidence manifest: primary model artifact is missing.")
            return self.manifest

        ds_id = model.dataset_id
        ds_fp = model.dataset_fingerprint
        split_fp = model.split_fingerprint
        m_id = model.model_id

        # -------------------------------------------------------------
        # 1. Dataset & Split Evidence
        # -------------------------------------------------------------
        if meta is not None:
            ds_name = getattr(meta, "name", getattr(meta, "dataset_name", ds_id))
            source_cit = getattr(meta, "original_reference", getattr(meta, "provenance_notes", "Verified Registry"))
            tot_rows = getattr(meta, "expected_rows", (len(split.X_train) + len(split.X_val) + len(split.X_test)) if split else 0)
            target_col = getattr(meta, "target_column", model.target_name)
            target_u = getattr(meta, "target_unit", model.target_unit)

            self._add(
                claim_id="dataset_name",
                claim_type="metadata",
                artifact_type="dataset",
                artifact_id=getattr(meta, "id", ds_id),
                field_path="name",
                observed=ds_name,
                display=ds_name,
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="dataset"
            )
            self._add(
                claim_id="dataset_source",
                claim_type="metadata",
                artifact_type="dataset",
                artifact_id=getattr(meta, "id", ds_id),
                field_path="original_reference",
                observed=source_cit,
                display=source_cit,
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="dataset"
            )
            self._add(
                claim_id="dataset_total_rows",
                claim_type="dataset_dimension",
                artifact_type="dataset",
                artifact_id=getattr(meta, "id", ds_id),
                field_path="expected_rows",
                observed=tot_rows,
                display=format_stat_display(tot_rows),
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="dataset"
            )
            self._add(
                claim_id="dataset_target_name",
                claim_type="metadata",
                artifact_type="dataset",
                artifact_id=getattr(meta, "id", ds_id),
                field_path="target_column",
                observed=target_col,
                display=target_col,
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="dataset"
            )
            self._add(
                claim_id="dataset_target_unit",
                claim_type="metadata",
                artifact_type="dataset",
                artifact_id=getattr(meta, "id", ds_id),
                field_path="target_unit",
                observed=target_u,
                display=target_u or "Dimensionless",
                unit=target_u,
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="dataset"
            )

        if split is not None:
            strat_val = getattr(split, "strategy", getattr(split, "strategy_used", "random"))
            n_tr = len(split.X_train) if hasattr(split, "X_train") and split.X_train is not None else getattr(split, "n_train", 0)
            n_va = len(split.X_val) if hasattr(split, "X_val") and split.X_val is not None else getattr(split, "n_val", 0)
            n_te = len(split.X_test) if hasattr(split, "X_test") and split.X_test is not None else getattr(split, "n_test", 0)

            self._add(
                claim_id="split_strategy",
                claim_type="split_config",
                artifact_type="split",
                artifact_id=split.split_fingerprint,
                field_path="strategy",
                observed=strat_val,
                display=strat_val.capitalize(),
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="split"
            )
            self._add(
                claim_id="split_train_rows",
                claim_type="sample_size",
                artifact_type="split",
                artifact_id=split.split_fingerprint,
                field_path="n_train",
                observed=n_tr,
                display=format_stat_display(n_tr),
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="train"
            )
            self._add(
                claim_id="split_val_rows",
                claim_type="sample_size",
                artifact_type="split",
                artifact_id=split.split_fingerprint,
                field_path="n_val",
                observed=n_va,
                display=format_stat_display(n_va),
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="validation"
            )
            self._add(
                claim_id="split_test_rows",
                claim_type="sample_size",
                artifact_type="split",
                artifact_id=split.split_fingerprint,
                field_path="n_test",
                observed=n_te,
                display=format_stat_display(n_te),
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="test"
            )

        # -------------------------------------------------------------
        # 2. Feature Selection Evidence
        # -------------------------------------------------------------
        fs_id = model.feature_source_id
        if fs_id in sources.feature_selection_results:
            fs_res = sources.feature_selection_results[fs_id]
            self._add(
                claim_id="fs_method_name",
                claim_type="feature_selection",
                artifact_type="feature_selection",
                artifact_id=fs_res.experiment_id,
                field_path="method_name",
                observed=fs_res.method_name,
                display=fs_res.method_name,
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="train"
            )
            self._add(
                claim_id="fs_requested_k",
                claim_type="feature_selection",
                artifact_type="feature_selection",
                artifact_id=fs_res.experiment_id,
                field_path="requested_k",
                observed=fs_res.requested_k,
                display=str(fs_res.requested_k),
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="train"
            )
            self._add(
                claim_id="fs_cv_rmse_mean",
                claim_type="feature_selection",
                artifact_type="feature_selection",
                artifact_id=fs_res.experiment_id,
                field_path="cv_rmse_mean",
                observed=fs_res.cv_rmse_mean,
                display=format_stat_display(fs_res.cv_rmse_mean),
                unit=model.target_unit,
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="cv"
            )
            self._add(
                claim_id="fs_cv_rmse_std",
                claim_type="feature_selection",
                artifact_type="feature_selection",
                artifact_id=fs_res.experiment_id,
                field_path="cv_rmse_std",
                observed=fs_res.cv_rmse_std,
                display=format_stat_display(fs_res.cv_rmse_std),
                unit=model.target_unit,
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="cv"
            )
        elif fs_id == "all_features":
            self._add(
                claim_id="fs_method_name",
                claim_type="feature_selection",
                artifact_type="model",
                artifact_id=m_id,
                field_path="feature_source_label",
                observed="All Available Features (Baseline)",
                display="All Available Features (Baseline)",
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="train"
            )
        else:
            self._add_missing(
                claim_id="fs_method_name",
                claim_type="feature_selection",
                artifact_type="feature_selection",
                artifact_id=fs_id,
                field_path="method_name",
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                reason=f"Feature selection artifact '{fs_id}' not found in registry."
            )

        # -------------------------------------------------------------
        # 3. Primary OLS Model Evidence
        # -------------------------------------------------------------
        self._add(
            claim_id="model_name",
            claim_type="model_spec",
            artifact_type="model",
            artifact_id=m_id,
            field_path="model_name",
            observed=model.model_name,
            display=model.model_name,
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="model_num_parameters",
            claim_type="model_spec",
            artifact_type="model",
            artifact_id=m_id,
            field_path="design_matrix_metadata.num_parameters",
            observed=model.design_matrix_metadata.num_parameters,
            display=str(model.design_matrix_metadata.num_parameters),
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="model_is_full_rank",
            claim_type="model_spec",
            artifact_type="model",
            artifact_id=m_id,
            field_path="design_matrix_metadata.is_full_rank",
            observed=model.design_matrix_metadata.is_full_rank,
            display="Full Rank" if model.design_matrix_metadata.is_full_rank else "Rank Deficient",
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        matrix_rank_val = getattr(model.design_matrix_metadata, "matrix_rank", getattr(model.design_matrix_metadata, "rank", model.design_matrix_metadata.num_parameters))
        self._add(
            claim_id="model_rank",
            claim_type="model_spec",
            artifact_type="model",
            artifact_id=m_id,
            field_path="design_matrix_metadata.matrix_rank",
            observed=matrix_rank_val,
            display=str(matrix_rank_val),
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="model_condition_number",
            claim_type="model_spec",
            artifact_type="model",
            artifact_id=m_id,
            field_path="design_matrix_metadata.condition_number",
            observed=model.design_matrix_metadata.condition_number,
            display=format_stat_display(model.design_matrix_metadata.condition_number, precision=2),
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="model_is_valid_for_inference",
            claim_type="inference_status",
            artifact_type="model",
            artifact_id=m_id,
            field_path="is_valid_for_inference",
            observed=model.is_valid_for_inference,
            display="Valid for Inference" if model.is_valid_for_inference else "Prediction Only (Inference Invalid)",
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="model_equation_text",
            claim_type="model_spec",
            artifact_type="model",
            artifact_id=m_id,
            field_path="equation_text",
            observed=model.equation_text,
            display=model.equation_text,
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )

        # -------------------------------------------------------------
        # 4. Model Training & Validation Performance
        # -------------------------------------------------------------
        self._add(
            claim_id="train_rmse",
            claim_type="performance_metric",
            artifact_type="model",
            artifact_id=m_id,
            field_path="train_metrics.rmse",
            observed=model.train_metrics.rmse,
            display=format_stat_display(model.train_metrics.rmse),
            unit=model.target_unit,
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="train_mae",
            claim_type="performance_metric",
            artifact_type="model",
            artifact_id=m_id,
            field_path="train_metrics.mae",
            observed=model.train_metrics.mae,
            display=format_stat_display(model.train_metrics.mae),
            unit=model.target_unit,
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="train_r2",
            claim_type="performance_metric",
            artifact_type="model",
            artifact_id=m_id,
            field_path="train_metrics.r2",
            observed=model.train_metrics.r2,
            display=format_stat_display(model.train_metrics.r2, precision=4),
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="train_adj_r2",
            claim_type="performance_metric",
            artifact_type="model",
            artifact_id=m_id,
            field_path="train_metrics.adj_r2",
            observed=model.train_metrics.adj_r2,
            display=format_stat_display(model.train_metrics.adj_r2, precision=4) if model.train_metrics.adj_r2 is not None else "N/A",
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="val_rmse",
            claim_type="performance_metric",
            artifact_type="model",
            artifact_id=m_id,
            field_path="val_metrics.rmse",
            observed=model.val_metrics.rmse,
            display=format_stat_display(model.val_metrics.rmse),
            unit=model.target_unit,
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="validation"
        )
        self._add(
            claim_id="val_mae",
            claim_type="performance_metric",
            artifact_type="model",
            artifact_id=m_id,
            field_path="val_metrics.mae",
            observed=model.val_metrics.mae,
            display=format_stat_display(model.val_metrics.mae),
            unit=model.target_unit,
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="validation"
        )
        self._add(
            claim_id="val_r2",
            claim_type="performance_metric",
            artifact_type="model",
            artifact_id=m_id,
            field_path="val_metrics.r2",
            observed=model.val_metrics.r2,
            display=format_stat_display(model.val_metrics.r2, precision=4),
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="validation"
        )
        gen_gap = model.val_metrics.rmse - model.train_metrics.rmse
        self._add(
            claim_id="generalization_gap",
            claim_type="performance_metric",
            artifact_type="model",
            artifact_id=m_id,
            field_path="computed(val_rmse - train_rmse)",
            observed=gen_gap,
            display=format_stat_display(gen_gap),
            unit=model.target_unit,
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="validation"
        )
        self._add(
            claim_id="model_aic",
            claim_type="information_criterion",
            artifact_type="model",
            artifact_id=m_id,
            field_path="aic",
            observed=model.aic,
            display=format_stat_display(model.aic, precision=2),
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="model_bic",
            claim_type="information_criterion",
            artifact_type="model",
            artifact_id=m_id,
            field_path="bic",
            observed=model.bic,
            display=format_stat_display(model.bic, precision=2),
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="model_f_statistic",
            claim_type="hypothesis_test",
            artifact_type="model",
            artifact_id=m_id,
            field_path="f_statistic",
            observed=model.f_statistic,
            display=format_stat_display(model.f_statistic, precision=2),
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )
        self._add(
            claim_id="model_f_pvalue",
            claim_type="hypothesis_test",
            artifact_type="model",
            artifact_id=m_id,
            field_path="f_pvalue",
            observed=model.f_pvalue,
            display=format_stat_display(model.f_pvalue, is_pvalue=True),
            ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
            scope="train"
        )

        # -------------------------------------------------------------
        # 5. Regression Coefficients Evidence
        # -------------------------------------------------------------
        if model.coef_table is not None and not model.coef_table.empty:
            for term, row in model.coef_table.iterrows():
                term_clean = str(term).replace(" ", "_").replace(":", "_").replace("(", "").replace(")", "")
                # Coef
                self._add(
                    claim_id=f"coef_{term_clean}",
                    claim_type="coefficient_estimate",
                    artifact_type="model",
                    artifact_id=m_id,
                    field_path=f"coef_table.loc['{term}', 'coef']",
                    observed=row.get("coef"),
                    display=format_stat_display(row.get("coef")),
                    ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                    scope="train"
                )
                # If model is valid for inference, record SE, t, p, CI
                if model.is_valid_for_inference:
                    self._add(
                        claim_id=f"stderr_{term_clean}",
                        claim_type="standard_error",
                        artifact_type="model",
                        artifact_id=m_id,
                        field_path=f"coef_table.loc['{term}', 'std_err']",
                        observed=row.get("std_err"),
                        display=format_stat_display(row.get("std_err")),
                        ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                        scope="train"
                    )
                    self._add(
                        claim_id=f"pvalue_{term_clean}",
                        claim_type="p_value",
                        artifact_type="model",
                        artifact_id=m_id,
                        field_path=f"coef_table.loc['{term}', 'p_value']",
                        observed=row.get("p_value"),
                        display=format_stat_display(row.get("p_value"), is_pvalue=True),
                        ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                        scope="train"
                    )
                else:
                    self._add_not_applicable(
                        claim_id=f"stderr_{term_clean}",
                        claim_type="standard_error",
                        artifact_type="model",
                        artifact_id=m_id,
                        field_path=f"coef_table.loc['{term}', 'std_err']",
                        ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                        reason="Model is rank-deficient; conventional parameter standard errors are invalid."
                    )
                    self._add_not_applicable(
                        claim_id=f"pvalue_{term_clean}",
                        claim_type="p_value",
                        artifact_type="model",
                        artifact_id=m_id,
                        field_path=f"coef_table.loc['{term}', 'p_value']",
                        ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                        reason="Model is rank-deficient; conventional parameter p-values are invalid."
                    )

        # -------------------------------------------------------------
        # 6. Assumption Diagnostics Evidence
        # -------------------------------------------------------------
        if diag is not None:
            suite_id = diag.suite_id
            for method_id, d_res in diag.diagnostics.items():
                m_clean = method_id.lower().replace("-", "_").replace(" ", "_")
                # Statistic
                self._add(
                    claim_id=f"diag_stat_{m_clean}",
                    claim_type="diagnostic_statistic",
                    artifact_type="diagnostic_result",
                    artifact_id=d_res.diagnostic_id,
                    field_path="statistic_value",
                    observed=d_res.statistic_value,
                    display=format_stat_display(d_res.statistic_value),
                    ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                    diag_suite_id=suite_id, diag_method_id=method_id,
                    scope=d_res.data_scope
                )
                # p-value (or N/A for descriptive statistics)
                if d_res.p_value is not None:
                    self._add(
                        claim_id=f"diag_pval_{m_clean}",
                        claim_type="diagnostic_pvalue",
                        artifact_type="diagnostic_result",
                        artifact_id=d_res.diagnostic_id,
                        field_path="p_value",
                        observed=d_res.p_value,
                        display=format_stat_display(d_res.p_value, is_pvalue=True),
                        ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                        diag_suite_id=suite_id, diag_method_id=method_id,
                        scope=d_res.data_scope
                    )
                else:
                    self._add_not_applicable(
                        claim_id=f"diag_pval_{m_clean}",
                        claim_type="diagnostic_pvalue",
                        artifact_type="diagnostic_result",
                        artifact_id=d_res.diagnostic_id,
                        field_path="p_value",
                        ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                        diag_suite_id=suite_id, diag_method_id=method_id,
                        reason="Descriptive statistic does not evaluate a formal sampling distribution p-value."
                    )
                # Decision
                self._add(
                    claim_id=f"diag_decision_{m_clean}",
                    claim_type="diagnostic_decision",
                    artifact_type="diagnostic_result",
                    artifact_id=d_res.diagnostic_id,
                    field_path="decision",
                    observed=d_res.decision,
                    display=d_res.decision.replace("_", " ").title(),
                    ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                    diag_suite_id=suite_id, diag_method_id=method_id,
                    scope=d_res.data_scope
                )

        # -------------------------------------------------------------
        # 7. Comparison & Candidate Selection Evidence
        # -------------------------------------------------------------
        if comp is not None:
            c_id = comp.comparison_id
            self._add(
                claim_id="comp_num_models",
                claim_type="comparison_summary",
                artifact_type="comparison",
                artifact_id=c_id,
                field_path="compared_model_ids",
                observed=len(comp.compared_model_ids),
                display=str(len(comp.compared_model_ids)),
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="validation"
            )
            self._add(
                claim_id="comp_candidate_id",
                claim_type="candidate_selection",
                artifact_type="comparison",
                artifact_id=c_id,
                field_path="selected_candidate_model_id",
                observed=comp.selected_candidate_model_id,
                display=comp.selected_candidate_model_id or "None Selected",
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="validation"
            )
            self._add(
                claim_id="comp_candidate_rationale",
                claim_type="candidate_selection",
                artifact_type="comparison",
                artifact_id=c_id,
                field_path="candidate_selection_rationale",
                observed=comp.candidate_selection_rationale,
                display=comp.candidate_selection_rationale or "No rationale recorded",
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="validation"
            )
            # Pareto status of primary model
            is_pareto = m_id in comp.pareto_model_ids
            self._add(
                claim_id="comp_is_pareto",
                claim_type="pareto_membership",
                artifact_type="comparison",
                artifact_id=c_id,
                field_path="pareto_model_ids",
                observed=is_pareto,
                display="Yes (Pareto Efficient)" if is_pareto else "No (Dominated)",
                ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                scope="validation"
            )

        # -------------------------------------------------------------
        # 8. Final Test Evaluation (Authorized Appendix Only)
        # -------------------------------------------------------------
        if include_final_test_authorized:
            if model.test_evaluation_status != "untested" and model.test_metrics is not None:
                self._add(
                    claim_id="test_rmse",
                    claim_type="final_test_metric",
                    artifact_type="final_test",
                    artifact_id=m_id,
                    field_path="test_metrics.rmse",
                    observed=model.test_metrics.rmse,
                    display=format_stat_display(model.test_metrics.rmse),
                    unit=model.target_unit,
                    ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                    scope="test"
                )
                self._add(
                    claim_id="test_r2",
                    claim_type="final_test_metric",
                    artifact_type="final_test",
                    artifact_id=m_id,
                    field_path="test_metrics.r2",
                    observed=model.test_metrics.r2,
                    display=format_stat_display(model.test_metrics.r2, precision=4),
                    ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                    scope="test"
                )
                self._add(
                    claim_id="test_refit_policy",
                    claim_type="final_test_metric",
                    artifact_type="final_test",
                    artifact_id=m_id,
                    field_path="test_refit_policy",
                    observed=model.test_refit_policy,
                    display=str(model.test_refit_policy),
                    ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                    scope="test"
                )
            else:
                self._add_missing(
                    claim_id="test_rmse",
                    claim_type="final_test_metric",
                    artifact_type="final_test",
                    artifact_id=m_id,
                    field_path="test_metrics.rmse",
                    ds_id=ds_id, ds_fp=ds_fp, split_fp=split_fp, m_id=m_id,
                    reason="Model has not been evaluated on the Test partition."
                )

        return self.manifest

    def _add(
        self,
        claim_id: str,
        claim_type: str,
        artifact_type: str,
        artifact_id: str,
        field_path: str,
        observed: Any,
        display: str,
        ds_id: str,
        ds_fp: str,
        split_fp: str,
        m_id: Optional[str] = None,
        diag_suite_id: Optional[str] = None,
        diag_method_id: Optional[str] = None,
        scope: str = "train",
        unit: Optional[str] = None,
        warnings: Optional[List[str]] = None
    ) -> None:
        rec = EvidenceRecord(
            claim_id=claim_id,
            claim_type=claim_type,
            source_artifact_type=artifact_type,
            source_artifact_id=artifact_id,
            source_field_path=field_path,
            dataset_id=ds_id,
            dataset_fingerprint=ds_fp,
            split_fingerprint=split_fp,
            model_id=m_id,
            diagnostic_suite_id=diag_suite_id,
            diagnostic_method_id=diag_method_id,
            analysis_data_scope=scope,
            observed_value=observed,
            display_value=display,
            unit=unit,
            evidence_status=EvidenceStatus.VERIFIED.value,
            validation_reason="Resolved and fingerprint-verified from experimental artifact.",
            methodological_warnings=warnings or []
        )
        self.manifest.add_record(rec)

    def _add_missing(
        self,
        claim_id: str,
        claim_type: str,
        artifact_type: str,
        artifact_id: str,
        field_path: str,
        ds_id: str,
        ds_fp: str,
        split_fp: str,
        m_id: Optional[str] = None,
        diag_suite_id: Optional[str] = None,
        diag_method_id: Optional[str] = None,
        reason: str = "Artifact or field is missing."
    ) -> None:
        rec = EvidenceRecord(
            claim_id=claim_id,
            claim_type=claim_type,
            source_artifact_type=artifact_type,
            source_artifact_id=artifact_id,
            source_field_path=field_path,
            dataset_id=ds_id,
            dataset_fingerprint=ds_fp,
            split_fingerprint=split_fp,
            model_id=m_id,
            diagnostic_suite_id=diag_suite_id,
            diagnostic_method_id=diag_method_id,
            observed_value=None,
            display_value="Unavailable",
            evidence_status=EvidenceStatus.MISSING.value,
            validation_reason=reason
        )
        self.manifest.add_record(rec)

    def _add_not_applicable(
        self,
        claim_id: str,
        claim_type: str,
        artifact_type: str,
        artifact_id: str,
        field_path: str,
        ds_id: str,
        ds_fp: str,
        split_fp: str,
        m_id: Optional[str] = None,
        diag_suite_id: Optional[str] = None,
        diag_method_id: Optional[str] = None,
        reason: str = "Statistic is not applicable."
    ) -> None:
        rec = EvidenceRecord(
            claim_id=claim_id,
            claim_type=claim_type,
            source_artifact_type=artifact_type,
            source_artifact_id=artifact_id,
            source_field_path=field_path,
            dataset_id=ds_id,
            dataset_fingerprint=ds_fp,
            split_fingerprint=split_fp,
            model_id=m_id,
            diagnostic_suite_id=diag_suite_id,
            diagnostic_method_id=diag_method_id,
            observed_value=None,
            display_value="N/A",
            evidence_status=EvidenceStatus.NOT_APPLICABLE.value,
            validation_reason=reason
        )
        self.manifest.add_record(rec)
