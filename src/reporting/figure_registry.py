"""
Figure Registry and Asset Generator for Academic Reports.
Reuses verified plotting routines from evaluation, diagnostics, and comparison modules.
Generates structured FigureReference objects and embeddable assets without inventing data.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.reporting.result_models import FigureReference
from src.reporting.source_resolver import ResolvedReportSources
from src.evaluation.plots import plot_actual_vs_predicted
from src.diagnostics.plots import (
    plot_residuals_vs_fitted_lowess,
    plot_normal_qq,
    plot_vif_horizontal_bar
)
from src.comparison.radar import create_radar_chart
from src.comparison.pareto import create_pareto_scatter_chart
from src.comparison.feature_overlap import create_jaccard_heatmap


class ReportFigureRegistry:
    """
    Manages generation, metadata registration, and embedding of academic figures.
    """

    def __init__(self, sources: ResolvedReportSources):
        self.sources = sources
        self.figures: Dict[str, go.Figure] = {}
        self.references: Dict[str, FigureReference] = {}

    def build_all_available_figures(self) -> List[FigureReference]:
        """
        Builds all supported academic figures that have sufficient underlying artifact data.
        """
        self.build_actual_vs_predicted()
        self.build_residuals_vs_fitted()
        self.build_normal_qq()
        self.build_vif_chart()
        self.build_radar_chart()
        self.build_pareto_chart()
        self.build_feature_overlap_heatmap()
        return list(self.references.values())

    def get_figure(self, figure_id: str) -> Optional[go.Figure]:
        return self.figures.get(figure_id)

    def get_reference(self, figure_id: str) -> Optional[FigureReference]:
        return self.references.get(figure_id)

    def render_figure_html(self, figure_id: str) -> str:
        """Renders an embeddable HTML string for standalone reports."""
        fig = self.figures.get(figure_id)
        if fig is None:
            ref = self.references.get(figure_id)
            reason = ref.caveats[0] if ref and ref.caveats else "Underlying data not available."
            return (
                f"<div class='figure-unavailable-box' style='padding: 20px; border: 1px dashed #94a3b8; "
                f"border-radius: 8px; background: #f8fafc; color: #64748b; text-align: center; margin: 15px 0;'>"
                f"<em>Figure [{figure_id}] Unavailable: {reason}</em></div>"
            )
        try:
            return fig.to_html(include_plotlyjs=False, full_html=False)
        except Exception as e:
            return f"<div class='figure-error-box'>Error rendering figure {figure_id}: {str(e)}</div>"

    # -------------------------------------------------------------------------
    # Individual Figure Builders
    # -------------------------------------------------------------------------

    def build_actual_vs_predicted(self) -> Optional[FigureReference]:
        fid = "fig_actual_vs_pred"
        model = self.sources.model
        split = self.sources.split_result

        if model is None or split is None or split.y_val is None:
            ref = FigureReference(
                figure_id=fid,
                title="Actual vs. Predicted Scatter Plot",
                caption="Validation set actual outcomes versus OLS model predictions.",
                figure_type="evaluation_scatter",
                model_id=model.model_id if model else None,
                data_scope="validation",
                provenance="Unavailable",
                caveats=["Validation predictions or target observations missing."],
                is_available=False
            )
            self.references[fid] = ref
            return ref

        try:
            # Predict on Validation
            if model.val_predictions is not None:
                val_preds = model.val_predictions
            elif model.builder is not None and model.fitted_model is not None and split.X_val is not None:
                X_val_des = model.builder.transform_design_matrix(split.X_val, is_training=False)
                val_preds = model.fitted_model.predict(X_val_des.values)
            else:
                ref = FigureReference(
                    figure_id=fid,
                    title="Actual vs. Predicted Scatter Plot",
                    caption="Validation set actual outcomes versus OLS model predictions.",
                    figure_type="evaluation_scatter",
                    model_id=model.model_id,
                    data_scope="validation",
                    provenance=f"Model {model.model_id}",
                    caveats=["Model predictor pipeline cannot compute validation predictions."],
                    is_available=False
                )
                self.references[fid] = ref
                return ref

            fig = plot_actual_vs_predicted(
                y_true=split.y_val,
                y_pred=val_preds,
                target_name=model.target_name,
                target_unit=model.target_unit or "",
                split_name="Validation",
                rmse=model.val_metrics.rmse,
                r2=model.val_metrics.r2
            )
            self.figures[fid] = fig
            ref = FigureReference(
                figure_id=fid,
                title="Actual vs. Predicted Scatter Plot",
                caption=(
                    f"Scatter plot of observed versus predicted {model.target_name} on the validation partition "
                    f"({model.val_metrics.rmse:.3f} {model.target_unit or ''} RMSE, R² = {model.val_metrics.r2:.4f}). "
                    "Red dashed line represents ideal 1:1 prediction."
                ),
                figure_type="evaluation_scatter",
                model_id=model.model_id,
                data_scope="validation",
                provenance=f"Model: {model.model_id} | Dataset: {model.dataset_id}",
                caveats=["Subsampled if sample count exceeds 5,000 for responsive rendering."],
                is_available=True
            )
            self.references[fid] = ref
            return ref
        except Exception as e:
            ref = FigureReference(
                figure_id=fid,
                title="Actual vs. Predicted Scatter Plot",
                caption="Validation set actual outcomes versus OLS model predictions.",
                figure_type="evaluation_scatter",
                model_id=model.model_id if model else None,
                data_scope="validation",
                provenance="Error during rendering",
                caveats=[f"Plot generation failed: {str(e)}"],
                is_available=False
            )
            self.references[fid] = ref
            return ref

    def build_residuals_vs_fitted(self) -> Optional[FigureReference]:
        fid = "fig_residuals_vs_fitted"
        model = self.sources.model
        split = self.sources.split_result

        if model is None or split is None or split.y_train is None:
            ref = FigureReference(
                figure_id=fid,
                title="Residuals vs. Fitted Values (LOWESS)",
                caption="In-sample training residuals against fitted values.",
                figure_type="diagnostic_residual",
                model_id=model.model_id if model else None,
                data_scope="train",
                provenance="Unavailable",
                caveats=["Training residuals not available."],
                is_available=False
            )
            self.references[fid] = ref
            return ref

        try:
            if model.fitted_model is not None:
                fitted_vals = np.asarray(model.fitted_model.fittedvalues)
                residuals = np.asarray(model.fitted_model.resid)
            elif model.train_predictions is not None:
                fitted_vals = np.asarray(model.train_predictions)
                residuals = np.asarray(split.y_train) - fitted_vals
            else:
                ref = FigureReference(
                    figure_id=fid,
                    title="Residuals vs. Fitted Values (LOWESS)",
                    caption="In-sample training residuals against fitted values.",
                    figure_type="diagnostic_residual",
                    model_id=model.model_id,
                    data_scope="train",
                    provenance=f"Model {model.model_id}",
                    caveats=["Residuals could not be retrieved from fitted model."],
                    is_available=False
                )
                self.references[fid] = ref
                return ref

            fig = plot_residuals_vs_fitted_lowess(
                fitted_values=fitted_vals,
                residuals=residuals,
                target_name=model.target_name,
                target_unit=model.target_unit or ""
            )
            self.figures[fid] = fig
            ref = FigureReference(
                figure_id=fid,
                title="Residuals vs. Fitted Values (LOWESS)",
                caption=(
                    "Diagnostic scatter of OLS training residuals against fitted values with overlaid "
                    "non-parametric LOWESS curve to identify potential nonlinear conditional mean drift."
                ),
                figure_type="diagnostic_residual",
                model_id=model.model_id,
                data_scope="train",
                provenance=f"Model: {model.model_id} | Design: {model.design_matrix_metadata.num_parameters} parameters",
                caveats=["Residuals are mathematically orthogonal to design columns by OLS construction."],
                is_available=True
            )
            self.references[fid] = ref
            return ref
        except Exception as e:
            ref = FigureReference(
                figure_id=fid,
                title="Residuals vs. Fitted Values (LOWESS)",
                caption="In-sample training residuals against fitted values.",
                figure_type="diagnostic_residual",
                model_id=model.model_id if model else None,
                data_scope="train",
                provenance="Error during rendering",
                caveats=[f"Plot generation failed: {str(e)}"],
                is_available=False
            )
            self.references[fid] = ref
            return ref

    def build_normal_qq(self) -> Optional[FigureReference]:
        fid = "fig_normal_qq"
        model = self.sources.model

        if model is None or model.fitted_model is None:
            ref = FigureReference(
                figure_id=fid,
                title="Normal Q-Q Plot",
                caption="Theoretical normal quantiles versus standardized empirical sample quantiles.",
                figure_type="diagnostic_normality",
                model_id=model.model_id if model else None,
                data_scope="train",
                provenance="Unavailable",
                caveats=["Fitted model residuals not available."],
                is_available=False
            )
            self.references[fid] = ref
            return ref

        try:
            residuals = np.asarray(model.fitted_model.resid)
            fig = plot_normal_qq(residuals)
            self.figures[fid] = fig
            ref = FigureReference(
                figure_id=fid,
                title="Normal Q-Q Plot",
                caption="Standardized residual quantiles compared to the standard Gaussian theoretical quantiles.",
                figure_type="diagnostic_normality",
                model_id=model.model_id,
                data_scope="train",
                provenance=f"Model: {model.model_id}",
                caveats=["Gauss-Markov BLUE property does not require normality; normality is required for exact finite-sample t/F tests."],
                is_available=True
            )
            self.references[fid] = ref
            return ref
        except Exception as e:
            ref = FigureReference(
                figure_id=fid,
                title="Normal Q-Q Plot",
                caption="Standardized residual quantiles compared to standard Gaussian theoretical quantiles.",
                figure_type="diagnostic_normality",
                model_id=model.model_id if model else None,
                data_scope="train",
                provenance="Error during rendering",
                caveats=[f"Plot generation failed: {str(e)}"],
                is_available=False
            )
            self.references[fid] = ref
            return ref

    def build_vif_chart(self) -> Optional[FigureReference]:
        fid = "fig_vif_bars"
        diag = self.sources.diagnostic_suite

        if diag is None:
            ref = FigureReference(
                figure_id=fid,
                title="Variance Inflation Factors (VIF)",
                caption="VIF bar chart for model regressors.",
                figure_type="diagnostic_multicollinearity",
                data_scope="train",
                provenance="Unavailable",
                caveats=["Diagnostic suite not run or missing."],
                is_available=False
            )
            self.references[fid] = ref
            return ref

        vif_res = diag.diagnostics.get("vif")
        if vif_res is None or not vif_res.details or ("vif_records" not in vif_res.details and "vif_series" not in vif_res.details):
            ref = FigureReference(
                figure_id=fid,
                title="Variance Inflation Factors (VIF)",
                caption="VIF bar chart for model regressors.",
                figure_type="diagnostic_multicollinearity",
                diagnostic_suite_id=diag.suite_id,
                data_scope="train",
                provenance=f"Suite: {diag.suite_id}",
                caveats=["VIF diagnostic was not run or has no regressor details."],
                is_available=False
            )
            self.references[fid] = ref
            return ref

        try:
            if "vif_records" in vif_res.details:
                vif_records = vif_res.details["vif_records"]
            else:
                vif_records = [{"feature": k, "vif": float(v)} for k, v in vif_res.details["vif_series"].items()]
            fig = plot_vif_horizontal_bar(vif_records)
            self.figures[fid] = fig
            ref = FigureReference(
                figure_id=fid,
                title="Variance Inflation Factors (VIF)",
                caption="Bar chart of regressor Variance Inflation Factors with standard 5.0 (moderate) and 10.0 (severe) collinearity thresholds.",
                figure_type="diagnostic_multicollinearity",
                diagnostic_suite_id=diag.suite_id,
                data_scope="train",
                provenance=f"Suite: {diag.suite_id} | Max VIF: {vif_res.statistic_value:.2f}",
                caveats=["High VIF inflates coefficient variance, but does not invalidate out-of-sample predictions."],
                is_available=True
            )
            self.references[fid] = ref
            return ref
        except Exception as e:
            ref = FigureReference(
                figure_id=fid,
                title="Variance Inflation Factors (VIF)",
                caption="VIF bar chart for model regressors.",
                figure_type="diagnostic_multicollinearity",
                diagnostic_suite_id=diag.suite_id,
                data_scope="train",
                provenance="Error during rendering",
                caveats=[f"Plot generation failed: {str(e)}"],
                is_available=False
            )
            self.references[fid] = ref
            return ref

    def build_radar_chart(self) -> Optional[FigureReference]:
        fid = "fig_radar_chart"
        comp = self.sources.comparison

        if comp is None or not comp.radar_scores:
            ref = FigureReference(
                figure_id=fid,
                title="Multi-Dimensional Radar Chart",
                caption="Normalized comparison across 5 performance and complexity dimensions.",
                figure_type="comparison_radar",
                data_scope="validation",
                provenance="Unavailable",
                caveats=["No comparison record or radar scores available."],
                is_available=False
            )
            self.references[fid] = ref
            return ref

        try:
            cand_id = comp.selected_candidate_model_id
            fig = create_radar_chart(comp.radar_scores, selected_model_id=cand_id)
            self.figures[fid] = fig
            ref = FigureReference(
                figure_id=fid,
                title="Multi-Dimensional Radar Chart",
                caption="Relative performance across 5 normalized dimensions. Values are strictly relative to the min/max among compared candidates.",
                figure_type="comparison_radar",
                experiment_id=comp.comparison_id,
                data_scope="validation",
                provenance=f"Comparison: {comp.comparison_id}",
                caveats=[
                    "Radar polygon area is an artifact of axis ordering and must NOT be interpreted as a composite statistical score."
                ],
                is_available=True
            )
            self.references[fid] = ref
            return ref
        except Exception as e:
            ref = FigureReference(
                figure_id=fid,
                title="Multi-Dimensional Radar Chart",
                caption="Normalized comparison across 5 dimensions.",
                figure_type="comparison_radar",
                experiment_id=comp.comparison_id if comp else None,
                data_scope="validation",
                provenance="Error during rendering",
                caveats=[f"Plot generation failed: {str(e)}"],
                is_available=False
            )
            self.references[fid] = ref
            return ref

    def build_pareto_chart(self) -> Optional[FigureReference]:
        fid = "fig_pareto_chart"
        comp = self.sources.comparison

        if comp is None or not comp.leaderboard:
            ref = FigureReference(
                figure_id=fid,
                title="Pareto Frontier (Loss vs. Complexity)",
                caption="Validation RMSE trade-off versus model parameter count.",
                figure_type="comparison_pareto",
                data_scope="validation",
                provenance="Unavailable",
                caveats=["No comparison record available."],
                is_available=False
            )
            self.references[fid] = ref
            return ref

        try:
            df_leaderboard = pd.DataFrame(comp.leaderboard)
            cand_id = comp.selected_candidate_model_id
            fig = create_pareto_scatter_chart(
                df_leaderboard,
                pareto_model_ids=comp.pareto_model_ids,
                selected_model_id=cand_id,
                target_unit=comp.target_unit
            )
            self.figures[fid] = fig
            ref = FigureReference(
                figure_id=fid,
                title="Pareto Frontier (Loss vs. Complexity)",
                caption=(
                    "Validation loss (RMSE) plotted against model complexity (OLS parameter count p). "
                    "Non-dominated Pareto-efficient models are connected by the efficiency frontier."
                ),
                figure_type="comparison_pareto",
                experiment_id=comp.comparison_id,
                data_scope="validation",
                provenance=f"Comparison: {comp.comparison_id}",
                caveats=["Pareto membership identifies efficient trade-offs, not a universally optimal model."],
                is_available=True
            )
            self.references[fid] = ref
            return ref
        except Exception as e:
            ref = FigureReference(
                figure_id=fid,
                title="Pareto Frontier (Loss vs. Complexity)",
                caption="Validation RMSE trade-off versus model parameter count.",
                figure_type="comparison_pareto",
                experiment_id=comp.comparison_id if comp else None,
                data_scope="validation",
                provenance="Error during rendering",
                caveats=[f"Plot generation failed: {str(e)}"],
                is_available=False
            )
            self.references[fid] = ref
            return ref

    def build_feature_overlap_heatmap(self) -> Optional[FigureReference]:
        fid = "fig_feature_overlap"
        comp = self.sources.comparison

        if comp is None or not comp.feature_overlap_matrix or "jaccard_matrix" not in comp.feature_overlap_matrix:
            ref = FigureReference(
                figure_id=fid,
                title="Feature Overlap Jaccard Similarity Heatmap",
                caption="Pairwise Jaccard similarity of selected original features across compared models.",
                figure_type="comparison_overlap",
                data_scope="validation",
                provenance="Unavailable",
                caveats=["Feature overlap matrix not available in comparison artifact."],
                is_available=False
            )
            self.references[fid] = ref
            return ref

        try:
            j_dict = comp.feature_overlap_matrix["jaccard_matrix"]
            df_j = pd.DataFrame(j_dict)
            fig = create_jaccard_heatmap(df_j)
            self.figures[fid] = fig
            ref = FigureReference(
                figure_id=fid,
                title="Feature Overlap Jaccard Similarity Heatmap",
                caption="Heatmap illustrating pairwise Jaccard similarity coefficients of selected original predictor sets across models.",
                figure_type="comparison_overlap",
                experiment_id=comp.comparison_id,
                data_scope="validation",
                provenance=f"Comparison: {comp.comparison_id}",
                caveats=["Evaluates original feature subsets prior to one-hot dummy column expansion."],
                is_available=True
            )
            self.references[fid] = ref
            return ref
        except Exception as e:
            ref = FigureReference(
                figure_id=fid,
                title="Feature Overlap Jaccard Similarity Heatmap",
                caption="Pairwise Jaccard similarity of selected original features.",
                figure_type="comparison_overlap",
                experiment_id=comp.comparison_id if comp else None,
                data_scope="validation",
                provenance="Error during rendering",
                caveats=[f"Plot generation failed: {str(e)}"],
                is_available=False
            )
            self.references[fid] = ref
            return ref
