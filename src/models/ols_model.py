"""
Ordinary Least Squares (OLS) Modeling Engine.

Integrates statsmodels.api.OLS for econometric parameter estimation, statistical inference,
hypothesis testing, coefficient interpretation, and LaTeX equation formulation.
"""
from datetime import datetime
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.design_matrix import DesignMatrixResult, OLSDesignMatrixBuilder
from src.models.model_result import OLSModelResult
from src.evaluation.regression_metrics import compute_regression_metrics, RegressionMetrics


def generate_regression_equation_latex(
    target_name: str,
    params: pd.Series,
    fit_intercept: bool = True,
    max_terms: Optional[int] = None
) -> Tuple[str, str]:
    """
    Generate general and fitted LaTeX equations for the estimated OLS model.
    
    Returns:
        (general_equation_latex, fitted_equation_latex)
    """
    clean_target = target_name.replace("_", r"\_")
    general_eq = r"\hat{" + clean_target + r"} = \hat{\beta}_0 + \sum_{j=1}^{p} \hat{\beta}_j X_j" if fit_intercept else r"\hat{" + clean_target + r"} = \sum_{j=1}^{p} \hat{\beta}_j X_j"

    # Fitted equation
    terms = []
    has_const = "const" in params.index and fit_intercept
    if has_const:
        c_val = params["const"]
        terms.append(f"{c_val:.4f}")

    slope_params = params.drop("const") if has_const else params
    total_slopes = len(slope_params)

    num_to_show = total_slopes if max_terms is None else min(max_terms, total_slopes)
    slopes_subset = slope_params.iloc[:num_to_show]

    for col, coef in slopes_subset.items():
        clean_col = str(col).replace("_", r"\_")
        sign = "+" if coef >= 0 else "-"
        abs_coef = abs(coef)
        terms.append(f"{sign} {abs_coef:.4f} \\cdot \\text{{{clean_col}}}")

    if max_terms is not None and total_slopes > max_terms:
        terms.append(f"+ \\dots \\; (\\text{{{total_slopes - max_terms} more terms}})")

    fitted_eq = r"\hat{" + clean_target + r"} = " + " ".join(terms)
    return general_eq, fitted_eq


def generate_regression_equation_text(
    target_name: str,
    params: pd.Series,
    fit_intercept: bool = True
) -> str:
    """Generate copyable plain-text representation of the fitted regression equation."""
    terms = []
    has_const = "const" in params.index and fit_intercept
    if has_const:
        c_val = params["const"]
        terms.append(f"{c_val:.4f}")

    slope_params = params.drop("const") if has_const else params
    for col, coef in slope_params.items():
        sign = "+" if coef >= 0 else "-"
        abs_coef = abs(coef)
        terms.append(f"{sign} {abs_coef:.4f} * {col}")

    return f"Estimated {target_name} = " + " ".join(terms)


def build_coefficient_table(
    ols_res: Any,
    design_result: DesignMatrixResult,
    scale_numeric: bool = False
) -> pd.DataFrame:
    """
    Construct comprehensive econometric coefficient summary table with interpretations.
    """
    params = ols_res.params
    bse = ols_res.bse
    tvalues = ols_res.tvalues
    pvalues = ols_res.pvalues
    conf_int = ols_res.conf_int(alpha=0.05)

    ref_cats = design_result.metadata.reference_categories
    orig_to_enc = design_result.metadata.original_to_encoded_mapping

    rows = []
    for col in params.index:
        coef = float(params[col])
        se = float(bse[col]) if col in bse and np.isfinite(bse[col]) else float("nan")
        t_stat = float(tvalues[col]) if col in tvalues and np.isfinite(tvalues[col]) else float("nan")
        p_val = float(pvalues[col]) if col in pvalues and np.isfinite(pvalues[col]) else float("nan")
        ci_lower = float(conf_int.loc[col, 0]) if col in conf_int.index and np.isfinite(conf_int.loc[col, 0]) else float("nan")
        ci_upper = float(conf_int.loc[col, 1]) if col in conf_int.index and np.isfinite(conf_int.loc[col, 1]) else float("nan")

        # Contextual interpretation
        if col == "const":
            interp = "Baseline response when all numeric predictors and categorical dummy indicators are 0."
        else:
            # Check if this column belongs to a categorical feature
            matched_cat = None
            for cat_feat, dummy_list in orig_to_enc.items():
                if col in dummy_list and cat_feat in ref_cats:
                    matched_cat = (cat_feat, ref_cats[cat_feat])
                    break

            if matched_cat:
                cat_feat, ref_level = matched_cat
                level_name = col[len(cat_feat) + 1:] if col.startswith(cat_feat + "_") else col
                interp = (
                    f"Expected change relative to reference level '{ref_level}', "
                    "holding all other included predictors constant."
                )
            else:
                scale_note = " (standardized SD units)" if scale_numeric else ""
                interp = f"Expected change in response per 1-unit increase{scale_note}, holding other predictors constant."

        rows.append({
            "Feature": col,
            "Coefficient": round(coef, 4),
            "Std. Error": round(se, 4) if np.isfinite(se) else "N/A",
            "t-statistic": round(t_stat, 3) if np.isfinite(t_stat) else "N/A",
            "p-value": f"{p_val:.4e}" if (np.isfinite(p_val) and p_val < 1e-4) else (round(p_val, 4) if np.isfinite(p_val) else "N/A"),
            "CI Lower (95%)": round(ci_lower, 4) if np.isfinite(ci_lower) else "N/A",
            "CI Upper (95%)": round(ci_upper, 4) if np.isfinite(ci_upper) else "N/A",
            "Interpretation": interp
        })

    return pd.DataFrame(rows)


def fit_ols_model(
    design_result: DesignMatrixResult,
    y_train: pd.Series,
    y_val: Optional[pd.Series] = None,
    target_name: str = "target",
    feature_source_id: str = "all_features",
    feature_source_label: str = "All Features Baseline",
    builder: Optional[OLSDesignMatrixBuilder] = None,
    is_post_selection: bool = False
) -> OLSModelResult:
    """
    Fits Ordinary Least Squares regression via statsmodels and produces OLSModelResult.
    """
    X_train_des = design_result.X_train_design
    metadata = design_result.metadata

    if len(X_train_des) != len(y_train):
        raise ValueError(f"Length mismatch: X_train has {len(X_train_des)} rows, y_train has {len(y_train)} rows.")

    warnings_list = list(metadata.warnings)

    if not metadata.is_full_rank:
        warnings_list.append(
            "Design matrix is rank-deficient. statsmodels least squares will use a pseudoinverse; "
            "individual coefficient significance tests (t-stats, p-values, standard errors) "
            "are not uniquely identifiable."
        )

    # Fit statsmodels OLS
    try:
        model = sm.OLS(y_train.values, X_train_des.values)
        ols_res = model.fit()
    except Exception as e:
        raise RuntimeError(f"statsmodels OLS fitting failed: {e}")

    # Wrap parameters with column names
    params_series = pd.Series(ols_res.params, index=X_train_des.columns)
    bse_series = pd.Series(ols_res.bse, index=X_train_des.columns)
    tval_series = pd.Series(ols_res.tvalues, index=X_train_des.columns)
    pval_series = pd.Series(ols_res.pvalues, index=X_train_des.columns)

    # Invert conf_int indexing if necessary
    try:
        ci_df = pd.DataFrame(ols_res.conf_int(alpha=0.05), index=X_train_des.columns)
    except Exception:
        ci_df = pd.DataFrame(np.nan, index=X_train_des.columns, columns=[0, 1])

    ols_res.params = params_series
    ols_res.bse = bse_series
    ols_res.tvalues = tval_series
    ols_res.pvalues = pval_series
    ols_res.conf_int = lambda alpha=0.05: ci_df

    # Generate equation representations
    general_eq, fitted_eq = generate_regression_equation_latex(
        target_name=target_name,
        params=params_series,
        fit_intercept=metadata.fit_intercept
    )
    equation_txt = generate_regression_equation_text(
        target_name=target_name,
        params=params_series,
        fit_intercept=metadata.fit_intercept
    )

    # Build coefficient summary table
    coef_table = build_coefficient_table(
        ols_res=ols_res,
        design_result=design_result,
        scale_numeric=metadata.scaling_applied
    )

    # In-sample predictions and metrics
    train_preds_arr = ols_res.predict(X_train_des.values)
    train_preds = pd.Series(train_preds_arr, index=X_train_des.index)

    p_slopes = metadata.num_parameters - (1 if metadata.fit_intercept else 0)
    train_metrics = compute_regression_metrics(
        y_true=y_train,
        y_pred=train_preds,
        num_non_intercept_parameters=p_slopes,
        split_name="Train",
        is_in_sample=True
    )

    # Validation predictions and metrics
    val_preds: Optional[pd.Series] = None
    if design_result.X_val_design is not None and y_val is not None:
        val_preds_arr = ols_res.predict(design_result.X_val_design.values)
        val_preds = pd.Series(val_preds_arr, index=design_result.X_val_design.index)
        val_metrics = compute_regression_metrics(
            y_true=y_val,
            y_pred=val_preds,
            num_non_intercept_parameters=p_slopes,
            split_name="Validation",
            is_in_sample=False
        )
    else:
        # Fallback dummy metrics if validation data was not supplied
        val_metrics = RegressionMetrics(
            split_name="Validation",
            sample_count=0,
            num_parameters=p_slopes,
            mse=0.0,
            rmse=0.0,
            mae=0.0,
            r2=0.0,
            adj_r2=None,
            notes="No validation set supplied."
        )

    # Statistical summary values from statsmodels
    f_stat = float(ols_res.fvalue) if np.isfinite(ols_res.fvalue) else float("nan")
    f_pval = float(ols_res.f_pvalue) if np.isfinite(ols_res.f_pvalue) else float("nan")
    llf = float(ols_res.llf) if np.isfinite(ols_res.llf) else float("nan")
    aic_val = float(ols_res.aic) if np.isfinite(ols_res.aic) else float("nan")
    bic_val = float(ols_res.bic) if np.isfinite(ols_res.bic) else float("nan")
    rss_val = float(ols_res.ssr) if np.isfinite(ols_res.ssr) else float("nan")

    # Post-selection inference notice
    if is_post_selection:
        warnings_list.append(
            "⚠️ Post-Selection Inference Notice: Predictor features were selected using data-driven algorithms on training data. "
            "Conventional p-values, t-statistics, and confidence intervals are computed conditional on the selected specification "
            "and do not account for model search uncertainty."
        )

    # Generate unique stable Model ID
    id_tokens = [
        metadata.dataset_fingerprint,
        metadata.split_fingerprint,
        feature_source_id,
        str(metadata.fit_intercept),
        str(metadata.scaling_applied),
        ",".join(metadata.selected_original_features),
        f"{train_metrics.rmse:.4f}"
    ]
    model_id = "ols_" + hashlib.sha256("||".join(id_tokens).encode("utf-8")).hexdigest()[:12]
    model_name = f"OLS Model ({feature_source_label})"

    return OLSModelResult(
        model_id=model_id,
        model_name=model_name,
        dataset_id=metadata.dataset_id,
        dataset_fingerprint=metadata.dataset_fingerprint,
        split_fingerprint=metadata.split_fingerprint,
        feature_source_id=feature_source_id,
        feature_source_label=feature_source_label,
        selected_original_features=metadata.selected_original_features,
        design_matrix_metadata=metadata,
        fit_intercept=metadata.fit_intercept,
        scale_numeric=metadata.scaling_applied,
        coef_table=coef_table,
        equation_general=general_eq,
        equation_fitted=fitted_eq,
        equation_text=equation_txt,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=None,
        r2=float(ols_res.rsquared),
        adj_r2=float(ols_res.rsquared_adj) if np.isfinite(ols_res.rsquared_adj) else None,
        f_statistic=f_stat,
        f_pvalue=f_pval,
        log_likelihood=llf,
        aic=aic_val,
        bic=bic_val,
        rss=rss_val,
        df_resid=int(ols_res.df_resid),
        df_model=int(ols_res.df_model),
        post_selection_inference_warning=is_post_selection,
        warnings=warnings_list,
        test_evaluation_status="untested",
        trained_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        fitted_model=ols_res,
        builder=builder,
        train_predictions=train_preds,
        val_predictions=val_preds,
        test_predictions=None
    )
