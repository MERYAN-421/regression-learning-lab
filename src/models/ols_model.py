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
    params: pd.Series,
    bse: pd.Series,
    tvalues: pd.Series,
    pvalues: pd.Series,
    conf_int: pd.DataFrame,
    design_result: DesignMatrixResult,
    scale_numeric: bool = False,
    is_valid_for_inference: bool = True
) -> pd.DataFrame:
    """
    Construct comprehensive econometric coefficient summary table with interpretations.
    When is_valid_for_inference=False, inference columns (SE, t, p, CI) are masked to prevent
    presenting unidentifiable pseudoinverse artifacts as valid scientific inference.
    """
    ref_cats = design_result.metadata.reference_categories
    orig_to_enc = design_result.metadata.original_to_encoded_mapping

    rows = []
    for col in params.index:
        coef = float(params[col])

        if is_valid_for_inference:
            se = float(bse[col]) if col in bse and np.isfinite(bse[col]) else float("nan")
            t_stat = float(tvalues[col]) if col in tvalues and np.isfinite(tvalues[col]) else float("nan")
            p_val = float(pvalues[col]) if col in pvalues and np.isfinite(pvalues[col]) else float("nan")
            ci_lower = float(conf_int.loc[col, 0]) if col in conf_int.index and np.isfinite(conf_int.loc[col, 0]) else float("nan")
            ci_upper = float(conf_int.loc[col, 1]) if col in conf_int.index and np.isfinite(conf_int.loc[col, 1]) else float("nan")

            se_str = round(se, 4) if np.isfinite(se) else "N/A"
            t_str = round(t_stat, 3) if np.isfinite(t_stat) else "N/A"
            p_str = f"{p_val:.4e}" if (np.isfinite(p_val) and p_val < 1e-4) else (round(p_val, 4) if np.isfinite(p_val) else "N/A")
            ci_low_str = round(ci_lower, 4) if np.isfinite(ci_lower) else "N/A"
            ci_up_str = round(ci_upper, 4) if np.isfinite(ci_upper) else "N/A"
        else:
            se_str = "N/A (Rank Deficient)"
            t_str = "N/A (Unidentifiable)"
            p_str = "N/A (Unidentifiable)"
            ci_low_str = "N/A"
            ci_up_str = "N/A"

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
            "Std. Error": se_str,
            "t-statistic": t_str,
            "p-value": p_str,
            "CI Lower (95%)": ci_low_str,
            "CI Upper (95%)": ci_up_str,
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
    is_post_selection: bool = False,
    allow_rank_deficient: bool = False
) -> OLSModelResult:
    """
    Fits Ordinary Least Squares regression via statsmodels and produces OLSModelResult.
    
    Parameters:
        allow_rank_deficient: When False (default), raises ValueError if the design matrix
            is rank-deficient or has non-positive residual degrees of freedom.
            When True, fits OLS via pseudoinverse for prediction only, masking unidentifiable
            inference statistics (standard errors, t-stats, p-values, CIs).
    """
    X_train_des = design_result.X_train_design
    metadata = design_result.metadata

    if len(X_train_des) != len(y_train):
        raise ValueError(f"Length mismatch: X_train has {len(X_train_des)} rows, y_train has {len(y_train)} rows.")

    warnings_list = list(metadata.warnings)

    # Backend guard for statistical inference validity
    if not metadata.is_valid_for_inference:
        dep_info = (
            f"Linearly dependent column(s): {', '.join(metadata.collinear_columns)}."
            if metadata.collinear_columns
            else "Exact linear dependency or non-positive degrees of freedom."
        )
        if not allow_rank_deficient:
            raise ValueError(
                f"OLS statistical inference blocked: Design matrix is rank-deficient "
                f"(matrix rank is {metadata.matrix_rank} for {metadata.num_parameters} columns, "
                f"residual df={metadata.residual_degrees_of_freedom}). {dep_info} "
                "Ordinary Least Squares requires full column rank to uniquely identify regression parameters. "
                "Enable allow_rank_deficient=True for prediction-only mode."
            )
        warnings_list.append(
            "⚠️ Prediction-Only Mode: Design matrix is rank-deficient. Standard errors, t-statistics, "
            "p-values, and confidence intervals are omitted because parameters are not uniquely identifiable."
        )

    # Fit statsmodels OLS without monkey-patching statsmodels objects
    try:
        y_train_s = pd.Series(y_train.values, name=target_name)
        X_train_df = pd.DataFrame(X_train_des.values, columns=X_train_des.columns)
        model = sm.OLS(y_train_s, X_train_df)
        ols_res = model.fit()
    except Exception as e:
        raise RuntimeError(f"statsmodels OLS fitting failed: {e}")

    # Named Series / DataFrame wrapper for display formatting
    params_series = pd.Series(ols_res.params, index=X_train_des.columns)
    bse_series = pd.Series(ols_res.bse, index=X_train_des.columns)
    tval_series = pd.Series(ols_res.tvalues, index=X_train_des.columns)
    pval_series = pd.Series(ols_res.pvalues, index=X_train_des.columns)

    try:
        ci_df = pd.DataFrame(ols_res.conf_int(alpha=0.05), index=X_train_des.columns)
    except Exception:
        ci_df = pd.DataFrame(np.nan, index=X_train_des.columns, columns=[0, 1])

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

    # Build coefficient summary table with explicit inference validity guard
    coef_table = build_coefficient_table(
        params=params_series,
        bse=bse_series,
        tvalues=tval_series,
        pvalues=pval_series,
        conf_int=ci_df,
        design_result=design_result,
        scale_numeric=metadata.scaling_applied,
        is_valid_for_inference=metadata.is_valid_for_inference
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
        is_in_sample=True,
        fit_intercept=metadata.fit_intercept
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
            is_in_sample=False,
            fit_intercept=metadata.fit_intercept
        )
    else:
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

    # Statistical summary values from statsmodels (guard invalid F-stat under rank deficiency)
    if metadata.is_valid_for_inference:
        f_stat = float(ols_res.fvalue) if np.isfinite(ols_res.fvalue) else float("nan")
        f_pval = float(ols_res.f_pvalue) if np.isfinite(ols_res.f_pvalue) else float("nan")
        adj_r2_val = float(ols_res.rsquared_adj) if np.isfinite(ols_res.rsquared_adj) else None
    else:
        f_stat = float("nan")
        f_pval = float("nan")
        adj_r2_val = None
        train_metrics.adj_r2 = None
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
        adj_r2=adj_r2_val,
        f_statistic=f_stat,
        f_pvalue=f_pval,
        log_likelihood=llf,
        aic=aic_val,
        bic=bic_val,
        rss=rss_val,
        df_resid=int(ols_res.df_resid),
        df_model=int(ols_res.df_model),
        post_selection_inference_warning=is_post_selection,
        is_valid_for_inference=metadata.is_valid_for_inference,
        target_name=target_name,
        warnings=warnings_list,
        test_evaluation_status="untested",
        trained_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        fitted_model=ols_res,
        builder=builder,
        train_predictions=train_preds,
        val_predictions=val_preds,
        test_predictions=None
    )
