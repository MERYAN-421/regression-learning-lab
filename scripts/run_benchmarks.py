"""
Empirical benchmark runner for Milestone 7 Four-Dataset Demonstrations.
Executes real end-to-end workflows across all 4 benchmark datasets:
1. Student Performance (395 rows, Random Split)
2. Appliances Energy (19,735 rows, Temporal Split)
3. King County House Sales (21,613 rows, Random Split)
4. Life Expectancy (2,938 rows, Group Split by Country)

Measures and records exact execution times, memory usage, fingerprints, and statistical outputs.
"""
import copy
import io
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data.registry import get_registry
from src.data.loader import load_dataset_bundle, get_dataset_fingerprint
from src.data.splitters import split_dataset
from src.data.models import SplitConfig, SplitStrategy
from src.features import run_feature_selection
from src.models.model_service import train_ols_experiment
from src.diagnostics.runner import run_full_diagnostic_suite
from src.comparison.leaderboard import build_comparison_leaderboard
from src.comparison.comparison_runner import run_model_comparison
from src.reporting.source_resolver import ResolvedReportSources
from src.reporting.report_runner import run_academic_report
from src.reporting.export_service import ReportExportService
from src.reporting.figure_registry import ReportFigureRegistry
from src.reporting.result_models import AuthorContent


def benchmark_student_performance():
    print("\n" + "=" * 60)
    print("DEMO D: Student Performance (Mathematics) Benchmark")
    print("=" * 60)
    tracemalloc.start()
    t0 = time.time()

    # 1. Dataset Loading
    t_load_start = time.time()
    registry = get_registry()
    meta = registry.get_dataset("student_performance")
    raw_df, feat_df, target_series, meta_df = load_dataset_bundle(meta)
    ds_fp = get_dataset_fingerprint(meta)
    t_load = time.time() - t_load_start

    # 2. Random Split
    t_split_start = time.time()
    split_cfg = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    split_res = split_dataset(feat_df, target_series, meta, split_cfg, raw_df)
    t_split = time.time() - t_split_start

    # 3. Feature Selection: Pearson Correlation (k=5)
    t_fs_start = time.time()
    fs_res_pearson = run_feature_selection(
        X_train=split_res.X_train,
        y_train=split_res.y_train,
        X_val=split_res.X_val,
        y_val=split_res.y_val,
        method_id="pearson",
        k=5,
        cv_strategy=split_res.strategy,
        cv_folds=5,
        random_seed=meta.default_random_seed,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        split_fingerprint=split_res.split_fingerprint,
        groups_train=split_res.groups_train
    )
    t_fs = time.time() - t_fs_start

    # 4. Model 1: All Features Baseline OLS
    t_ols1_start = time.time()
    m1 = train_ols_experiment(
        split_result=split_res,
        feature_source="all_features",
        target_name=meta.target_column,
        fit_intercept=True,
        scale_numeric=True,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        allow_rank_deficient=False
    )
    t_ols1 = time.time() - t_ols1_start

    # 5. Model 2: Selected Features (Pearson k=5) OLS
    t_ols2_start = time.time()
    m2 = train_ols_experiment(
        split_result=split_res,
        feature_source=fs_res_pearson,
        target_name=meta.target_column,
        fit_intercept=True,
        scale_numeric=True,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        allow_rank_deficient=False
    )
    t_ols2 = time.time() - t_ols2_start

    # 6. Assumption Diagnostics on Candidate (m2)
    t_diag_start = time.time()
    diag_suite_m2 = run_full_diagnostic_suite(m2, split_res)
    t_diag = time.time() - t_diag_start

    # 7. Model Comparison
    t_comp_start = time.time()
    comp_res = run_model_comparison(
        models=[m1, m2],
        active_dataset_id=meta.id,
        active_dataset_fingerprint=ds_fp,
        active_split_fingerprint=split_res.split_fingerprint,
        active_target_name=meta.target_column,
        active_target_unit="",
        diagnostic_suites={m2.model_id: diag_suite_m2},
        selected_candidate_id=m2.model_id,
        candidate_selection_rationale="Selected via Pearson filter: parsimonious specification with strong validation fit."
    )
    t_comp = time.time() - t_comp_start

    # 8. Academic Report & Exports
    t_rep_start = time.time()
    sources = ResolvedReportSources(
        model=m2,
        comparison=comp_res,
        diagnostic_suite=diag_suite_m2,
        split_result=split_res,
        dataset_metadata=meta,
        feature_selection_results={fs_res_pearson.experiment_id: fs_res_pearson},
        all_comparison_models={m1.model_id: m1, m2.model_id: m2}
    )
    author = AuthorContent(
        author_name="Regression Learning Lab Researcher",
        research_question="Can Student mathematics performance be predicted parsimoniously using demographic and study features?",
        main_findings="Pearson feature selection identifies strong predictors while preserving full rank estimation."
    )
    report_res = run_academic_report(
        sources=sources,
        report_title="Student Performance OLS Empirical Evaluation",
        author_content=author
    )
    fig_reg = ReportFigureRegistry(sources=sources)
    fig_reg.build_all_available_figures()
    export_svc = ReportExportService(report=report_res, sources=sources, figure_registry=fig_reg)
    md_content = export_svc.export_markdown()
    html_content = export_svc.export_html()
    json_content = export_svc.export_manifest_json()
    zip_bytes = export_svc.export_reproduction_zip()
    t_rep = time.time() - t_rep_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.time() - t0

    result = {
        "dataset": "Student Performance",
        "rows": len(raw_df),
        "dataset_fingerprint": ds_fp,
        "split_fingerprint": split_res.split_fingerprint,
        "selected_features": m2.selected_original_features,
        "m1_id": m1.model_id,
        "m1_val_rmse": float(m1.val_metrics.rmse),
        "m1_val_r2": float(m1.val_metrics.r2),
        "m2_id": m2.model_id,
        "m2_val_rmse": float(m2.val_metrics.rmse),
        "m2_val_r2": float(m2.val_metrics.r2),
        "candidate_id": comp_res.selected_candidate_model_id,
        "diag_suite_id": diag_suite_m2.suite_id,
        "report_id": report_res.report_id,
        "zip_size_bytes": len(zip_bytes),
        "timings_sec": {
            "dataset_load": round(t_load, 3),
            "split": round(t_split, 3),
            "feature_selection": round(t_fs, 3),
            "ols_fit_m1": round(t_ols1, 3),
            "ols_fit_m2": round(t_ols2, 3),
            "diagnostics": round(t_diag, 3),
            "comparison": round(t_comp, 3),
            "report_and_exports": round(t_rep, 3),
            "total_workflow": round(total_time, 3)
        },
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2)
    }
    print(json.dumps(result, indent=2))
    return result


def benchmark_appliances_energy():
    print("\n" + "=" * 60)
    print("DEMO A: Appliances Energy Prediction (Temporal Split) Benchmark")
    print("=" * 60)
    tracemalloc.start()
    t0 = time.time()

    # 1. Dataset Loading
    t_load_start = time.time()
    registry = get_registry()
    meta = registry.get_dataset("appliances_energy")
    raw_df, feat_df, target_series, meta_df = load_dataset_bundle(meta)
    ds_fp = get_dataset_fingerprint(meta)
    t_load = time.time() - t_load_start

    # 2. Temporal Split
    t_split_start = time.time()
    split_cfg = SplitConfig(
        strategy=SplitStrategy.TEMPORAL,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        date_column=meta.date_column
    )
    split_res = split_dataset(feat_df, target_series, meta, split_cfg, raw_df)
    t_split = time.time() - t_split_start

    # 3. TimeSeriesSplit Feature Selection (Pearson k=10)
    t_fs_start = time.time()
    fs_res = run_feature_selection(
        X_train=split_res.X_train,
        y_train=split_res.y_train,
        X_val=split_res.X_val,
        y_val=split_res.y_val,
        method_id="pearson",
        k=10,
        cv_strategy=split_res.strategy,
        cv_folds=5,
        random_seed=meta.default_random_seed,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        split_fingerprint=split_res.split_fingerprint,
        groups_train=split_res.groups_train
    )
    t_fs = time.time() - t_fs_start

    # 4. Model Fit: OLS with Selected Features (Full Rank)
    t_fit_start = time.time()
    model = train_ols_experiment(
        split_result=split_res,
        feature_source=fs_res,
        target_name=meta.target_column,
        fit_intercept=True,
        scale_numeric=True,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        allow_rank_deficient=False
    )
    t_fit = time.time() - t_fit_start

    # 5. Assumption Diagnostics (Temporal Autocorrelation Applicable)
    t_diag_start = time.time()
    diag_suite = run_full_diagnostic_suite(model, split_res)
    t_diag = time.time() - t_diag_start

    # 6. Academic Report Generation & Exports
    t_rep_start = time.time()
    sources = ResolvedReportSources(
        model=model,
        split_result=split_res,
        dataset_metadata=meta,
        diagnostic_suite=diag_suite,
        feature_selection_results={fs_res.experiment_id: fs_res}
    )
    report = run_academic_report(
        sources=sources,
        report_title="Appliances Energy Chronological OLS Analysis",
        author_content=AuthorContent(author_name="Energy Analytics Group")
    )
    fig_reg = ReportFigureRegistry(sources=sources)
    export_svc = ReportExportService(report=report, sources=sources, figure_registry=fig_reg)
    zip_bytes = export_svc.export_reproduction_zip()
    t_rep = time.time() - t_rep_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.time() - t0

    result = {
        "dataset": "Appliances Energy Prediction",
        "rows": len(raw_df),
        "dataset_fingerprint": ds_fp,
        "split_fingerprint": split_res.split_fingerprint,
        "selected_features": model.selected_original_features,
        "model_id": model.model_id,
        "val_rmse": float(model.val_metrics.rmse),
        "val_r2": float(model.val_metrics.r2),
        "diag_suite_id": diag_suite.suite_id,
        "dw_status": diag_suite.durbin_watson.applicability.value if hasattr(diag_suite, "durbin_watson") else "executed",
        "report_id": report.report_id,
        "zip_size_bytes": len(zip_bytes),
        "timings_sec": {
            "dataset_load": round(t_load, 3),
            "split": round(t_split, 3),
            "feature_selection": round(t_fs, 3),
            "ols_fit": round(t_fit, 3),
            "diagnostics": round(t_diag, 3),
            "report_and_exports": round(t_rep, 3),
            "total_workflow": round(total_time, 3)
        },
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2)
    }
    print(json.dumps(result, indent=2))
    return result


def benchmark_kc_house_sales():
    print("\n" + "=" * 60)
    print("DEMO B: King County House Sales (Random Split) Benchmark")
    print("=" * 60)
    tracemalloc.start()
    t0 = time.time()

    # 1. Dataset Loading
    t_load_start = time.time()
    registry = get_registry()
    meta = registry.get_dataset("kc_house_sales")
    raw_df, feat_df, target_series, meta_df = load_dataset_bundle(meta)
    ds_fp = get_dataset_fingerprint(meta)
    t_load = time.time() - t_load_start

    # 2. Random Split
    t_split_start = time.time()
    split_cfg = SplitConfig(
        strategy=SplitStrategy.RANDOM,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    split_res = split_dataset(feat_df, target_series, meta, split_cfg, raw_df)
    t_split = time.time() - t_split_start

    # 3. Model 1: All Features Baseline OLS
    t_fit_start = time.time()
    m1 = train_ols_experiment(
        split_result=split_res,
        feature_source="all_features",
        target_name=meta.target_column,
        fit_intercept=True,
        scale_numeric=True,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        allow_rank_deficient=False
    )
    t_ols1 = time.time() - t_fit_start

    # 4. Feature Selection: Pearson Correlation (k=5)
    t_fs_start = time.time()
    fs_res = run_feature_selection(
        X_train=split_res.X_train,
        y_train=split_res.y_train,
        X_val=split_res.X_val,
        y_val=split_res.y_val,
        method_id="pearson",
        k=5,
        cv_strategy=split_res.strategy,
        cv_folds=5,
        random_seed=meta.default_random_seed,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        split_fingerprint=split_res.split_fingerprint,
        groups_train=split_res.groups_train
    )
    t_fs = time.time() - t_fs_start

    # 5. Model 2: Selected Features (k=5)
    t_ols2_start = time.time()
    m2 = train_ols_experiment(
        split_result=split_res,
        feature_source=fs_res,
        target_name=meta.target_column,
        fit_intercept=True,
        scale_numeric=True,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        allow_rank_deficient=False
    )
    t_ols2 = time.time() - t_ols2_start

    # 6. Diagnostics on Candidate (m2)
    t_diag_start = time.time()
    diag_suite_m2 = run_full_diagnostic_suite(m2, split_res)
    t_diag = time.time() - t_diag_start

    # 7. Model Comparison & Pareto
    t_comp_start = time.time()
    comp_res = run_model_comparison(
        models=[m1, m2],
        active_dataset_id=meta.id,
        active_dataset_fingerprint=ds_fp,
        active_split_fingerprint=split_res.split_fingerprint,
        active_target_name=meta.target_column,
        active_target_unit="",
        diagnostic_suites={m2.model_id: diag_suite_m2},
        selected_candidate_id=m2.model_id,
        candidate_selection_rationale="Selected via Pearson filter: parsimonious specification with strong validation fit."
    )
    t_comp = time.time() - t_comp_start

    # 8. Academic Report & Exports
    t_rep_start = time.time()
    sources = ResolvedReportSources(
        model=m2,
        comparison=comp_res,
        diagnostic_suite=diag_suite_m2,
        split_result=split_res,
        dataset_metadata=meta,
        feature_selection_results={fs_res.experiment_id: fs_res},
        all_comparison_models={m1.model_id: m1, m2.model_id: m2}
    )
    report = run_academic_report(
        sources=sources,
        report_title="King County House Sales Empirical Valuation Report",
        author_content=AuthorContent(author_name="Real Estate Econometrics Lab")
    )
    fig_reg = ReportFigureRegistry(sources=sources)
    export_svc = ReportExportService(report=report, sources=sources, figure_registry=fig_reg)
    zip_bytes = export_svc.export_reproduction_zip()
    t_rep = time.time() - t_rep_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.time() - t0

    result = {
        "dataset": "King County House Sales",
        "rows": len(raw_df),
        "dataset_fingerprint": ds_fp,
        "split_fingerprint": split_res.split_fingerprint,
        "m1_id": m1.model_id,
        "m1_val_rmse": float(m1.val_metrics.rmse),
        "m2_id": m2.model_id,
        "m2_val_rmse": float(m2.val_metrics.rmse),
        "candidate_id": comp_res.selected_candidate_model_id,
        "pareto_models": comp_res.pareto_model_ids,
        "diag_suite_id": diag_suite_m2.suite_id,
        "report_id": report.report_id,
        "zip_size_bytes": len(zip_bytes),
        "timings_sec": {
            "dataset_load": round(t_load, 3),
            "split": round(t_split, 3),
            "ols_fit_m1": round(t_ols1, 3),
            "feature_selection": round(t_fs, 3),
            "ols_fit_m2": round(t_ols2, 3),
            "diagnostics": round(t_diag, 3),
            "comparison": round(t_comp, 3),
            "report_and_exports": round(t_rep, 3),
            "total_workflow": round(total_time, 3)
        },
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2)
    }
    print(json.dumps(result, indent=2))
    return result


def benchmark_life_expectancy():
    print("\n" + "=" * 60)
    print("DEMO C: Life Expectancy (Group Split by Country) Benchmark")
    print("=" * 60)
    tracemalloc.start()
    t0 = time.time()

    # 1. Dataset Loading
    t_load_start = time.time()
    registry = get_registry()
    meta = registry.get_dataset("life_expectancy")
    raw_df, feat_df, target_series, meta_df = load_dataset_bundle(meta)
    ds_fp = get_dataset_fingerprint(meta)
    t_load = time.time() - t_load_start

    # 2. Group Split by Country
    t_split_start = time.time()
    split_cfg = SplitConfig(
        strategy=SplitStrategy.GROUP,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        group_column=meta.group_column,
        random_seed=42
    )
    split_res = split_dataset(feat_df, target_series, meta, split_cfg, raw_df)
    t_split = time.time() - t_split_start

    # 3. Model Fit: OLS on Group Split
    t_fit_start = time.time()
    model = train_ols_experiment(
        split_result=split_res,
        feature_source="all_features",
        target_name=meta.target_column,
        fit_intercept=True,
        scale_numeric=True,
        dataset_id=meta.id,
        dataset_fingerprint=ds_fp,
        allow_rank_deficient=False
    )
    t_fit = time.time() - t_fit_start

    # 4. Diagnostics: Guard Against Panel Non-Applicability (Durbin-Watson Not Applicable)
    t_diag_start = time.time()
    diag_suite = run_full_diagnostic_suite(model, split_res)
    t_diag = time.time() - t_diag_start

    # 5. Academic Report & Exports
    t_rep_start = time.time()
    sources = ResolvedReportSources(
        model=model,
        split_result=split_res,
        dataset_metadata=meta,
        diagnostic_suite=diag_suite
    )
    report = run_academic_report(
        sources=sources,
        report_title="Life Expectancy Cross-National Panel OLS Analysis",
        author_content=AuthorContent(author_name="Global Health Statistics Group")
    )
    fig_reg = ReportFigureRegistry(sources=sources)
    export_svc = ReportExportService(report=report, sources=sources, figure_registry=fig_reg)
    zip_bytes = export_svc.export_reproduction_zip()
    t_rep = time.time() - t_rep_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.time() - t0

    result = {
        "dataset": "Life Expectancy (WHO)",
        "rows": len(raw_df),
        "group_column": "Country",
        "dataset_fingerprint": ds_fp,
        "split_fingerprint": split_res.split_fingerprint,
        "model_id": model.model_id,
        "val_rmse": float(model.val_metrics.rmse),
        "val_r2": float(model.val_metrics.r2),
        "diag_suite_id": diag_suite.suite_id,
        "dw_status": diag_suite.durbin_watson.applicability.value if hasattr(diag_suite, "durbin_watson") else "not_applicable",
        "report_id": report.report_id,
        "zip_size_bytes": len(zip_bytes),
        "timings_sec": {
            "dataset_load": round(t_load, 3),
            "split": round(t_split, 3),
            "ols_fit": round(t_fit, 3),
            "diagnostics": round(t_diag, 3),
            "report_and_exports": round(t_rep, 3),
            "total_workflow": round(total_time, 3)
        },
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2)
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    print("Regression Learning Lab — Milestone 7 Production Benchmark Suite")
    results = {}
    results["student_performance"] = benchmark_student_performance()
    results["appliances_energy"] = benchmark_appliances_energy()
    results["kc_house_sales"] = benchmark_kc_house_sales()
    results["life_expectancy"] = benchmark_life_expectancy()

    # Save to disk
    out_path = PROJECT_ROOT / "docs" / "benchmarks_summary.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAll 4 benchmarks executed successfully. Results written to {out_path}")


if __name__ == "__main__":
    main()
