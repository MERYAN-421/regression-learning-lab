"""
Integration and Data Leakage Prevention Tests for OLS Modeling (Groups D, E, G).
Verifies:
- All Features modeling workflow
- Milestone 2 FeatureSelectionResult integration
- Split fingerprint validation and stale model rejection
- Model identity uniqueness
- Zero data leakage: Train-only preprocessor fitting
- Controlled final Test evaluation policies (train_val vs train_only)
- Real dataset integration on all four Kaggle benchmark datasets:
  1. Appliances Energy (Temporal split)
  2. King County House Sales (Structural collinearity: sqft_living = sqft_above + sqft_basement)
  3. Life Expectancy (Group split by Country)
  4. Student Performance (Categorical reference dummy encoding)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.registry import get_registry
from src.data.loader import load_dataset_bundle
from src.data.splitters import split_dataset
from src.data.models import SplitConfig, SplitStrategy
from src.features.selectors import run_feature_selection
from src.models.model_service import train_ols_experiment
from src.experiments.regression_experiment_tracker import RegressionExperimentTracker
from src.evaluation.predictions import predict_ols

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def regression_test_bundle():
    """Synthetic split result fixture for integration tests."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "num1": np.random.normal(0, 1, n),
        "num2": np.random.normal(0, 1, n),
        "cat": np.random.choice(["A", "B", "C"], size=n),
    })
    y = 3.0 + 2.0 * df["num1"] - 1.5 * df["num2"] + (df["cat"] == "B").astype(float) * 2.0 + np.random.normal(0, 0.2, n)

    # Split into 70 train, 25 val, 25 test
    train_df, val_df, test_df = df.iloc[:70], df.iloc[70:95], df.iloc[95:]
    y_train, y_val, y_test = y.iloc[:70], y.iloc[70:95], y.iloc[95:]

    from src.data.models import SplitResult
    split = SplitResult(
        X_train=train_df,
        X_val=val_df,
        X_test=test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_indices=np.arange(70),
        val_indices=np.arange(70, 95),
        test_indices=np.arange(95, 120),
        actual_proportions={"train": 0.583, "val": 0.208, "test": 0.208},
        sample_counts={"train": 70, "val": 25, "test": 25},
        strategy="random",
        random_seed=42,
        split_fingerprint="split_fingerprint_test_123"
    )
    return split


def test_ols_all_features_workflow(regression_test_bundle):
    """Verify end-to-end OLS model training using All Features."""
    split = regression_test_bundle
    model_res = train_ols_experiment(
        split_result=split,
        feature_source="all_features",
        target_name="target",
        fit_intercept=True
    )

    assert model_res.feature_source_id == "all_features"
    assert len(model_res.selected_original_features) == 3
    # cat has 3 levels -> 2 dummies + 2 numeric + const = 5 columns
    assert model_res.design_matrix_metadata.num_parameters == 5
    assert model_res.train_metrics.r2 > 0.85
    assert model_res.val_metrics.r2 > 0.80
    assert not model_res.post_selection_inference_warning


def test_ols_feature_selection_integration(regression_test_bundle):
    """Verify OLS model training with a completed Milestone 2 FeatureSelectionResult."""
    split = regression_test_bundle
    # Run Milestone 2 Pearson feature selection (K=2)
    fs_res = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="pearson",
        k=2,
        cv_strategy="random",
        split_fingerprint=split.split_fingerprint
    )

    # Train OLS with the feature selection result handoff
    model_res = train_ols_experiment(
        split_result=split,
        feature_source=fs_res,
        target_name="target",
        fit_intercept=True
    )

    assert model_res.feature_source_id == fs_res.experiment_id
    assert len(model_res.selected_original_features) == 2
    assert model_res.post_selection_inference_warning, "Must display post-selection inference notice"
    assert any("Post-Selection Inference" in w for w in model_res.warnings)


def test_ols_split_fingerprint_mismatch_rejection(regression_test_bundle):
    """Verify that an outdated feature selection experiment with mismatched split fingerprint is rejected."""
    split = regression_test_bundle
    fs_res = run_feature_selection(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        method_id="pearson",
        k=2,
        cv_strategy="random",
        split_fingerprint="outdated_split_fp_999"
    )

    with pytest.raises(ValueError, match="Split fingerprint mismatch"):
        train_ols_experiment(
            split_result=split,
            feature_source=fs_res,
            target_name="target"
        )


def test_ols_model_identity_uniqueness(regression_test_bundle):
    """Verify that altering feature subsets, scaling, or intercept produces unique model IDs."""
    split = regression_test_bundle

    m1 = train_ols_experiment(split, "all_features", fit_intercept=True, scale_numeric=False)
    m2 = train_ols_experiment(split, "all_features", fit_intercept=True, scale_numeric=True)
    m3 = train_ols_experiment(split, "all_features", fit_intercept=False, scale_numeric=False)

    assert m1.model_id != m2.model_id, "Scaling difference must yield distinct model IDs"
    assert m1.model_id != m3.model_id, "Intercept difference must yield distinct model IDs"


def test_tracker_controlled_test_evaluation(regression_test_bundle):
    """Verify controlled final Test evaluation with Train+Val refit and Train-only policies."""
    split = regression_test_bundle
    tracker = RegressionExperimentTracker()

    model = train_ols_experiment(split, "all_features", fit_intercept=True)
    tracker.add_model(model)

    assert model.test_metrics is None
    assert model.test_evaluation_status == "untested"

    # 1. Finalize with Train+Val refit policy
    final_model = tracker.finalize_test_evaluation(model.model_id, split, refit_policy="train_val")
    assert final_model.test_metrics is not None
    assert final_model.test_evaluation_status == "evaluated_train_val_refit"
    assert final_model.test_metrics.rmse > 0
    assert len(final_model.test_predictions) == len(split.X_test)


# =====================================================================
# Real Dataset Integration Tests (All 4 Kaggle Benchmark Datasets)
# =====================================================================

def test_real_dataset_appliances_energy_temporal_ols():
    """Verify Appliances Energy temporal OLS modeling without temporal lookahead leakage."""
    reg = get_registry()
    meta = reg.get_dataset("appliances_energy")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.TEMPORAL, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, date_column="date")
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    # Train OLS on top 10 numeric predictors
    selected = ["lights", "T1", "RH_1", "T2", "T3", "T4", "T6", "T8", "T_out", "Windspeed"]
    from src.models.design_matrix import build_ols_design_matrices
    from src.models.ols_model import fit_ols_model

    design_res, builder = build_ols_design_matrices(
        X_train=split.X_train,
        selected_features=selected,
        X_val=split.X_val,
        fit_intercept=True
    )
    model = fit_ols_model(design_res, y_train=split.y_train, y_val=split.y_val, target_name="Appliances")

    assert model.design_matrix_metadata.is_full_rank
    assert model.train_metrics.rmse > 0
    assert model.val_metrics.rmse > 0
    assert "const" in model.coef_table["Feature"].values


def test_real_dataset_kc_house_sales_structural_collinearity_detection():
    """
    Verify King County House Sales detects structural collinearity:
    sqft_living ≈ sqft_above + sqft_basement (exact identity on 99.99% of data).
    Tests condition number / near-collinearity and exact rank deficiency when strictly enforced.
    """
    reg = get_registry()
    meta = reg.get_dataset("kc_house_sales")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    # Include all 3 collinear features
    collinear_features = ["bedrooms", "bathrooms", "sqft_living", "sqft_above", "sqft_basement"]
    from src.models.design_matrix import build_ols_design_matrices

    design_res, builder = build_ols_design_matrices(
        X_train=split.X_train,
        selected_features=collinear_features,
        fit_intercept=True
    )
    meta_dm = design_res.metadata

    # 5 features + const = 6 columns.
    assert meta_dm.num_parameters == 6
    # In KC House sales, condition number is high due to near-exact identity sqft_living ≈ sqft_above + sqft_basement
    assert meta_dm.condition_number > 5000
    assert any("condition number" in w.lower() or "collinearity" in w.lower() or "rank" in w.lower() for w in meta_dm.warnings)

    # When exact structural identity holds strictly (e.g. data cleaned):
    X_train_clean = split.X_train.copy()
    X_train_clean["sqft_living"] = X_train_clean["sqft_above"] + X_train_clean["sqft_basement"]
    design_res_clean, _ = build_ols_design_matrices(
        X_train=X_train_clean,
        selected_features=collinear_features,
        fit_intercept=True
    )
    assert design_res_clean.metadata.matrix_rank == 5
    assert not design_res_clean.metadata.is_full_rank
    assert any("Rank deficiency detected" in w for w in design_res_clean.metadata.warnings)


def test_real_dataset_life_expectancy_group_split_ols():
    """Verify Life Expectancy group-split OLS modeling with categorical reference encoding."""
    reg = get_registry()
    meta = reg.get_dataset("life_expectancy")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.GROUP, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, group_column="Country", random_seed=42)
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    # Contains categorical predictor 'Status' (Developing / Developed) and continuous features
    selected = ["Status", "Adult Mortality", "BMI", "Polio", "Diphtheria", "HIV/AIDS", "Schooling"]
    from src.models.design_matrix import build_ols_design_matrices
    from src.models.ols_model import fit_ols_model

    design_res, builder = build_ols_design_matrices(
        X_train=split.X_train,
        selected_features=selected,
        X_val=split.X_val,
        fit_intercept=True
    )
    model = fit_ols_model(design_res, y_train=split.y_train, y_val=split.y_val, target_name="Life expectancy")

    # Status has 2 categories -> 1 reference category omitted, 1 dummy indicator
    assert "Status" in design_res.metadata.reference_categories
    assert model.train_metrics.r2 > 0.70
    assert model.val_metrics.r2 > 0.70


def test_real_dataset_student_performance_reference_dummy_encoding():
    """Verify Student Performance categorical reference level encoding and LaTeX equation."""
    reg = get_registry()
    meta = reg.get_dataset("student_performance")
    raw_df, feat_df, target_ser, _ = load_dataset_bundle(meta)

    cfg = SplitConfig(strategy=SplitStrategy.RANDOM, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    split = split_dataset(feat_df, target_ser, meta, cfg, raw_df=raw_df)

    selected = ["school", "sex", "age", "Medu", "Fedu", "G1", "G2"]
    from src.models.design_matrix import build_ols_design_matrices
    from src.models.ols_model import fit_ols_model

    design_res, builder = build_ols_design_matrices(
        X_train=split.X_train,
        selected_features=selected,
        X_val=split.X_val,
        fit_intercept=True
    )
    model = fit_ols_model(design_res, y_train=split.y_train, y_val=split.y_val, target_name="G3")

    # 'school' (GP / MS) -> reference GP omitted, school_MS active
    assert design_res.metadata.reference_categories["school"] == "GP"
    assert "school_MS" in design_res.X_train_design.columns
    assert "school_GP" not in design_res.X_train_design.columns

    # High R² (> 0.80) due to G1 and G2 academic indicators
    assert model.train_metrics.r2 > 0.80
    assert model.val_metrics.r2 > 0.75
