"""
Experiment tracking and lifecycle management for Ordinary Least Squares (OLS) regression models.
Ensures session isolation, split fingerprint invalidation, and controlled final Test evaluation.
"""
from typing import Dict, List, Optional
import pandas as pd

from src.data.models import SplitResult
from src.models.design_matrix import build_ols_design_matrices
from src.models.ols_model import fit_ols_model
from src.models.model_result import OLSModelResult
from src.evaluation.regression_metrics import compute_regression_metrics
from src.evaluation.comparison import build_model_comparison_table


class RegressionExperimentTracker:
    """
    Manages collection of fitted OLS regression models within a user session.
    Guarantees that stale models from past splits or dataset switches are invalidated,
    and enforces controlled, explicit final Test evaluation.
    """

    def __init__(self):
        self._models: Dict[str, OLSModelResult] = {}

    def add_model(self, model: OLSModelResult) -> None:
        """Store or update a fitted OLS model result."""
        self._models[model.model_id] = model

    def get_model(self, model_id: str) -> Optional[OLSModelResult]:
        """Retrieve model by unique model_id."""
        return self._models.get(model_id)

    def get_all_models(self) -> List[OLSModelResult]:
        """Return all stored models."""
        return list(self._models.values())

    def get_valid_models(
        self,
        current_split_fingerprint: str,
        current_dataset_id: Optional[str] = None,
        current_dataset_fingerprint: Optional[str] = None
    ) -> List[OLSModelResult]:
        """
        Return models strictly compatible with the active dataset and split fingerprint.
        Prevents displaying outdated models or comparing models from different partitions.
        """
        valid = []
        for m in self._models.values():
            if m.split_fingerprint == current_split_fingerprint:
                if current_dataset_id is not None and m.dataset_id != current_dataset_id:
                    continue
                if current_dataset_fingerprint is not None and m.dataset_fingerprint != current_dataset_fingerprint:
                    continue
                valid.append(m)
        return valid

    def clear(self) -> None:
        """Clear all stored models in the tracker."""
        self._models.clear()

    def generate_comparison_table(
        self,
        current_split_fingerprint: Optional[str] = None,
        current_dataset_id: Optional[str] = None,
        current_dataset_fingerprint: Optional[str] = None
    ) -> pd.DataFrame:
        """Build comparative summary leaderboard of valid models."""
        models_to_compare = (
            self.get_valid_models(current_split_fingerprint, current_dataset_id, current_dataset_fingerprint)
            if current_split_fingerprint is not None
            else self.get_all_models()
        )
        return build_model_comparison_table(models_to_compare)

    def finalize_test_evaluation(
        self,
        model_id: str,
        split_result: SplitResult,
        refit_policy: str = "train_val"
    ) -> OLSModelResult:
        """
        Executes controlled final out-of-sample Test evaluation.
        
        Parameters:
            model_id: ID of the model to finalize.
            split_result: Active SplitResult containing X_test, y_test.
            refit_policy: 'train_val' (refits fixed specification on Train+Val before test)
                          or 'train_only' (evaluates train-fitted model directly on test).
        """
        if refit_policy not in ("train_val", "train_only"):
            raise ValueError(f"Unknown refit_policy: '{refit_policy}'. Must be 'train_val' or 'train_only'.")

        model = self.get_model(model_id)
        if model is None:
            raise KeyError(f"Model ID '{model_id}' not found in tracker.")

        # Backend lock: prevent repeated final evaluation or refit policy alterations
        if model.test_evaluation_status != "untested":
            raise RuntimeError(
                f"Model '{model_id}' has already been evaluated on the Test set "
                f"(status: '{model.test_evaluation_status}', policy: '{model.test_refit_policy}'). "
                "Repeated test evaluation and post-hoc refit policy modifications are strictly locked "
                "to prevent test set data snooping and maintain statistical validity."
            )

        if model.split_fingerprint != split_result.split_fingerprint:
            raise ValueError("Split fingerprint mismatch! Cannot evaluate test data from a different split.")

        if split_result.X_test is None or split_result.y_test is None or len(split_result.X_test) == 0:
            raise ValueError("Active split does not contain test partition observations.")

        p_slopes = model.design_matrix_metadata.num_parameters - (1 if model.fit_intercept else 0)

        if refit_policy == "train_val":
            # Combine Train and Validation partitions for final refit
            X_train_val = pd.concat([split_result.X_train, split_result.X_val], axis=0)
            y_train_val = pd.concat([split_result.y_train, split_result.y_val], axis=0)

            # Build design matrix fitted on Train+Val
            design_res_final, builder_final = build_ols_design_matrices(
                X_train=X_train_val,
                selected_features=model.selected_original_features,
                X_val=None,
                X_test=split_result.X_test,
                fit_intercept=model.fit_intercept,
                scale_numeric=model.scale_numeric,
                dataset_id=model.dataset_id,
                dataset_fingerprint=model.dataset_fingerprint,
                split_fingerprint=split_result.split_fingerprint
            )

            # Fit OLS on combined Train+Val
            import statsmodels.api as sm
            ols_final = sm.OLS(y_train_val.values, design_res_final.X_train_design.values).fit()

            test_preds_arr = ols_final.predict(design_res_final.X_test_design.values)
            test_preds = pd.Series(test_preds_arr, index=split_result.X_test.index)

            test_metrics = compute_regression_metrics(
                y_true=split_result.y_test,
                y_pred=test_preds,
                num_non_intercept_parameters=p_slopes,
                split_name="Test (Train+Val Refit)",
                is_in_sample=False,
                fit_intercept=model.fit_intercept
            )

            model.test_metrics = test_metrics
            model.test_predictions = test_preds
            model.test_evaluation_status = "evaluated_train_val_refit"
            model.test_refit_policy = "train_val"
            model.warnings.append(
                f"🔒 Final Test Evaluation completed (Train+Val Refit): Test RMSE={test_metrics.rmse:.3f}, "
                f"Test R²={test_metrics.r2:.4f}."
            )

        elif refit_policy == "train_only":
            if model.builder is None or model.fitted_model is None:
                raise RuntimeError("Model does not retain fitted builder/estimator references.")

            # Transform X_test directly using the Train-only builder
            X_test_des = model.builder.transform(split_result.X_test)
            test_preds_arr = model.fitted_model.predict(X_test_des.values)
            test_preds = pd.Series(test_preds_arr, index=split_result.X_test.index)

            test_metrics = compute_regression_metrics(
                y_true=split_result.y_test,
                y_pred=test_preds,
                num_non_intercept_parameters=p_slopes,
                split_name="Test (Train-Only Model)",
                is_in_sample=False,
                fit_intercept=model.fit_intercept
            )

            model.test_metrics = test_metrics
            model.test_predictions = test_preds
            model.test_evaluation_status = "evaluated_train_only"
            model.test_refit_policy = "train_only"
            model.warnings.append(
                f"🔒 Final Test Evaluation completed (Train-Only Model): Test RMSE={test_metrics.rmse:.3f}, "
                f"Test R²={test_metrics.r2:.4f}."
            )

        self.add_model(model)
        return model
