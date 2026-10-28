"""
Experiment runner and multi-method tracking registry for Feature Selection.
Stores completed experiment results, invalidates stale runs, and generates comparative audits.
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from src.features.result_models import FeatureSelectionResult


class FeatureSelectionExperimentTracker:
    """
    Manages history of Feature Selection experiments for the active dataset session.
    Provides comparison summaries and feature overlap matrices across algorithms.
    """

    def __init__(self):
        self._experiments: Dict[str, FeatureSelectionResult] = {}

    def add_result(self, result: FeatureSelectionResult) -> None:
        """Store an experiment result keyed by experiment ID."""
        self._experiments[result.experiment_id] = result

    def get_result(self, experiment_id: str) -> Optional[FeatureSelectionResult]:
        return self._experiments.get(experiment_id)

    def list_results(self) -> List[FeatureSelectionResult]:
        return list(self._experiments.values())

    def clear(self) -> None:
        self._experiments.clear()

    def invalidate_stale_experiments(
        self,
        current_dataset_id: str,
        current_split_fingerprint: str
    ) -> int:
        """
        Purge experiments that were computed on a different dataset or split configuration.
        Returns the number of purged stale experiments.
        """
        to_delete = []
        for exp_id, res in self._experiments.items():
            if res.dataset_id != current_dataset_id or res.split_fingerprint != current_split_fingerprint:
                to_delete.append(exp_id)

        for exp_id in to_delete:
            del self._experiments[exp_id]

        return len(to_delete)

    def get_valid_experiments(
        self,
        current_split_fingerprint: str,
        current_dataset_id: Optional[str] = None
    ) -> List[FeatureSelectionResult]:
        """Return all experiment results that match the active split fingerprint (and dataset ID if given)."""
        valid = []
        for res in self._experiments.values():
            if res.split_fingerprint == current_split_fingerprint:
                if current_dataset_id is None or res.dataset_id == current_dataset_id:
                    valid.append(res)
        return valid

    def generate_comparison_table(self) -> pd.DataFrame:
        """Construct side-by-side comparison summary table across all stored experiments."""
        if not self._experiments:
            return pd.DataFrame()

        rows = []
        for res in self._experiments.values():
            rows.append({
                "Experiment ID": res.experiment_id,
                "Method": res.method_name,
                "Category": res.category,
                "Req K": res.requested_k,
                "Total Orig": res.total_original_count,
                "Selected Orig": res.selected_original_count,
                "Full Enc Dim": res.full_encoded_dimension,
                "Selected Enc Dim": res.selected_encoded_dimension,
                "CV RMSE Mean": round(res.cv_rmse_mean, 3),
                "CV RMSE Std": round(res.cv_rmse_std, 3),
                "Val MSE": round(res.val_mse, 3),
                "Val RMSE": round(res.val_rmse, 3),
                "Val R²": round(res.val_r2, 3),
                "Runtime (s)": round(res.execution_time_seconds, 2)
            })

        df = pd.DataFrame(rows)
        return df.sort_values(by="Val RMSE", ascending=True).reset_index(drop=True)

    def generate_feature_overlap_matrix(self) -> pd.DataFrame:
        """
        Construct binary indicator matrix showing which original features were selected
        by each experiment method.
        """
        if not self._experiments:
            return pd.DataFrame()

        # Collect all unique original features across experiments
        all_features = sorted(list(set(
            feat
            for res in self._experiments.values()
            for feat in res.all_original_features
        )))

        overlap_dict = {"Original Feature": all_features}
        for exp_id, res in self._experiments.items():
            col_label = f"{res.method_name} (K={res.selected_original_count})"
            selected_set = set(res.selected_original_features)
            overlap_dict[col_label] = ["✅" if f in selected_set else "—" for f in all_features]

        df = pd.DataFrame(overlap_dict)
        return df
