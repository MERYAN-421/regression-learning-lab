"""
OLS Design Matrix Builder for Regression Learning Lab.

Constructs statistically rigorous and identifiable numerical design matrices for
Ordinary Least Squares (OLS) regression:
- Train-only median imputation for numeric features and mode imputation for categorical features
- Categorical reference-level encoding (drop='first') when an intercept is included, preventing the dummy-variable trap
- Explicit intercept handling ('const')
- Exact column-rank and numerical conditioning diagnostics
- Collinearity and perfect linear dependency detection (e.g. sqft_living = sqft_above + sqft_basement)
- Reusable pipeline for transforming Validation and Test datasets without data leakage
"""
from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DesignMatrixMetadata:
    """Detailed structural and numerical diagnostics of the constructed design matrix."""
    dataset_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    selected_original_features: List[str]
    encoded_feature_names: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    reference_categories: Dict[str, str]
    original_to_encoded_mapping: Dict[str, List[str]]
    imputed_numeric_values: Dict[str, float]
    imputed_categorical_values: Dict[str, str]
    scaling_applied: bool
    scaling_params: Optional[Dict[str, Tuple[float, float]]]
    fit_intercept: bool
    matrix_rank: int
    num_parameters: int
    residual_degrees_of_freedom: int
    condition_number: float
    is_full_rank: bool
    is_valid_for_inference: bool
    collinear_columns: List[str]
    warnings: List[str] = field(default_factory=list)
    transformation_fingerprint: str = ""


@dataclass
class DesignMatrixResult:
    """Encapsulates the generated training, validation, and test design matrices with metadata."""
    X_train_design: pd.DataFrame
    X_val_design: Optional[pd.DataFrame]
    X_test_design: Optional[pd.DataFrame]
    metadata: DesignMatrixMetadata


class OLSDesignMatrixBuilder:
    """
    Transforms raw tabular predictors into an OLS numerical design matrix.
    Fits all preprocessing strictly on training data.
    """

    def __init__(
        self,
        selected_original_features: List[str],
        fit_intercept: bool = True,
        scale_numeric: bool = False,
        dataset_id: str = "dataset",
        dataset_fingerprint: str = "",
        split_fingerprint: str = ""
    ):
        self.selected_original_features = list(selected_original_features)
        self.fit_intercept = fit_intercept
        self.scale_numeric = scale_numeric
        self.dataset_id = dataset_id
        self.dataset_fingerprint = dataset_fingerprint
        self.split_fingerprint = split_fingerprint

        # Fitted state
        self.is_fitted = False
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.num_imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.cat_imputer: Optional[SimpleImputer] = None
        self.cat_encoder: Optional[OneHotEncoder] = None
        self.reference_categories: Dict[str, str] = {}
        self.original_to_encoded_mapping: Dict[str, List[str]] = {}
        self.encoded_feature_names: List[str] = []
        self.metadata: Optional[DesignMatrixMetadata] = None

    def fit(self, X_train: pd.DataFrame) -> "OLSDesignMatrixBuilder":
        """
        Fit imputers, encoders, and scalers strictly on the training partition.
        """
        # Validate selected features
        missing_cols = [c for c in self.selected_original_features if c not in X_train.columns]
        if missing_cols:
            raise ValueError(f"Selected features {missing_cols} not found in training DataFrame.")

        if not self.selected_original_features:
            raise ValueError("Cannot build OLS design matrix with an empty feature list.")

        X_sub = X_train[self.selected_original_features].copy()

        # Identify numeric vs categorical columns
        self.numeric_features = X_sub.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_features = [c for c in self.selected_original_features if c not in self.numeric_features]

        self.original_to_encoded_mapping = {}
        imputed_numeric_values: Dict[str, float] = {}
        imputed_categorical_values: Dict[str, str] = {}
        scaling_params: Optional[Dict[str, Tuple[float, float]]] = {} if self.scale_numeric else None
        warnings_list: List[str] = []

        # 1. Numeric Preprocessing (Median imputation, optional scaling)
        if self.numeric_features:
            self.num_imputer = SimpleImputer(strategy="median")
            self.num_imputer.fit(X_sub[self.numeric_features])
            for idx, col in enumerate(self.numeric_features):
                imputed_numeric_values[col] = float(self.num_imputer.statistics_[idx])
                # In original units or scaled, numeric feature maps to its own name
                self.original_to_encoded_mapping[col] = [col]

            if self.scale_numeric:
                self.scaler = StandardScaler()
                self.scaler.fit(self.num_imputer.transform(X_sub[self.numeric_features]))
                for idx, col in enumerate(self.numeric_features):
                    mean = float(self.scaler.mean_[idx])
                    scale = float(self.scaler.scale_[idx])
                    scaling_params[col] = (mean, scale)

        # 2. Categorical Preprocessing (Mode imputation, reference-category OHE)
        self.reference_categories = {}
        if self.categorical_features:
            cat_df = X_sub[self.categorical_features].copy()
            for col in self.categorical_features:
                cat_df[col] = cat_df[col].replace({None: np.nan})

            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            self.cat_imputer.fit(cat_df)
            for idx, col in enumerate(self.categorical_features):
                imputed_categorical_values[col] = str(self.cat_imputer.statistics_[idx])

            cat_imputed = self.cat_imputer.transform(cat_df)

            # When intercept is included, use drop='first' to prevent the dummy-variable trap
            drop_strategy = "first" if self.fit_intercept else None
            self.cat_encoder = OneHotEncoder(
                drop=drop_strategy,
                handle_unknown="ignore",
                sparse_output=False
            )
            self.cat_encoder.fit(cat_imputed)

            # Map categories and record omitted reference levels
            for idx, col in enumerate(self.categorical_features):
                categories = self.cat_encoder.categories_[idx]
                if len(categories) == 0:
                    warnings_list.append(f"Categorical feature '{col}' has zero unique levels.")
                    self.original_to_encoded_mapping[col] = []
                    continue

                if self.fit_intercept:
                    ref_cat = str(categories[0])
                    self.reference_categories[col] = ref_cat
                    remaining_cats = categories[1:]
                    dummy_names = [f"{col}_{cat}" for cat in remaining_cats]
                    self.original_to_encoded_mapping[col] = dummy_names
                    if len(categories) == 1:
                        warnings_list.append(
                            f"Categorical feature '{col}' has only 1 level ('{ref_cat}'). "
                            "It has no active non-reference dummy indicators."
                        )
                else:
                    dummy_names = [f"{col}_{cat}" for cat in categories]
                    self.original_to_encoded_mapping[col] = dummy_names

        # 3. Assemble full list of non-intercept encoded names
        ordered_encoded: List[str] = []
        for orig in self.selected_original_features:
            ordered_encoded.extend(self.original_to_encoded_mapping.get(orig, []))

        # Add intercept column 'const' at position 0 if requested
        if self.fit_intercept:
            self.encoded_feature_names = ["const"] + ordered_encoded
        else:
            self.encoded_feature_names = ordered_encoded

        self.is_fitted = True

        # Perform transformation and matrix diagnostics on training data
        X_train_design = self.transform(X_train)
        metadata = self._compute_matrix_diagnostics(
            X_train_design,
            imputed_numeric_values,
            imputed_categorical_values,
            scaling_params,
            warnings_list
        )
        self.metadata = metadata
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a feature DataFrame using fitted imputers, encoders, and intercept rules.
        """
        if not self.is_fitted:
            raise RuntimeError("OLSDesignMatrixBuilder must be fitted before transforming data.")

        missing_cols = [c for c in self.selected_original_features if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Required features {missing_cols} missing from DataFrame.")

        X_sub = X[self.selected_original_features].copy()
        arrays_to_concat: List[np.ndarray] = []
        col_names: List[str] = []

        # Add intercept column
        if self.fit_intercept:
            intercept_arr = np.ones((len(X_sub), 1), dtype=float)
            arrays_to_concat.append(intercept_arr)
            col_names.append("const")

        # Transform numeric features
        if self.numeric_features and self.num_imputer is not None:
            num_imp = self.num_imputer.transform(X_sub[self.numeric_features])
            if self.scale_numeric and self.scaler is not None:
                num_arr = self.scaler.transform(num_imp)
            else:
                num_arr = num_imp
            arrays_to_concat.append(num_arr)
            col_names.extend(self.numeric_features)

        # Transform categorical features
        if self.categorical_features and self.cat_imputer is not None and self.cat_encoder is not None:
            cat_df = X_sub[self.categorical_features].copy()
            for col in self.categorical_features:
                cat_df[col] = cat_df[col].replace({None: np.nan})
            cat_imp = self.cat_imputer.transform(cat_df)
            cat_arr = self.cat_encoder.transform(cat_imp)
            arrays_to_concat.append(cat_arr)

            # Get encoder output feature names
            cat_feature_names = []
            for idx, col in enumerate(self.categorical_features):
                cat_feature_names.extend(self.original_to_encoded_mapping.get(col, []))
            col_names.extend(cat_feature_names)

        if not arrays_to_concat:
            full_matrix = np.empty((len(X_sub), 0), dtype=float)
        else:
            full_matrix = np.hstack(arrays_to_concat)

        return pd.DataFrame(full_matrix, columns=col_names, index=X.index)

    def _compute_matrix_diagnostics(
        self,
        X_train_design: pd.DataFrame,
        imputed_numeric_values: Dict[str, float],
        imputed_categorical_values: Dict[str, str],
        scaling_params: Optional[Dict[str, Tuple[float, float]]],
        base_warnings: List[str]
    ) -> DesignMatrixMetadata:
        """
        Evaluate full column rank, residual degrees of freedom, conditioning, and exact linear dependencies.
        """
        warnings_list = list(base_warnings)
        n_obs = len(X_train_design)
        p_cols = X_train_design.shape[1]
        mat_vals = X_train_design.values.astype(float)

        # Check for non-finite entries
        if not np.all(np.isfinite(mat_vals)):
            warnings_list.append("Design matrix contains non-finite (NaN or Inf) values.")

        # Check for constant non-intercept columns
        for c in X_train_design.columns:
            if c != "const" and np.nanstd(X_train_design[c]) < 1e-12:
                warnings_list.append(f"Predictor column '{c}' has zero variance (constant).")

        # Rank and condition number
        if p_cols > 0 and n_obs > 0:
            rank = int(np.linalg.matrix_rank(mat_vals))
            try:
                cond_num = float(np.linalg.cond(mat_vals))
            except Exception:
                cond_num = float("inf")
        else:
            rank = 0
            cond_num = float("inf")

        df_resid = n_obs - p_cols
        is_full_rank = (rank == p_cols)
        collinear_cols: List[str] = []

        if not is_full_rank and p_cols > 1:
            # Diagnose which columns are linearly dependent via QR decomposition with column pivoting
            try:
                # Q, R, P = scipy.linalg.qr(mat_vals, pivoting=True)
                from scipy.linalg import qr
                _, r_mat, p_piv = qr(mat_vals, pivoting=True)
                diag_r = np.abs(np.diag(r_mat))
                tol = max(mat_vals.shape) * np.spacing(np.max(diag_r)) if len(diag_r) > 0 else 1e-7
                dep_indices = p_piv[diag_r < tol]
                collinear_cols = [X_train_design.columns[i] for i in dep_indices]
            except Exception:
                # Fallback simple collinearity scanner
                collinear_cols = []

            warnings_list.append(
                f"Rank deficiency detected: Matrix rank is {rank} but {p_cols} columns were specified. "
                f"Exact linear dependence prevents unique OLS parameter identification."
            )
            if collinear_cols:
                warnings_list.append(
                    f"Linearly dependent column(s): {', '.join(collinear_cols)}. "
                    "(e.g., structural identities such as sqft_living = sqft_above + sqft_basement)."
                )

        if df_resid <= 0:
            warnings_list.append(
                f"Insufficient residual degrees of freedom ({df_resid}). "
                f"Number of parameters ({p_cols}) >= sample size ({n_obs})."
            )

        if is_full_rank and cond_num > 1e4:
            warnings_list.append(
                f"High condition number ({cond_num:.2e}): Near-multicollinearity detected. "
                "Parameter estimates may be sensitive to slight numerical perturbations."
            )

        is_valid_for_inference = is_full_rank and (df_resid > 0)

        # Transformation fingerprint
        fp_components = [
            self.dataset_fingerprint,
            self.split_fingerprint,
            str(self.fit_intercept),
            str(self.scale_numeric),
            ",".join(self.selected_original_features),
            str(p_cols),
            str(rank)
        ]
        trans_fp = "dm_" + hashlib.sha256("||".join(fp_components).encode("utf-8")).hexdigest()[:12]

        return DesignMatrixMetadata(
            dataset_id=self.dataset_id,
            dataset_fingerprint=self.dataset_fingerprint,
            split_fingerprint=self.split_fingerprint,
            selected_original_features=list(self.selected_original_features),
            encoded_feature_names=list(self.encoded_feature_names),
            numeric_features=list(self.numeric_features),
            categorical_features=list(self.categorical_features),
            reference_categories=dict(self.reference_categories),
            original_to_encoded_mapping=dict(self.original_to_encoded_mapping),
            imputed_numeric_values=imputed_numeric_values,
            imputed_categorical_values=imputed_categorical_values,
            scaling_applied=self.scale_numeric,
            scaling_params=scaling_params,
            fit_intercept=self.fit_intercept,
            matrix_rank=rank,
            num_parameters=p_cols,
            residual_degrees_of_freedom=df_resid,
            condition_number=cond_num,
            is_full_rank=is_full_rank,
            is_valid_for_inference=is_valid_for_inference,
            collinear_columns=collinear_cols,
            warnings=warnings_list,
            transformation_fingerprint=trans_fp
        )


def build_ols_design_matrices(
    X_train: pd.DataFrame,
    selected_features: List[str],
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    fit_intercept: bool = True,
    scale_numeric: bool = False,
    dataset_id: str = "dataset",
    dataset_fingerprint: str = "",
    split_fingerprint: str = ""
) -> Tuple[DesignMatrixResult, OLSDesignMatrixBuilder]:
    """
    Convenience function to fit an OLSDesignMatrixBuilder on X_train and transform
    X_train, and optionally X_val and X_test.
    """
    builder = OLSDesignMatrixBuilder(
        selected_original_features=selected_features,
        fit_intercept=fit_intercept,
        scale_numeric=scale_numeric,
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_fingerprint
    )
    builder.fit(X_train)

    X_train_design = builder.transform(X_train)
    X_val_design = builder.transform(X_val) if X_val is not None else None
    X_test_design = builder.transform(X_test) if X_test is not None else None

    result = DesignMatrixResult(
        X_train_design=X_train_design,
        X_val_design=X_val_design,
        X_test_design=X_test_design,
        metadata=builder.metadata
    )
    return result, builder
