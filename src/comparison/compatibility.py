"""
Experiment Compatibility Engine for Model Comparison Laboratory.
Ensures models are only compared when trained under identical data partitions,
identical target definitions, and identical evaluation scopes.
"""
from typing import List, Optional
from src.models.model_result import OLSModelResult
from src.comparison.result_models import CompatibilityResult


def validate_model_compatibility(
    models: List[OLSModelResult],
    active_dataset_id: Optional[str] = None,
    active_dataset_fingerprint: Optional[str] = None,
    active_split_fingerprint: Optional[str] = None,
    active_target_name: Optional[str] = None,
    active_target_unit: Optional[str] = None
) -> CompatibilityResult:
    """
    Validates whether a collection of fitted OLS models can be meaningfully and fairly compared.
    
    Mandatory compatibility requirements:
    1. At least 1 model supplied (at least 2 recommended for comparison).
    2. Identical dataset ID and dataset content fingerprint.
    3. Identical split fingerprint (guarantees identical Train, Validation, and Test row partitions).
    4. Identical target variable name and measurement scale.
    5. Identical training and validation observation counts.
    
    Permitted variations (these are comparison dimensions, not incompatibilities):
    - Differing feature selection algorithms (Pearson, ANOVA, RFE, SFS, Lasso).
    - Differing feature counts (K) and OLS parameter dimensions (p).
    - Differing numerical scaling (unscaled vs StandardScaler).
    - Differing inference validity (full rank vs prediction-only).
    """
    if not models:
        return CompatibilityResult(
            is_compatible=False,
            reasons=["No models provided for comparison. At least one fitted model is required."]
        )

    reasons: List[str] = []
    warnings: List[str] = []

    # Reference values from first model or active environment
    ref_model = models[0]
    expected_dataset_id = active_dataset_id or ref_model.dataset_id
    expected_dataset_fp = active_dataset_fingerprint or ref_model.dataset_fingerprint
    expected_split_fp = active_split_fingerprint or ref_model.split_fingerprint
    expected_target_name = active_target_name or getattr(ref_model, "target_name", "target")
    expected_train_count = len(ref_model.train_predictions) if ref_model.train_predictions is not None else ref_model.train_metrics.sample_count
    expected_val_count = len(ref_model.val_predictions) if ref_model.val_predictions is not None else ref_model.val_metrics.sample_count

    # Verify each model against reference
    for idx, m in enumerate(models):
        m_id = m.model_id
        # 1. Dataset ID
        if m.dataset_id != expected_dataset_id:
            reasons.append(
                f"Dataset ID mismatch: Model '{m_id}' was trained on '{m.dataset_id}', "
                f"expected '{expected_dataset_id}'."
            )

        # 2. Dataset Fingerprint
        if expected_dataset_fp and m.dataset_fingerprint and m.dataset_fingerprint != expected_dataset_fp:
            reasons.append(
                f"Dataset fingerprint mismatch: Model '{m_id}' ({m.dataset_fingerprint[:8]}) "
                f"does not match active dataset version ({expected_dataset_fp[:8]})."
            )

        # 3. Split Fingerprint
        if expected_split_fp and m.split_fingerprint and m.split_fingerprint != expected_split_fp:
            reasons.append(
                f"Split fingerprint mismatch: Model '{m_id}' was trained on split '{m.split_fingerprint[:8]}', "
                f"expected '{expected_split_fp[:8]}'. Models from different splits cannot be ranked together."
            )

        # 4. Target Name
        m_target = getattr(m, "target_name", "target")
        if expected_target_name and m_target != expected_target_name:
            reasons.append(
                f"Target variable mismatch: Model '{m_id}' predicts '{m_target}', "
                f"expected '{expected_target_name}'."
            )

        # 5. Observation Counts
        m_train_n = len(m.train_predictions) if m.train_predictions is not None else m.train_metrics.sample_count
        m_val_n = len(m.val_predictions) if m.val_predictions is not None else m.val_metrics.sample_count

        if m_train_n != expected_train_count:
            reasons.append(
                f"Training sample count mismatch: Model '{m_id}' has {m_train_n} training samples, "
                f"expected {expected_train_count}."
            )
        if m_val_n != expected_val_count:
            reasons.append(
                f"Validation sample count mismatch: Model '{m_id}' has {m_val_n} validation samples, "
                f"expected {expected_val_count}."
            )

        # Check for prediction-only mode warnings
        if not m.is_valid_for_inference:
            warnings.append(
                f"Notice: Model '{m_id}' is in prediction-only mode (rank deficient). "
                "Conventional coefficient inference is unidentifiable, but predictive metrics remain comparable."
            )

        # Check for target unit difference
        m_unit = getattr(m, "target_unit", None)
        if m_unit and active_target_unit and m_unit != active_target_unit:
            warnings.append(
                f"Target measurement unit differs: Model '{m_id}' uses '{m_unit}', "
                f"expected '{active_target_unit}'. Ensure metrics are interpreted under comparable physical units."
            )

        # Check for intercept differences
        if m.fit_intercept != ref_model.fit_intercept:
            warnings.append(
                f"Model '{m_id}' has fit_intercept={m.fit_intercept}, differing from reference model "
                f"'{ref_model.model_id}' (fit_intercept={ref_model.fit_intercept}). R² and AIC definitions differ."
            )

    is_compatible = (len(reasons) == 0)

    return CompatibilityResult(
        is_compatible=is_compatible,
        reasons=reasons,
        warnings=warnings,
        dataset_id=expected_dataset_id,
        dataset_fingerprint=expected_dataset_fp,
        split_fingerprint=expected_split_fp,
        target_name=expected_target_name,
        target_unit=active_target_unit or ""
    )
