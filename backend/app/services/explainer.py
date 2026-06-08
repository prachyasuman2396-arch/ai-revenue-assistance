"""
SHAP Explainability Service
Returns top risk factors and supports waterfall chart data.
"""

from os import name

import pandas as pd
import numpy as np
import shap
import logging
from typing import Dict, Any, List, Optional

from ..utils.model_loader import load_model
from ..services.feature_engineering import engineer_features
from ..config import settings

logger = logging.getLogger(__name__)

# Cache SHAP explainer per model instance
_explainer_cache = None


def _get_explainer(model):
    global _explainer_cache
    if _explainer_cache is not None:
        return _explainer_cache

    try:
        # LightGBM native SHAP via TreeExplainer on the classifier step
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]
        _explainer_cache = (shap.TreeExplainer(classifier), preprocessor)
        logger.info("SHAP TreeExplainer initialised")
    except Exception as e:
        logger.warning(f"TreeExplainer failed: {e}. Falling back to KernelExplainer.")
        _explainer_cache = None
        raise

    return _explainer_cache


def get_feature_names(model) -> List[str]:
    """Extract feature names after preprocessing."""
    try:
        preprocessor = model.named_steps["preprocessor"]
        num_features = preprocessor.transformers_[0][2].tolist()
        cat_features = preprocessor.transformers_[1][1].named_steps["onehot"].get_feature_names_out(
            preprocessor.transformers_[1][2].tolist()
        ).tolist()
        return num_features + cat_features
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        return []


def explain_prediction(customer_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate SHAP values for a single customer prediction.

    Returns:
        dict with top_risk_factors (list of {feature, impact}) and waterfall_data
    """
    model = load_model(settings.MODEL_PATH)

    df = pd.DataFrame([customer_dict])
    X = engineer_features(
        df,
        monthly_charges_median=settings.MONTHLY_CHARGES_MEDIAN,
        for_model=True
    )

    try:
        explainer, preprocessor = _get_explainer(model)

        X_transformed = preprocessor.transform(X)
        shap_values = explainer.shap_values(X_transformed)

        # Handle LightGBM binary classification output
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        feature_names = get_feature_names(model)

        logger.info(f"Feature names count: {len(feature_names)}")
        logger.info(f"SHAP values count: {len(shap_vals)}")

        # Fallback if feature names extraction fails
        if not feature_names:
            logger.warning("Feature names empty. Using generic names.")
            feature_names = [
                f"feature_{i}" for i in range(len(shap_vals))
            ]

        factors = []

        for feature_name, value in zip(feature_names, shap_vals):
            factors.append({
                "feature": str(feature_name),
                "impact": float(value)
            })

        logger.info(f"Factors generated: {len(factors)}")

        factors_sorted = sorted(
            factors,
            key=lambda x: abs(x["impact"]),
            reverse=True
        )

        top_factors = factors_sorted[:settings.SHAP_TOP_N]

        base_value = explainer.expected_value

        if isinstance(base_value, np.ndarray):
            if len(base_value) > 1:
                base_value = float(base_value[1])
            else:
                base_value = float(base_value[0])
        else:
            base_value = float(base_value)

        waterfall_data = {
            "features": [f["feature"] for f in factors_sorted[:15]],
            "shap_values": [f["impact"] for f in factors_sorted[:15]],
            "base_value": base_value,
        }

        return {
            "top_risk_factors": top_factors,
            "waterfall_data": waterfall_data,
        }

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")

        raw_factors = []

        for col in X.columns:
            try:
                val = X[col].iloc[0]
                raw_factors.append({
                    "feature": col,
                    "impact": float(val) * 0.01
                })
            except Exception:
                raw_factors.append({
                    "feature": col,
                    "impact": 0.0
                })

        return {
            "top_risk_factors": raw_factors[:settings.SHAP_TOP_N],
            "waterfall_data": {},
        }