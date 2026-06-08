"""
Prediction Service
Pipeline: Raw Customer Data → Feature Engineering → Preprocessing → LightGBM → Probability
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Any

from ..utils.model_loader import load_model
from ..services.feature_engineering import engineer_features
from ..config import settings

logger = logging.getLogger(__name__)


def predict_single(customer_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full prediction pipeline for a single customer.
    
    Returns:
        dict with churn_probability, churn_prediction, confidence, inference_time_ms
    """
    start = time.perf_counter()

    model = load_model(settings.MODEL_PATH)

    # Build raw dataframe (single row)
    df = pd.DataFrame([customer_dict])

    # Feature engineering → returns X columns as used during training
    X = engineer_features(df, monthly_charges_median=settings.MONTHLY_CHARGES_MEDIAN, for_model=True)

    # Predict
    prob = float(model.predict_proba(X)[0, 1])
    prediction = prob >= 0.3  # threshold from notebook

    confidence = "High" if prob >= 0.7 or prob <= 0.3 else "Medium" if prob >= 0.5 or prob <= 0.4 else "Low"

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": bool(prediction),
        "confidence": confidence,
        "inference_time_ms": round(elapsed_ms, 2),
    }


def predict_batch(customers: list) -> list:
    """
    Run prediction for a batch of customers.
    Returns list of prediction dicts.
    """
    model = load_model(settings.MODEL_PATH)

    df = pd.DataFrame(customers)
    X = engineer_features(df, monthly_charges_median=settings.MONTHLY_CHARGES_MEDIAN, for_model=True)

    probs = model.predict_proba(X)[:, 1]
    results = []
    for i, prob in enumerate(probs):
        prob = float(prob)
        prediction = prob >= 0.3
        confidence = "High" if prob >= 0.7 or prob <= 0.3 else "Medium"
        results.append({
            "churn_probability": round(prob, 4),
            "churn_prediction": bool(prediction),
            "confidence": confidence,
        })
    return results
