from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.services.predictor import ModelPredictor

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "champion_model.joblib"


def test_champion_model_artifact_exists():
    assert MODEL_PATH.exists(), f"Champion model artifact missing at: {MODEL_PATH}"


def test_champion_model_inference_pipeline():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba"), "Model must implement predict_proba"

    # Synthetic sample row
    sample_df = pd.DataFrame(
        [
            {
                "gender": "Male",
                "senior_citizen": 0,
                "partner": True,
                "dependents": False,
                "tenure": 12,
                "phone_service": True,
                "multiple_lines": "No",
                "internet_service": "DSL",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "Yes",
                "tech_support": "Yes",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "One year",
                "paperless_billing": False,
                "payment_method": "Credit card (automatic)",
                "monthly_charges": 55.0,
                "total_charges": 660.0,
            }
        ]
    )

    probs = model.predict_proba(sample_df)
    assert probs.shape == (1, 2)
    assert np.isclose(probs.sum(), 1.0)
    assert 0.0 <= probs[0, 1] <= 1.0


def test_predictor_service_integration():
    predictor = ModelPredictor()
    assert predictor.model is not None
    assert predictor.model_name is not None

    sample_dict = {
        "customer_id": "INTEGRATION_TEST_USER",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": False,
        "dependents": False,
        "tenure": 4,
        "phone_service": True,
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": True,
        "payment_method": "Electronic check",
        "monthly_charges": 75.0,
        "total_charges": 300.0,
    }

    result = predictor.predict_and_assess(sample_dict, save_audit=False)
    assert result["customer_id"] == "INTEGRATION_TEST_USER"
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_analysis"]["risk_category"] in ("HIGH", "MEDIUM", "LOW")
    assert "PLAYBOOK" in result["recommendation"]["playbook_code"]
    assert len(result["explanation"]["summary"]) > 0
