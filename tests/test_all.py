"""
Test suite for AI Revenue Risk Intelligence Platform.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ──────────────────────────────────────────────
# Sample customer fixture
# ──────────────────────────────────────────────

@pytest.fixture
def sample_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0,
    }


@pytest.fixture
def sample_df(sample_customer):
    return pd.DataFrame([sample_customer])


# ──────────────────────────────────────────────
# Feature Engineering Tests
# ──────────────────────────────────────────────

class TestFeatureEngineering:

    def test_fix_total_charges_coerces_string(self):
        from app.services.feature_engineering import fix_total_charges
        df = pd.DataFrame([{"TotalCharges": " ", "MonthlyCharges": 50.0, "tenure": 10}])
        result = fix_total_charges(df)
        assert result["TotalCharges"].dtype == float
        assert not result["TotalCharges"].isnull().any()

    def test_add_clv(self, sample_df):
        from app.services.feature_engineering import add_clv
        result = add_clv(sample_df)
        assert "CLV" in result.columns
        expected = 85.5 * 12
        assert abs(result["CLV"].iloc[0] - expected) < 0.01

    def test_add_avg_spend(self, sample_df):
        from app.services.feature_engineering import add_avg_spend
        result = add_avg_spend(sample_df)
        assert "AvgSpend" in result.columns
        expected = 1026.0 / (12 + 1)
        assert abs(result["AvgSpend"].iloc[0] - expected) < 0.01

    def test_add_tenure_group(self, sample_df):
        from app.services.feature_engineering import add_tenure_group
        result = add_tenure_group(sample_df)
        assert "TenureGroup" in result.columns
        assert str(result["TenureGroup"].iloc[0]) == "0-12"

    def test_add_high_charges_above_median(self, sample_df):
        from app.services.feature_engineering import add_high_charges
        result = add_high_charges(sample_df, median_monthly=70.35)
        # 85.5 > 70.35 → HighCharges = 1
        assert result["HighCharges"].iloc[0] == 1

    def test_add_high_charges_below_median(self):
        from app.services.feature_engineering import add_high_charges
        df = pd.DataFrame([{"MonthlyCharges": 30.0}])
        result = add_high_charges(df, median_monthly=70.35)
        assert result["HighCharges"].iloc[0] == 0

    def test_add_auto_payment_automatic(self, sample_df):
        from app.services.feature_engineering import add_auto_payment
        df = sample_df.copy()
        df["PaymentMethod"] = "Bank transfer (automatic)"
        result = add_auto_payment(df)
        assert result["AutoPayment"].iloc[0] == 1

    def test_add_auto_payment_non_automatic(self, sample_df):
        from app.services.feature_engineering import add_auto_payment
        result = add_auto_payment(sample_df)  # Electronic check = not automatic
        assert result["AutoPayment"].iloc[0] == 0

    def test_add_protection_bundle(self):
        from app.services.feature_engineering import add_protection_bundle
        df = pd.DataFrame([{
            "OnlineSecurity": "Yes",
            "TechSupport": "Yes",
            "DeviceProtection": "No",
        }])
        result = add_protection_bundle(df)
        assert result["ProtectionBundle"].iloc[0] == 2

    def test_add_streaming_bundle(self):
        from app.services.feature_engineering import add_streaming_bundle
        df = pd.DataFrame([{"StreamingTV": "Yes", "StreamingMovies": "Yes"}])
        result = add_streaming_bundle(df)
        assert result["StreamingBundle"].iloc[0] == 2

    def test_add_new_customer_short_tenure(self, sample_df):
        from app.services.feature_engineering import add_new_customer
        result = add_new_customer(sample_df)  # tenure=12 → NewCustomer=1
        assert result["NewCustomer"].iloc[0] == 1

    def test_add_new_customer_long_tenure(self):
        from app.services.feature_engineering import add_new_customer
        df = pd.DataFrame([{"tenure": 36}])
        result = add_new_customer(df)
        assert result["NewCustomer"].iloc[0] == 0

    def test_add_revenue_exposure(self, sample_df):
        from app.services.feature_engineering import add_revenue_exposure
        result = add_revenue_exposure(sample_df)
        assert "RevenueExposure" in result.columns
        expected = 85.5 * np.log1p(12)
        assert abs(result["RevenueExposure"].iloc[0] - expected) < 0.01

    def test_engineer_features_returns_raw_columns(self, sample_df):
        from app.services.feature_engineering import engineer_features, MODEL_INPUT_COLUMNS
        result = engineer_features(sample_df, for_model=True)
        for col in MODEL_INPUT_COLUMNS:
            if col in sample_df.columns:
                assert col in result.columns

    def test_engineer_features_full_has_engineered_cols(self, sample_df):
        from app.services.feature_engineering import engineer_features
        result = engineer_features(sample_df, for_model=False)
        for col in ["CLV", "AvgSpend", "HighCharges", "AutoPayment", "ProtectionBundle",
                    "StreamingBundle", "NewCustomer", "RevenueExposure"]:
            assert col in result.columns


# ──────────────────────────────────────────────
# Revenue Risk Engine Tests
# ──────────────────────────────────────────────

class TestRevenueRiskEngine:

    def test_calculate_clv(self):
        from app.services.revenue_risk import calculate_clv
        assert calculate_clv(100.0, 12) == 1200.0

    def test_calculate_clv_zero_tenure(self):
        from app.services.revenue_risk import calculate_clv
        assert calculate_clv(100.0, 0) == 0.0

    def test_calculate_avg_spend(self):
        from app.services.revenue_risk import calculate_avg_spend
        assert abs(calculate_avg_spend(1026.0, 12) - (1026.0 / 13)) < 0.01

    def test_calculate_revenue_at_risk(self):
        from app.services.revenue_risk import calculate_revenue_at_risk
        assert calculate_revenue_at_risk(0.8, 1000.0) == 800.0

    def test_risk_band_critical(self):
        from app.services.revenue_risk import calculate_risk_band
        assert calculate_risk_band(0.85) == "Critical"

    def test_risk_band_high(self):
        from app.services.revenue_risk import calculate_risk_band
        assert calculate_risk_band(0.65) == "High"

    def test_risk_band_medium(self):
        from app.services.revenue_risk import calculate_risk_band
        assert calculate_risk_band(0.45) == "Medium"

    def test_risk_band_low(self):
        from app.services.revenue_risk import calculate_risk_band
        assert calculate_risk_band(0.2) == "Low"

    def test_risk_category_from_score(self):
        from app.services.revenue_risk import calculate_risk_category_from_score
        assert calculate_risk_category_from_score(85) == "Critical"
        assert calculate_risk_category_from_score(65) == "High"
        assert calculate_risk_category_from_score(45) == "Medium"
        assert calculate_risk_category_from_score(10) == "Low"

    def test_revenue_risk_score_capped_at_100(self):
        from app.services.revenue_risk import calculate_revenue_risk_score
        score = calculate_revenue_risk_score(99999, max_revenue_at_risk=1000)
        assert score == 100.0

    def test_calculate_revenue_metrics_full(self, sample_customer):
        from app.services.revenue_risk import calculate_revenue_metrics
        result = calculate_revenue_metrics(sample_customer, churn_probability=0.75)
        assert "customer_lifetime_value" in result
        assert "revenue_at_risk" in result
        assert "revenue_risk_score" in result
        assert "risk_band" in result
        assert result["risk_band"] == "High"  # 0.75 → High
        assert result["customer_lifetime_value"] == 85.5 * 12

    def test_basic_recommendation_critical(self):
        from app.services.revenue_risk import basic_recommendation
        assert basic_recommendation("Critical") == "Immediate retention campaign"

    def test_basic_recommendation_low(self):
        from app.services.revenue_risk import basic_recommendation
        assert basic_recommendation("Low") == "No action"

    def test_calculate_dashboard_metrics(self, sample_customer):
        from app.services.revenue_risk import calculate_dashboard_metrics
        customers = [sample_customer, sample_customer]
        predictions = [
            {"churn_probability": 0.8},
            {"churn_probability": 0.2},
        ]
        result = calculate_dashboard_metrics(customers, predictions)
        assert result["total_customers"] == 2
        assert 0.0 < result["avg_churn_probability"] < 1.0
        assert result["total_revenue_at_risk"] >= 0


# ──────────────────────────────────────────────
# Groq Recommendation Fallback Tests
# ──────────────────────────────────────────────

class TestGroqRecommendation:

    def test_rule_based_fallback_returns_4_actions(self, sample_customer):
        from app.services.groq_recommendation import _rule_based_fallback
        result = _rule_based_fallback(
            customer_profile=sample_customer,
            prediction={"churn_probability": 0.8, "risk_band": "Critical"},
            revenue_metrics={"customer_lifetime_value": 1026.0, "revenue_at_risk": 820.8},
            shap_factors=[{"feature": "Contract_Month-to-month", "impact": 0.3}],
        )
        assert len(result["retention_strategy"]) == 4
        assert result["generated_summary"] != ""

    def test_groq_without_api_key_uses_fallback(self, sample_customer):
        from app.services.groq_recommendation import get_retention_recommendation
        result = get_retention_recommendation(
            customer_profile=sample_customer,
            prediction={"churn_probability": 0.7, "risk_band": "High", "confidence": "High"},
            revenue_metrics={"customer_lifetime_value": 1026.0, "revenue_at_risk": 718.2, "revenue_risk_score": 60.0},
            shap_factors=[],
            groq_api_key=None,
        )
        assert result["source"] == "rule_based"
        assert len(result["retention_strategy"]) > 0


# ──────────────────────────────────────────────
# API Route Tests (with mock model)
# ──────────────────────────────────────────────

class TestAPIRoutes:

    @pytest.fixture
    def client(self, tmp_path):
        """FastAPI test client with a mocked model."""
        from fastapi.testclient import TestClient

        # Create a mock pipeline that returns a fixed probability
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.named_steps = {
            "preprocessor": MagicMock(),
            "classifier": MagicMock(),
        }
        mock_model.named_steps["preprocessor"].transform.return_value = np.zeros((1, 30))
        mock_model.named_steps["preprocessor"].transformers_ = [
            ("num", MagicMock(), ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]),
            ("cat", MagicMock(), ["gender", "Contract", "PaymentMethod"]),
        ]
        mock_model.named_steps["classifier"].predict.return_value = np.array([1])

        # Patch load_model everywhere it's used
        with patch("app.utils.model_loader.load_model", return_value=mock_model), \
             patch("app.utils.model_loader.is_model_loaded", return_value=True):

            # Also patch the shap explainer to avoid SHAP errors in unit tests
            with patch("app.services.explainer._get_explainer") as mock_explainer:
                mock_exp = MagicMock()
                mock_exp.shap_values.return_value = [np.zeros((1, 30)), np.zeros((1, 30))]
                mock_exp.expected_value = np.array([0.3, 0.5])
                mock_explainer.return_value = (mock_exp, mock_model.named_steps["preprocessor"])

                from backend.main import app
                yield TestClient(app)

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_endpoint(self, client, sample_customer):
        resp = client.post("/predict", json=sample_customer)
        assert resp.status_code == 200
        data = resp.json()
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_full_analysis_endpoint(self, client, sample_customer):
        payload = {"customer": sample_customer}
        resp = client.post("/full-analysis", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "revenue_metrics" in data
        assert "risk_band" in data
        assert "retention_strategy" in data

    def test_dashboard_endpoint(self, client, sample_customer):
        payload = {"customers": [sample_customer, sample_customer]}
        resp = client.post("/dashboard", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_customers"] == 2
        assert "risk_distribution" in data
