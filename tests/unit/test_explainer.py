import pytest

from backend.app.services.explainer import ChurnExplainer


@pytest.fixture
def high_risk_customer():
    return {
        "gender": "Female",
        "senior_citizen": 1,
        "partner": False,
        "dependents": False,
        "tenure": 2,
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
        "monthly_charges": 70.7,
        "total_charges": 151.65,
    }


def test_explainer_high_risk_customer(high_risk_customer):
    explainer = ChurnExplainer()
    explanation = explainer.explain_instance(high_risk_customer, top_k=3)

    assert "top_risk_factors" in explanation
    assert "top_protective_factors" in explanation
    assert "summary" in explanation
    assert len(explanation["top_risk_factors"]) > 0

    # High-risk profile should identify month-to-month or electronic check
    risk_feats = [r["feature"] for r in explanation["top_risk_factors"]]
    assert any(
        "contract" in f
        or "month_to_month" in f
        or "payment_method" in f
        or "has_fiber" in f
        for f in risk_feats
    )
    assert len(explanation["summary"]) > 20
