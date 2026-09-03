from backend.app.services.recommendation import generate_recommendations


def test_recommendation_high_risk_early_tenure():
    customer = {
        "tenure": 2,
        "contract": "Month-to-month",
        "monthly_charges": 80.0,
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }
    risk_info = {
        "risk_category": "HIGH",
        "revenue_at_risk": 384.0,
        "churn_probability": 0.80,
    }
    rec = generate_recommendations(customer, risk_info)
    assert rec["priority"] == "P1_URGENT"
    assert rec["playbook_code"] == "PLAYBOOK_ONBOARDING_CONCIERGE"
    assert "Early-Life" in rec["primary_action"]


def test_recommendation_high_risk_high_bill():
    customer = {
        "tenure": 18,
        "contract": "Month-to-month",
        "monthly_charges": 105.0,
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }
    risk_info = {
        "risk_category": "HIGH",
        "revenue_at_risk": 441.0,
        "churn_probability": 0.70,
    }
    rec = generate_recommendations(customer, risk_info)
    assert rec["priority"] == "P1_URGENT"
    assert rec["playbook_code"] == "PLAYBOOK_ANNUAL_CONTRACT_MIGRATION"


def test_recommendation_medium_risk_tech_support():
    customer = {
        "tenure": 24,
        "contract": "One year",
        "monthly_charges": 90.0,
        "internet_service": "Fiber optic",
        "tech_support": "No",
        "online_security": "No",
        "payment_method": "Credit card (automatic)",
    }
    risk_info = {
        "risk_category": "MEDIUM",
        "revenue_at_risk": 216.0,
        "churn_probability": 0.40,
    }
    rec = generate_recommendations(customer, risk_info)
    assert rec["priority"] == "P2_ELEVATED"
    assert rec["playbook_code"] == "PLAYBOOK_TECH_SUPPORT_BUNDLE"


def test_recommendation_low_risk():
    customer = {
        "tenure": 60,
        "contract": "Two year",
        "monthly_charges": 20.0,
        "internet_service": "No",
        "payment_method": "Bank transfer (automatic)",
    }
    risk_info = {
        "risk_category": "LOW",
        "revenue_at_risk": 24.0,
        "churn_probability": 0.05,
    }
    rec = generate_recommendations(customer, risk_info)
    assert rec["priority"] == "P3_STANDARD"
    assert rec["playbook_code"] == "PLAYBOOK_LOYALTY_CADENCE"
