from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to AI Revenue Assistance" in response.json()["message"]


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "model_version" in data


def test_predict_endpoint_payload():
    payload = {
        "customer_id": "TEST_PROSPECT_001",
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
        "monthly_charges": 85.50,
        "total_charges": 171.0,
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "TEST_PROSPECT_001"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["prediction"] in (0, 1)

    # Check risk analysis
    assert "risk_analysis" in data
    assert data["risk_analysis"]["risk_category"] in ("HIGH", "MEDIUM", "LOW")
    assert data["risk_analysis"]["revenue_at_risk"] >= 0.0

    # Check recommendation
    assert "recommendation" in data
    assert "primary_action" in data["recommendation"]
    assert "playbook_code" in data["recommendation"]

    # Check explainability
    assert "explanation" in data
    assert "top_risk_factors" in data["explanation"]
    assert len(data["explanation"]["summary"]) > 10


def test_get_customers():
    response = client.get("/api/customers?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "customer_id" in data[0]


def test_predict_for_existing_customer_end_to_end():
    # 1. Fetch a known customer from the database
    cust_res = client.get("/api/customers?limit=1")
    assert cust_res.status_code == 200
    test_customer = cust_res.json()[0]
    test_id = test_customer["customer_id"]

    # 2. Score the customer through the database integration endpoint
    pred_res = client.post(f"/api/predict/{test_id}")
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert pred_data["customer_id"] == test_id
    assert "risk_analysis" in pred_data
    assert "recommendation" in pred_data

    # 3. Verify prediction was persisted to PostgreSQL
    hist_res = client.get(f"/api/predictions/{test_id}")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert len(hist_data) >= 1
    assert hist_data[0]["customer_id"] == test_id

    # 4. Verify recommendation was persisted to PostgreSQL
    rec_res = client.get(f"/api/recommendations/{test_id}")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert len(rec_data) >= 1
    assert rec_data[0]["customer_id"] == test_id
