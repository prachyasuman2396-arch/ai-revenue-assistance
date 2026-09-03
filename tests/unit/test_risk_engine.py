from backend.app.services.revenue_risk import calculate_revenue_risk


def test_calculate_revenue_risk_high():
    # Very high probability
    res1 = calculate_revenue_risk(
        churn_probability=0.75, monthly_charges=50.0, contract_type="Month-to-month"
    )
    assert res1["risk_category"] == "HIGH"
    assert res1["revenue_at_risk"] == 225.0  # 0.75 * (50.0 * 6)

    # Moderate probability but high bill ($95/mo)
    res2 = calculate_revenue_risk(
        churn_probability=0.45, monthly_charges=95.0, contract_type="Month-to-month"
    )
    assert res2["risk_category"] == "HIGH"
    assert res2["revenue_at_risk"] == 256.5  # 0.45 * (95.0 * 6)


def test_calculate_revenue_risk_medium():
    # Moderate churn probability
    res = calculate_revenue_risk(
        churn_probability=0.35, monthly_charges=50.0, contract_type="Month-to-month"
    )
    assert res["risk_category"] == "MEDIUM"

    # Low probability (0.22) but very high bill ($100/mo)
    res_high_bill = calculate_revenue_risk(
        churn_probability=0.22, monthly_charges=100.0, contract_type="Month-to-month"
    )
    assert res_high_bill["risk_category"] == "MEDIUM"


def test_calculate_revenue_risk_low():
    res = calculate_revenue_risk(
        churn_probability=0.10, monthly_charges=20.0, contract_type="Two year"
    )
    assert res["risk_category"] == "LOW"
    assert res["horizon_months"] == 24
    assert res["annual_revenue_exposure"] == 240.0
