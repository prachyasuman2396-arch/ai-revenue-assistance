from typing import Any, Dict


def calculate_revenue_risk(
    churn_probability: float,
    monthly_charges: float,
    contract_type: str = "Month-to-month",
    tenure_months: int = 12,
) -> Dict[str, Any]:
    """Translates calibrated churn probability and customer subscription value

    into quantifiable financial risk and an operational risk tier.

    Parameters:
        churn_probability: Calibrated probability P(Churn) in [0.0, 1.0].
        monthly_charges: Current monthly billing amount in USD.
        contract_type: "Month-to-month", "One year", or "Two year".
        tenure_months: Total months of active tenancy.

    Returns:
        Dict containing:
            - risk_category: "HIGH", "MEDIUM", or "LOW"
            - revenue_at_risk: Expected financial loss in USD
            - annual_revenue_exposure: Total 12-month value at stake
            - risk_multiplier_factors: Explanation of horizon weighting
    """
    prob = max(0.0, min(1.0, float(churn_probability)))
    m_charges = max(0.0, float(monthly_charges))

    # Determine projected revenue horizon (months) based on contract commitment:
    # Month-to-month: High near-term defection window (6 months expected lifetime lost)
    # One year: 12 months annual contract value
    # Two year: 24 months multi-year contract value
    contract_horizon_map = {
        "Month-to-month": 6,
        "One year": 12,
        "Two year": 24,
    }
    horizon_months = contract_horizon_map.get(contract_type, 6)

    # Calculate Total Monetary Exposure over the decision horizon
    total_exposure = m_charges * horizon_months

    # Expected Value Formula: EV(Loss) = P(Churn) * Exposure
    revenue_at_risk = round(prob * total_exposure, 2)
    annual_revenue_exposure = round(m_charges * 12, 2)

    # 2D Tiering Matrix (Probability x Billing Volume)
    # Rationale: A $110/mo customer with 40% churn risk represents higher actual dollar risk
    # than a $20/mo customer with 70% churn risk.
    if prob >= 0.60 or (prob >= 0.40 and m_charges >= 75.0):
        risk_category = "HIGH"
    elif prob >= 0.30 or (prob >= 0.20 and m_charges >= 85.0):
        risk_category = "MEDIUM"
    else:
        risk_category = "LOW"

    return {
        "risk_category": risk_category,
        "revenue_at_risk": revenue_at_risk,
        "monthly_charges": round(m_charges, 2),
        "horizon_months": horizon_months,
        "annual_revenue_exposure": annual_revenue_exposure,
        "churn_probability": round(prob, 4),
    }
