from unittest.mock import MagicMock, patch

from backend.app.schemas.prediction import PersonalizedOutreach
from backend.app.services.llm_retention import GroqRetentionService, retention_service


@patch("backend.app.services.llm_retention.settings.groq_api_key", None)
def test_retention_service_deterministic_fallback():
    service = GroqRetentionService()
    service._llm = None

    customer_data = {
        "customer_id": "TEST_CUST_100",
        "tenure": 5,
        "contract": "Month-to-month",
        "monthly_charges": 65.0,
        "payment_method": "Electronic check",
    }
    risk_info = {
        "risk_category": "HIGH",
        "revenue_at_risk": 250.0,
        "churn_probability": 0.64,
    }
    rec_info = {"primary_action": "Issue $25 courtesy loyalty credit"}
    exp_info = {"summary": "Month-to-month contract and low tenure"}

    outreach = service.generate_outreach(customer_data, risk_info, rec_info, exp_info)

    assert outreach is not None
    assert "retention_email_subject" in outreach
    assert "retention_email_body" in outreach
    assert "call_script_talking_points" in outreach
    assert "counter_objection_strategy" in outreach
    assert len(outreach["call_script_talking_points"]) >= 3
    assert "TEST_CUST_100" in outreach["retention_email_subject"] or "TEST_CUST_100" in outreach["retention_email_body"]


def test_retention_service_mocked_groq_success():
    service = GroqRetentionService()

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    expected_outreach = PersonalizedOutreach(
        retention_email_subject="Exclusive 20% discount for our valued member",
        retention_email_body="Dear Customer, we value your loyalty. Here is a 20% discount.",
        call_script_talking_points=["Acknowledge loyalty", "Offer price lock guarantee"],
        counter_objection_strategy="Highlight lifetime savings over competitor plans",
    )

    # When prompt | structured_llm is invoked, it invokes the chain
    mock_structured_llm.invoke.return_value = expected_outreach
    service._llm = mock_llm

    customer_data = {
        "customer_id": "VIP_CUST_200",
        "tenure": 24,
        "contract": "One year",
        "monthly_charges": 95.0,
    }
    risk_info = {
        "risk_category": "MEDIUM",
        "revenue_at_risk": 380.0,
        "churn_probability": 0.40,
    }
    rec_info = {"primary_action": "Renew with 20% discount"}
    exp_info = {"summary": "Pricing sensitivity"}

    with patch("backend.app.services.llm_retention.ChatPromptTemplate.from_messages") as mock_prompt:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_outreach
        mock_prompt.return_value.__or__.return_value = mock_chain

        outreach = service.generate_outreach(customer_data, risk_info, rec_info, exp_info)

        assert outreach is not None
        assert outreach["retention_email_subject"] == "Exclusive 20% discount for our valued member"
        assert len(outreach["call_script_talking_points"]) == 2
