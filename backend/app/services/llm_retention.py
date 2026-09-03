import logging
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from backend.app.config.settings import settings
from backend.app.schemas.prediction import PersonalizedOutreach

logger = logging.getLogger(__name__)


class GroqRetentionService:
    """Retention intelligence service powered by Groq and LangChain.

    Synthesizes ML churn probabilities, financial risk exposure, and SHAP
    explainability drivers into hyper-personalized customer outreach strategies.
    """

    def __init__(self):
        self._llm = None
        self._active_model_name = settings.groq_model_name
        self._init_llm()

    def _init_llm(self, model_name: Optional[str] = None):
        api_key = settings.groq_api_key
        if not api_key or not api_key.strip():
            logger.info("Groq API key not configured. LLM retention outreach running in fallback mode.")
            self._llm = None
            return

        target_model = model_name or self._active_model_name
        try:
            self._llm = ChatGroq(
                model=target_model,
                groq_api_key=api_key.strip(),
                temperature=0.3,
                max_retries=1,
                timeout=12.0,
            )
            self._active_model_name = target_model
            logger.info(f"Initialized ChatGroq with model '{target_model}'.")
        except Exception as e:
            logger.warning(f"Could not initialize ChatGroq for {target_model}: {e}")
            self._llm = None

    def generate_outreach(
        self,
        customer_data: Dict[str, Any],
        risk_info: Dict[str, Any],
        recommendation_info: Dict[str, Any],
        explanation_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generates tailored retention email, call talking points, and objection handling."""
        # Check if Groq is available
        if self._llm is None:
            self._init_llm()

        if self._llm is None:
            return self._build_deterministic_fallback(
                customer_data, risk_info, recommendation_info, explanation_info
            )

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an executive customer retention strategist for a major subscription telecom service. "
                        "Your mission is to formulate personalized, empathetic, high-converting retention outreach "
                        "grounded in machine learning risk drivers, financial exposure, and SHAP explainability insights. "
                        "Never sound generic. Speak directly to the customer's exact pain points and tenure status.",
                    ),
                    (
                        "human",
                        "Generate a personalized retention outreach plan for this customer:\n\n"
                        "Customer Profile:\n"
                        "- Customer ID: {customer_id}\n"
                        "- Tenure: {tenure} months\n"
                        "- Contract: {contract}\n"
                        "- Monthly Charges: ${monthly_charges:.2f}\n"
                        "- Payment Method: {payment_method}\n"
                        "- Internet Service: {internet_service}\n\n"
                        "ML Model Assessment:\n"
                        "- Churn Probability: {churn_prob:.1%}\n"
                        "- Risk Category: {risk_category}\n"
                        "- Revenue At Risk: ${revenue_at_risk:.2f}\n"
                        "- Primary Playbook Action: {primary_action}\n\n"
                        "SHAP Root Causes:\n"
                        "- Key Risk Drivers: {risk_summary}\n\n"
                        "Provide a structured JSON output with:\n"
                        "1. A compelling email subject line\n"
                        "2. An empathetic, value-focused email body offering the playbook concession\n"
                        "3. 3-4 concrete phone talking points for the account representative\n"
                        "4. A proactive counter-objection strategy addressing their primary risk driver",
                    ),
                ]
            )

            candidate_models = [
                self._active_model_name,
                "openai/gpt-oss-120b",
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-20b",
            ]
            seen = set()
            ordered_candidates = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

            cust_id = customer_data.get("customer_id", "Customer")
            tenure = int(customer_data.get("tenure", 1))
            contract = str(customer_data.get("contract", "Month-to-month"))
            monthly_charges = float(customer_data.get("monthly_charges", 50.0))
            payment_method = str(customer_data.get("payment_method", "Electronic check"))
            internet_service = str(customer_data.get("internet_service", "DSL"))

            churn_prob = float(risk_info.get("churn_probability", 0.5))
            risk_cat = str(risk_info.get("risk_category", "HIGH"))
            rev_at_risk = float(risk_info.get("revenue_at_risk", 0.0))
            primary_action = str(recommendation_info.get("primary_action", "Retention Review"))
            risk_summary = str(explanation_info.get("summary", "Account tenancy and contract flexibility."))

            payload = {
                "customer_id": cust_id,
                "tenure": tenure,
                "contract": contract,
                "monthly_charges": monthly_charges,
                "payment_method": payment_method,
                "internet_service": internet_service,
                "churn_prob": churn_prob,
                "risk_category": risk_cat,
                "revenue_at_risk": rev_at_risk,
                "primary_action": primary_action,
                "risk_summary": risk_summary,
            }

            for model_name in ordered_candidates:
                try:
                    if self._llm is None or self._active_model_name != model_name:
                        self._init_llm(model_name)
                    if self._llm is None:
                        continue

                    structured_llm = self._llm.with_structured_output(PersonalizedOutreach)
                    chain = prompt | structured_llm
                    result = chain.invoke(payload)

                    if isinstance(result, PersonalizedOutreach):
                        return result.model_dump()
                    elif isinstance(result, dict):
                        return result
                except Exception as model_err:
                    logger.warning(f"Groq model '{model_name}' failed ({model_err}). Trying next candidate...")
                    continue

            return self._build_deterministic_fallback(
                customer_data, risk_info, recommendation_info, explanation_info
            )

        except Exception as e:
            logger.warning(
                f"Groq retention generation failed ({e}). Reverting to playbook fallback."
            )
            return self._build_deterministic_fallback(
                customer_data, risk_info, recommendation_info, explanation_info
            )

    def _build_deterministic_fallback(
        self,
        customer_data: Dict[str, Any],
        risk_info: Dict[str, Any],
        recommendation_info: Dict[str, Any],
        explanation_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rule-based fallback when Groq API is offline or unconfigured."""
        cust_id = customer_data.get("customer_id", "Valued Subscriber")
        contract = customer_data.get("contract", "Month-to-month")
        tenure = customer_data.get("tenure", 12)
        primary_action = recommendation_info.get(
            "primary_action", "Personalized account consultation."
        )

        return {
            "retention_email_subject": f"An exclusive loyalty update for your account ({cust_id})",
            "retention_email_body": (
                f"Dear {cust_id},\n\n"
                f"We noticed you have been with us for {tenure} months under your {contract} plan. "
                "We want to ensure your experience exceeds expectations. "
                f"As a token of our appreciation, we would like to offer: {primary_action}\n\n"
                "Please reply to this email or speak with your dedicated specialist to activate this offer.\n\n"
                "Warm regards,\nYour VIP Customer Care Team"
            ),
            "call_script_talking_points": [
                f"Acknowledge {tenure} months of account tenancy with genuine gratitude.",
                f"Address primary contract flexibility friction ({contract}) by offering annual stability perks.",
                f"Present the core retention concession: {primary_action}",
                "Inquire about network performance and offer complimentary speed optimization check.",
            ],
            "counter_objection_strategy": (
                f"If customer cites pricing or contract flexibility, pivot to: {primary_action} "
                "paired with zero penalty commitments."
            ),
        }


# Singleton instance
retention_service = GroqRetentionService()
