"""
Groq AI Retention Recommendation Service
Uses llama-3.3-70b-versatile to generate personalized retention strategies.
"""

import json
import time
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _build_prompt(
    customer_profile: Dict[str, Any],
    prediction: Dict[str, Any],
    revenue_metrics: Dict[str, Any],
    shap_factors: List[Dict[str, Any]],
) -> str:
    factors_str = "\n".join(
        [f"  - {f['feature']}: SHAP impact = {f['impact']:.4f}" for f in shap_factors[:5]]
    )

    return f"""You are a senior customer retention analyst at a telecom company.

CUSTOMER PROFILE:
{json.dumps(customer_profile, indent=2)}

CHURN PREDICTION:
- Churn Probability: {prediction['churn_probability']:.1%}
- Risk Band: {prediction.get('risk_band', 'N/A')}
- Confidence: {prediction.get('confidence', 'N/A')}

REVENUE METRICS:
- Customer Lifetime Value: ${revenue_metrics['customer_lifetime_value']:,.2f}
- Revenue At Risk: ${revenue_metrics['revenue_at_risk']:,.2f}
- Revenue Risk Score: {revenue_metrics['revenue_risk_score']}/100

TOP RISK FACTORS (SHAP):
{factors_str}

Your task:
1. Explain the top 3 reasons this customer is likely to churn based on their profile and SHAP factors.
2. Estimate the business impact if this customer churns.
3. Generate exactly 4 personalized retention actions, ranked by expected impact (highest first).
4. Write a 2-sentence executive summary.

Respond ONLY with a valid JSON object in this exact format, no markdown, no preamble:
{{
  "churn_reasons": ["reason1", "reason2", "reason3"],
  "business_impact": "string describing dollar impact and risk",
  "retention_actions": [
    {{"priority": 1, "action": "...", "rationale": "...", "expected_impact": "..."}},
    {{"priority": 2, "action": "...", "rationale": "...", "expected_impact": "..."}},
    {{"priority": 3, "action": "...", "rationale": "...", "expected_impact": "..."}},
    {{"priority": 4, "action": "...", "rationale": "...", "expected_impact": "..."}}
  ],
  "executive_summary": "string"
}}"""


def _parse_groq_response(content: str) -> Dict[str, Any]:
    """Robustly parse JSON from Groq response, handling markdown fences."""
    # Strip markdown code fences if present
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON object with regex
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from response: {content[:200]}")


def get_retention_recommendation(
    customer_profile: Dict[str, Any],
    prediction: Dict[str, Any],
    revenue_metrics: Dict[str, Any],
    shap_factors: List[Dict[str, Any]],
    groq_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Groq API to generate AI-powered retention recommendations.
    Falls back to rule-based recommendations if Groq is unavailable.
    """
    from ..config import settings

    api_key = groq_api_key or settings.GROQ_API_KEY

    if not api_key:
        logger.warning("GROQ_API_KEY not set. Using rule-based fallback.")
        return _rule_based_fallback(customer_profile, prediction, revenue_metrics, shap_factors)

    prompt = _build_prompt(customer_profile, prediction, revenue_metrics, shap_factors)

    for attempt in range(1, settings.GROQ_MAX_RETRIES + 1):
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": settings.GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": settings.GROQ_MAX_TOKENS,
                "temperature": 0.3,
            }

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = _parse_groq_response(content)

            return {
                "retention_strategy": parsed.get("retention_actions", []),
                "churn_reasons": parsed.get("churn_reasons", []),
                "business_impact": parsed.get("business_impact", ""),
                "generated_summary": parsed.get("executive_summary", ""),
                "source": "groq",
            }

        except Exception as e:
            logger.warning(f"Groq attempt {attempt} failed: {e}")
            if attempt < settings.GROQ_MAX_RETRIES:
                time.sleep(2 ** attempt)

    logger.error("All Groq retries exhausted. Using rule-based fallback.")
    return _rule_based_fallback(customer_profile, prediction, revenue_metrics, shap_factors)


def _rule_based_fallback(
    customer_profile: Dict[str, Any],
    prediction: Dict[str, Any],
    revenue_metrics: Dict[str, Any],
    shap_factors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Rule-based retention recommendations when Groq is unavailable."""
    prob = prediction["churn_probability"]
    clv = revenue_metrics["customer_lifetime_value"]
    rar = revenue_metrics["revenue_at_risk"]
    risk_band = prediction.get("risk_band", "Medium")

    contract = customer_profile.get("Contract", "Month-to-month")
    payment = customer_profile.get("PaymentMethod", "")
    internet = customer_profile.get("InternetService", "")
    tenure = customer_profile.get("tenure", 0)

    actions = []

    if contract == "Month-to-month":
        actions.append({
            "priority": 1,
            "action": "Offer annual contract upgrade with 15% discount",
            "rationale": "Month-to-month customers churn at significantly higher rates. Annual contracts reduce churn by ~40%.",
            "expected_impact": "Estimated 30-40% churn risk reduction",
        })

    if "automatic" not in payment.lower():
        actions.append({
            "priority": 2,
            "action": "Incentivize switch to automatic payment (bank transfer or credit card)",
            "rationale": "Auto-payment customers are stickier and have lower churn rates.",
            "expected_impact": "Estimated 10-15% churn risk reduction",
        })

    if internet == "Fiber optic":
        actions.append({
            "priority": 3,
            "action": "Bundle OnlineSecurity + TechSupport at promotional rate",
            "rationale": "Fiber customers without security services churn at high rates. Value-added bundles increase stickiness.",
            "expected_impact": "Estimated 15-20% churn risk reduction",
        })

    if tenure <= 12:
        actions.append({
            "priority": 4,
            "action": "Assign dedicated onboarding success manager for first 90 days",
            "rationale": "New customers are at highest risk. Proactive engagement in early tenure reduces churn by ~25%.",
            "expected_impact": "Estimated 20-25% churn risk reduction",
        })

    # Ensure we always have 4 actions
    generic_actions = [
        {
            "priority": len(actions) + 1,
            "action": "Personalised outreach call from retention specialist",
            "rationale": "Direct human engagement is highly effective for at-risk customers.",
            "expected_impact": "Estimated 10-20% churn risk reduction",
        },
        {
            "priority": len(actions) + 2,
            "action": "Offer loyalty reward: one month free or service upgrade",
            "rationale": "Immediate tangible value can re-engage dissatisfied customers.",
            "expected_impact": "Estimated 5-15% churn risk reduction",
        },
    ]

    for ga in generic_actions:
        if len(actions) >= 4:
            break
        ga["priority"] = len(actions) + 1
        actions.append(ga)

    actions = actions[:4]

    summary = (
        f"Customer shows a {prob:.1%} churn probability with ${rar:,.2f} revenue at risk. "
        f"Immediate retention intervention is {'critical' if risk_band == 'Critical' else 'recommended'} "
        f"to protect ${clv:,.2f} in lifetime value."
    )

    return {
        "retention_strategy": actions,
        "churn_reasons": [
            f"High churn probability of {prob:.1%}",
            f"Contract type: {contract}",
            f"Payment method: {payment}",
        ],
        "business_impact": f"Revenue at risk: ${rar:,.2f}. CLV: ${clv:,.2f}.",
        "generated_summary": summary,
        "source": "rule_based",
    }
