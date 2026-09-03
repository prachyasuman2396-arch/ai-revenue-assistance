from typing import Any, Dict, List


def generate_recommendations(
    customer_data: Dict[str, Any],
    risk_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic, business-rules recommendation engine that outputs prescriptive

    retention playbooks tailored to customer friction points and financial risk.

    Parameters:
        customer_data: Customer account features (contract, internet_service, etc.).
        risk_info: Output from Revenue Risk Engine (risk_category, revenue_at_risk, churn_probability).

    Returns:
        Dict containing:
            - primary_action: Core prescriptive retention step
            - playbook_code: Machine-readable action code
            - priority: "P1_URGENT", "P2_ELEVATED", or "P3_STANDARD"
            - secondary_actions: List of complementary tactics
            - estimated_roi_impact: Business value projection
    """
    risk_category = risk_info.get("risk_category", "LOW")
    revenue_at_risk = risk_info.get("revenue_at_risk", 0.0)
    churn_prob = risk_info.get("churn_probability", 0.0)

    contract = customer_data.get("contract", "Month-to-month")
    internet = customer_data.get("internet_service", "DSL")
    tech_support = customer_data.get("tech_support", "No")
    online_sec = customer_data.get("online_security", "No")
    payment_method = customer_data.get("payment_method", "Electronic check")
    tenure = int(customer_data.get("tenure", 12))
    monthly_charges = float(customer_data.get("monthly_charges", 50.0))

    secondary_actions: List[str] = []

    # Priority 1: High Risk Accounts
    if risk_category == "HIGH":
        priority = "P1_URGENT"

        if tenure <= 6:
            primary_action = (
                "Deploy Early-Life Concierge Call: Contact subscriber within 48 hours "
                "to resolve onboarding setup friction and provide a $20 goodwill bill credit."
            )
            playbook_code = "PLAYBOOK_ONBOARDING_CONCIERGE"
        elif contract == "Month-to-month" and monthly_charges >= 75.0:
            primary_action = (
                "Executive Retention Offer: Offer 15% discount on an annual contract lock-in, "
                "protecting high monthly recurring revenue from immediate defection."
            )
            playbook_code = "PLAYBOOK_ANNUAL_CONTRACT_MIGRATION"
        else:
            primary_action = (
                "Priority Retention Specialist Outreach: Direct phone call offering "
                "tailored rate review and service bundle adjustment."
            )
            playbook_code = "PLAYBOOK_SPECIALIST_INTERVENTION"

    # Priority 2: Medium Risk Accounts
    elif risk_category == "MEDIUM":
        priority = "P2_ELEVATED"

        if internet == "Fiber optic" and (tech_support != "Yes" or online_sec != "Yes"):
            primary_action = (
                "Ecosystem Protection Bundle: Offer complimentary TechSupport and "
                "OnlineSecurity for 6 months to improve fiber network satisfaction."
            )
            playbook_code = "PLAYBOOK_TECH_SUPPORT_BUNDLE"
        elif payment_method == "Electronic check":
            primary_action = (
                "Autopay Migration Incentive: Offer a one-time $15 bill credit "
                "to switch from manual Electronic Check to automated credit card/bank debit."
            )
            playbook_code = "PLAYBOOK_AUTOPAY_INCENTIVE"
        elif contract == "Month-to-month":
            primary_action = (
                "Annual Agreement Upgrade: Present in-app promotion offering free speed "
                "upgrade in exchange for 1-year contract extension."
            )
            playbook_code = "PLAYBOOK_CONTRACT_UPGRADE"
        else:
            primary_action = "Targeted Digital Engagement: Send personalized loyalty satisfaction survey with $10 reward."
            playbook_code = "PLAYBOOK_DIGITAL_NUDGE"

    # Priority 3: Low Risk Accounts
    else:
        priority = "P3_STANDARD"
        primary_action = (
            "Standard Customer Success Cadence: Account is stable. Eligible for "
            "quarterly loyalty perk and cross-sell of value-add entertainment tiers."
        )
        playbook_code = "PLAYBOOK_LOYALTY_CADENCE"

    # Secondary Action Rules
    if (
        payment_method == "Electronic check"
        and playbook_code != "PLAYBOOK_AUTOPAY_INCENTIVE"
    ):
        secondary_actions.append(
            "Prompt user to enroll in automated bank transfer for $10 autopay credit."
        )

    if (
        internet == "Fiber optic"
        and tech_support != "Yes"
        and playbook_code != "PLAYBOOK_TECH_SUPPORT_BUNDLE"
    ):
        secondary_actions.append("Suggest discounted premium 24/7 TechSupport add-on.")

    if contract == "Month-to-month" and playbook_code not in [
        "PLAYBOOK_ANNUAL_CONTRACT_MIGRATION",
        "PLAYBOOK_CONTRACT_UPGRADE",
    ]:
        secondary_actions.append(
            "Present annual commitment discount banner on next billing invoice."
        )

    # Business Value Projection
    if risk_category in ("HIGH", "MEDIUM"):
        estimated_roi = (
            f"Preserves up to ${revenue_at_risk:.2f} at-risk revenue by intercepting a "
            f"{churn_prob:.1%} flight probability."
        )
    else:
        estimated_roi = "Maintain account LTV; account currently represents minimal revenue flight risk."

    return {
        "primary_action": primary_action,
        "playbook_code": playbook_code,
        "priority": priority,
        "secondary_actions": secondary_actions,
        "estimated_roi_impact": estimated_roi,
    }
