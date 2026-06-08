"""
Revenue Risk Engine
Implements CLV, Revenue At Risk, Risk Banding, Revenue Risk Score from the notebook.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

from ..config import settings
from ..services.feature_engineering import engineer_features_with_revenue

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Single-customer helpers
# ──────────────────────────────────────────────

def calculate_clv(monthly_charges: float, tenure: int) -> float:
    """Customer Lifetime Value = MonthlyCharges * tenure (from notebook)"""
    return round(monthly_charges * tenure, 2)


def calculate_avg_spend(total_charges: float, tenure: int) -> float:
    """AvgSpend = TotalCharges / (tenure + 1)"""
    return round(total_charges / (tenure + 1), 2)


def calculate_revenue_exposure(monthly_charges: float, tenure: int) -> float:
    """RevenueExposure = MonthlyCharges * log1p(tenure)"""
    return round(monthly_charges * np.log1p(tenure), 2)


def calculate_revenue_at_risk(churn_probability: float, clv: float) -> float:
    """RevenueAtRisk = Churn_Probability * CLV"""
    return round(churn_probability * clv, 2)


def calculate_risk_band(churn_probability: float) -> str:
    """
    Risk band based on churn probability (from notebook risk_band function).
    Critical: >= 0.8
    High:     >= 0.6
    Medium:   >= 0.4
    Low:      < 0.4
    """
    if churn_probability >= settings.RISK_THRESHOLD_CRITICAL:
        return "Critical"
    elif churn_probability >= settings.RISK_THRESHOLD_HIGH:
        return "High"
    elif churn_probability >= settings.RISK_THRESHOLD_MEDIUM:
        return "Medium"
    else:
        return "Low"


def calculate_risk_category_from_score(score: float) -> str:
    """
    Risk category based on RevenueRiskScore (0-100) (from notebook risk_category function).
    """
    if score >= settings.SCORE_THRESHOLD_CRITICAL:
        return "Critical"
    elif score >= settings.SCORE_THRESHOLD_HIGH:
        return "High"
    elif score >= settings.SCORE_THRESHOLD_MEDIUM:
        return "Medium"
    else:
        return "Low"


def calculate_revenue_risk_score(revenue_at_risk: float, max_revenue_at_risk: float = None) -> float:
    """
    RevenueRiskScore = (RevenueAtRisk / max_RevenueAtRisk) * 100 (MinMax scaled from notebook).
    For single-customer: use a fixed reference max from dataset statistics.
    """
    if max_revenue_at_risk is None:
        # Typical max CLV in Telco dataset (72 months * ~$100/mo ~ $7200)
        max_revenue_at_risk = 7200.0

    score = (revenue_at_risk / max_revenue_at_risk) * 100
    return round(min(score, 100.0), 1)


def calculate_revenue_metrics(customer_dict: Dict[str, Any], churn_probability: float) -> Dict[str, Any]:
    """
    Compute all revenue risk metrics for a single customer.
    """
    monthly_charges = float(customer_dict.get("MonthlyCharges", 0))
    tenure = int(customer_dict.get("tenure", 0))
    total_charges = float(customer_dict.get("TotalCharges") or monthly_charges * max(tenure, 1))

    clv = calculate_clv(monthly_charges, tenure)
    avg_spend = calculate_avg_spend(total_charges, tenure)
    revenue_exposure = calculate_revenue_exposure(monthly_charges, tenure)
    revenue_at_risk = calculate_revenue_at_risk(churn_probability, clv)
    revenue_risk_score = calculate_revenue_risk_score(revenue_at_risk)
    risk_band = calculate_risk_band(churn_probability)
    risk_category = calculate_risk_category_from_score(revenue_risk_score)

    return {
        "customer_lifetime_value": clv,
        "revenue_at_risk": revenue_at_risk,
        "revenue_risk_score": revenue_risk_score,
        "avg_spend": avg_spend,
        "revenue_exposure": revenue_exposure,
        "risk_band": risk_band,
        "risk_category": risk_category,
    }


# ──────────────────────────────────────────────
# Basic recommendation (from notebook)
# ──────────────────────────────────────────────

def basic_recommendation(risk_category: str) -> str:
    if risk_category == "Critical":
        return "Immediate retention campaign"
    elif risk_category == "High":
        return "Offer discount"
    elif risk_category == "Medium":
        return "Monitor customer"
    else:
        return "No action"


# ──────────────────────────────────────────────
# Batch dashboard calculations
# ──────────────────────────────────────────────

def calculate_dashboard_metrics(customers: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
    """
    Compute portfolio-level revenue risk metrics for dashboard.
    Uses MinMaxScaler across the batch for RevenueRiskScore (matches notebook).
    """
    rows = []
    for i, (cust, pred) in enumerate(zip(customers, predictions)):
        monthly = float(cust.get("MonthlyCharges", 0))
        tenure = int(cust.get("tenure", 0))
        total = float(cust.get("TotalCharges") or monthly * max(tenure, 1))
        prob = float(pred["churn_probability"])

        clv = monthly * tenure
        revenue_at_risk = prob * clv
        risk_band = calculate_risk_band(prob)

        rows.append({
            "customer_index": i,
            "churn_probability": round(prob, 4),
            "risk_band": risk_band,
            "clv": round(clv, 2),
            "revenue_at_risk": round(revenue_at_risk, 2),
        })

    df = pd.DataFrame(rows)

    # MinMaxScale RevenueAtRisk to get RevenueRiskScore (as per notebook)
    max_rar = df["revenue_at_risk"].max()
    min_rar = df["revenue_at_risk"].min()
    if max_rar > min_rar:
        df["revenue_risk_score"] = ((df["revenue_at_risk"] - min_rar) / (max_rar - min_rar) * 100).round(1)
    else:
        df["revenue_risk_score"] = 0.0

    df["risk_category"] = df["revenue_risk_score"].apply(calculate_risk_category_from_score)
    df["recommendation"] = df["risk_category"].apply(basic_recommendation)

    # Risk distribution
    risk_dist = df["risk_band"].value_counts().to_dict()

    # Top 10 risk customers
    top_risk = df.nlargest(10, "revenue_risk_score").to_dict(orient="records")

    # Revenue recovery opportunity: top 10% of customers by RevenueAtRisk
    top_10_pct = max(1, int(len(df) * 0.1))
    recovery_opportunity = df.nlargest(top_10_pct, "revenue_at_risk")["revenue_at_risk"].sum()

    return {
        "total_customers": len(df),
        "avg_churn_probability": round(df["churn_probability"].mean(), 4),
        "total_revenue_at_risk": round(df["revenue_at_risk"].sum(), 2),
        "avg_revenue_at_risk": round(df["revenue_at_risk"].mean(), 2),
        "risk_distribution": risk_dist,
        "top_risk_customers": top_risk,
        "revenue_recovery_opportunity": round(recovery_opportunity, 2),
    }
