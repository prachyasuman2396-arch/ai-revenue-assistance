"""
FastAPI API Routes
All business logic delegated to services.
"""

import time
import logging
from fastapi import APIRouter, HTTPException
from typing import List

from ..schemas import (
    CustomerInput,
    FullAnalysisRequest,
    FullAnalysisResponse,
    DashboardRequest,
    DashboardResponse,
    HealthResponse,
    PredictionResult,
    ExplainabilityResult,
    RevenueMetrics,
    RetentionAction,
    RiskFactor,
    DashboardCustomerRow,
)
from ..services.predictor import predict_single, predict_batch
from ..services.explainer import explain_prediction
from ..services.revenue_risk import calculate_revenue_metrics, calculate_dashboard_metrics
from ..services.groq_recommendation import get_retention_recommendation
from ..utils.model_loader import is_model_loaded, load_model
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ──────────────────────────────────────────────
# Health / Root
# ──────────────────────────────────────────────

@router.get("/", tags=["System"])
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    try:
        load_model(settings.MODEL_PATH)
        model_ok = True
    except Exception:
        model_ok = False

    return HealthResponse(
        status="healthy" if model_ok else "degraded",
        model_loaded=model_ok,
        version=settings.APP_VERSION,
    )


# ──────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(customer: CustomerInput):
    """Predict churn probability for a single customer."""
    try:
        result = predict_single(customer.dict())
        return PredictionResult(
            churn_probability=result["churn_probability"],
            churn_prediction=result["churn_prediction"],
            confidence=result["confidence"],
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Explain
# ──────────────────────────────────────────────

@router.post("/explain", response_model=ExplainabilityResult, tags=["Explainability"])
def explain(customer: CustomerInput):
    """Return SHAP-based top risk factors for a customer."""
    try:
        result = explain_prediction(customer.dict())
        factors = [RiskFactor(**f) for f in result["top_risk_factors"]]
        return ExplainabilityResult(top_risk_factors=factors)
    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Recommend
# ──────────────────────────────────────────────

@router.post("/recommend", tags=["Retention"])
def recommend(request: FullAnalysisRequest):
    """Generate AI-powered retention recommendations."""
    try:
        customer_dict = request.customer.dict()
        pred = predict_single(customer_dict)
        revenue = calculate_revenue_metrics(customer_dict, pred["churn_probability"])
        shap_result = explain_prediction(customer_dict)

        pred_with_band = {**pred, "risk_band": revenue["risk_band"]}
        groq_result = get_retention_recommendation(
            customer_profile=customer_dict,
            prediction=pred_with_band,
            revenue_metrics=revenue,
            shap_factors=shap_result["top_risk_factors"],
        )

        return {
            "customer_id": request.customer_id,
            "retention_strategy": groq_result["retention_strategy"],
            "churn_reasons": groq_result["churn_reasons"],
            "business_impact": groq_result["business_impact"],
            "generated_summary": groq_result["generated_summary"],
            "source": groq_result["source"],
        }
    except Exception as e:
        logger.error(f"Recommend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────

@router.post("/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
def dashboard(request: DashboardRequest):
    """Portfolio-level revenue risk dashboard for a batch of customers."""
    try:
        customers = [c.dict() for c in request.customers]
        predictions = predict_batch(customers)
        metrics = calculate_dashboard_metrics(customers, predictions)

        top_risk = [DashboardCustomerRow(**row) for row in metrics["top_risk_customers"]]

        return DashboardResponse(
            total_customers=metrics["total_customers"],
            avg_churn_probability=metrics["avg_churn_probability"],
            total_revenue_at_risk=metrics["total_revenue_at_risk"],
            avg_revenue_at_risk=metrics["avg_revenue_at_risk"],
            risk_distribution=metrics["risk_distribution"],
            top_risk_customers=top_risk,
            revenue_recovery_opportunity=metrics["revenue_recovery_opportunity"],
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Full Analysis
# ──────────────────────────────────────────────

@router.post("/full-analysis", response_model=FullAnalysisResponse, tags=["Analysis"])
def full_analysis(request: FullAnalysisRequest):
    """
    Complete end-to-end analysis:
    Prediction + SHAP + Revenue Metrics + AI Retention Strategy
    """
    start = time.perf_counter()
    try:
        customer_dict = request.customer.dict()

        # 1. Prediction
        pred = predict_single(customer_dict)

        # 2. SHAP Explainability
        shap_result = explain_prediction(customer_dict)

        # 3. Revenue Metrics
        revenue = calculate_revenue_metrics(customer_dict, pred["churn_probability"])

        # 4. AI Retention Strategy
        pred_with_band = {**pred, "risk_band": revenue["risk_band"]}
        groq_result = get_retention_recommendation(
            customer_profile=customer_dict,
            prediction=pred_with_band,
            revenue_metrics=revenue,
            shap_factors=shap_result["top_risk_factors"],
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return FullAnalysisResponse(
            customer_id=request.customer_id,
            prediction=PredictionResult(
                churn_probability=pred["churn_probability"],
                churn_prediction=pred["churn_prediction"],
                confidence=pred["confidence"],
            ),
            explainability=ExplainabilityResult(
                top_risk_factors=[RiskFactor(**f) for f in shap_result["top_risk_factors"]]
            ),
            revenue_metrics=RevenueMetrics(
                customer_lifetime_value=revenue["customer_lifetime_value"],
                revenue_at_risk=revenue["revenue_at_risk"],
                revenue_risk_score=revenue["revenue_risk_score"],
                avg_spend=revenue["avg_spend"],
                revenue_exposure=revenue["revenue_exposure"],
            ),
            risk_band=revenue["risk_band"],
            retention_strategy=[RetentionAction(**a) for a in groq_result["retention_strategy"]],
            generated_summary=groq_result["generated_summary"],
            inference_time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
