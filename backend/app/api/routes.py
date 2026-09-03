from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.models import (
    CustomerPrediction,
    CustomerRecommendation,
    RawCustomerData,
)
from backend.app.schemas.customer import CustomerResponse
from backend.app.schemas.prediction import PredictionRequest, PredictionResponse
from backend.app.schemas.recommendation import RecommendationResponse
from backend.app.services.predictor import predictor

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service liveness and readiness probe."""
    return {
        "status": "healthy",
        "model_name": predictor.model_name,
        "model_version": predictor.model_version,
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_churn(
    request: PredictionRequest,
    db: Session = Depends(get_db),
):
    """Generates real-time churn prediction, financial risk analysis,

    and retention recommendation from payload telemetry.
    """
    try:
        customer_dict = request.model_dump()
        result = predictor.predict_and_assess(
            customer_data=customer_dict,
            db=db,
            save_audit=False,  # Payloads may be prospective or anonymous
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        ) from e


@router.post(
    "/predict/{customer_id}",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_for_customer(
    customer_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves existing customer from PostgreSQL, evaluates churn probability,

    computes revenue risk, generates recommendation, and persists audit logs.
    """
    customer = (
        db.query(RawCustomerData)
        .filter(RawCustomerData.customer_id == customer_id)
        .first()
    )
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer ID '{customer_id}' not found in database.",
        )

    # Convert ORM row to dictionary
    customer_dict = {
        c.name: getattr(customer, c.name) for c in customer.__table__.columns
    }
    # Handle total_charges type coercion
    try:
        raw_tc = str(customer_dict.get("total_charges", "")).strip()
        customer_dict["total_charges"] = float(raw_tc) if raw_tc else 0.0
    except ValueError:
        customer_dict["total_charges"] = 0.0

    customer_dict["monthly_charges"] = float(customer_dict.get("monthly_charges", 0.0))
    customer_dict["tenure"] = int(customer_dict.get("tenure", 0))

    result = predictor.predict_and_assess(
        customer_data=customer_dict,
        db=db,
        save_audit=True,
    )
    return result


@router.get(
    "/customers",
    response_model=List[CustomerResponse],
    status_code=status.HTTP_200_OK,
)
def list_customers(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Paginated retrieval of raw customer accounts."""
    customers = db.query(RawCustomerData).offset(skip).limit(limit).all()
    return customers


@router.get(
    "/customers/{customer_id}",
    response_model=CustomerResponse,
    status_code=status.HTTP_200_OK,
)
def get_customer(
    customer_id: str,
    db: Session = Depends(get_db),
):
    """Fetches customer profile by ID."""
    customer = (
        db.query(RawCustomerData)
        .filter(RawCustomerData.customer_id == customer_id)
        .first()
    )
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found.",
        )
    return customer


@router.get(
    "/predictions/{customer_id}",
    status_code=status.HTTP_200_OK,
)
def get_customer_predictions(
    customer_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves historical prediction audit log for a given customer."""
    preds = (
        db.query(CustomerPrediction)
        .filter(CustomerPrediction.customer_id == customer_id)
        .order_by(CustomerPrediction.created_at.desc())
        .all()
    )
    if not preds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No predictions logged for customer '{customer_id}'.",
        )
    return [
        {
            "id": p.id,
            "customer_id": p.customer_id,
            "churn_probability": float(p.churn_probability),
            "prediction": p.prediction,
            "model_name": p.model_name,
            "model_version": p.model_version,
            "created_at": p.created_at,
        }
        for p in preds
    ]


@router.get(
    "/recommendations/{customer_id}",
    response_model=List[RecommendationResponse],
    status_code=status.HTTP_200_OK,
)
def get_customer_recommendations(
    customer_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves historical retention recommendations for a given customer."""
    recs = (
        db.query(CustomerRecommendation)
        .filter(CustomerRecommendation.customer_id == customer_id)
        .order_by(CustomerRecommendation.created_at.desc())
        .all()
    )
    if not recs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No recommendations logged for customer '{customer_id}'.",
        )
    return recs
