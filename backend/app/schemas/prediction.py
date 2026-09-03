from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Customer telemetry input schema for real-time model prediction."""

    customer_id: Optional[str] = Field(default="ANONYMOUS_PROSPECT")
    gender: str = Field(..., examples=["Female"])
    senior_citizen: int = Field(default=0, ge=0, le=1)
    partner: bool = Field(default=False)
    dependents: bool = Field(default=False)
    tenure: int = Field(..., ge=0, le=120, examples=[12])
    phone_service: bool = Field(default=True)
    multiple_lines: str = Field(
        default="No", examples=["No", "Yes", "No phone service"]
    )
    internet_service: str = Field(default="DSL", examples=["DSL", "Fiber optic", "No"])
    online_security: str = Field(
        default="No", examples=["Yes", "No", "No internet service"]
    )
    online_backup: str = Field(
        default="No", examples=["Yes", "No", "No internet service"]
    )
    device_protection: str = Field(
        default="No", examples=["Yes", "No", "No internet service"]
    )
    tech_support: str = Field(
        default="No", examples=["Yes", "No", "No internet service"]
    )
    streaming_tv: str = Field(
        default="No", examples=["Yes", "No", "No internet service"]
    )
    streaming_movies: str = Field(
        default="No", examples=["Yes", "No", "No internet service"]
    )
    contract: str = Field(
        default="Month-to-month", examples=["Month-to-month", "One year", "Two year"]
    )
    paperless_billing: bool = Field(default=True)
    payment_method: str = Field(default="Electronic check")
    monthly_charges: float = Field(..., ge=0.0, examples=[70.5])
    total_charges: float = Field(..., ge=0.0, examples=[846.0])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "7590-VHVEG",
                "gender": "Female",
                "senior_citizen": 0,
                "partner": True,
                "dependents": False,
                "tenure": 1,
                "phone_service": False,
                "multiple_lines": "No phone service",
                "internet_service": "DSL",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": True,
                "payment_method": "Electronic check",
                "monthly_charges": 29.85,
                "total_charges": 29.85,
            }
        }
    )


class RiskDetailSchema(BaseModel):
    risk_category: str
    revenue_at_risk: float
    monthly_charges: float
    annual_revenue_exposure: float
    horizon_months: int


class PersonalizedOutreach(BaseModel):
    retention_email_subject: str = Field(..., description="Subject line for retention email")
    retention_email_body: str = Field(..., description="Customer-facing personalized email body")
    call_script_talking_points: List[str] = Field(
        default_factory=list, description="Bullet points for customer success call"
    )
    counter_objection_strategy: str = Field(
        ..., description="Strategy to counter customer objections based on SHAP risk drivers"
    )


class RecommendationDetailSchema(BaseModel):
    primary_action: str
    playbook_code: str
    priority: str
    secondary_actions: List[str]
    estimated_roi_impact: str
    personalized_outreach: Optional[PersonalizedOutreach] = None


class ExplanationDetailSchema(BaseModel):
    top_risk_factors: List[Dict[str, Any]]
    top_protective_factors: List[Dict[str, Any]]
    summary: str


class PredictionResponse(BaseModel):
    """End-to-end response combining model probability, revenue risk,

    actionable recommendation, and explainability.
    """

    customer_id: str
    churn_probability: float
    prediction: int  # 1 or 0
    model_name: str
    model_version: str
    risk_analysis: RiskDetailSchema
    recommendation: RecommendationDetailSchema
    explanation: ExplanationDetailSchema
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
