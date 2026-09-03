from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RecommendationResponse(BaseModel):
    """Schema for individual customer recommendation history queries matching DB table."""

    id: int
    prediction_id: int
    customer_id: str
    risk_category: str
    revenue_at_risk: float
    recommended_action: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
