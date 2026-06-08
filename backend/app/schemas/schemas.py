from pydantic import BaseModel, Field, validator
from typing import Optional, List, Any
from enum import Enum


class ContractType(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class InternetService(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class PaymentMethod(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class YesNoPhone(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone_service = "No phone service"


class YesNoInternet(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet_service = "No internet service"


class CustomerInput(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., gt=0, example=85.5)
    TotalCharges: Optional[float] = Field(None, example=1026.0)

    @validator("TotalCharges", pre=True, always=True)
    def fill_total_charges(cls, v, values):
        if v is None and "MonthlyCharges" in values and "tenure" in values:
            return values["MonthlyCharges"] * max(values["tenure"], 1)
        return v

    class Config:
        use_enum_values = True


class RiskFactor(BaseModel):
    feature: str
    impact: float


class PredictionResult(BaseModel):
    churn_probability: float
    churn_prediction: bool
    confidence: str


class ExplainabilityResult(BaseModel):
    top_risk_factors: List[RiskFactor]


class RevenueMetrics(BaseModel):
    customer_lifetime_value: float
    revenue_at_risk: float
    revenue_risk_score: float
    avg_spend: float
    revenue_exposure: float


class RetentionAction(BaseModel):
    priority: int
    action: str
    rationale: str
    expected_impact: str


class FullAnalysisRequest(BaseModel):
    customer_id: Optional[str] = Field(None, example="CUST-001")
    customer: CustomerInput


class FullAnalysisResponse(BaseModel):
    customer_id: Optional[str]
    prediction: PredictionResult
    explainability: ExplainabilityResult
    revenue_metrics: RevenueMetrics
    risk_band: str
    retention_strategy: List[RetentionAction]
    generated_summary: str
    inference_time_ms: float


class DashboardRequest(BaseModel):
    customers: List[CustomerInput]


class DashboardCustomerRow(BaseModel):
    customer_index: int
    churn_probability: float
    risk_band: str
    clv: float
    revenue_at_risk: float
    revenue_risk_score: float
    recommendation: str


class DashboardResponse(BaseModel):
    total_customers: int
    avg_churn_probability: float
    total_revenue_at_risk: float
    avg_revenue_at_risk: float
    risk_distribution: dict
    top_risk_customers: List[DashboardCustomerRow]
    revenue_recovery_opportunity: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
