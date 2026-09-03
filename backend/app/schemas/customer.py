from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CustomerBase(BaseModel):
    customer_id: str
    gender: str | None = None
    senior_citizen: int | None = None
    partner: str | None = None
    dependents: str | None = None
    tenure: int | None = None
    phone_service: str | None = None
    multiple_lines: str | None = None
    internet_service: str | None = None
    online_security: str | None = None
    online_backup: str | None = None
    device_protection: str | None = None
    tech_support: str | None = None
    streaming_tv: str | None = None
    streaming_movies: str | None = None
    contract: str | None = None
    paperless_billing: str | None = None
    payment_method: str | None = None
    monthly_charges: float | None = None
    total_charges: str | None = None
    churn: str | None = None


class CustomerResponse(CustomerBase):
    ingested_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
