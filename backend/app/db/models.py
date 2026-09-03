from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from backend.app.db.database import Base


class RawCustomerData(Base):
    """Raw ingestion layer: preserves source fidelity without loss."""

    __tablename__ = "raw_customer_data"

    customer_id = Column(String(20), primary_key=True, index=True)
    gender = Column(String(10), nullable=True)
    senior_citizen = Column(SmallInteger, nullable=True)
    partner = Column(String(3), nullable=True)
    dependents = Column(String(3), nullable=True)
    tenure = Column(Integer, nullable=True)
    phone_service = Column(String(30), nullable=True)
    multiple_lines = Column(String(30), nullable=True)
    internet_service = Column(String(30), nullable=True)
    online_security = Column(String(30), nullable=True)
    online_backup = Column(String(30), nullable=True)
    device_protection = Column(String(30), nullable=True)
    tech_support = Column(String(30), nullable=True)
    streaming_tv = Column(String(30), nullable=True)
    streaming_movies = Column(String(30), nullable=True)
    contract = Column(String(20), nullable=True)
    paperless_billing = Column(String(3), nullable=True)
    payment_method = Column(String(50), nullable=True)
    monthly_charges = Column(Numeric(10, 2), nullable=True)
    total_charges = Column(Text, nullable=True)  # Text to handle empty whitespace
    churn = Column(String(3), nullable=True)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    predictions = relationship(
        "CustomerPrediction", back_populates="customer", cascade="all, delete-orphan"
    )


class CleanCustomerFeatures(Base):
    """Processed layer: validated, cast to proper numeric and boolean types."""

    __tablename__ = "clean_customer_features"

    customer_id = Column(
        String(20),
        ForeignKey("raw_customer_data.customer_id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    gender = Column(String(10), nullable=False)
    senior_citizen = Column(SmallInteger, nullable=False)
    partner = Column(Boolean, nullable=False)
    dependents = Column(Boolean, nullable=False)
    tenure = Column(Integer, nullable=False)
    phone_service = Column(Boolean, nullable=False)
    multiple_lines = Column(String(30), nullable=False)
    internet_service = Column(String(30), nullable=False)
    online_security = Column(String(30), nullable=False)
    online_backup = Column(String(30), nullable=False)
    device_protection = Column(String(30), nullable=False)
    tech_support = Column(String(30), nullable=False)
    streaming_tv = Column(String(30), nullable=False)
    streaming_movies = Column(String(30), nullable=False)
    contract = Column(String(20), nullable=False, index=True)
    paperless_billing = Column(Boolean, nullable=False)
    payment_method = Column(String(50), nullable=False)
    monthly_charges = Column(Numeric(10, 2), nullable=False)
    total_charges = Column(Numeric(10, 2), nullable=False)
    churn = Column(SmallInteger, nullable=False, index=True)  # 1 for Yes, 0 for No
    processed_at = Column(DateTime(timezone=True), server_default=func.now())


class CustomerPrediction(Base):
    """Inference audit log: records every prediction, probability, and model version."""

    __tablename__ = "customer_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(
        String(20),
        ForeignKey("raw_customer_data.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    churn_probability = Column(Numeric(5, 4), nullable=False)
    prediction = Column(SmallInteger, nullable=False)  # 1 or 0
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    customer = relationship("RawCustomerData", back_populates="predictions")
    recommendations = relationship(
        "CustomerRecommendation",
        back_populates="prediction",
        cascade="all, delete-orphan",
    )


class CustomerRecommendation(Base):
    """Revenue risk & intervention log: records action playbooks and exposure."""

    __tablename__ = "customer_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(
        Integer,
        ForeignKey("customer_predictions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    customer_id = Column(
        String(20),
        ForeignKey("raw_customer_data.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    risk_category = Column(String(20), nullable=False, index=True)  # LOW, MEDIUM, HIGH
    revenue_at_risk = Column(Numeric(10, 2), nullable=False)
    recommended_action = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    prediction = relationship("CustomerPrediction", back_populates="recommendations")
