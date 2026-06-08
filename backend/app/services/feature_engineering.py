"""
Feature Engineering Service
Implements all transformations from the notebook exactly.
"""

import pandas as pd
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)

# Raw columns expected before feature engineering (excludes customerID which was dropped)
RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Columns used for model training (X before preprocessing)
MODEL_INPUT_COLUMNS = RAW_FEATURE_COLUMNS  # same after dropping Churn


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric, fill NaN with median."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_val = df["TotalCharges"].median()
    if pd.isna(median_val):
        # Fallback if all NaN (e.g., single-row inference)
        median_val = df["MonthlyCharges"].iloc[0] * df["tenure"].iloc[0]
    df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
    return df


def add_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Customer Lifetime Value = MonthlyCharges * tenure"""
    df = df.copy()
    df["CLV"] = df["MonthlyCharges"] * df["tenure"]
    return df


def add_clv_log(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transformed CLV to reduce skew."""
    df = df.copy()
    df["CLV_log"] = np.log1p(df["CLV"])
    return df


def add_avg_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Average monthly spend = TotalCharges / (tenure + 1)"""
    df = df.copy()
    df["AvgSpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Tenure group: 0-12, 12-24, 24-48, 48-72"""
    df = df.copy()
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48-72"],
    )
    return df


def add_high_charges(df: pd.DataFrame, median_monthly: float = 70.35) -> pd.DataFrame:
    """HighCharges: 1 if MonthlyCharges > dataset median, else 0"""
    df = df.copy()
    df["HighCharges"] = (df["MonthlyCharges"] > median_monthly).astype(int)
    return df


def add_auto_payment(df: pd.DataFrame) -> pd.DataFrame:
    """AutoPayment: 1 if PaymentMethod contains 'automatic'"""
    df = df.copy()
    df["AutoPayment"] = (
        df["PaymentMethod"].str.lower().str.contains("automatic", na=False)
    ).astype(int)
    return df


def add_protection_bundle(df: pd.DataFrame) -> pd.DataFrame:
    """ProtectionBundle: count of OnlineSecurity + TechSupport + DeviceProtection"""
    df = df.copy()
    df["ProtectionBundle"] = (
        (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
    )
    return df


def add_streaming_bundle(df: pd.DataFrame) -> pd.DataFrame:
    """StreamingBundle: count of StreamingTV + StreamingMovies"""
    df = df.copy()
    df["StreamingBundle"] = (
        (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )
    return df


def add_new_customer(df: pd.DataFrame) -> pd.DataFrame:
    """NewCustomer: 1 if tenure <= 12"""
    df = df.copy()
    df["NewCustomer"] = (df["tenure"] <= 12).astype(int)
    return df


def add_revenue_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """RevenueExposure = MonthlyCharges * log1p(tenure)"""
    df = df.copy()
    df["RevenueExposure"] = df["MonthlyCharges"] * np.log1p(df["tenure"])
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """service_count: number of add-on services subscribed"""
    services = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df = df.copy()
    df["service_count"] = df[services].eq("Yes").sum(axis=1)
    return df


def engineer_features(
    df: pd.DataFrame,
    monthly_charges_median: float = 70.35,
    for_model: bool = True,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps from the notebook in order.
    
    Args:
        df: Raw customer dataframe
        monthly_charges_median: Dataset median for HighCharges feature
        for_model: If True, return only the columns needed for model input (X)

    Returns:
        Engineered dataframe
    """
    df = fix_total_charges(df)
    df = add_clv(df)
    df = add_clv_log(df)
    df = add_avg_spend(df)
    df = add_tenure_group(df)
    df = add_high_charges(df, monthly_charges_median)
    df = add_auto_payment(df)
    df = add_protection_bundle(df)
    df = add_streaming_bundle(df)
    df = add_new_customer(df)
    df = add_revenue_exposure(df)
    df = add_service_count(df)

    if for_model:
        # Return only the original raw columns for the sklearn preprocessor
        # (matches what was used during training: X = df.drop('Churn', axis=1))
        available = [c for c in MODEL_INPUT_COLUMNS if c in df.columns]
        return df[available]

    return df


def engineer_features_with_revenue(
    df: pd.DataFrame,
    monthly_charges_median: float = 70.35,
) -> pd.DataFrame:
    """
    Full feature engineering including revenue features.
    Used for revenue risk calculations (not model input).
    """
    df = fix_total_charges(df)
    df = add_clv(df)
    df = add_clv_log(df)
    df = add_avg_spend(df)
    df = add_tenure_group(df)
    df = add_high_charges(df, monthly_charges_median)
    df = add_auto_payment(df)
    df = add_protection_bundle(df)
    df = add_streaming_bundle(df)
    df = add_new_customer(df)
    df = add_revenue_exposure(df)
    df = add_service_count(df)
    return df
