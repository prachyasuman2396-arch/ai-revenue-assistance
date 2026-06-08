"""
Train the LightGBM churn prediction model from the IBM Telco Customer Churn dataset.
Saves best_lightgbm_churn.pkl to the models/ directory.

Usage:
    python train_model.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

The script replicates the notebook pipeline exactly:
  1. Load and clean data
  2. Feature engineering
  3. Preprocessing (ColumnTransformer)
  4. LightGBM training
  5. Evaluate on test set
  6. Save with joblib
"""

import argparse
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Feature Engineering (mirrors notebook exactly)
# ──────────────────────────────────────────────

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Engineered features (notebook Phase 1)
    df["CLV"] = df["MonthlyCharges"] * df["tenure"]
    df["CLV_log"] = np.log1p(df["CLV"])
    df["AvgSpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48-72"],
    )
    df["HighCharges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    df["AutoPayment"] = df["PaymentMethod"].str.lower().str.contains("automatic", na=False).astype(int)
    df["ProtectionBundle"] = (
        (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
    )
    df["StreamingBundle"] = (
        (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )
    df["NewCustomer"] = (df["tenure"] <= 12).astype(int)
    df["RevenueExposure"] = df["MonthlyCharges"] * np.log1p(df["tenure"])

    services = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = df[services].eq("Yes").sum(axis=1)

    return df


def train(data_path: str, output_path: str = "models/best_lightgbm_churn.pkl"):
    logger.info(f"Loading data from {data_path}")
    # Handle files with .xls/.xlsx extension that are actually CSVs
    # (a common issue when downloading from Kaggle on some browsers)
    ext = os.path.splitext(data_path)[1].lower()
    if ext in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(data_path)
            logger.info("Loaded as Excel file")
        except Exception:
            # File may be a CSV with a wrong extension — try CSV
            logger.info("Excel read failed, trying CSV parser...")
            df = pd.read_csv(data_path)
    else:
        df = pd.read_csv(data_path)

    logger.info(f"Dataset shape: {df.shape}")

    # Drop customerID if present
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Feature engineering
    df = feature_engineer(df)

    # Notebook keeps only the original raw columns for X (before engineered features)
    # This matches: X = df.drop('Churn', axis=1) [run before adding engineered cols]
    # However, since we add features to df we need the original raw columns:
    raw_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]

    X = df[raw_cols]
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Column types
    num_col = X.select_dtypes(exclude="object").columns.tolist()
    cat_col = X.select_dtypes(include="object").columns.tolist()

    logger.info(f"Numeric cols: {num_col}")
    logger.info(f"Categorical cols: {cat_col}")

    # Preprocessing pipelines (exact notebook setup)
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_col),
        ("cat", cat_pipeline, cat_col),
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # LightGBM pipeline (exact notebook parameters)
    lgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
        )),
    ])

    logger.info("Training LightGBM pipeline...")
    lgb_pipeline.fit(X_train, y_train)

    # Evaluate (notebook threshold = 0.3)
    y_prob = lgb_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.3).astype(int)

    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(lgb_pipeline, output_path)
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--data", required=True, help="Path to Telco churn CSV")
    parser.add_argument("--output", default="models/best_lightgbm_churn.pkl", help="Output model path")
    args = parser.parse_args()
    train(args.data, args.output)
