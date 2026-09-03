import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

# Ensure project root in path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from backend.app.db.database import engine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_raw_dataframe(df: pd.DataFrame) -> None:
    """Pre-cleaning validation: verify structural contracts on incoming raw data."""
    logger.info("Executing pre-cleaning schema validation...")
    required_cols = [
        "customer_id",
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "tenure",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract",
        "paperless_billing",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "churn",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")

    # Check primary key duplicates
    duplicates = df[df.duplicated(subset=["customer_id"], keep=False)]
    if not duplicates.empty:
        raise ValueError(
            f"Duplicate customer_ids detected in raw data: {len(duplicates)}"
        )

    logger.info(
        f"Raw data validation passed: {len(df)} rows, all {len(required_cols)} columns present."
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and transforms raw records into typed analytical features.

    Preserves original raw data by working on an explicit copy.
    """
    logger.info("Starting data cleaning and type coercion pipeline...")
    clean_df = df.copy()

    # 1. Total Charges handling:
    # Source contains empty strings ' ' for accounts with tenure=0.
    # Replace empty spaces with NaN, then fill with 0.0, and cast to float64.
    clean_df["total_charges"] = (
        clean_df["total_charges"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )
    # Zero tenure accounts naturally have 0 total billed
    zero_tenure_mask = clean_df["tenure"] == 0
    clean_df.loc[
        zero_tenure_mask & clean_df["total_charges"].isna(), "total_charges"
    ] = 0.0

    # Ensure no remaining NaNs in total_charges
    if clean_df["total_charges"].isna().any():
        median_val = clean_df["total_charges"].median()
        clean_df["total_charges"] = clean_df["total_charges"].fillna(median_val)
        logger.warning(
            f"Imputed {clean_df['total_charges'].isna().sum()} unexpected missing total_charges with median: {median_val}"
        )

    # 2. Enforce Numeric Types & Range Checks
    clean_df["tenure"] = clean_df["tenure"].astype(int)
    clean_df["monthly_charges"] = clean_df["monthly_charges"].astype(float)
    clean_df["senior_citizen"] = clean_df["senior_citizen"].astype(int)

    # 3. Standardize Binary Flags to booleans
    binary_map = {"Yes": True, "No": False}
    for col in ["partner", "dependents", "phone_service", "paperless_billing"]:
        clean_df[col] = clean_df[col].map(binary_map).astype(bool)

    # 4. Standardize Target Variable: Churn (1 = Yes, 0 = No)
    clean_df["churn"] = clean_df["churn"].map({"Yes": 1, "No": 0}).astype(int)

    # 5. Clean categorical string fields
    cat_cols = [
        "gender",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract",
        "payment_method",
    ]
    for col in cat_cols:
        clean_df[col] = clean_df[col].astype(str).str.strip()

    logger.info("Data cleaning transformations completed successfully.")
    return clean_df


def validate_clean_dataframe(df: pd.DataFrame) -> None:
    """Post-cleaning assertions: strict checks to ensure data is model-ready."""
    logger.info("Executing post-cleaning validation assertions...")

    # Assert no nulls anywhere
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise AssertionError(
            f"Post-cleaning nulls detected:\n{null_counts[null_counts > 0]}"
        )

    # Range and logic constraints
    assert (df["tenure"] >= 0).all(), "Negative tenure values detected!"
    assert (df["monthly_charges"] >= 0).all(), "Negative monthly charges detected!"
    assert (df["total_charges"] >= 0).all(), "Negative total charges detected!"
    assert set(df["churn"].unique()).issubset({0, 1}), "Invalid churn target values!"
    assert set(df["senior_citizen"].unique()).issubset({0, 1}), (
        "Invalid senior_citizen values!"
    )

    logger.info(
        "Post-cleaning validation passed: 0 nulls, all domain constraints satisfied."
    )


def run_cleaning_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts from PostgreSQL raw layer, cleans, validates, and persists to processed layer."""
    logger.info("Fetching data from PostgreSQL 'raw_customer_data'...")
    with engine.connect() as conn:
        raw_df = pd.read_sql(
            text(
                "SELECT customer_id, gender, senior_citizen, partner, dependents, tenure, "
                "phone_service, multiple_lines, internet_service, online_security, online_backup, "
                "device_protection, tech_support, streaming_tv, streaming_movies, contract, "
                "paperless_billing, payment_method, monthly_charges, total_charges, churn "
                "FROM raw_customer_data"
            ),
            conn,
        )

    # Step 1: Pre-validation
    validate_raw_dataframe(raw_df)

    # Step 2: Clean and Transform
    clean_df = clean_dataframe(raw_df)

    # Step 3: Post-validation
    validate_clean_dataframe(clean_df)

    # Step 4: Persist to PostgreSQL 'clean_customer_features'
    logger.info(
        "Persisting processed features to PostgreSQL 'clean_customer_features' table..."
    )
    with engine.begin() as conn:
        # Clear existing processed data for idempotent re-runs
        conn.execute(text("DELETE FROM clean_customer_features"))
        records = clean_df.to_dict(orient="records")
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            conn.execute(
                text(
                    "INSERT INTO clean_customer_features ("
                    "customer_id, gender, senior_citizen, partner, dependents, tenure, "
                    "phone_service, multiple_lines, internet_service, online_security, "
                    "online_backup, device_protection, tech_support, streaming_tv, "
                    "streaming_movies, contract, paperless_billing, payment_method, "
                    "monthly_charges, total_charges, churn) VALUES ("
                    ":customer_id, :gender, :senior_citizen, :partner, :dependents, :tenure, "
                    ":phone_service, :multiple_lines, :internet_service, :online_security, "
                    ":online_backup, :device_protection, :tech_support, :streaming_tv, "
                    ":streaming_movies, :contract, :paperless_billing, :payment_method, "
                    ":monthly_charges, :total_charges, :churn)"
                ),
                batch,
            )
            logger.info(
                f"  Inserted processed batch {i // batch_size + 1}: {len(batch)} rows"
            )

    # Step 5: Save Parquet snapshot for fast offline training
    processed_dir = BASE_DIR / "ml" / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = processed_dir / "clean_customers.parquet"
    csv_path = processed_dir / "clean_customers.csv"
    clean_df.to_parquet(parquet_path, index=False)
    clean_df.to_csv(csv_path, index=False)
    logger.info(f"Saved local processed snapshot to: {parquet_path}")

    return raw_df, clean_df


if __name__ == "__main__":
    run_cleaning_pipeline()
