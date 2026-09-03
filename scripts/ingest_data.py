import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from backend.app.db.database import engine
from backend.app.db.models import RawCustomerData

DATA_PATH = BASE_DIR / "ml" / "data" / "raw" / "customer_churn.csv"


def ingest_raw_data(file_path: Path = DATA_PATH):
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {file_path}")

    print(f"Reading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    total_csv_rows = len(df)
    print(f"Total rows in CSV: {total_csv_rows}")

    # Map CSV headers to database column names
    column_mapping = {
        "customerID": "customer_id",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Churn": "churn",
    }

    df_renamed = df.rename(columns=column_mapping)
    records = df_renamed.to_dict(orient="records")

    print("Ingesting records into PostgreSQL 'raw_customer_data' table in batches...")
    batch_size = 1000
    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = insert(RawCustomerData).values(batch)
            stmt = stmt.on_conflict_do_nothing(index_elements=["customer_id"])
            conn.execute(stmt)
            print(f"  Ingested batch {i // batch_size + 1}: {len(batch)} rows")

    print("Ingestion completed.")
    verify_ingestion(total_csv_rows)


def verify_ingestion(expected_row_count: int):
    print("\n" + "=" * 50)
    print("DATA INTEGRITY & INGESTION VERIFICATION")
    print("=" * 50)

    with engine.connect() as conn:
        # 1. Row count
        count_res = conn.execute(
            text("SELECT COUNT(*) FROM raw_customer_data")
        ).scalar()
        print(
            f"Total Rows in raw_customer_data: {count_res} (Expected: {expected_row_count})"
        )
        assert count_res == expected_row_count, (
            f"Row count mismatch! Expected {expected_row_count}, got {count_res}"
        )

        # 2. Duplicate ID Check
        dup_res = conn.execute(
            text(
                "SELECT customer_id, COUNT(*) FROM raw_customer_data GROUP BY customer_id HAVING COUNT(*) > 1"
            )
        ).fetchall()
        print(f"Duplicate Customer IDs: {len(dup_res)}")
        assert len(dup_res) == 0, f"Found duplicate IDs: {dup_res}"

        # 3. Target Distribution
        target_res = conn.execute(
            text(
                "SELECT churn, COUNT(*), ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM raw_customer_data), 2) AS pct "
                "FROM raw_customer_data GROUP BY churn"
            )
        ).fetchall()
        print("\nTarget Distribution (Churn):")
        for churn_val, cnt, pct in target_res:
            print(f"  - Churn='{churn_val}': {cnt} ({pct}%)")

        # 4. Null & Blank Value Check
        blank_res = conn.execute(
            text(
                "SELECT COUNT(*) FROM raw_customer_data WHERE total_charges = ' ' OR total_charges IS NULL"
            )
        ).scalar()
        print(f"\nRecords with empty whitespace/null in total_charges: {blank_res}")

        # 5. Contract Distribution Check
        contract_res = conn.execute(
            text(
                "SELECT contract, COUNT(*) FROM raw_customer_data GROUP BY contract ORDER BY COUNT(*) DESC"
            )
        ).fetchall()
        print("\nContract Distribution:")
        for contract_type, cnt in contract_res:
            print(f"  - {contract_type}: {cnt}")

    print("=" * 50)
    print("VERIFICATION SUCCESSFUL: Zero data loss verified.")
    print("=" * 50)


if __name__ == "__main__":
    ingest_raw_data()
