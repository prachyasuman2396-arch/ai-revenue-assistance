import sys
from pathlib import Path

import pandas as pd

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

DATA_PATH = BASE_DIR / "ml" / "data" / "processed" / "clean_customers.parquet"


def run_eda(data_path: Path = DATA_PATH):
    if not data_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {data_path}. Run Phase 5 first."
        )

    df = pd.read_parquet(data_path)
    total_customers = len(df)
    churn_count = df["churn"].sum()
    churn_rate = (churn_count / total_customers) * 100

    print("=" * 65)
    print("PHASE 6: FOCUSED BUSINESS & STATISTICAL EDA")
    print("=" * 65)
    print(f"Total Customer Population: {total_customers:,}")
    print(f"Total Churned Customers:   {churn_count:,} ({churn_rate:.2f}%)")
    print(
        f"Retained Customers:        {total_customers - churn_count:,} ({100 - churn_rate:.2f}%)"
    )

    # 1. Financial Exposure & Monthly Revenue Loss
    total_monthly_revenue = df["monthly_charges"].sum()
    churned_monthly_revenue = df[df["churn"] == 1]["monthly_charges"].sum()
    revenue_loss_pct = (churned_monthly_revenue / total_monthly_revenue) * 100

    print("\n--- 1. FINANCIAL & REVENUE EXPOSURE ---")
    print(f"Total Monthly Recurring Revenue (MRR):      ${total_monthly_revenue:,.2f}")
    print(
        f"Monthly Revenue Lost to Churn:              ${churned_monthly_revenue:,.2f} ({revenue_loss_pct:.2f}% of MRR)"
    )
    print(
        f"Mean Monthly Charge (Churners):             ${df[df['churn'] == 1]['monthly_charges'].mean():.2f}"
    )
    print(
        f"Mean Monthly Charge (Retained):             ${df[df['churn'] == 0]['monthly_charges'].mean():.2f}"
    )

    # 2. Risk by Contract Type (Highest Impact Operational Segment)
    print("\n--- 2. CHURN RISK BY CONTRACT TYPE ---")
    contract_grp = (
        df.groupby("contract")
        .agg(
            total_customers=("customer_id", "count"),
            churned_customers=("churn", "sum"),
            churn_rate=("churn", lambda x: round(x.mean() * 100, 2)),
            monthly_revenue_at_risk=(
                "monthly_charges",
                lambda x: round(x[df.loc[x.index, "churn"] == 1].sum(), 2),
            ),
        )
        .sort_values(by="churn_rate", ascending=False)
    )
    print(contract_grp.to_string())

    # 3. Risk by Internet Service Type
    print("\n--- 3. CHURN RISK BY INTERNET SERVICE TYPE ---")
    internet_grp = (
        df.groupby("internet_service")
        .agg(
            total_customers=("customer_id", "count"),
            churned_customers=("churn", "sum"),
            churn_rate=("churn", lambda x: round(x.mean() * 100, 2)),
            avg_monthly_charge=("monthly_charges", "mean"),
        )
        .sort_values(by="churn_rate", ascending=False)
    )
    print(internet_grp.to_string())

    # 4. Impact of Value-Added Protections (TechSupport & OnlineSecurity)
    print("\n--- 4. SERVICE ECOSYSTEM RETENTION IMPACT ---")
    for service in ["tech_support", "online_security", "online_backup"]:
        svc_grp = df.groupby(service)["churn"].agg(
            total_customers="count", churn_rate=lambda x: round(x.mean() * 100, 2)
        )
        print(f"\nFeature: {service}")
        print(svc_grp.to_string())

    # 5. Tenure Dynamics (Early Life vs Matured Tenancy)
    print("\n--- 5. TENURE COHORT RISK BREAKDOWN ---")
    df["tenure_cohort"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6 mos", "7-12 mos", "13-24 mos", "25-48 mos", "49-72 mos"],
    )
    tenure_grp = df.groupby("tenure_cohort", observed=False).agg(
        total_customers=("customer_id", "count"),
        churn_rate=("churn", lambda x: round(x.mean() * 100, 2)),
        mrr_lost=(
            "monthly_charges",
            lambda x: round(x[df.loc[x.index, "churn"] == 1].sum(), 2),
        ),
    )
    print(tenure_grp.to_string())

    # 6. Payment Method & Billing Channel Risk
    print("\n--- 6. PAYMENT METHOD RISK ---")
    pay_grp = (
        df.groupby("payment_method")
        .agg(
            total_customers=("customer_id", "count"),
            churn_rate=("churn", lambda x: round(x.mean() * 100, 2)),
        )
        .sort_values(by="churn_rate", ascending=False)
    )
    print(pay_grp.to_string())

    # 7. Numerical Correlations with Target
    print("\n--- 7. NUMERICAL CORRELATION WITH CHURN ---")
    numeric_cols = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "senior_citizen",
        "churn",
    ]
    corr = df[numeric_cols].corr()["churn"].sort_values(ascending=False)
    for col, val in corr.items():
        if col != "churn":
            print(f"  {col:<20}: {val:+.4f}")

    print("=" * 65)
    print("EDA EXECUTION COMPLETED.")
    print("=" * 65)


if __name__ == "__main__":
    run_eda()
