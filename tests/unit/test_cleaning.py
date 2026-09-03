import pandas as pd
import pytest

from ml.data.clean_data import (
    clean_dataframe,
    validate_clean_dataframe,
    validate_raw_dataframe,
)


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        {
            "customer_id": ["0001-TEST", "0002-TEST"],
            "gender": ["Male", "Female"],
            "senior_citizen": [0, 1],
            "partner": ["Yes", "No"],
            "dependents": ["No", "Yes"],
            "tenure": [12, 0],
            "phone_service": ["Yes", "No"],
            "multiple_lines": ["No", "No phone service"],
            "internet_service": ["DSL", "Fiber optic"],
            "online_security": ["Yes", "No"],
            "online_backup": ["No", "Yes"],
            "device_protection": ["No", "No"],
            "tech_support": ["Yes", "No"],
            "streaming_tv": ["No", "No"],
            "streaming_movies": ["No", "No"],
            "contract": ["One year", "Month-to-month"],
            "paperless_billing": ["Yes", "No"],
            "payment_method": ["Mailed check", "Electronic check"],
            "monthly_charges": [45.5, 20.0],
            "total_charges": ["546.0", " "],  # Test blank space for tenure=0
            "churn": ["No", "Yes"],
        }
    )


def test_clean_dataframe_whitespace_handling(sample_raw_df):
    validate_raw_dataframe(sample_raw_df)
    clean_df = clean_dataframe(sample_raw_df)
    validate_clean_dataframe(clean_df)

    # Assert tenure=0 row had whitespace replaced with 0.0
    zero_tenure_row = clean_df[clean_df["tenure"] == 0].iloc[0]
    assert zero_tenure_row["total_charges"] == 0.0
    assert clean_df["churn"].tolist() == [0, 1]
    assert clean_df["partner"].tolist() == [True, False]
    assert clean_df["dependents"].tolist() == [False, True]
