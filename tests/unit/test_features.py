import numpy as np
import pandas as pd
import pytest

from ml.features.feature_pipeline import (
    DomainFeatureGenerator,
    build_preprocessor_pipeline,
)


@pytest.fixture
def sample_clean_df():
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "gender": ["Female", "Male", "Female"],
            "senior_citizen": [0, 1, 0],
            "partner": [True, False, True],
            "dependents": [False, True, False],
            "tenure": [12, 1, 48],
            "phone_service": [True, True, False],
            "multiple_lines": ["No", "Yes", "No phone service"],
            "internet_service": ["DSL", "Fiber optic", "DSL"],
            "online_security": ["Yes", "No", "Yes"],
            "online_backup": ["No", "No", "Yes"],
            "device_protection": ["No", "No", "Yes"],
            "tech_support": ["Yes", "No", "Yes"],
            "streaming_tv": ["No", "Yes", "Yes"],
            "streaming_movies": ["No", "Yes", "No"],
            "contract": ["One year", "Month-to-month", "Two year"],
            "paperless_billing": [True, True, False],
            "payment_method": [
                "Mailed check",
                "Electronic check",
                "Bank transfer (automatic)",
            ],
            "monthly_charges": [55.0, 95.5, 60.0],
            "total_charges": [660.0, 95.5, 2880.0],
            "churn": [0, 1, 0],
        }
    )


def test_domain_feature_generator(sample_clean_df):
    gen = DomainFeatureGenerator()
    out_df = gen.transform(sample_clean_df)

    assert "num_services" in out_df.columns
    assert "has_fiber_no_tech_support" in out_df.columns
    assert "is_month_to_month_electronic_check" in out_df.columns
    assert "tenure_years" in out_df.columns
    assert "monthly_to_total_ratio" in out_df.columns

    # C2 has Fiber optic and tech_support == 'No'
    assert out_df.loc[1, "has_fiber_no_tech_support"] == 1
    assert out_df.loc[0, "has_fiber_no_tech_support"] == 0

    # C2 has Month-to-month and Electronic check
    assert out_df.loc[1, "is_month_to_month_electronic_check"] == 1
    assert out_df.loc[0, "is_month_to_month_electronic_check"] == 0

    # Tenure years
    assert out_df.loc[0, "tenure_years"] == 1.0


def test_preprocessor_pipeline_fit_transform(sample_clean_df):
    preprocessor, num_cols, cat_cols = build_preprocessor_pipeline()

    # Exclude target and ID from feature matrix
    X = sample_clean_df.drop(columns=["customer_id", "churn"])
    transformed = preprocessor.fit_transform(X)

    # Assert matrix is 2D, non-empty, and has zero NaNs
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_clean_df)
    assert not np.isnan(transformed).any()

    # Test inference with unseen category
    X_new = X.copy()
    X_new["payment_method"] = "Cryptocurrency"  # Unseen category
    transformed_new = preprocessor.transform(X_new)
    assert transformed_new.shape == transformed.shape
    assert not np.isnan(transformed_new).any()
