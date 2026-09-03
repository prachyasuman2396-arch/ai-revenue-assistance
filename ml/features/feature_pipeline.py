from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generates domain-specific features based on business EDA findings.

    Fully compatible with scikit-learn Pipeline and prevents training-serving skew.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # 1. Total active ecosystem services count (switching cost proxy)
        active_services = np.zeros(len(X_df), dtype=int)

        if "phone_service" in X_df.columns:
            active_services += (X_df["phone_service"].isin([True, "Yes", 1])).astype(
                int
            )
        if "multiple_lines" in X_df.columns:
            active_services += (X_df["multiple_lines"] == "Yes").astype(int)
        if "internet_service" in X_df.columns:
            active_services += (X_df["internet_service"] != "No").astype(int)

        for col in [
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ]:
            if col in X_df.columns:
                active_services += (X_df[col] == "Yes").astype(int)

        X_df["num_services"] = active_services

        # 2. Premium high-risk segment: Fiber optic without dedicated TechSupport
        if "internet_service" in X_df.columns and "tech_support" in X_df.columns:
            X_df["has_fiber_no_tech_support"] = (
                (X_df["internet_service"] == "Fiber optic")
                & (X_df["tech_support"] != "Yes")
            ).astype(int)
        else:
            X_df["has_fiber_no_tech_support"] = 0

        # 3. Highest churn payment + contract combination
        if "contract" in X_df.columns and "payment_method" in X_df.columns:
            X_df["is_month_to_month_electronic_check"] = (
                (X_df["contract"] == "Month-to-month")
                & (X_df["payment_method"] == "Electronic check")
            ).astype(int)
        else:
            X_df["is_month_to_month_electronic_check"] = 0

        # 4. Tenure years
        if "tenure" in X_df.columns:
            X_df["tenure_years"] = X_df["tenure"].astype(float) / 12.0
        else:
            X_df["tenure_years"] = 0.0

        # 5. Monthly charge to total charge ratio (high for early life accounts)
        if "monthly_charges" in X_df.columns and "total_charges" in X_df.columns:
            X_df["monthly_to_total_ratio"] = X_df["monthly_charges"].astype(float) / (
                X_df["total_charges"].astype(float) + 1.0
            )
        else:
            X_df["monthly_to_total_ratio"] = 0.0

        return X_df


def build_preprocessor_pipeline(
    numeric_features: List[str] = None,
    categorical_features: List[str] = None,
) -> Tuple[Pipeline, List[str], List[str]]:
    """Builds a scikit-learn Pipeline encapsulating domain feature generation,

    imputation, standard scaling, and one-hot encoding.
    """
    if numeric_features is None:
        numeric_features = [
            "tenure",
            "monthly_charges",
            "total_charges",
            "num_services",
            "tenure_years",
            "monthly_to_total_ratio",
            "has_fiber_no_tech_support",
            "is_month_to_month_electronic_check",
            "senior_citizen",
        ]

    if categorical_features is None:
        categorical_features = [
            "gender",
            "partner",
            "dependents",
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
        ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    full_preprocessor = Pipeline(
        steps=[
            ("domain_features", DomainFeatureGenerator()),
            ("column_transforms", column_transformer),
        ]
    )

    return full_preprocessor, numeric_features, categorical_features
