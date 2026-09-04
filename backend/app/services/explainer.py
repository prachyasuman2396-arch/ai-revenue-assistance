import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "champion_model.joblib"

# Human-friendly dictionary mapping raw/transformed feature keys to plain business narratives
FEATURE_DESCRIPTIONS = {
    "contract_Month-to-month": "Customer has a flexible Month-to-Month agreement with no cancellation penalty.",
    "contract_One year": "Customer committed to a 1-Year contract.",
    "contract_Two year": "Customer committed to a 2-Year long-term contract.",
    "has_fiber_no_tech_support": "High-margin Fiber Optic user lacking dedicated Tech Support.",
    "is_month_to_month_electronic_check": "Dangerous combination of Month-to-Month contract and manual Electronic Check payments.",
    "payment_method_Electronic check": "Manual Electronic Check billing creates monthly friction.",
    "payment_method_Credit card (automatic)": "Enrolled in automatic credit card payments.",
    "payment_method_Bank transfer (automatic)": "Enrolled in automatic bank debit.",
    "internet_service_Fiber optic": "Fiber optic service tier (high churn segment across portfolio).",
    "internet_service_DSL": "Subscribed to reliable DSL service.",
    "internet_service_No": "No internet service (low attrition segment).",
    "online_security_No": "Lacks Online Security add-on protection.",
    "online_security_Yes": "Protected with active Online Security.",
    "tech_support_No": "Lacks specialized Tech Support assistance.",
    "tech_support_Yes": "Active Tech Support coverage creates strong retention stickiness.",
    "tenure": "Account tenancy duration.",
    "tenure_years": "Total years of customer relationship.",
    "monthly_charges": "Monthly recurring bill commitment.",
    "total_charges": "Cumulative lifetime charges billed.",
    "num_services": "Total bundle services actively adopted.",
    "monthly_to_total_ratio": "Ratio of current monthly bill to cumulative lifetime billing.",
}


class ChurnExplainer:
    """Production model explainer converting mathematical contributions

    into actionable, business-friendly factor analyses.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._feature_names = None
        self._coefficients = None
        self._intercept = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            logger.warning(f"Champion model not found at {self.model_path}")
            return

        self._model = joblib.load(self.model_path)

        # Extract underlying fitted pipeline and coefficients
        # CalibratedClassifierCV wraps calibrated_classifiers_
        try:
            base_calibrator = self._model.calibrated_classifiers_[0]
            base_pipeline = base_calibrator.estimator
            preprocessor = base_pipeline.named_steps["preprocessor"]

            # Extract transformed feature names
            col_transform = preprocessor.named_steps["column_transforms"]
            self._feature_names = col_transform.get_feature_names_out()

            # For linear models: extract average weights across folds
            weights = []
            intercepts = []
            for c in self._model.calibrated_classifiers_:
                w = c.estimator.named_steps["classifier"].coef_[0]
                b = c.estimator.named_steps["classifier"].intercept_[0]
                weights.append(w)
                intercepts.append(b)

            self._coefficients = np.mean(weights, axis=0)
            self._intercept = np.mean(intercepts)
            self._base_preprocessor = preprocessor
            logger.info("ChurnExplainer initialized with fitted feature attributions.")
        except Exception as e:
            logger.warning(f"Could not extract linear coefficients directly: {e}")

    def explain_instance(
        self,
        customer_data: Dict[str, Any],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Explains why an individual customer received their churn risk score.

        Returns:
            - top_risk_factors: features pushing risk UP
            - top_protective_factors: features pushing risk DOWN
            - summary: cohesive business paragraph
        """
        if self._model is None or self._coefficients is None:
            self._load_model()
            if self._model is None:
                return {
                    "top_risk_factors": [],
                    "top_protective_factors": [],
                    "summary": "Model explainability is currently initializing.",
                }

        df = pd.DataFrame([customer_data])
        # Clean expected fields if necessary
        if "customer_id" in df.columns:
            df = df.drop(columns=["customer_id"])
        if "churn" in df.columns:
            df = df.drop(columns=["churn"])

        # Transform features
        X_trans = self._base_preprocessor.transform(df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

        x_row = X_trans[0]
        # Linear contribution: w_i * x_i
        contributions = self._coefficients * x_row

        # Pair feature names with contributions
        clean_names = [name.split("__")[-1] for name in self._feature_names]
        feature_impacts = list(zip(clean_names, contributions, x_row))

        # Sort into positive (risk increasing) and negative (risk decreasing)
        risk_factors = [
            (feat, float(contrib))
            for feat, contrib, val in feature_impacts
            if contrib > 0.01 and val != 0
        ]
        protective_factors = [
            (feat, float(contrib))
            for feat, contrib, val in feature_impacts
            if contrib < -0.01 and val != 0
        ]

        risk_factors.sort(key=lambda x: x[1], reverse=True)
        protective_factors.sort(key=lambda x: x[1])

        top_risk = []
        for feat, score in risk_factors[:top_k]:
            desc = FEATURE_DESCRIPTIONS.get(
                feat, f"Attribute '{feat}' increases churn likelihood."
            )
            top_risk.append(
                {
                    "feature": feat,
                    "attribution_score": round(score, 3),
                    "description": desc,
                }
            )

        top_protective = []
        for feat, score in protective_factors[:top_k]:
            desc = FEATURE_DESCRIPTIONS.get(
                feat, f"Attribute '{feat}' fosters customer loyalty."
            )
            top_protective.append(
                {
                    "feature": feat,
                    "attribution_score": round(score, 3),
                    "description": desc,
                }
            )

        # Synthesize narrative summary
        risk_reasons = [r["description"] for r in top_risk]
        if risk_reasons:
            summary = "Key Risk Drivers: " + " ".join(risk_reasons)
        else:
            summary = "Customer maintains strong loyalty characteristics with minimal churn indicators."

        if top_protective:
            summary += " Protective Offsets: " + " ".join(
                [p["description"] for p in top_protective[:2]]
            )

        return {
            "top_risk_factors": top_risk,
            "top_protective_factors": top_protective,
            "summary": summary,
        }


# Singleton instance for API dependency injection
explainer = ChurnExplainer()
