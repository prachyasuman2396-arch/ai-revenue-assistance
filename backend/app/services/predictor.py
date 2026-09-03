import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.config.settings import settings
from backend.app.db.models import CustomerPrediction, CustomerRecommendation
from backend.app.services.explainer import explainer
from backend.app.services.llm_retention import retention_service
from backend.app.services.recommendation import generate_recommendations
from backend.app.services.revenue_risk import calculate_revenue_risk

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "champion_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"


class ModelPredictor:
    """Production prediction service orchestrating inference, revenue risk calculation,

    recommendation playbooks, explainability, and database persistence.
    """

    def __init__(
        self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.model_name = settings.model_name
        self.model_version = settings.model_version
        self._load_model()

    def _load_model(self):
        loaded_from_registry = False

        # 1. Primary: Attempt loading from MLflow Model Registry
        try:
            tracking_uri = settings.get_resolved_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            logger.info(
                f"Loading registered model from MLflow: {model_uri} via {tracking_uri}..."
            )
            self.model = mlflow.sklearn.load_model(model_uri)
            loaded_from_registry = True
            logger.info(
                f"Successfully loaded '{self.model_name}' (version: {self.model_version}) from MLflow Model Registry."
            )
        except Exception as e:
            logger.warning(
                f"Could not load model from MLflow Model Registry ({e}). Falling back to local artifact."
            )

        # 2. Fallback to local artifact if registry load failed or offline
        if not loaded_from_registry:
            if not self.model_path.exists():
                logger.warning(
                    f"Champion model artifact not found at {self.model_path}. Train the model first."
                )
                return

            logger.info(f"Loading champion model from {self.model_path}...")
            self.model = joblib.load(self.model_path)

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    meta = json.load(f)
                    if not loaded_from_registry:
                        self.model_name = meta.get("model_name", settings.model_name)
                        self.model_version = meta.get(
                            "champion_run_id", settings.model_version
                        )[:8]
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

        logger.info(
            f"ModelPredictor ready: {self.model_name} (Version: {self.model_version})"
        )

    def predict_and_assess(
        self,
        customer_data: Dict[str, Any],
        db: Optional[Session] = None,
        save_audit: bool = True,
    ) -> Dict[str, Any]:
        """Full end-to-end inference flow.

        Executes:
          1. Feature Assembly & Model Scoring
          2. Financial Risk Calculation (Revenue Risk Engine)
          3. Prescriptive Recommendation Generation
          4. Feature Attribution & Explanation (SHAP)
          5. PostgreSQL Audit Logging (if customer exists in raw layer)
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise RuntimeError(
                    "Prediction model is not initialized. Please train the model."
                )

        cust_id = customer_data.get("customer_id", "ANONYMOUS_PROSPECT")

        # 1. Feature Assembly & Probability Scoring
        df = pd.DataFrame([customer_data])
        # Drop identifiers and labels if present
        for col in ["customer_id", "churn", "ingested_at", "processed_at"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        prob = float(self.model.predict_proba(df)[0, 1])
        pred_label = int(prob >= 0.5)

        # 2. Revenue Risk Calculation
        monthly_charges = float(customer_data.get("monthly_charges", 50.0))
        contract = customer_data.get("contract", "Month-to-month")
        tenure = int(customer_data.get("tenure", 12))
        risk_info = calculate_revenue_risk(
            churn_probability=prob,
            monthly_charges=monthly_charges,
            contract_type=contract,
            tenure_months=tenure,
        )

        # 3. Targeted Recommendation Playbook
        recommendation_info = generate_recommendations(customer_data, risk_info)

        # 4. Feature Explanations
        explanation_info = explainer.explain_instance(customer_data, top_k=3)

        # 5. Generative Retention Outreach via Groq & LangChain
        outreach_info = retention_service.generate_outreach(
            customer_data=customer_data,
            risk_info=risk_info,
            recommendation_info=recommendation_info,
            explanation_info=explanation_info,
        )
        recommendation_info["personalized_outreach"] = outreach_info

        now_utc = datetime.now(timezone.utc)

        # 6. Database Persistence (Audit Traceability)
        if save_audit and db is not None:
            try:
                pred_record = CustomerPrediction(
                    customer_id=cust_id,
                    churn_probability=round(prob, 4),
                    prediction=pred_label,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    created_at=now_utc,
                )
                db.add(pred_record)
                db.flush()  # Generates pred_record.id

                rec_record = CustomerRecommendation(
                    prediction_id=pred_record.id,
                    customer_id=cust_id,
                    risk_category=risk_info["risk_category"],
                    revenue_at_risk=risk_info["revenue_at_risk"],
                    recommended_action=recommendation_info["primary_action"],
                    created_at=now_utc,
                )
                db.add(rec_record)
                db.commit()
                logger.info(
                    f"Audit record persisted for customer {cust_id} (Pred ID: {pred_record.id})"
                )
            except Exception as e:
                db.rollback()
                logger.warning(
                    f"Could not persist prediction audit (e.g. anonymous or foreign key constraint): {e}"
                )

        return {
            "customer_id": cust_id,
            "churn_probability": round(prob, 4),
            "prediction": pred_label,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "risk_analysis": risk_info,
            "recommendation": recommendation_info,
            "explanation": explanation_info,
            "created_at": now_utc,
        }


# Singleton instance
predictor = ModelPredictor()
