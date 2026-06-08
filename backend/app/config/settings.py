from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

# Project root = two levels up from this file (backend/app/config/settings.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # App
    APP_NAME: str = "AI Revenue Risk Intelligence"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API Keys
    GROQ_API_KEY: Optional[str] = None

    # Model — default resolves to <project_root>/models/best_lightgbm_churn.pkl
    MODEL_PATH: str = str(_PROJECT_ROOT / "models" / "best_lightgbm_churn.pkl")

    # Risk Thresholds (churn probability)
    RISK_THRESHOLD_CRITICAL: float = 0.8
    RISK_THRESHOLD_HIGH: float = 0.6
    RISK_THRESHOLD_MEDIUM: float = 0.4

    # Risk Score Thresholds (0-100 score)
    SCORE_THRESHOLD_CRITICAL: float = 80.0
    SCORE_THRESHOLD_HIGH: float = 60.0
    SCORE_THRESHOLD_MEDIUM: float = 30.0

    # SHAP
    SHAP_TOP_N: int = 5

    # Groq
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 1500
    GROQ_MAX_RETRIES: int = 3

    # Monthly Charges median (from dataset; used for HighCharges feature)
    MONTHLY_CHARGES_MEDIAN: float = 70.35

    class Config:
        env_file = str(_PROJECT_ROOT / ".env")
        case_sensitive = True
        extra = "ignore"  # silently ignore unknown vars (e.g. API_URL used by frontend)


settings = Settings()
