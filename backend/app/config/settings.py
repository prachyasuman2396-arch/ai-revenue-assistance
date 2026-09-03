from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "AI Revenue Assistance"
    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = (
        "postgresql+psycopg://revenue_user:revenue_password@localhost:5433/revenue_ai"
    )
    mlflow_tracking_uri: str = "http://localhost:5001"
    model_name: str = "ChurnPredictor"
    model_version: str = "latest"
    groq_api_key: Optional[str] = None
    groq_model_name: str = "openai/gpt-oss-120b"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_resolved_tracking_uri(self) -> str:
        v = self.mlflow_tracking_uri
        if v.startswith("sqlite:///") and not v.startswith("sqlite:////"):
            rel_path = v[len("sqlite:///") :]
            if not Path(rel_path).is_absolute():
                abs_path = (BASE_DIR / rel_path).resolve()
                return f"sqlite:///{abs_path}"
        return v


settings = Settings()
