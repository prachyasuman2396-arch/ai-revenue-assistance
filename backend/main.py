"""
AI Revenue Risk Intelligence - FastAPI Backend
"""

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# ──────────────────────────────────────────────
# Path fix: allow running from project root OR from backend/
# When running as `uvicorn backend.main:app` from project root,
# Python can't find the `app` package inside backend/.
# We insert backend/ into sys.path so `from app.xxx` always resolves.
# ──────────────────────────────────────────────
_backend_dir = Path(__file__).parent.resolve()
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.utils.model_loader import load_model

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan: pre-load model on startup
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model(settings.MODEL_PATH)
        logger.info(f"Model loaded from {settings.MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(
            f"Model not found at {settings.MODEL_PATH}. "
            "Train and save the model first, or mount the models/ directory."
        )
    yield
    logger.info("Application shutdown")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-Driven Revenue Risk Intelligence Platform. "
        "Predicts customer churn, quantifies revenue exposure, "
        "and generates AI-powered retention strategies."
    ),
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)
