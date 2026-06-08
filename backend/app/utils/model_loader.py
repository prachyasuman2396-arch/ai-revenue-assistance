"""
Model loader: supports joblib and pickle formats, auto-detects format.
"""

import joblib
import pickle
import os
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_model_cache: Optional[Any] = None


def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk. Supports joblib and pickle formats.
    Auto-detects format based on file header.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}")

    # Try joblib first (default for sklearn/lightgbm pipelines)
    try:
        model = joblib.load(model_path)
        _model_cache = model
        logger.info("Model loaded with joblib")
        return model
    except Exception as e:
        logger.warning(f"joblib load failed: {e}. Trying pickle...")

    # Fallback to pickle
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        _model_cache = model
        logger.info("Model loaded with pickle")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def is_model_loaded() -> bool:
    return _model_cache is not None


def clear_model_cache():
    global _model_cache
    _model_cache = None
