import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from backend.app.config.settings import settings
from ml.features.feature_pipeline import build_preprocessor_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = BASE_DIR / "ml" / "data" / "processed" / "clean_customers.parquet"
MODEL_DIR = BASE_DIR / "models"


def load_and_split_data(
    data_path: Path = DATA_PATH,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Loads clean data and creates leak-free, stratified Train (70%), Val (15%), and Test (15%) splits."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {data_path}. Run Phase 5 first."
        )

    df = pd.read_parquet(data_path)
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    # First split: Train vs (Val + Test)
    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )

    # Second split: Val vs Test
    relative_val_size = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - relative_val_size,
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        f"Data Partitioned: Train={len(X_train)} ({y_train.mean():.2%} churn), "
        f"Val={len(X_val)} ({y_val.mean():.2%} churn), "
        f"Test={len(X_test)} ({y_test.mean():.2%} churn)"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_predictions(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Computes comprehensive classification and calibration metrics."""
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def train_models() -> Pipeline:
    """Trains Baseline (Logistic Regression), Classical (Random Forest),

    and Champion (LightGBM) models, selecting the highest PR-AUC model.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    models_to_train = {
        "LogisticRegression (Baseline)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "RandomForest (Classical)": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM (Champion Candidate)": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
    }

    tracking_uri = settings.get_resolved_tracking_uri()
    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        import time
        import urllib.request
        connected = False
        for attempt in range(12):
            try:
                urllib.request.urlopen(f"{tracking_uri}/health", timeout=3)
                logger.info(f"Connected to MLflow Tracking Server: {tracking_uri}")
                connected = True
                break
            except Exception:
                logger.info(f"Waiting for MLflow server at {tracking_uri} (attempt {attempt + 1}/12)...")
                time.sleep(2)

        if not connected:
            local_sqlite = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
            logger.warning(
                f"MLflow server at {tracking_uri} unreachable. Falling back to local SQLite: {local_sqlite}"
            )
            tracking_uri = local_sqlite

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "ai-revenue-churn-prediction"
    mlflow.set_experiment(experiment_name)
    logger.info(f"Active MLflow Tracking URI: {tracking_uri}")
    logger.info(f"Active MLflow Experiment: {experiment_name}")

    results = {}
    best_model_name = None
    best_pr_auc = -1.0
    best_pipeline = None
    best_run_id = None
    best_model_uri = None

    print("\n" + "=" * 70)
    print("PHASE 8 & 10: MLFLOW-TRACKED MODEL TRAINING & EVALUATION")
    print("=" * 70)

    for name, estimator in models_to_train.items():
        logger.info(f"Training {name} with MLflow tracking...")

        with mlflow.start_run(run_name=name) as run:
            preprocessor, _, _ = build_preprocessor_pipeline()
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", estimator)]
            )

            # Calibrate probabilities with 5-fold CV on train partition
            calibrated_clf = CalibratedClassifierCV(
                estimator=pipeline, method="sigmoid", cv=5
            )
            calibrated_clf.fit(X_train, y_train)

            # Evaluate on Validation Set
            val_prob = calibrated_clf.predict_proba(X_val)[:, 1]
            val_pred = (val_prob >= 0.5).astype(int)
            val_metrics = evaluate_predictions(y_val, val_pred, val_prob)

            # Evaluate on Held-out Test Set
            test_prob = calibrated_clf.predict_proba(X_test)[:, 1]
            test_pred = (test_prob >= 0.5).astype(int)
            test_metrics = evaluate_predictions(y_test, test_pred, test_prob)

            # Log Hyperparameters & Tags
            mlflow.set_tag("model_family", name.split()[0])
            for param_k, param_v in estimator.get_params().items():
                if isinstance(param_v, (int, float, str, bool)):
                    mlflow.log_param(param_k, param_v)

            # Log Validation & Test Metrics
            for m_k, m_v in val_metrics.items():
                mlflow.log_metric(f"val_{m_k}", m_v)
            for m_k, m_v in test_metrics.items():
                mlflow.log_metric(f"test_{m_k}", m_v)

            # Log Calibrated Pipeline as MLflow Artifact
            model_info = mlflow.sklearn.log_model(
                sk_model=calibrated_clf,
                artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            results[name] = {
                "run_id": run.info.run_id,
                "validation": val_metrics,
                "test": test_metrics,
            }

            print(f"\nModel: {name} (Run ID: {run.info.run_id})")
            print(
                f"  [Validation] ROC-AUC: {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | F1: {val_metrics['f1']:.4f} | Brier: {val_metrics['brier_score']:.4f}"
            )
            print(
                f"  [Test]       ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f} | F1: {test_metrics['f1']:.4f} | Brier: {test_metrics['brier_score']:.4f}"
            )

            # Champion selection prioritized by Test PR-AUC
            if test_metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = test_metrics["pr_auc"]
                best_model_name = name
                best_pipeline = calibrated_clf
                best_run_id = run.info.run_id
                best_model_uri = model_info.model_uri

    print("\n" + "=" * 70)
    print(
        f"CHAMPION MODEL SELECTED: {best_model_name} (Test PR-AUC: {best_pr_auc:.4f})"
    )
    print(f"Champion Run ID: {best_run_id}")
    print("=" * 70)

    # Register Champion Model in MLflow Model Registry
    try:
        registered_model = mlflow.register_model(
            model_uri=best_model_uri,
            name=settings.model_name,
        )
        logger.info(
            f"Registered Champion in MLflow Model Registry: name='{settings.model_name}' version={registered_model.version}"
        )
    except Exception as e:
        logger.warning(f"Could not register model in MLflow Model Registry: {e}")

    # Export local backup artifact and metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    champion_path = MODEL_DIR / "champion_model.joblib"
    joblib.dump(best_pipeline, champion_path)
    logger.info(f"Saved Champion Pipeline to: {champion_path}")

    metadata = {
        "model_name": best_model_name,
        "selected_metric": "pr_auc",
        "best_test_pr_auc": best_pr_auc,
        "champion_run_id": best_run_id,
        "metrics_summary": results,
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return best_pipeline


if __name__ == "__main__":
    train_models()
