# AI Revenue Assistance System — Architecture Specification

## 1. System Mission & Context
The AI Revenue Assistance platform is an enterprise-grade machine learning system designed to:
1. Ingest telecommunications/subscription customer data reliably.
2. Store immutable raw events and curated feature sets in PostgreSQL.
3. Train, calibrate, evaluate, and version customer churn models using MLflow.
4. Translate raw ML probabilities into quantifiable financial metrics (**Revenue at Risk**).
5. Generate deterministic, high-impact business retention recommendations.
6. Serve predictions with low latency via FastAPI with full auditability.

---

## 2. End-to-End Architectural Data Flow

```
[Raw Customer Data] 
       │
       ▼
[Data Ingestion Script / Stream]
       │
       ▼
[PostgreSQL: raw_customer_data] (Immutable)
       │
       ▼
[Data Validation & Cleaning Pipeline]
       │
       ▼
[PostgreSQL: clean_customer_features] (Processed / Analytical)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[MLflow Training & Feature Pipeline]    [FastAPI Prediction Microservice]
  - Stratified 80/10/10 Split             - POST /api/v1/predict
  - Sklearn Preprocessor Pipeline         - In-memory Calibrated Model
  - Hyperparameter Search                 - Pydantic Validation
  - Calibration (Sigmoid / Isotonic)      - Real-time Feature Assembly
  - Artifact & Metrics Logging                   │
       │                                         ▼
       ▼                                [Revenue Risk Engine]
[MLflow Model Registry]                   - Monetary Exposure Calculation
  - Model: "ChurnPredictor"               - Risk Classification (Low/Med/High)
  - Staging -> Production Transition             │
                                                 ▼
                                        [Recommendation Engine]
                                          - Segment-driven Action Playbook
                                                 │
                                                 ▼
                                        [PostgreSQL: Predictions & Audit]
                                          - customer_predictions
                                          - customer_recommendations
```

---

## 3. Core Component Responsibilities

### 3.1 Data Layer (`backend/app/db/`)
* **`raw_customer_data`**: Stores the raw, untouched ingestion records. Preserves source fidelity (including whitespace in numeric strings and raw identifiers).
* **`clean_customer_features`**: Cleaned, typed, and normalized records ready for analytical consumption and batch feature extraction.
* **`customer_predictions`**: Logs every generated inference: customer ID, predicted class, calibrated probability, model name, model version, timestamp, and request latency.
* **`customer_recommendations`**: Logs the business risk category, computed revenue at risk, and recommended action.

### 3.2 MLOps & Training Layer (`ml/`)
* **Validation (`ml/src/validation.py`)**: Enforces Pydantic/schema contracts and assertions on data bounds and null distributions.
* **Feature Engineering (`ml/src/features.py`)**: Encapsulates `ColumnTransformer` with numeric scaling, categorical encoding, and handling of new categories.
* **Model Training (`ml/src/train.py`)**: Trains baseline (Logistic Regression) and advanced (LightGBM/XGBoost) models with MLflow run tracking.
* **Model Registry (`ml/src/registry.py`)**: Programmatically registers models and transitions the champion to production.

### 3.3 Serving & Business Logic Layer (`backend/app/`)
* **API Engine (`backend/app/main.py`, `backend/app/api/`)**: High-performance FastAPI application handling incoming requests, validation, and database connection pooling.
* **Revenue Risk Engine (`backend/app/services/risk_engine.py`)**: Computes:
  $$\text{Revenue At Risk} = P(\text{Churn}) \times \text{Exposure}$$
* **Recommendation Engine (`backend/app/services/recommendation_engine.py`)**: Evaluates customer contract type, tenure, support tickets, and risk tier to output prescriptive actions.

### 3.4 Operational Infrastructure
* **Docker**: Reproducible containerization of the FastAPI application, PostgreSQL database, and MLflow server.
* **GitHub Actions**: Linting (`ruff`), unit tests, integration tests, and container image builds.
* **Target AWS Architecture**: Amazon ECS (Fargate) for stateless API serving, Amazon RDS (PostgreSQL) for transactional storage, Amazon S3 for MLflow artifacts, and AWS Secrets Manager for credentials.
