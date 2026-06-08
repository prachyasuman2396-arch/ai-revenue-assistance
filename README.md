# 🧠 AI Revenue Risk Intelligence Platform

An end-to-end production ML system that predicts customer churn, quantifies revenue exposure, and generates AI-powered retention strategies using the IBM Telco Customer Churn dataset.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│  Single Analysis │ Batch Upload │ Dashboard │ Retention  │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP REST
┌──────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend                       │
│                                                         │
│  /predict  /explain  /recommend  /dashboard  /full-    │
│  analysis                                               │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Feature Eng │  │  LightGBM    │  │  Revenue     │  │
│  │  Service     │  │  Predictor   │  │  Risk Engine │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │  SHAP        │  │  Groq AI     │                     │
│  │  Explainer   │  │  Retention   │                     │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
           │                          │
    ┌──────▼──────┐           ┌───────▼──────┐
    │  LightGBM   │           │   Groq API   │
    │  .pkl model │           │ llama-3.3-70b│
    └─────────────┘           └──────────────┘
```

---

## Project Structure

```
ai-revenue-risk-intelligence/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # FastAPI endpoints
│   │   ├── services/
│   │   │   ├── feature_engineering.py   # All notebook transformations
│   │   │   ├── predictor.py             # LightGBM prediction pipeline
│   │   │   ├── explainer.py             # SHAP explainability
│   │   │   ├── revenue_risk.py          # CLV, Revenue At Risk, Risk Score
│   │   │   └── groq_recommendation.py  # AI retention strategies
│   │   ├── schemas/schemas.py     # Pydantic request/response models
│   │   ├── config/settings.py     # Configurable thresholds & settings
│   │   └── utils/model_loader.py  # joblib/pickle model loader
│   └── main.py                    # FastAPI app entrypoint
├── frontend/
│   └── streamlit_app.py           # Multi-page Streamlit UI
├── models/                        # Trained model (.pkl) goes here
├── data/                          # CSV data files
├── tests/test_all.py              # pytest test suite
├── train_model.py                 # Training script
├── Dockerfile
├── render.yaml
├── requirements.txt
└── streamlit_app.py               # Streamlit Cloud entrypoint
```

---

## Quick Start (Local)

### 1. Clone & Install

```bash
git clone https://github.com/your-org/ai-revenue-risk-intelligence.git
cd ai-revenue-risk-intelligence
pip install -r requirements.txt
```

### 2. Get the Dataset

Download the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Train the Model

```bash
python train_model.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
# Saves to models/best_lightgbm_churn.pkl
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 5. Start Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### 6. Start Frontend

```bash
cd frontend
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

---

## Docker

```bash
# Build
docker build -t revenue-risk-ai .

# Run (mount your trained model)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e GROQ_API_KEY=your_key \
  revenue-risk-ai
```

---

## Render Deployment (Backend)

1. Push your repository to GitHub (include `models/best_lightgbm_churn.pkl`)
2. Create a new **Web Service** on [Render](https://ai-revenue-assistance-1.onrender.com/docs)
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml`
5. Set `GROQ_API_KEY` in the **Environment** tab of the Render dashboard
6. Your API will be live at `https://your-app.onrender.com`

**render.yaml** configures:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

---

## Streamlit Cloud Deployment (Frontend)

1. Push your repository to GitHub
2. Go to [share.streamlit.io](https://prachyasuman2396-arch-ai-revenue-a-frontendstreamlit-app-5t0g1k.streamlit.app)
3. Connect your repo, set main file to `streamlit_app.py`
4. Add secrets in the Streamlit Cloud dashboard:
   ```toml
   API_URL = "https://your-render-app.onrender.com"
   ```

---

## API Documentation

| Method | Endpoint        | Description                                      |
|--------|----------------|--------------------------------------------------|
| GET    | /              | Root / welcome                                   |
| GET    | /health        | Health check + model status                      |
| POST   | /predict       | Churn probability for one customer               |
| POST   | /explain       | SHAP top risk factors                            |
| POST   | /recommend     | AI retention strategy                            |
| POST   | /dashboard     | Portfolio metrics for a batch of customers       |
| POST   | /full-analysis | Complete analysis: predict + explain + revenue + AI |

Full interactive docs: `http://localhost:8000/docs`

### Example: Full Analysis

**Request:**
```json
POST /full-analysis
{
  "customer_id": "CUST-001",
  "customer": {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 427.5
  }
}
```

**Response:**
```json
{
  "customer_id": "CUST-001",
  "prediction": {
    "churn_probability": 0.78,
    "churn_prediction": true,
    "confidence": "High"
  },
  "explainability": {
    "top_risk_factors": [
      {"feature": "Contract_Month-to-month", "impact": 0.312},
      {"feature": "tenure", "impact": -0.241}
    ]
  },
  "revenue_metrics": {
    "customer_lifetime_value": 427.5,
    "revenue_at_risk": 333.45,
    "revenue_risk_score": 78.3,
    "avg_spend": 71.25,
    "revenue_exposure": 137.2
  },
  "risk_band": "High",
  "retention_strategy": [
    {
      "priority": 1,
      "action": "Offer annual contract upgrade with 15% discount",
      "rationale": "Month-to-month contract is the strongest churn driver.",
      "expected_impact": "30-40% churn risk reduction"
    }
  ],
  "generated_summary": "Customer shows 78% churn risk with $333 revenue at risk...",
  "inference_time_ms": 45.2
}
```

---

## Environment Variables

| Variable                    | Default                                  | Description                        |
|-----------------------------|------------------------------------------|------------------------------------|
| `GROQ_API_KEY`              | None                                     | Groq API key for AI recommendations|
| `MODEL_PATH`                | `models/best_lightgbm_churn.pkl`         | Path to trained model              |
| `RISK_THRESHOLD_CRITICAL`   | `0.8`                                    | Churn prob threshold for Critical  |
| `RISK_THRESHOLD_HIGH`       | `0.6`                                    | Churn prob threshold for High      |
| `RISK_THRESHOLD_MEDIUM`     | `0.4`                                    | Churn prob threshold for Medium    |
| `SCORE_THRESHOLD_CRITICAL`  | `80.0`                                   | Score threshold for Critical       |
| `SCORE_THRESHOLD_HIGH`      | `60.0`                                   | Score threshold for High           |
| `SCORE_THRESHOLD_MEDIUM`    | `30.0`                                   | Score threshold for Medium         |
| `API_URL`                   | `http://localhost:8000`                  | Backend URL (frontend)             |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Feature Engineering Reference

All features match the notebook exactly:

| Feature           | Formula                                      |
|-------------------|----------------------------------------------|
| CLV               | `MonthlyCharges × tenure`                    |
| CLV_log           | `log1p(CLV)`                                 |
| AvgSpend          | `TotalCharges / (tenure + 1)`                |
| TenureGroup       | `pd.cut(tenure, [0,12,24,48,72])`            |
| HighCharges       | `MonthlyCharges > dataset_median`            |
| AutoPayment       | `PaymentMethod contains "automatic"`         |
| ProtectionBundle  | `OnlineSecurity + TechSupport + DeviceProtection count` |
| StreamingBundle   | `StreamingTV + StreamingMovies count`        |
| NewCustomer       | `tenure <= 12`                               |
| RevenueExposure   | `MonthlyCharges × log1p(tenure)`             |
| RevenueAtRisk     | `Churn_Probability × CLV`                    |
| RevenueRiskScore  | `MinMaxScaled(RevenueAtRisk) × 100`          |

---

## Screenshots

> Add screenshots here after deploying.

- [ ] Single Customer Analysis
- [ ] Revenue Dashboard
- [ ] SHAP Explainability
- [ ] Retention Strategy
- [ ] Model Monitoring
