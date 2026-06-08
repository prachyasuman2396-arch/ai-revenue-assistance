#!/bin/bash
# Run FastAPI backend from the project root directory.
# Usage: bash run_backend.sh [--port 8000]

set -e

# Navigate to the project root (where this script lives)
cd "$(dirname "$0")"

echo "Starting AI Revenue Risk Intelligence backend..."
echo "Project root: $(pwd)"
echo "Model path: models/best_lightgbm_churn.pkl"

if [ ! -f "models/best_lightgbm_churn.pkl" ]; then
  echo ""
  echo "⚠️  Model not found. Train it first:"
  echo "   python train_model.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv.xls"
  echo ""
fi

# Run uvicorn — backend/main.py adds backend/ to sys.path automatically
uvicorn backend.main:app --reload "$@"
