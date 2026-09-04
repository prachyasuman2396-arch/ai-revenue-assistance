#!/usr/bin/env bash
# ==============================================================================
# Run Streamlit Executive Frontend
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "📦 Ensuring Streamlit & Plotly are installed..."
pip install --quiet streamlit plotly

echo "🚀 Launching AI Revenue Assistance Streamlit Dashboard on port 8501..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
