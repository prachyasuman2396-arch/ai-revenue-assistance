#!/bin/bash
# Run Streamlit frontend from the project root directory.
# Usage: bash run_frontend.sh

set -e

cd "$(dirname "$0")"

echo "Starting Streamlit frontend..."
echo "Backend API: ${API_URL:-http://localhost:8000}"

streamlit run frontend/streamlit_app.py
