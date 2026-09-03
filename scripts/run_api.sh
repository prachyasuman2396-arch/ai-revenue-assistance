#!/usr/bin/env bash
set -e

# Project root directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

echo "Starting FastAPI server with Uvicorn..."
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
