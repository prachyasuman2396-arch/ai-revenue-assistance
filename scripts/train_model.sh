#!/usr/bin/env bash
set -e

# Project root directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

echo "Running model training and MLflow tracking pipeline..."
python ml/training/train.py "$@"
