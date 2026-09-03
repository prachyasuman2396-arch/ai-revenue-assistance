#!/usr/bin/env bash
# ==============================================================================
# AWS EC2 One-Click Production Deployment Script
# Project: AI Revenue Assistance (MLflow, FastAPI, PostgreSQL, Groq LLM)
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "🚀 Starting AI Revenue Assistance Deployment on AWS EC2"
echo "======================================================================"

# 1. Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release git

# 2. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker Engine..."
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Allow current user to run docker without sudo
    sudo usermod -aG docker "$USER"
    echo "✅ Docker installed successfully."
fi

# 3. Setup Python virtual environment for initial model training
echo "🐍 Setting up Python environment..."
sudo apt-get install -y python3-pip python3-venv

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure .env if not exists
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️ Please ensure you set GROQ_API_KEY in .env if you wish to use live LLM generation."
fi

# 5. Start Infrastructure (PostgreSQL & MLflow)
echo "🐘 Starting PostgreSQL and MLflow containers..."
sudo docker compose up -d postgres mlflow

echo "⏳ Waiting 10s for MLflow and PostgreSQL to become healthy..."
sleep 10

# 6. Train champion model and log into MLflow Model Registry
echo "🧠 Training Champion Model and registering in MLflow..."
export PYTHONPATH=.
python ml/training/train.py

# 7. Start API Service
echo "⚡ Building and starting FastAPI container..."
sudo docker compose up -d --build api

echo "======================================================================"
echo "🎉 Deployment Complete!"
echo "======================================================================"
echo "👉 Interactive Swagger API Docs : http://<EC2-PUBLIC-IP>:8000/docs"
echo "👉 MLflow Experiment Dashboard   : http://<EC2-PUBLIC-IP>:5001"
echo "👉 Health Check Endpoint         : http://<EC2-PUBLIC-IP>:8000/api/health"
echo "======================================================================"
