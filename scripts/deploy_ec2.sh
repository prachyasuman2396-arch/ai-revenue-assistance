#!/usr/bin/env bash
# ==============================================================================
# AWS EC2 Universal Production Deployment Script
# Supports: Amazon Linux 2023 / Amazon Linux 2 / Ubuntu / Debian
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "🚀 Starting AI Revenue Assistance Deployment on AWS EC2"
echo "======================================================================"

# 1. Reclaim disk space and manage swap efficiently on small 8GB root disks
echo "🧹 Reclaiming disk space and cleaning package caches..."
sudo swapoff -a 2>/dev/null || true
sudo rm -f /swapfile 2>/dev/null || true
sudo docker system prune -f 2>/dev/null || true
sudo docker builder prune -af 2>/dev/null || true

TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM" -lt 2000 ]; then
    echo "💾 Creating lightweight 512MB swap file..."
    sudo dd if=/dev/zero of=/swapfile bs=64M count=8 2>/dev/null || true
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile 2>/dev/null || true
    sudo swapon /swapfile 2>/dev/null || true
fi

# 2. Detect OS and install Docker + Git + Python
if command -v dnf &> /dev/null; then
    echo "📦 Detected Amazon Linux 2023 / RHEL (dnf)..."
    sudo dnf update -y
    sudo dnf install -y git python3 python3-pip docker gcc
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER" || true

    # Install docker compose CLI plugin if missing
    if ! docker compose version &> /dev/null; then
        echo "🐳 Installing Docker Compose plugin..."
        sudo mkdir -p /usr/local/lib/docker/cli-plugins
        sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" -o /usr/local/lib/docker/cli-plugins/docker-compose
        sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    fi

elif command -v yum &> /dev/null; then
    echo "📦 Detected Amazon Linux 2 (yum)..."
    sudo yum update -y
    sudo yum install -y git python3 python3-pip docker
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER" || true

    if ! docker compose version &> /dev/null; then
        sudo mkdir -p /usr/local/lib/docker/cli-plugins
        sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" -o /usr/local/lib/docker/cli-plugins/docker-compose
        sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    fi

elif command -v apt-get &> /dev/null; then
    echo "📦 Detected Ubuntu / Debian (apt-get)..."
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl gnupg lsb-release git python3-pip python3-venv

    if ! command -v docker &> /dev/null; then
        echo "🐳 Installing Docker Engine..."
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update -y
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        sudo usermod -aG docker "$USER" || true
    fi
fi

# 3. Create .env if missing and ensure clean file structures
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env from .env.example..."
    cp .env.example .env
fi

# Clean up previous containers and ensure clean directories
echo "🧹 Resetting containers and setting directory permissions..."
sudo docker compose down 2>/dev/null || true
if [ -e "mlflow.db" ]; then
    sudo rm -rf mlflow.db
fi
mkdir -p models mlruns
sudo chmod -R 777 models mlruns

# 4. Start Infrastructure (PostgreSQL & MLflow)
echo "🐘 Starting PostgreSQL and MLflow containers..."
sudo docker compose up -d postgres mlflow

echo "⏳ Waiting for MLflow server to be healthy on http://localhost:5001..."
MLFLOW_READY=false
for i in {1..35}; do
    if curl -s http://localhost:5001/health > /dev/null 2>&1; then
        echo "✅ MLflow server is ready and responding!"
        MLFLOW_READY=true
        break
    fi
    echo "Waiting for MLflow server image & container to start (attempt $i/35)..."
    sleep 3
done

if [ "$MLFLOW_READY" = false ]; then
    echo "❌ MLflow container failed to report healthy. Container logs:"
    sudo docker logs ai_revenue_mlflow --tail 25
    exit 1
fi

# Ensure current user owns project files and directories
sudo chown -R "$USER":"$USER" .
sudo chmod -R 777 models mlruns 2>/dev/null || true

# 5. Setup Python virtual environment & train champion model
echo "🐍 Setting up Python environment for champion model training..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "🧠 Training Champion Model and registering into MLflow..."
export PYTHONPATH=.
python ml/training/train.py

# 6. Start API Service via systemd daemon (zero disk duplication, auto-restart)
echo "⚡ Configuring and starting FastAPI systemd production service..."
CURRENT_DIR="$(pwd)"
CURRENT_USER="$(whoami)"

sudo bash -c "cat <<EOF > /etc/systemd/system/ai-revenue-api.service
[Unit]
Description=AI Revenue Assistance Production FastAPI Service
After=network.target docker.service

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${CURRENT_DIR}
Environment=\"PYTHONPATH=${CURRENT_DIR}\"
ExecStart=${CURRENT_DIR}/.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
sudo systemctl enable --now ai-revenue-api.service
sudo systemctl restart ai-revenue-api.service

# Verify API health
echo "⏳ Verifying FastAPI health..."
for i in {1..15}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "✅ FastAPI service is healthy and responding on port 8000!"
        break
    fi
    sleep 2
done

echo "======================================================================"
echo "🎉 Deployment Complete!"
echo "======================================================================"
echo "👉 Interactive Swagger API Docs : http://<EC2-PUBLIC-IP>:8000/docs"
echo "👉 MLflow Experiment Dashboard   : http://<EC2-PUBLIC-IP>:5001"
echo "👉 Health Check Endpoint         : http://<EC2-PUBLIC-IP>:8000/api/health"
echo "======================================================================"
