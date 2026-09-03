# AWS EC2 Production Deployment Guide

This guide walks you through deploying the **AI Revenue Assistance** platform on an **AWS EC2** instance so that your FastAPI documentation and MLflow tracking server are publicly accessible via a live URL for portfolio reviews and interviewers.

---

## Step 1: Launch an EC2 Instance on AWS

1. Log into your [AWS Management Console](https://console.aws.amazon.com/).
2. Navigate to **EC2** > Click **Launch Instance**.
3. Configure the instance settings:
   * **Name**: `ai-revenue-assistance-prod`
   * **Application & OS Image (AMI)**: Select **Ubuntu 24.04 LTS (x86_64)**.
   * **Instance Type**:
     * Recommended: **`t3.small`** (2 vCPU, 2 GiB RAM) or **`t2.medium`** (2 vCPU, 4 GiB RAM) for smooth LightGBM training and Docker containers.
     * Free Tier fallback: `t2.micro` (Note: `t2.micro` has 1GB RAM, which requires enabling 2GB Swap space before running Docker builds).
   * **Key Pair (login)**:
     * Select an existing `.pem` key pair or click **Create new key pair** (e.g., `revenue-ai-key.pem`).
     * Save the `.pem` file to your computer.

4. **Network Settings (Security Group)**:
   Click **Edit** and add the following **Inbound Security Group Rules**:
   | Type | Protocol | Port Range | Source | Purpose |
   | :--- | :--- | :--- | :--- | :--- |
   | **SSH** | TCP | `22` | My IP | Secure SSH terminal access |
   | **Custom TCP** | TCP | `8000` | `0.0.0.0/0` | **FastAPI Swagger Docs & Endpoints** |
   | **Custom TCP** | TCP | `5001` | `0.0.0.0/0` | **MLflow Experiment & Model Registry UI** |

5. **Storage**:
   * Set root volume to **25 GiB gp3** (AWS Free Tier includes up to 30 GiB gp3 storage).
6. Click **Launch Instance**.

---

## Step 2: Connect to Your EC2 Instance

Open your local terminal (macOS/Linux) and connect using your `.pem` key:

```bash
# 1. Set read-only permissions on your downloaded key
chmod 400 ~/Downloads/revenue-ai-key.pem

# 2. Connect via SSH (replace with your EC2 Public IPv4 DNS or IP)
ssh -i ~/Downloads/revenue-ai-key.pem ubuntu@<YOUR-EC2-PUBLIC-IP>
```

---

## Step 3: Clone Repository & Run One-Click Deployment

Once connected to your EC2 instance, run the following commands:

```bash
# 1. Clone your updated repository
git clone https://github.com/prachyasuman2396-arch/ai-revenue-assistance.git
cd ai-revenue-assistance

# 2. (Optional) If using t2.micro or t3.micro (1GB RAM), enable 2GB swap space:
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

# 3. Run the automated deployment script
bash scripts/deploy_ec2.sh
```

### What `scripts/deploy_ec2.sh` automatically does:
1. Installs Docker, Docker Compose, and build tools.
2. Creates the Python environment and installs dependencies.
3. Launches PostgreSQL and MLflow 3 containers in Docker.
4. Trains the champion model and registers `ChurnPredictor` into the MLflow Model Registry.
5. Builds and starts the FastAPI container with health checks and persistence.

---

## Step 4: Configure Groq API Key on EC2

To enable live LLM generation for retention outreach on EC2:
```bash
# Edit .env on the EC2 instance
nano .env
```
Add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL_NAME=openai/gpt-oss-120b
```
Save with `Ctrl + O`, `Enter`, and exit with `Ctrl + X`.

Then restart the API container:
```bash
sudo docker compose up -d api
```

---

## Step 5: Verify Your Live Public Endpoints

In your web browser, navigate to your EC2 public IP:

* **Interactive API Documentation (Swagger)**:
  `http://<YOUR-EC2-PUBLIC-IP>:8000/docs`

* **MLflow UI & Model Registry**:
  `http://<YOUR-EC2-PUBLIC-IP>:5001`

* **Health Check**:
  `http://<YOUR-EC2-PUBLIC-IP>:8000/api/health`

### Test Live Prediction via curl:
```bash
curl -s -X POST http://<YOUR-EC2-PUBLIC-IP>:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "AWS_PROD_USER",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": false,
    "dependents": false,
    "tenure": 2,
    "phone_service": true,
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": true,
    "payment_method": "Electronic check",
    "monthly_charges": 85.0,
    "total_charges": 170.0
  }'
```

---

## Step 6: Maintain Continuous Uptime

The Docker containers are configured with `restart: unless-stopped`:
* If the EC2 instance reboots or stops/starts, all containers (`postgres`, `mlflow`, and `api`) will **automatically restart** and maintain their state.
* Model artifacts and SQLite databases are safely persisted in mounted volumes (`./models`, `./mlruns`, `./mlflow.db`).
