FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY backend/ ./backend/
COPY models/ ./models/
COPY train_model.py .

# Set working directory to backend
WORKDIR /app/backend

ENV PYTHONPATH=/app/backend
ENV MODEL_PATH=/app/models/best_lightgbm_churn.pkl

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
