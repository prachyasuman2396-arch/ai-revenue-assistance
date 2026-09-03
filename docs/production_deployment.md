# Production Architecture & MLOps Lifecycle Blueprint

## Overview
This document details the production cloud deployment architecture (AWS) and ongoing MLOps operational lifecycle for the **AI Revenue Assistance** platform.

---

## 1. AWS Cloud Architecture (Phase 19)

```
                            [ Route 53 DNS ]
                                   │
                                   ▼
                    [ Application Load Balancer (ALB) ]
                            (TLS 1.3 Termination)
                                   │
                ┌──────────────────┴──────────────────┐
                │ VPC Public Subnets (NAT Gateways)   │
                └──────────────────┬──────────────────┘
                                   │
     ┌─────────────────────────────┴─────────────────────────────┐
     │ VPC Private Application Subnet (Isolated)                 │
     │                                                           │
     │   ┌───────────────────────────────────────────────────┐   │
     │   │ AWS ECS Fargate Cluster (Auto-Scaling Service)   │   │
     │   │  ├─ Task Replica 1 [FastAPI + Predictor]          │   │
     │   │  ├─ Task Replica 2 [FastAPI + Predictor]          │   │
     │   │  └─ Task Replica N [FastAPI + Predictor]          │   │
     │   └─────────────────────────┬─────────────────────────┘   │
     │                             │                             │
     │   ┌─────────────────────────┴─────────────────────────┐   │
     │   │ AWS RDS PostgreSQL (Multi-AZ Replica)             │   │
     │   │  ├─ raw_customer_data                             │   │
     │   │  ├─ clean_customer_features                       │   │
     │   │  ├─ customer_predictions                          │   │
     │   │  └─ customer_recommendations                      │   │
     │   └───────────────────────────────────────────────────┘   │
     └─────────────────────────────┬─────────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │ AWS S3 + Secrets Manager + KMS      │
                │  ├─ S3: Model Registry Artifacts    │
                │  ├─ Secrets: DB Credentials         │
                │  └─ KMS: Envelope Encryption        │
                └─────────────────────────────────────┘
```

### 1.1 Key Infrastructure Components
1. **Compute (AWS ECS Fargate)**:
   - *Why Fargate over EKS or EC2?*: Serverless container execution eliminates Kubernetes control plane maintenance overhead while providing instant horizontal auto-scaling (scaling up on CPU > 70% or ALB Target Request Count > 500 req/min).
2. **Database (AWS RDS PostgreSQL 16 Multi-AZ)**:
   - Automated synchronous replication across two Availability Zones for 99.95% SLA. Encrypted with AWS KMS.
3. **Model Storage (AWS S3 & MLflow)**:
   - S3 bucket versioning for immutable champion artifacts with S3 Object Lock for regulatory auditability.
4. **Security & Least Privilege**:
   - Containers run in private subnets with egress routed strictly through NAT Gateways.
   - Database credentials managed via AWS Secrets Manager with automatic 30-day credential rotation.
   - Zero hardcoded credentials in environment variables or container layers.

### 1.2 Zero-Downtime Deployment & Rollback Strategy
* **Blue/Green Deployments with AWS CodeDeploy**:
  - New container revisions receive 10% of production traffic for 10 minutes (Canary testing).
  - Automated CloudWatch alarms monitor 5xx error rate and p99 latency (< 150ms).
  - If 5xx errors breach 0.5%, CodeDeploy automatically rolls back to the previous stable task definition in < 30 seconds.
* **Model Rollback**:
  - MLflow Model Registry tracks aliases (`@champion`, `@challenger`). Reverting a model regression requires flipping the pointer to the previous version without redeploying container code.

---

## 2. Production MLOps & Lifecycle Management (Phase 20)

### 2.1 Data Drift & Concept Drift Detection
* **Data Drift (Input Covariate Shift $P(X)$)**:
  - Daily batch job calculates **Population Stability Index (PSI)** and **Wasserstein Distance** on numerical features (`monthly_charges`, `tenure`) and **Chi-Square Goodness-of-Fit** on categorical features (`contract`, `payment_method`).
  - *Threshold*: $\text{PSI} \ge 0.20$ triggers an automated alert indicating significant feature distribution shift.
* **Concept Drift (Target Conditional Shift $P(Y|X)$)**:
  - Customer churn ground truth has a 30-to-60-day lag (a customer must be inactive for a billing cycle to be labeled churned).
  - Monitored by matching delayed billing cancellations against historical predictions to compute rolling **PR-AUC decay** and **Brier Score Calibration Error**.

### 2.2 Model Retraining Strategy
* **Cadence**: Scheduled monthly retraining or event-driven retraining triggered when data drift $\text{PSI} > 0.25$.
* **Shadow Deployment (Dark Launch)**:
  - New candidate models run in shadow mode alongside the production champion for 7 days, receiving mirrored live traffic without serving user-facing recommendations.
  - Candidate model must demonstrate superior PR-AUC on delayed ground truth before promotion to `@champion`.

### 2.3 Production Observability & Monitoring
* **Telemetry Metrics (Prometheus / CloudWatch)**:
  - `inference_latency_ms_bucket` (p50, p95, p99)
  - `prediction_churn_probability_histogram` (detects real-time prediction distribution shifts)
  - `recommendation_playbook_counter` (tracks distribution of issued playbooks)
* **Structured JSON Logging**:
  - Every API call outputs structured JSON logs containing `request_id`, `customer_id`, `model_version`, `latency_ms`, and `http_status`.

### 2.4 Incident Response Playbook
| Severity | Trigger | Immediate Containment Action | Root Cause Investigation |
| :--- | :--- | :--- | :--- |
| **SEV-1 (Critical)** | API 5xx rate > 2% or p99 latency > 1,000ms | Rollback ALB to previous stable container revision immediately. | Inspect database connection pool exhaustion or unhandled schema exception. |
| **SEV-2 (High)** | Disproportionate churn predictions (> 80% flagged HIGH risk) | Revert MLflow model registry alias to previous champion model. | Check for unannounced upstream billing schema change or currency formatting drift. |
| **SEV-3 (Medium)** | Daily drift alert ($\text{PSI} > 0.20$) | Review drift dashboard; schedule retraining pipeline on latest 90-day window. | Analyze whether marketing launched a new promotional pricing campaign altering `monthly_charges`. |
