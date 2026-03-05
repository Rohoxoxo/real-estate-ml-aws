# 🏠 Bengaluru House Price Estimator
### End-to-End ML + AWS Serverless Portfolio Project

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Site-0969da?style=for-the-badge)](https://d1py9yte5490xy.cloudfront.net)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/Rohoxoxo/real-estate-ml-aws/actions)
[![AWS](https://img.shields.io/badge/Deployed%20on-AWS-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![scikit-learn](https://img.shields.io/badge/Model-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

> A fully serverless, production-grade machine learning application that predicts property prices across Bengaluru, India. Built to demonstrate real-world ML engineering and AWS cloud skills.

---

## 🌐 Live Demo

**→ [https://d1py9yte5490xy.cloudfront.net](https://d1py9yte5490xy.cloudfront.net)**

Enter a property's area type, location, square footage, BHK, bathrooms, and balconies — get an instant ML-powered price estimate in Indian Lakhs and USD.

---

## 🏗️ Architecture

```
Browser
  │
  ▼
CloudFront (HTTPS + CDN)          ← Edge caching, HTTP→HTTPS redirect
  │
  ▼
S3 (Private Frontend Bucket)      ← index.html served via OAC (no public access)
  │
  ▼
API Gateway (POST /predict)       ← CORS-enabled REST endpoint
  │
  ▼
Lambda (Docker Container)         ← Runs inside private VPC subnet
  │           │
  ▼           ▼
S3 (Model    DynamoDB             ← Model artifacts (private) + prediction logs
 Artifacts)   │
              ▼
           CloudWatch             ← Dashboard + alarms + SNS email alerts
```

**Full flow:** `Browser → CloudFront → S3 Frontend → API Gateway → Lambda (VPC) → S3 Model → DynamoDB → CloudWatch`

---

## ✅ Features

- **Instant price prediction** powered by a HistGradientBoosting ML model
- **Fully serverless** — no EC2, no always-on servers, scales to zero
- **Private & secure** — S3 buckets use OAC (no public URLs), Lambda runs in a private VPC subnet
- **Prediction logging** — every request is stored in DynamoDB with timestamp, inputs, and predicted price
- **Monitoring & alerting** — CloudWatch dashboard + SNS email alerts for Lambda errors and slow responses
- **Auto-deploy CI/CD** — GitHub Actions deploys Lambda and frontend automatically on every push to `main`
- **Dual currency output** — results shown in ₹ Lakhs and USD equivalent

---

## 🤖 ML Model

| Detail | Value |
|---|---|
| Dataset | Bengaluru_House_Data.csv (~13,000 rows) |
| Algorithm | HistGradientBoostingRegressor (scikit-learn 1.4.2) |
| Target | Price in Indian Lakhs |
| Location encoding | Target encoding via `location_encoding.pkl` |
| Area type encoding | One-hot encoding (`drop_first=True`) |
| Artifacts | `hgb_model.pkl`, `model_columns.pkl`, `location_encoding.pkl` |

The model is stored as `.pkl` artifacts in a **private S3 bucket** and loaded by Lambda at cold start.

---

## ☁️ AWS Infrastructure

| Service | Role |
|---|---|
| **CloudFront** | HTTPS CDN, HTTP→HTTPS redirect, OAC for private S3 |
| **S3** (frontend) | Hosts `index.html` — private, served only via CloudFront |
| **S3** (model) | Stores ML model artifacts — private, accessed only by Lambda |
| **API Gateway** | REST endpoint — `POST /predict` with CORS enabled |
| **Lambda** | Docker container running prediction logic (1024MB, 3min timeout) |
| **ECR** | Stores the Lambda Docker image |
| **VPC** | Lambda runs in private subnet with NAT Gateway + VPC Endpoints for S3 & DynamoDB |
| **DynamoDB** | Logs every prediction (inputs, price, timestamp, request ID) |
| **CloudWatch** | Dashboard + 2 alarms (error rate, duration) |
| **SNS** | Email alerts when alarms trigger |
| **GitHub Actions** | CI/CD — auto-deploys Lambda image and frontend on push to `main` |

---

## 🚀 CI/CD Pipeline

Every push to `main` automatically:
1. Builds the Docker image for `linux/amd64`
2. Pushes the image to ECR
3. Updates the Lambda function with the new image
4. Uploads `index.html` to S3
5. Invalidates the CloudFront cache

No manual deployment steps needed.

---

## 📡 API Reference

**Endpoint:** `POST https://vu70cn63qj.execute-api.us-east-1.amazonaws.com/predict`

**Request body:**
```json
{
  "area_type": "Super built-up Area",
  "location": "Whitefield",
  "total_sqft": 1200,
  "bath": 2,
  "balcony": 1,
  "BHK": 2
}
```

**Response:**
```json
{
  "ok": true,
  "predicted_price_lakhs": 65.33,
  "request_id": "uuid-here"
}
```

---

## 🗂️ Project Structure

```
real-estate-ml-aws/
├── deployment/
│   ├── lambda_handler.py       # Lambda function (prediction + DynamoDB logging)
│   ├── Dockerfile              # Docker image for Lambda (python:3.11)
│   └── index.html              # Frontend (deployed to S3)
├── notebooks/
│   └── train_model.ipynb       # Model training + artifact export
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions CI/CD pipeline
└── README.md
```

---

## 🔧 Local Development

### Prerequisites
- AWS CLI configured (`aws configure`)
- Docker Desktop
- Python 3.11

### Run prediction locally
```bash
# Install dependencies
pip install scikit-learn==1.4.2 numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 joblib==1.4.2 boto3

# Run lambda handler directly
python deployment/lambda_handler.py
```

### Build & push Docker image (PowerShell)
```powershell
# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 241706223312.dkr.ecr.us-east-1.amazonaws.com

# Build and push (single line — PowerShell does not support backslash continuation)
docker buildx build --platform linux/amd64 --provenance=false -t 241706223312.dkr.ecr.us-east-1.amazonaws.com/real-estate-ml:latest --push .
```

### Deploy frontend manually
```powershell
aws s3 cp deployment/index.html s3://real-estate-frontend-rc/index.html; aws cloudfront create-invalidation --distribution-id EA1BPKGZSVI28 --paths "/*"
```

---

## 📊 Completed Milestones

| Step | Description | Status |
|---|---|---|
| ML Model | Trained HistGradientBoostingRegressor, saved 3 artifacts to S3 | ✅ Done |
| S3 + CloudFront | Private S3 frontend, CloudFront HTTPS with OAC | ✅ Done |
| Lambda Container | Docker image on ECR, Lambda with 1024MB / 3min timeout | ✅ Done |
| API Gateway | POST /predict endpoint with CORS enabled | ✅ Done |
| DynamoDB Logging | Logs every prediction with Decimal-safe floats | ✅ Done |
| CloudWatch | Dashboard + 2 alarms + SNS email alerts | ✅ Done |
| VPC Isolation | Lambda in private subnet, NAT Gateway, VPC Endpoints | ✅ Done |
| CI/CD Pipeline | GitHub Actions auto-deploy on push to main | ✅ Done |

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `Docker` · `AWS Lambda` · `API Gateway` · `S3` · `CloudFront` · `DynamoDB` · `CloudWatch` · `ECR` · `VPC` · `GitHub Actions` · `HTML/CSS/JS`

---

## 👤 Author

**Rohit Chandel**
- GitHub: [@Rohoxoxo](https://github.com/Rohoxoxo)
- Project repo: [real-estate-ml-aws](https://github.com/Rohoxoxo/real-estate-ml-aws)
