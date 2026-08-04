# ❤️ Heart Disease Risk Prediction Platform
### End-to-End Machine Learning Operations (MLOps) Pipeline with Real-Time Monitoring

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-MLOps-0194E2)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Cloud_Run-4285F4?logo=googlecloud)
![Evidently AI](https://img.shields.io/badge/Evidently-Data_Drift-red)

</p>

---

# 📌 Problem Statement

Cardiovascular disease is one of the leading causes of death worldwide, making **early risk prediction** essential for improving patient outcomes.

While Machine Learning models can accurately predict heart disease using historical clinical data, **building a model alone is not enough**. In production, data continuously changes due to shifts in patient demographics, hospital procedures, and data collection methods. These changes can gradually reduce prediction accuracy without being immediately visible.

The objective of this project is to build a **production-ready MLOps pipeline** that not only predicts heart disease risk but also addresses real-world operational challenges by enabling:

- Reliable model deployment
- Experiment tracking
- Model version management
- Automated deployment
- Production logging
- Continuous monitoring
- Data drift detection
- Cloud-native scalability

This project demonstrates how Machine Learning systems are built, deployed, monitored, and maintained in real production environments.

---

# 🎯 Project Objectives

- Develop an accurate Heart Disease Prediction model.
- Deploy the model as a scalable REST API.
- Automate deployment using CI/CD.
- Track experiments and manage model versions using MLflow.
- Monitor production inference logs.
- Detect data drift using Evidently AI.
- Build a production-ready cloud-native ML application.

---

# 🚀 Live Application

## Prediction API

https://heart-disease-api-235131289843.us-central1.run.app

## Interactive API Documentation

https://heart-disease-api-235131289843.us-central1.run.app/docs

---

# 📂 GitHub Repository

https://github.com/amoghjainda2000-coder/heart-disease-mlops

---

# 🌟 Solution Overview

This project implements a complete **Machine Learning Operations (MLOps)** workflow that transforms a trained machine learning model into a reliable production service.

The solution integrates:

- Machine Learning Model Training
- Experiment Tracking
- Feature Store
- Model Registry
- Model Versioning
- REST API Development
- Docker Containerization
- Google Cloud Deployment
- CI/CD Automation
- Production Logging
- Data Drift Monitoring

Unlike traditional ML projects that stop after model training, this project focuses on the **entire lifecycle of a production ML system.**

---

# ✨ Key Features

## Machine Learning

- Random Forest Classifier
- Feature Engineering
- Data Preprocessing
- Probability-based Prediction
- Serialized Production Model

---

## MLOps

- MLflow Experiment Tracking
- Model Registry
- Model Versioning
- Feature Store Integration
- Automated CI/CD Pipeline
- Production Deployment Workflow

---

## API & Cloud Deployment

- FastAPI REST API
- OpenAPI / Swagger Documentation
- Docker Containerization
- Google Cloud Run Deployment
- Serverless Auto Scaling
- HTTPS Endpoint

---

## Production Monitoring

Every prediction request generates structured logs containing:

- Request ID
- Timestamp
- Input Features
- Prediction
- Prediction Probability
- Response Latency
- API Status

These logs are stored in **Google Cloud Logging** for operational monitoring and debugging.

---

## Data Drift Detection

Incoming production data is continuously compared against the original training dataset using Evidently AI.

The monitoring pipeline identifies:

- Feature Drift
- Distribution Drift
- Missing Values
- Data Quality Issues
- Statistical Distribution Changes

Interactive HTML reports provide detailed insights into production model health.

---

# 🏗️ Solution Architecture

```
                  User / Client
                        │
                        ▼
              FastAPI Prediction API
               (Google Cloud Run)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
 Heart Disease Prediction      Structured JSON Logs
                                        │
                                        ▼
                             Google Cloud Logging
                                        │
                                        ▼
                          Production Inference Data
                                        │
                                        ▼
                          Evidently AI Monitoring
                                        │
                                        ▼
                           Data Drift HTML Reports
```

---

# ⚙️ Technology Stack

| Layer | Technology |
|--------|------------|
| Language | Python |
| Machine Learning | Scikit-Learn |
| Model | Random Forest |
| API | FastAPI |
| Experiment Tracking | MLflow |
| Model Registry | MLflow |
| Feature Store | MLflow |
| Deployment | Google Cloud Run |
| Containerization | Docker |
| Monitoring | Evidently AI |
| Logging | Google Cloud Logging |
| CI/CD | GitHub Actions |
| Data Processing | Pandas, NumPy |

---

# 🔄 End-to-End Workflow

```
Training Dataset
       │
       ▼
Data Validation
       │
       ▼
Feature Engineering
       │
       ▼
Model Training
       │
       ▼
MLflow Experiment Tracking
       │
       ▼
Model Registry
       │
       ▼
Docker Image
       │
       ▼
CI/CD Pipeline
       │
       ▼
Google Cloud Run
       │
       ▼
REST API
       │
       ▼
Prediction Requests
       │
       ▼
Google Cloud Logging
       │
       ▼
Evidently AI
       │
       ▼
Data Drift Detection
```

---

# 📡 REST API Example

### Request

```bash
curl -X POST \
'https://heart-disease-api-235131289843.us-central1.run.app/predict' \
-H 'Content-Type: application/json' \
-d '{
"age":58,
"sex":"Male",
"resting_bp_systolic":140,
"cholesterol_total":240,
"bmi":29.5,
"max_heart_rate_achieved":150
}'
```

### Response

```json
{
  "request_id":"283f2cb3",
  "prediction":1,
  "probability":0.823,
  "model_used":"Random Forest"
}
```

---

# 📊 Monitoring Pipeline

Run the drift detection pipeline:

```bash
python src/monitoring/check_data_drift.py
```

Output:

```
drift_report.html
```

The report provides feature-wise statistical comparisons between training data and production inference data.

---

# 💼 Skills Demonstrated

- Machine Learning
- MLOps
- Production ML Systems
- MLflow
- Feature Store
- Model Registry
- Model Versioning
- FastAPI
- REST APIs
- Docker
- Google Cloud Platform
- Cloud Run
- GitHub Actions
- CI/CD
- Cloud Logging
- Evidently AI
- Data Drift Detection
- Production Monitoring
- Python
- Pandas
- NumPy
- Scikit-Learn

---

# 🎯 Business Impact

This project demonstrates how organizations can operationalize Machine Learning by building reliable, scalable, and observable AI systems.

It showcases enterprise-grade MLOps practices including automated deployment, experiment management, model governance, monitoring, and drift detection—capabilities that help maintain prediction quality and support long-term production reliability.

---

# 👨‍💻 Author

**Amogh Jain**

Senior Software Engineer(Machine Learning and Gen AI) • MBA (Business Analytics)

GitHub:
https://github.com/amoghjainda2000-coder

---

⭐ If you found this project useful, consider giving it a star.
