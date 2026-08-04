# End-to-End Heart Disease Risk Prediction & MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/)
[![GCP Cloud Run](https://img.shields.io/badge/GCP-Cloud_Run-4285F4)](https://cloud.google.com/run)
[![Evidently AI](https://img.shields.io/badge/Evidently_AI-Monitoring-FF6F61)](https://www.evidentlyai.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)](https://www.docker.com/)

An end-to-end Machine Learning Operations (MLOps) system designed for heart disease risk prediction. The project features automated model training, a containerized REST API deployed to **Google Cloud Run**, structured Cloud Logging for real-time inference telemetry, and continuous monitoring for **Data Drift** using **Evidently AI**.

---

## Key Features

* **RESTful Inference Engine:** Low-latency API developed with **FastAPI** to serve model predictions (`/predict`).
* **Containerization & Cloud Deployment:** Fully containerized with **Docker** and deployed on **GCP Cloud Run** for serverless scalability.
* **Production Telemetry:** Structured JSON logging integrated into GCP Cloud Logging for every prediction request (`input_features`, `predictions`, `latency_ms`).
* **Continuous Drift Monitoring:** Automated Python monitoring pipeline comparing real-time inference logs against baseline training data using statistical distance metrics (Wasserstein & Jensen-Shannon distance).
* **Automated HTML Reporting:** Generates interactive data health and feature drift dashboards using **Evidently AI**.

---

## System Architecture

```text
  [ User / Client ]
         │
         ▼
[ FastAPI App (GCP Cloud Run) ] ──(Log Stream)──► [ GCP Cloud Logging ]
         │                                                │
         ▼                                                ▼
  [ Risk Score / Prediction ]                     [ Current Logs Export (CSV/JSON) ]
                                                          │
                                                          ▼
                                              [ Evidently Monitoring Pipeline ]
                                                          │
                                                          ▼
                                             [ Interactive HTML Drift Report ]
```

---

## Data Drift & Monitoring Workflow

Production logs are continuously monitored against the SQL baseline dataset (`training_reference.csv`) to detect statistical shifts in clinical feature distributions (e.g., `bmi`, `resting_bp_systolic`, `cholesterol_total`).

### Running the Drift Detection Locally

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/heart-disease-mlops.git](https://github.com/your-username/heart-disease-mlops.git)
   cd heart-disease-mlops
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute Drift Analysis:**
   ```bash
   python src/monitoring/check_data_drift.py
   ```

4. **View Output:**
   Open the generated `drift_report_YYYYMMDD_HHMMSS.html` file in any web browser to view feature-level distribution comparisons and statistical test results.

---

## Sample API Usage

### Send a Prediction Request

```bash
curl -X 'POST' \
  'https://<your-cloud-run-url>/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 58,
    "sex": "Male",
    "resting_bp_systolic": 140,
    "cholesterol_total": 240,
    "bmi": 29.5,
    "max_heart_rate_achieved": 150
  }'
```

### Sample API Response

```json
{
  "request_id": "283f2cb3-4a68-4d7d-8c71-e8615492c9ff",
  "prediction": 1,
  "probability": 0.823,
  "model_used": "Champion (Random Forest)"
}
```

---

## Project Structure

```text
.
├── data/                  # Baseline and inference sample data
├── src/
│   ├── api/               # FastAPI endpoints & schemas
│   ├── models/            # Model training & inference logic
│   └── monitoring/        # Evidently AI data drift scripts
├── Dockerfile             # Container definition for Cloud Run deployment
├── requirements.txt       # Python dependencies
└── README.md
```
