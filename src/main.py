import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production MLOps Inference Service supporting Champion & Challenger models",
    version="1.0.0",
)

# ----------------------------------------------------
# 1. Load Artifacts on Service Startup
# ----------------------------------------------------
MODELS_DIR = "models"

try:
    champion_model = joblib.load(os.path.join(MODELS_DIR, "champion_model.pkl"))
    challenger_model = joblib.load(os.path.join(MODELS_DIR, "challenger_model.pkl"))
    challenger_model_v2 = joblib.load(os.path.join(MODELS_DIR, "model_svc.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("[INFO] All model artifacts successfully loaded!")
except Exception as e:
    print(f" Warning: Model artifacts missing or failed to load: {e}")
    champion_model = None
    challenger_model = None
    challenger_model_v2 = None


# ----------------------------------------------------
# 2. Define Pydantic Schema for Input Request
# ----------------------------------------------------
class PatientData(BaseModel):
    age: int = Field(..., example=52)
    sex: str = Field(..., example="Male")  # "Male" or "Female"
    resting_bp_systolic: int = Field(..., example=125)
    resting_bp_diastolic: int = Field(..., example=80)
    cholesterol_total: int = Field(..., example=210)
    hdl: int = Field(..., example=50)
    ldl: int = Field(..., example=130)
    triglycerides: int = Field(..., example=150)
    fasting_blood_sugar: int = Field(..., example=100)
    hba1c: float = Field(..., example=5.6)
    bmi: float = Field(..., example=27.4)
    resting_heart_rate: int = Field(..., example=72)
    max_heart_rate_achieved: int = Field(..., example=160)
    chest_pain_type: str = Field(..., example="Asymptomatic")
    exercise_induced_angina: bool = Field(..., example=False)
    st_depression: float = Field(..., example=1.0)
    family_history: bool = Field(..., example=True)
    smoker_status: str = Field(..., example="Never")
    alcohol_units_per_week: float = Field(..., example=2.0)
    exercise_minutes_per_week: int = Field(..., example=150)
    sleep_hours: float = Field(..., example=7.5)
    stress_score: float = Field(..., example=35.0)
    wearable_owner: bool = Field(..., example=True)
    daily_steps: int = Field(..., example=8500)
    diet_quality_score: float = Field(..., example=75.0)


def preprocess_patient_input(data: PatientData) -> pd.DataFrame:
    """Helper to transform raw input JSON into matching encoded columns."""
    input_dict = data.model_dump()
    df_raw = pd.DataFrame([input_dict])

    # Categorical encoding matching training phase
    categorical_cols = ["sex", "chest_pain_type", "smoker_status"]
    df_encoded = pd.get_dummies(df_raw, columns=categorical_cols, drop_first=True)

    # Reindex to guarantee exact order and missing dummy columns
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_aligned


# ----------------------------------------------------
# 3. Define API Endpoints
# ----------------------------------------------------
@app.get("/health")
def health_check():
    """Health check endpoint for GCP Cloud Run / Kubernetes readiness probes."""
    if champion_model is None or challenger_model is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    return {"status": "healthy", "service": "HeartDiseasePredictionAPI"}


@app.post("/predict/champion")
def predict_champion(patient: PatientData):
    """Prediction endpoint using CHAMPION Model (Random Forest)."""
    if champion_model is None:
        raise HTTPException(status_code=500, detail="Champion model unavailable")

    df_input = preprocess_patient_input(patient)
    prediction = int(champion_model.predict(df_input)[0])
    probability = float(champion_model.predict_proba(df_input)[0][1])

    return {
        "model": "Champion (Random Forest)",
        "prediction": prediction,
        "risk_status": "High Risk" if prediction == 1 else "Low Risk",
        "probability": round(probability, 4),
    }


@app.post("/predict/challenger")
def predict_challenger(patient: PatientData):
    """Prediction endpoint using CHALLENGER Model (Logistic Regression)."""
    if challenger_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Challenger model unavailable")

    df_input = preprocess_patient_input(patient)
    df_scaled = scaler.transform(df_input)

    prediction = int(challenger_model.predict(df_scaled)[0])
    probability = float(challenger_model.predict_proba(df_scaled)[0][1])

    return {
        "model": "Challenger (Logistic Regression)",
        "prediction": prediction,
        "risk_status": "High Risk" if prediction == 1 else "Low Risk",
        "probability": round(probability, 4),
    }

@app.post("/predict/challenger_v2")
def predict_challenger(patient: PatientData):
    """Prediction endpoint using CHALLENGER Model V2 (SVM)."""
    if challenger_model_v2 is None or scaler is None:
        raise HTTPException(status_code=500, detail="Challenger model unavailable")

    df_input = preprocess_patient_input(patient)
    df_scaled = scaler.transform(df_input)

    prediction = int(challenger_model_v2.predict(df_scaled)[0])
    probability = float(challenger_model_v2.predict_proba(df_scaled)[0][1])

    return {
        "model": "Challenger_V2 (SVM)",
        "prediction": prediction,
        "risk_status": "High Risk" if prediction == 1 else "Low Risk",
        "probability": round(probability, 4),
    }