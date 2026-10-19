from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Heart Disease Prediction API",
    description="REST API for serving heart disease classification model predictions.",
    version="1.0.0"
)

# Load model and scaler artifacts
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or Scaler artifact missing. Re-run training pipeline.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class PatientData(BaseModel):
    age: float = Field(..., example=63.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=1.0)
    trestbps: float = Field(..., example=145.0)
    chol: float = Field(..., example=233.0)
    fbs: float = Field(..., example=1.0)
    restecg: float = Field(..., example=2.0)
    thalach: float = Field(..., example=150.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=2.3)
    slope: float = Field(..., example=3.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=6.0)

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Heart Disease MLOps API"}

@app.post("/predict")
def predict(data: PatientData):
    try:
        features = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]])
        
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] if hasattr(model, "predict_proba") else None
        
        return {
            "prediction": int(prediction),
            "label": "Heart Disease Present" if prediction == 1 else "No Heart Disease",
            "probability": float(probability) if probability is not None else "N/A"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))