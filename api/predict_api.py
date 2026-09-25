import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

# Security (C9 / OWASP best practices)
API_KEY_NAME = "X-API-Key"
SERVER_API_KEY = os.getenv("DATA_API_KEY", "simplon_secure_token_2024")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

MODEL_PATH = os.path.join("models", "match_predictor.joblib")

app = FastAPI(
    title="FootPredict ML Inference API",
    description="REST API exposing the football match prediction model (Simplon RNCP37827 - C9).",
    version="1.0.0",
    docs_url="/docs"
)

# Load the model on startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == SERVER_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Unauthorized: Invalid API Key"
    )

# Input Schema for Validation
class PredictionRequest(BaseModel):
    home_points: float
    away_points: float
    home_goal_diff: float
    away_goal_diff: float

@app.post("/predict", tags=["Prediction"])
def predict_outcome(payload: PredictionRequest, auth: str = Depends(get_api_key)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")

    # Calculate the derived features exactly as we did in training
    diff_points = payload.home_points - payload.away_points
    diff_goal_diff = payload.home_goal_diff - payload.away_goal_diff

    input_df = pd.DataFrame([{
        "home_points": payload.home_points,
        "away_points": payload.away_points,
        "home_goal_diff": payload.home_goal_diff,
        "away_goal_diff": payload.away_goal_diff,
        "diff_points": diff_points,
        "diff_goal_diff": diff_goal_diff
    }])

    # Generate predictions
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    # Map probabilities to class names
    prob_dict = {classes[i]: round(float(probabilities[i]), 2) for i in range(len(classes))}

    return {
        "predicted_outcome": prediction,
        "probabilities": prob_dict
    }