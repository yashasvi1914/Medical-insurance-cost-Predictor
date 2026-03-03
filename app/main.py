"""
Medical Insurance Cost Prediction — FastAPI Backend
Author: Senior Data Scientist
Description: REST API serving a trained ML model to predict insurance charges.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# ── FastAPI imports ──────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")


# ── Load model bundle ────────────────────────────────────────────────────────
MODEL_PATH = BASE_DIR / "model" / "model_bundle.pkl"
RESULTS_PATH = BASE_DIR / "model" / "model_results.json"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model       = bundle["model"]
scaler      = bundle["scaler"]
le_sex      = bundle["le_sex"]
le_smoker   = bundle["le_smoker"]
le_region   = bundle["le_region"]
use_scaler  = bundle["use_scaler"]
best_name   = bundle["best_model_name"]

with open(RESULTS_PATH) as f:
    model_results = json.load(f)

print(f"✅ Loaded model: {best_name}")


# ── Pydantic schema ──────────────────────────────────────────────────────────
class InsuranceInput(BaseModel):
    age:      int   = Field(..., ge=18, le=100,   description="Age of the individual")
    sex:      str   = Field(...,                  description="'male' or 'female'")
    bmi:      float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    children: int   = Field(..., ge=0,   le=10,   description="Number of dependents")
    smoker:   str   = Field(...,                  description="'yes' or 'no'")
    region:   str   = Field(...,                  description="southwest|southeast|northwest|northeast")

    @validator("sex")
    def validate_sex(cls, v):
        if v.lower() not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")
        return v.lower()

    @validator("smoker")
    def validate_smoker(cls, v):
        if v.lower() not in ("yes", "no"):
            raise ValueError("smoker must be 'yes' or 'no'")
        return v.lower()

    @validator("region")
    def validate_region(cls, v):
        valid = {"southwest", "southeast", "northwest", "northeast"}
        if v.lower() not in valid:
            raise ValueError(f"region must be one of {valid}")
        return v.lower()


class PredictionResponse(BaseModel):
    predicted_charge: float
    model_used:       str
    r2_score:         float
    rmse:             float
    input_summary:    dict


# ── App setup ────────────────────────────────────────────────────────────────
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Medical Insurance Cost Predictor",
        description="Predict annual medical insurance charges using ML regression models.",
        version="1.0.0",
    )

    # Static files & templates
    app.mount("/static",    StaticFiles(directory=str(BASE_DIR / "static")),    name="static")
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    # ── Routes ───────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Serve the frontend UI."""
        return templates.TemplateResponse("index.html", {
            "request":    request,
            "model_name": best_name,
            "r2_score":   model_results["results"][best_name]["r2"],
        })

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Serve the EDA dashboard."""
        return templates.TemplateResponse("dashboard.html", {
            "request":      request,
            "model_results": model_results,
        })

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(data: InsuranceInput):
        """
        Predict insurance cost.
        Accepts JSON with: age, sex, bmi, children, smoker, region
        Returns predicted_charge in USD.
        """
        try:
            sex_enc    = le_sex.transform([data.sex])[0]
            smoker_enc = le_smoker.transform([data.smoker])[0]
            region_enc = le_region.transform([data.region])[0]

            features = np.array([[data.age, sex_enc, data.bmi,
                                   data.children, smoker_enc, region_enc]])

            if use_scaler:
                features = scaler.transform(features)

            prediction = float(model.predict(features)[0])
            prediction = max(0.0, round(prediction, 2))

            return PredictionResponse(
                predicted_charge = prediction,
                model_used       = best_name,
                r2_score         = model_results["results"][best_name]["r2"],
                rmse             = model_results["results"][best_name]["rmse"],
                input_summary    = data.dict(),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "model": best_name}

    @app.get("/model-info")
    async def model_info():
        """Return all model comparison metrics."""
        return model_results


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print("Install FastAPI first: pip install fastapi uvicorn")
