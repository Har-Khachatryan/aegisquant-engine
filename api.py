"""
AegisQuant – Stable production API (global feature importances).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aegis_api")

ARTIFACT_PATH = "aegis_quant_artifacts.pkl"

REASON_MAP = {
    "f0": "Low account balance",
    "f1": "Negative balance velocity (outflows)",
    "f2": "Elevated market pain index",
    "f3": "Sudden drop in login frequency",
    "f4": "Aggressive trading profile",
    "f5": "Balanced trading profile",
    "f6": "Conservative trading profile",
}

# ── Pydantic models ──
class ClientFeatures(BaseModel):
    account_balance:   float = Field(..., gt=0)
    balance_velocity:  float = Field(..., ge=0, le=5)
    market_pain_index: float = Field(..., ge=0, le=1)
    login_freq_drop:   float = Field(..., ge=0, le=5)
    avg_holding_days:  float = Field(..., ge=1)
    crypto_ratio:      float = Field(..., ge=0, le=1)
    tech_stocks_ratio: float = Field(..., ge=0, le=1)

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_drivers: List[str]
    model_version: str

# ── Predictor ──
class ChurnPredictor:
    def __init__(self, artifact_path: str) -> None:
        if not os.path.exists(artifact_path):
            from churn_model import run_training_pipeline
            run_training_pipeline()
        artifacts = joblib.load(artifact_path)
        self.pipeline = artifacts["pipeline"]
        self.model = self.pipeline.named_steps["xgb_churn"]
        self.booster = self.model.get_booster()
        # Имена признаков из бустера (если None — создаём f0..fN)
        feature_names = self.booster.feature_names
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(self.model.n_features_in_)]
        self.feature_names = feature_names
        self.importances = self.model.feature_importances_
        log.info("✅ Predictor ready (global feature importances).")

    def predict_and_explain(self, features: ClientFeatures) -> PredictionResponse:
        # 1. DataFrame для пайплайна
        input_df = pd.DataFrame([[
            features.avg_holding_days,
            features.crypto_ratio,
            features.tech_stocks_ratio,
            features.account_balance,
            features.balance_velocity,
            features.market_pain_index,
            features.login_freq_drop,
        ]], columns=[
            "avg_holding_days", "crypto_ratio", "tech_stocks_ratio",
            "account_balance", "balance_velocity", "market_pain_index", "login_freq_drop"
        ])

        # 2. Вероятность оттока
        churn_prob = float(self.pipeline.predict_proba(input_df)[0, 1])

        # 3. Топ-2 драйвера по глобальной важности
        pairs = list(zip(self.feature_names, self.importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        risk_drivers = [REASON_MAP.get(name, f"Feature '{name}'") for name, _ in pairs[:2]]

        return PredictionResponse(
            churn_probability=round(churn_prob, 4),
            risk_drivers=risk_drivers,
            model_version="aegis_quant_v3.1_cross_pollination",
        )

# ── FastAPI app ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🔧 Loading AegisQuant model...")
    predictor = ChurnPredictor(ARTIFACT_PATH)
    app.state.predictor = predictor
    log.info("✅ Model loaded.")
    yield
    log.info("Shutdown.")

app = FastAPI(title="AegisQuant Churn Prediction API", version="3.1.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/predict")
async def predict_get():
    return {"message": "Send a POST request with JSON body. See /docs"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_post(features: ClientFeatures):
    if not hasattr(app.state, "predictor"):
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return app.state.predictor.predict_and_explain(features)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)