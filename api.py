"""
AegisQuant — Production REST API (v4.0)

Upgrade v3.2 → v4.0
────────────────────
  - Request body is the bank-customer ClientFeatures contract (config.py).
  - Inference and SHAP explanations come from inference.ChurnPredictor, the
    same component the dashboard and the optimisation engine use.
  - The response adds the risk tier, the retention decision (probability vs the
    threshold learned in training), the investor profile and detailed drivers.
  - Handlers are plain `def`: SHAP and XGBoost are CPU-bound, so FastAPI runs
    them in its thread pool instead of blocking the event loop (v3.2 used
    `async def`, which serialised every request).

Endpoints
─────────
  GET  /            → redirect to the interactive docs
  GET  /health      → liveness + model version, test AUC, threshold
  GET  /model-card  → training metadata and hold-out metrics
  POST /predict     → churn probability, risk drivers (local SHAP), profile
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from config import MODEL_VERSION, ClientFeatures
from feature_cross_pollination import load_evaluation
from inference import ChurnPredictor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aegis_api")


# ═════════════════════════════════════════════════════════════════════════════
# Response models
# ═════════════════════════════════════════════════════════════════════════════
class Driver(BaseModel):
    feature: str                 # business concept (a raw input + its engineered features)
    label: str
    value: str                   # the customer's own value, formatted
    contribution: float          # summed SHAP value in log-odds; > 0 pushes churn up
    text: str


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_tier: str               # Low / Elevated / High
    retention_action: bool       # probability ≥ decision threshold
    decision_threshold: float
    investor_profile: str        # conservative / balanced / aggressive
    segment_id: int
    risk_drivers: List[str]      # top-2 plain-language drivers (v3.x field)
    drivers: List[Driver]        # top-5 signed contributions
    base_value: float
    model_version: str


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI application
# ═════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the predictor once at startup (trains first if artifacts are missing)."""
    log.info("  AegisQuant API — loading predictor...")
    app.state.predictor = ChurnPredictor()
    log.info("  AegisQuant API — predictor ready, serving requests.")
    yield
    log.info("  AegisQuant API — shutdown.")


app = FastAPI(
    title="AegisQuant — AI Portfolio Shield & Risk Engine",
    version="4.0.0",
    description=(
        "Bank-customer churn prediction trained on the Kaggle Churn Modelling dataset, "
        "with local SHAP explanations and KMeans investor profiles. "
        "POST /predict with a ClientFeatures JSON body."
    ),
    lifespan=lifespan,
)


def _predictor(request: Request) -> ChurnPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialised.")
    return predictor


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(request: Request) -> dict:
    p = _predictor(request)
    return {
        "status": "healthy",
        "version": "4.0.0",
        "model_version": p.meta.get("model_version", MODEL_VERSION),
        "trained_at": p.meta.get("trained_at"),
        "test_auc": p.meta.get("test_auc"),
        "decision_threshold": p.threshold,
    }


@app.get("/model-card")
def model_card(request: Request) -> dict:
    p = _predictor(request)
    evaluation = load_evaluation()
    return {
        "meta": p.meta,
        "test_metrics": evaluation.get("test"),
        "baseline_logistic": evaluation.get("baseline_logistic"),
        "fairness": evaluation.get("fairness"),
        "segments": evaluation.get("segments"),
    }


@app.get("/predict")
def predict_get() -> dict:
    return {"message": "POST to /predict with a JSON body. See /docs for the schema."}


@app.post("/predict", response_model=PredictionResponse)
def predict_post(features: ClientFeatures, request: Request) -> PredictionResponse:
    """Churn probability, retention decision, investor profile and local SHAP drivers."""
    return PredictionResponse(**_predictor(request).explain(features))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
