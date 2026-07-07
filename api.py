"""
AegisQuant — Production REST API (v3.2)

Upgrade v3.1 → v3.2
────────────────────
[1] DRY / SINGLE SOURCE OF TRUTH
    ClientFeatures is imported directly from config.py.  The duplicate
    ClientFeatures model that previously lived in this file is removed.
    Both the FastAPI request body and the internal inference path now use
    the same canonical Pydantic model, including the cross-field validator
    (crypto_ratio + tech_stocks_ratio <= 1.0) and all Field bounds.

[2] LOCAL EXPLAINABILITY VIA SHAP (TreeExplainer)
    The v3.1 implementation ranked features by their global feature_importances_
    vector — the same top-2 features appeared for EVERY client regardless of
    their specific input values.  This is:
      a) Scientifically incorrect: global importance captures average tree-split
         frequency across the training set, not the marginal contribution of
         this client's feature values to their specific churn score.
      b) Legally problematic: regulators (GDPR Art. 22, SR 11-7) require
         individual explanations for automated decisions; aggregate statistics
         do not qualify.

    v3.2 fix: shap.TreeExplainer with local SHAP values
    ─────────────────────────────────────────────────────
    • shap.TreeExplainer(xgb_booster) is created ONCE at predictor init.
      TreeExplainer uses the exact tree structure (no approximation) and
      runs in O(n_leaves) per sample — nanoseconds for a 7-feature model.
    • For each request, shap_values = explainer.shap_values(X_transformed)[0]
      gives the signed contribution of each feature to this client's log-odds.
    • Top-2 features by POSITIVE SHAP value are selected (positive = drives
      churn up for this specific client), mapped via SHAP_FEATURE_REASON_MAP.
    • If fewer than 2 features have positive SHAP values, the top-2 by
      absolute SHAP magnitude are used as a fallback.

[3] SECURE ARTIFACT LOADING
    ChurnPredictor loads the XGBoost model via load_xgb_model() (native JSON)
    and the ProcessorBundle via load_processor() (sklearn-only joblib).
    No full-pipeline pickle is loaded in the API process.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
import shap
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── AegisQuant modules ────────────────────────────────────────────────────────
from config import (
    CLUSTER_FEATURES,
    CHURN_FEATURES,
    ALL_FEATURES,
    SHAP_FEATURE_REASON_MAP,
    ClientFeatures,      # single canonical model — no duplicate definition here
    PROFILE_RESOLVER_PATH,
    ARTIFACT_PATH,
)
from feature_cross_pollination import (
    ProcessorBundle,
    load_xgb_model,
    load_processor,
    load_artifact_meta,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aegis_api")


# ═════════════════════════════════════════════════════════════════════════════
# Response model
# ═════════════════════════════════════════════════════════════════════════════
class PredictionResponse(BaseModel):
    churn_probability: float
    risk_drivers:      List[str]
    model_version:     str


# ═════════════════════════════════════════════════════════════════════════════
# ChurnPredictor
# ═════════════════════════════════════════════════════════════════════════════
class ChurnPredictor:
    """
    Self-contained predictor used by the FastAPI endpoint.

    Artifact loading
    ────────────────
    Uses the v3.2 modular loaders (no pickle on the XGBoost model):
      • load_xgb_model()  → XGBClassifier from aegis_xgb.json
      • load_processor()  → ProcessorBundle (scaler + kmeans) from joblib

    SHAP explainer
    ──────────────
    shap.TreeExplainer is initialised once at startup using the XGBoost booster.
    TreeExplainer works directly on the booster's internal tree representation:
      - No approximation: exact Shapley values via the polynomial-time
        tree SHAP algorithm (Lundberg & Lee, 2017).
      - Thread-safe for reads: the explainer object is immutable post-init.
      - Complexity: O(TLD) per sample where T=trees, L=max_leaves, D=depth.
        For this model (300 trees, depth 4): ~4,800 operations per inference.
    """

    def __init__(self, artifact_path: str = ARTIFACT_PATH) -> None:
        # Bootstrap if artifacts are missing
        if not os.path.exists("aegis_xgb.json"):
            log.warning("  Native artifacts missing — running training pipeline...")
            from churn_model import run_training_pipeline
            run_training_pipeline()

        # Load XGBoost native JSON
        self.xgb_model = load_xgb_model()
        self.booster   = self.xgb_model.get_booster()

        # Load ProcessorBundle (sklearn components only, no XGBoost binary)
        self.processor: ProcessorBundle = load_processor()

        # Resolve feature names for SHAP labelling
        self.feature_names: list[str] = self.processor.output_feature_names
        # If the booster has explicit feature_names, prefer those
        booster_fn = self.booster.feature_names
        if booster_fn:
            self.feature_names = list(booster_fn)

        # Initialise TreeExplainer ONCE — immutable, thread-safe for reads
        self.explainer = shap.TreeExplainer(self.booster)

        # Load metadata for version string
        meta = load_artifact_meta()
        self.model_version = (
            f"aegis_quant_v3.2_cross_pollination "
            f"(trained {meta.get('trained_at', 'unknown')[:10]})"
        )

        log.info(
            f"  ChurnPredictor ready | "
            f"Val AUC: {meta.get('val_auc', 'N/A')} | "
            f"SHAP TreeExplainer initialised | "
            f"Features: {self.feature_names}"
        )

    def _build_input(self, features: ClientFeatures) -> np.ndarray:
        """
        Build the full transformed feature matrix that the XGBoost model expects.
        Replicates ClusterInjector.transform() via ProcessorBundle.transform().
        Shape: (1, len(CHURN_FEATURES) + N_CLUSTERS)
        """
        feat = pd.DataFrame(
            [[
                features.avg_holding_days,
                features.crypto_ratio,
                features.tech_stocks_ratio,
                features.account_balance,
                features.balance_velocity,
                features.market_pain_index,
                features.login_freq_drop,
            ]],
            columns=ALL_FEATURES,
        )
        return self.processor.transform(feat)   # shape: (1, 7)

    def predict_and_explain(self, features: ClientFeatures) -> PredictionResponse:
        """
        Compute churn probability and LOCAL SHAP-based risk drivers.

        Steps
        ─────
        1. Transform input via ProcessorBundle (ClusterInjector logic).
        2. Predict churn probability via XGBClassifier.predict_proba().
        3. Compute local SHAP values via TreeExplainer.shap_values().
           shap_values[0] is a 1D array of shape (n_features,) where each
           element is the marginal contribution of that feature to the
           log-odds of churn for THIS specific client.
        4. Select top-2 features by POSITIVE SHAP value.
           Positive SHAP → feature increases churn probability for this client.
           If fewer than 2 features are positive, fall back to top-2 by
           absolute magnitude (ensures we always return 2 drivers).
        5. Map feature names through SHAP_FEATURE_REASON_MAP for human-readable
           risk driver strings.

        Why positive SHAP values?
        ─────────────────────────
        We want to tell the relationship manager: "these are the factors that
        are ACTIVELY DRIVING this client toward churn right now."  Features
        with negative SHAP values are protective factors — reporting them as
        "risk drivers" would be misleading.  Only positive contributions are
        surfaced.  The fallback to absolute magnitude handles edge cases where
        a client's profile is so atypical that no individual feature pushes
        strongly positive (e.g. a perfectly balanced client where all marginal
        contributions are near-zero).
        """
        # Step 1: transform input
        X_transformed = self._build_input(features)   # shape: (1, 7)

        # Step 2: churn probability
        churn_prob = float(self.xgb_model.predict_proba(X_transformed)[0, 1])

        # Step 3: local SHAP values for this sample
        # shap_values returns ndarray of shape (n_samples, n_features) for
        # binary classification using TreeExplainer with XGBoost.
        # We take [0] to get the single-sample vector.
        shap_matrix = self.explainer.shap_values(X_transformed)
        # Handle both (n, f) and (f,) output shapes across shap versions
        if shap_matrix.ndim == 2:
            shap_vec: np.ndarray = shap_matrix[0]
        else:
            shap_vec = shap_matrix

        # Step 4: top-2 features by positive SHAP value
        paired: list[tuple[str, float]] = list(zip(self.feature_names, shap_vec.tolist()))
        positive_pairs = [(name, val) for name, val in paired if val > 0.0]
        positive_pairs.sort(key=lambda x: x[1], reverse=True)

        if len(positive_pairs) >= 2:
            top_pairs = positive_pairs[:2]
        elif len(positive_pairs) == 1:
            # Supplement with the next-best absolute-value feature
            abs_pairs = sorted(paired, key=lambda x: abs(x[1]), reverse=True)
            used = {positive_pairs[0][0]}
            top_pairs = [positive_pairs[0]]
            for name, val in abs_pairs:
                if name not in used:
                    top_pairs.append((name, val))
                    break
        else:
            # No positive SHAP values — use top-2 by absolute magnitude
            abs_pairs = sorted(paired, key=lambda x: abs(x[1]), reverse=True)
            top_pairs = abs_pairs[:2]

        # Step 5: map to human-readable strings
        risk_drivers = [
            SHAP_FEATURE_REASON_MAP.get(name, f"Feature '{name}'")
            for name, _ in top_pairs
        ]

        return PredictionResponse(
            churn_probability=round(churn_prob, 4),
            risk_drivers=risk_drivers,
            model_version=self.model_version,
        )


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI application
# ═════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    ASGI lifespan: initialise ChurnPredictor once at startup, store on
    app.state for reuse across requests.  The predictor is thread-safe for
    concurrent reads (SHAP TreeExplainer is immutable post-init).
    """
    log.info("  AegisQuant API — loading predictor...")
    predictor = ChurnPredictor()
    app.state.predictor = predictor
    log.info("  AegisQuant API — predictor ready, serving requests.")
    yield
    log.info("  AegisQuant API — shutdown.")


app = FastAPI(
    title="AegisQuant — AI Portfolio Shield & Risk Engine",
    version="3.2.0",
    description=(
        "Churn prediction with local SHAP explainability. "
        "POST /predict with a ClientFeatures JSON body."
    ),
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "version": "3.2.0"}


@app.get("/predict")
async def predict_get() -> dict:
    return {"message": "POST to /predict with a JSON body. See /docs for the schema."}


@app.post("/predict", response_model=PredictionResponse)
async def predict_post(features: ClientFeatures) -> PredictionResponse:
    """
    Churn prediction endpoint.

    Request body: ClientFeatures JSON (all seven numeric fields required;
    client_id and description are optional and ignored by the model).

    Response:
      churn_probability — float in [0, 1]
      risk_drivers      — list[str] of top-2 LOCAL SHAP risk driver descriptions
      model_version     — string identifying the artifact version
    """
    if not hasattr(app.state, "predictor"):
        raise HTTPException(status_code=503, detail="Predictor not initialised.")
    return app.state.predictor.predict_and_explain(features)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
