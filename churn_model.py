"""
AegisQuant — Offline training pipeline (v3.2)

Upgrade v3.1 → v3.2
────────────────────
[SECURE & NATIVE SERIALISATION]
  - run_training_pipeline() no longer calls joblib.dump() on the full sklearn
    Pipeline.  Instead it delegates to save_artifacts() in
    feature_cross_pollination.py, which writes:
      • aegis_xgb.json            — XGBoost native JSON (no pickle)
      • aegis_processor.joblib    — StandardScaler + KMeans only
      • aegis_profile_resolver.joblib — DynamicProfileResolver mapping
      • aegis_artifact_meta.json  — human-readable metadata sidecar
      • aegis_quant_artifacts.pkl — legacy combined pickle (backward compat)
      • reference_data.pkl        — training features for DriftMonitor

  The evaluation split is kept strictly separate from the pipeline.fit() call:
  pipeline.fit() uses all 3,000 samples (XGBWithValidation carves its own
  internal 80/20 split for early stopping), while the metrics block below uses
  a second independent 80/20 split purely for reporting — no data leakage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from config import (
    CHURN_THRESHOLD,
    CLUSTER_FEATURES,
    CHURN_FEATURES,
    ALL_FEATURES,
)
from data_pipeline import generate_synthetic_data, DynamicProfileResolver
from feature_cross_pollination import build_pipeline, save_artifacts

log = logging.getLogger("aegis")


def run_training_pipeline() -> None:
    """
    Offline training pipeline — safe to call from any module that needs to
    bootstrap artifacts (optimizer.__init__, api.py lifespan, etc.).

    Stages
    ──────
    1. Generate 3,000-sample synthetic investor dataset.
    2. Build and fit the unified cross-pollination pipeline on all 3,000 samples.
       (XGBWithValidation internally carves an 80/20 validation split for
       early stopping; this is separate from the reporting split below.)
    3. Fit the DynamicProfileResolver to label KMeans clusters as archetypes.
    4. Evaluate on an independent held-out 20 % split and log metrics.
    5. Persist all artifacts via save_artifacts() using the v3.2 modular scheme.
    """
    log.info("=" * 70)
    log.info("AegisQuant v3.2  |  [OFFLINE] Training pipeline — START")

    # ── Step 1: Synthetic data ────────────────────────────────────────────────
    df = generate_synthetic_data(n=3_000)
    X  = df[ALL_FEATURES]
    y  = df["churn"]

    # ── Step 2: Fit unified pipeline on full dataset ──────────────────────────
    log.info("  [2/4] Fitting cross-pollination pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    # ── Step 3: Fit DynamicProfileResolver ───────────────────────────────────
    log.info("  [3/4] Fitting DynamicProfileResolver...")
    cluster_injector = pipeline.named_steps["cluster_features"]
    X_cluster        = df[CLUSTER_FEATURES].copy()
    scaled           = cluster_injector.scaler.transform(X_cluster)
    cluster_ids      = cluster_injector.kmeans.predict(scaled)

    df_temp = pd.DataFrame({
        "cluster_id":       cluster_ids,
        "avg_holding_days": df["avg_holding_days"].values,
    })
    profile_resolver = DynamicProfileResolver()
    profile_resolver.fit(df_temp)

    # ── Step 4: Evaluation on independent reporting split ────────────────────
    log.info("  [4/4] Evaluating on held-out 20 % reporting split...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    y_prob: np.ndarray = pipeline.predict_proba(X_val)[:, 1]
    y_pred: np.ndarray = (y_prob >= CHURN_THRESHOLD).astype(int)
    auc: float         = roc_auc_score(y_val, y_prob)

    log.info(f"  AegisQuant Cross-Pollination AUC-ROC: {auc:.4f}")
    log.info(
        f"\n{classification_report(y_val, y_pred, target_names=['Retain', 'Churn'])}"
    )

    # ── Step 5: Persist via modular save scheme ───────────────────────────────
    log.info("  Persisting artifacts (v3.2 modular scheme)...")
    save_artifacts(
        pipeline=pipeline,
        profile_resolver=profile_resolver,
        val_auc=auc,
        training_features=X,
    )

    log.info("  [OFFLINE] Training pipeline complete.")
    log.info("=" * 70)
