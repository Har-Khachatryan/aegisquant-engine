"""
AegisQuant — Training pipeline using cross‑pollination (KMeans features injected).
Saves: 'aegis_quant_artifacts.pkl' containing
    - pipeline (ClusterInjector + XGBoost)
    - profile_resolver (DynamicProfileResolver)
"""

from __future__ import annotations

import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from config import ARTIFACT_PATH, CHURN_THRESHOLD, CLUSTER_FEATURES, CHURN_FEATURES
from data_pipeline import generate_synthetic_data, DynamicProfileResolver
from feature_cross_pollination import build_pipeline

log = logging.getLogger("aegis")


def run_training_pipeline() -> None:
    log.info("═" * 70)
    log.info("⚙️  [OFFLINE] Training pipeline (Cross‑Pollination) — START")

    # 1. Synthetic data
    df = generate_synthetic_data(n=3_000)

    # 2. All features
    X = df[CLUSTER_FEATURES + CHURN_FEATURES]
    y = df["churn"]

    # 3. Build and fit the unified pipeline
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    # 4. Evaluate on a held‑out set (for logging)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    y_pred = (y_prob >= CHURN_THRESHOLD).astype(int)
    log.info(f"✅ Cross‑pollination AUC-ROC: {auc:.4f}")
    log.info(f"\n{classification_report(y_val, y_pred, target_names=['Retain', 'Churn'])}")

    # 5. Fit the profile resolver (for UI labels)
    cluster_injector = pipeline.named_steps["cluster_features"]
    X_cluster = df[CLUSTER_FEATURES].copy()
    scaled = cluster_injector.scaler.transform(X_cluster)
    cluster_ids = cluster_injector.kmeans.predict(scaled)
    df_temp = pd.DataFrame({
        "cluster_id": cluster_ids,
        "avg_holding_days": df["avg_holding_days"]
    })
    profile_resolver = DynamicProfileResolver()
    profile_resolver.fit(df_temp)
    # 6. Save artifacts
    joblib.dump(
        {
            "pipeline":         pipeline,
            "profile_resolver": profile_resolver,
            "val_auc":          auc,
            "trained_at":       datetime.now(timezone.utc).isoformat(),
        },
        ARTIFACT_PATH,
    )
    X.to_pickle("reference_data.pkl")
    log.info("Референсные данные сохранены в reference_data.pkl")
    log.info("💾  Pipeline + profile resolver saved.")
    log.info("═" * 70)