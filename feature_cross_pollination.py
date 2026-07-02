"""
AegisQuant – Feature‑Cross‑Pollination Pipeline (with Early Stopping)
Integrates KMeans cluster membership (one‑hot encoded) as features into the
XGBoost churn classifier.  Monotonic constraints are fully preserved.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Column names ──────────────────────────────────────────────────────────
CLUSTER_FEATURES = ["avg_holding_days", "crypto_ratio", "tech_stocks_ratio"]
CHURN_FEATURES   = ["account_balance", "balance_velocity",
                    "market_pain_index", "login_freq_drop"]

N_CLUSTERS = 3
ORIGINAL_CONSTRAINTS = (-1, -1, 1, 1)   # balance↓, velocity↓, pain↑, login_drop↑


class ClusterInjector(BaseEstimator, TransformerMixin):
    """
    Fits StandardScaler + KMeans on cluster features,
    then appends one‑hot encoded cluster IDs to the churn features.
    Output shape: (n_samples, 4 + N_CLUSTERS)
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=random_state,
                             n_init=15, max_iter=500)

    def fit(self, X: pd.DataFrame, y=None) -> "ClusterInjector":
        X_cluster = X[CLUSTER_FEATURES].copy()
        scaled = self.scaler.fit_transform(X_cluster)
        self.kmeans.fit(scaled)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_cluster = X[CLUSTER_FEATURES].copy()
        scaled = self.scaler.transform(X_cluster)
        cluster_ids = self.kmeans.predict(scaled)

        # One‑hot encode
        n = len(cluster_ids)
        one_hot = np.zeros((n, N_CLUSTERS), dtype=np.int8)
        one_hot[np.arange(n), cluster_ids] = 1

        churn_arr = X[CHURN_FEATURES].values.astype(np.float64)
        return np.hstack([churn_arr, one_hot])


class XGBWithValidation(XGBClassifier):
    """
    Thin wrapper that automatically creates a validation set from the training data
    for early stopping.  The split is done after the cluster features are injected.
    """
    def fit(self, X, y, **kwargs):
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        return super().fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            **kwargs
        )


def build_pipeline() -> Pipeline:
    """
    Full pipeline:
        ClusterInjector  →  XGBWithValidation (monotonic constraints + early stopping)
    """
    constraints = tuple(ORIGINAL_CONSTRAINTS) + (0,) * N_CLUSTERS

    xgb = XGBWithValidation(
        max_depth=4,
        learning_rate=0.02,
        n_estimators=300,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=1.5,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
        monotone_constraints=constraints,
    )

    pipeline = Pipeline([
        ("cluster_features", ClusterInjector(random_state=42)),
        ("xgb_churn", xgb),
    ])

    return pipeline