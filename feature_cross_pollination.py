"""
AegisQuant — Feature-Cross-Pollination Pipeline (v3.2)

Upgrade v3.1 → v3.2
────────────────────
[SECURE & NATIVE SERIALISATION]
  The previous build_pipeline() returned a single sklearn Pipeline that was
  persisted entirely via joblib/pickle. This creates two problems:
    1. Security: pickle executes arbitrary code on deserialisation; an attacker
       who can write to the artifact file can achieve RCE.
    2. Version fragility: sklearn Pipeline pickles embed class paths; any
       version mismatch between training and serving environments silently
       corrupts the model.

  Resolution in v3.2:
    - The XGBoost model is saved/loaded using its own native JSON format
      (xgb_model.save_model / XGBClassifier.load_model), which is safe,
      human-readable, and cross-version stable.
    - The sklearn preprocessing components (StandardScaler + KMeans inside
      ClusterInjector) are isolated into a dedicated ProcessorBundle dataclass
      and saved as a separate joblib file containing ONLY lightweight sklearn
      objects — no XGBoost binary.
    - The sklearn Pipeline wrapper is retained for training convenience (so
      ClusterInjector.fit/transform wiring is unchanged) but is NEVER pickled
      as a unit for production serving.

Architecture
────────────
  ClusterInjector   — sklearn transformer: scales cluster features, runs KMeans,
                      one-hot encodes cluster IDs, prepends them to churn features.
  XGBWithValidation — thin XGBClassifier subclass that carves out an internal
                      validation split for early stopping, preserving monotonic
                      constraints on the original 4 churn features.
  build_pipeline()  — assembles the two steps for training only.
  ProcessorBundle   — serialisable dataclass containing the fitted sklearn objects
                      (scaler + kmeans) extracted from ClusterInjector post-training.
  save_artifacts()  — writes XGB JSON + processor joblib + profile resolver joblib
                      + JSON metadata sidecar; also writes the legacy pickle for
                      DriftMonitor backward compatibility.
  load_processor()  — reconstructs a ClusterInjector from a saved ProcessorBundle
                      without touching any XGBoost binary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    CLUSTER_FEATURES,
    CHURN_FEATURES,
    N_CLUSTERS,
    XGB_MODEL_PATH,
    PROCESSOR_PATH,
    PROFILE_RESOLVER_PATH,
    ARTIFACT_META_PATH,
    ARTIFACT_PATH,   # legacy pickle path
)

log = logging.getLogger("aegis")

# Monotonic constraints on the 4 original churn features
# (balance↓, velocity↓, pain↑, login_drop↑) + 0 for each one-hot cluster column
_CHURN_CONSTRAINTS: tuple[int, ...] = (-1, -1, 1, 1)


# ═════════════════════════════════════════════════════════════════════════════
# ClusterInjector — sklearn transformer
# ═════════════════════════════════════════════════════════════════════════════
class ClusterInjector(BaseEstimator, TransformerMixin):
    """
    Stage 1 of the cross-pollination pipeline.

    Fit:
      1. StandardScaler on CLUSTER_FEATURES.
      2. KMeans(n_clusters=N_CLUSTERS) on scaled cluster features.

    Transform:
      1. Scale CLUSTER_FEATURES with the fitted scaler.
      2. Predict cluster IDs with the fitted KMeans.
      3. One-hot encode cluster IDs → shape (n, N_CLUSTERS).
      4. Horizontally stack [CHURN_FEATURES | one-hot] →
         output shape (n, len(CHURN_FEATURES) + N_CLUSTERS).

    Output feature order (matches monotone_constraints):
      [account_balance, balance_velocity, market_pain_index, login_freq_drop,
       cluster_0, cluster_1, cluster_2]
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler: StandardScaler = StandardScaler()
        self.kmeans: KMeans = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=random_state,
            n_init=15,
            max_iter=500,
        )

    def fit(self, X: pd.DataFrame, y=None) -> "ClusterInjector":
        X_cluster = X[CLUSTER_FEATURES].copy()
        scaled = self.scaler.fit_transform(X_cluster)
        self.kmeans.fit(scaled)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_cluster = X[CLUSTER_FEATURES].copy()
        scaled = self.scaler.transform(X_cluster)
        cluster_ids = self.kmeans.predict(scaled)

        n = len(cluster_ids)
        one_hot = np.zeros((n, N_CLUSTERS), dtype=np.float64)
        one_hot[np.arange(n), cluster_ids] = 1.0

        churn_arr = X[CHURN_FEATURES].values.astype(np.float64)
        return np.hstack([churn_arr, one_hot])

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """
        Returns the canonical feature names of the transformed output.
        Used by SHAP to label columns correctly.
        """
        cluster_cols = [f"cluster_{i}" for i in range(N_CLUSTERS)]
        return CHURN_FEATURES + cluster_cols


# ═════════════════════════════════════════════════════════════════════════════
# XGBWithValidation — thin XGBClassifier subclass
# ═════════════════════════════════════════════════════════════════════════════
class XGBWithValidation(XGBClassifier):
    """
    Wraps XGBClassifier to auto-carve an internal 80/20 validation split
    for early stopping during pipeline.fit().

    Note on native serialisation:
      When saving, we call get_booster().save_model(XGB_MODEL_PATH) which
      writes a pure JSON representation of the booster.  Loading is done via
      XGBClassifier.load_model(), reconstructing the exact same booster without
      any Python pickle.
    """

    def fit(self, X, y, **kwargs):
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        return super().fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            **kwargs,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline builder
# ═════════════════════════════════════════════════════════════════════════════
def build_pipeline() -> Pipeline:
    """
    Assemble the cross-pollination training pipeline.

    Used during offline training ONLY.  Never pickled as a unit for serving.

    Monotonic constraints cover the 4 original churn features followed by
    N_CLUSTERS zeros (cluster columns have no monotonic constraint because
    cluster membership is a categorical, not an ordinal, signal).
    """
    constraints = _CHURN_CONSTRAINTS + (0,) * N_CLUSTERS

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

    return Pipeline([
        ("cluster_features", ClusterInjector(random_state=42)),
        ("xgb_churn",        xgb),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# ProcessorBundle — lightweight sklearn-only serialisable container
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ProcessorBundle:
    """
    Holds the fitted sklearn preprocessing components extracted from a trained
    ClusterInjector.  Saved via joblib (no XGBoost binary inside).

    Fields
    ──────
    scaler  — fitted StandardScaler (means + scales for CLUSTER_FEATURES)
    kmeans  — fitted KMeans (n_clusters centroids in scaled space)
    """
    scaler: StandardScaler
    kmeans: KMeans

    def predict_cluster(self, X_cluster: pd.DataFrame) -> np.ndarray:
        """Return integer cluster IDs for a CLUSTER_FEATURES DataFrame."""
        scaled = self.scaler.transform(X_cluster)
        return self.kmeans.predict(scaled)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Replicate ClusterInjector.transform() for serving.
        Input: full 7-feature DataFrame (ALL_FEATURES order).
        Output: (n, len(CHURN_FEATURES) + N_CLUSTERS) array — identical to what
                the training pipeline fed into XGBoost.
        """
        X_cluster = X[CLUSTER_FEATURES].copy()
        scaled    = self.scaler.transform(X_cluster)
        ids       = self.kmeans.predict(scaled)

        n       = len(ids)
        one_hot = np.zeros((n, N_CLUSTERS), dtype=np.float64)
        one_hot[np.arange(n), ids] = 1.0

        churn_arr = X[CHURN_FEATURES].values.astype(np.float64)
        return np.hstack([churn_arr, one_hot])

    @property
    def output_feature_names(self) -> list[str]:
        """Canonical names for the transformed output columns (SHAP-compatible)."""
        return CHURN_FEATURES + [f"cluster_{i}" for i in range(N_CLUSTERS)]


# ═════════════════════════════════════════════════════════════════════════════
# Serialisation helpers
# ═════════════════════════════════════════════════════════════════════════════
def save_artifacts(
    pipeline: Pipeline,
    profile_resolver,          # DynamicProfileResolver — import avoided to prevent circular dep
    val_auc: float,
    training_features: pd.DataFrame,
) -> None:
    """
    Persist all artifacts using the v3.2 modular scheme:

      1. XGBoost native JSON  (aegis_xgb.json)
         — Human-readable, cross-version safe, no pickle.
         — Saved via booster.save_model() NOT joblib.

      2. sklearn processor bundle  (aegis_processor.joblib)
         — Contains ONLY StandardScaler + KMeans (no XGBoost binary).
         — joblib on lightweight numpy arrays; negligible security surface.

      3. Profile resolver  (aegis_profile_resolver.joblib)
         — DynamicProfileResolver: a dict[int, str] mapping + sklearn base.
         — Small; safe to joblib-serialise.

      4. Metadata JSON sidecar  (aegis_artifact_meta.json)
         — val_auc, trained_at timestamp, feature names list.
         — Plain text; auditable without loading any binary.

      5. Legacy combined pickle  (aegis_quant_artifacts.pkl)
         — Kept for backward compatibility with DriftMonitor/monitor_and_retrain.
         — Contains the full sklearn Pipeline + profile_resolver.
         — Should be phased out in v4.0 once dependent modules are updated.

    Parameters
    ──────────
    pipeline          — fitted sklearn Pipeline from build_pipeline()
    profile_resolver  — fitted DynamicProfileResolver
    val_auc           — held-out AUC-ROC from churn_model.run_training_pipeline()
    training_features — X DataFrame (ALL_FEATURES) used for reference_data.pkl
    """
    import joblib as _joblib

    cluster_injector = pipeline.named_steps["cluster_features"]
    xgb_step         = pipeline.named_steps["xgb_churn"]

    # 1. XGBoost native JSON
    booster = xgb_step.get_booster()
    booster.save_model(XGB_MODEL_PATH)
    log.info(f"  ✅ XGBoost model saved → {XGB_MODEL_PATH}")

    # 2. sklearn processor bundle
    bundle = ProcessorBundle(
        scaler=cluster_injector.scaler,
        kmeans=cluster_injector.kmeans,
    )
    _joblib.dump(bundle, PROCESSOR_PATH)
    log.info(f"  ✅ Processor bundle (scaler + kmeans) saved → {PROCESSOR_PATH}")

    # 3. Profile resolver
    _joblib.dump(profile_resolver, PROFILE_RESOLVER_PATH)
    log.info(f"  ✅ Profile resolver saved → {PROFILE_RESOLVER_PATH}")

    # 4. Metadata JSON sidecar
    feature_names = bundle.output_feature_names
    meta = {
        "val_auc":       round(val_auc, 6),
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "n_features":    len(feature_names),
        "xgb_model_path":    XGB_MODEL_PATH,
        "processor_path":    PROCESSOR_PATH,
        "profile_resolver_path": PROFILE_RESOLVER_PATH,
    }
    with open(ARTIFACT_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info(f"  ✅ Artifact metadata saved → {ARTIFACT_META_PATH}")

    # 5. Legacy combined pickle (backward compat)
    _joblib.dump(
        {
            "pipeline":         pipeline,
            "profile_resolver": profile_resolver,
            "val_auc":          val_auc,
            "trained_at":       meta["trained_at"],
        },
        ARTIFACT_PATH,
    )
    log.info(f"  ✅ Legacy combined pickle saved → {ARTIFACT_PATH} (backward compat)")

    # 6. Reference data for DriftMonitor
    training_features.to_pickle("reference_data.pkl")
    log.info("  ✅ Reference data saved → reference_data.pkl")


def load_processor() -> ProcessorBundle:
    """
    Load the sklearn ProcessorBundle from aegis_processor.joblib.
    Raises FileNotFoundError with a clear message if artifacts are missing.
    """
    try:
        bundle: ProcessorBundle = joblib.load(PROCESSOR_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Processor artifact '{PROCESSOR_PATH}' not found. "
            "Run churn_model.run_training_pipeline() first."
        )
    log.info(f"  ✅ Processor bundle loaded from {PROCESSOR_PATH}")
    return bundle


def load_xgb_model() -> XGBClassifier:
    """
    Load the XGBoost model from its native JSON format.
    A fresh XGBClassifier shell is created and populated via load_model();
    no pickle is involved.
    """
    if not __import__("os").path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(
            f"XGBoost artifact '{XGB_MODEL_PATH}' not found. "
            "Run churn_model.run_training_pipeline() first."
        )
    xgb = XGBClassifier()
    xgb.load_model(XGB_MODEL_PATH)
    log.info(f"  ✅ XGBoost model loaded from {XGB_MODEL_PATH}")
    return xgb


def load_artifact_meta() -> dict:
    """Load the JSON metadata sidecar. Returns {} if file is missing."""
    try:
        with open(ARTIFACT_META_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        log.warning(f"  ⚠️  Metadata file '{ARTIFACT_META_PATH}' not found.")
        return {}
