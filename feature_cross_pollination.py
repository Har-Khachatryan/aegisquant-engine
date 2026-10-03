"""
AegisQuant — Feature-Cross-Pollination Pipeline (v4.0)

Architecture
────────────
  ClusterInjector   — sklearn transformer: scales the life-stage CLUSTER_FEATURES,
                      runs KMeans, one-hot encodes the cluster ID and appends it
                      to the CHURN_FEATURES (the "cross-pollination").
  XGBWithValidation — XGBClassifier that carves an internal stratified 80/20
                      split for early stopping during pipeline.fit().
  build_pipeline()  — assembles the two steps for training only.
  ProcessorBundle   — the fitted scaler + KMeans reduced to plain arrays.
                      Serving replicates ClusterInjector.transform() with NumPy
                      (nearest centroid), so no sklearn object is ever unpickled.

Upgrade v3.2 → v4.0
────────────────────
  - The v3.2 docstring promised "secure serialisation" but still wrote a full
    pipeline pickle and two joblib files. v4.0 writes JSON only:
      aegis_xgb.json               XGBoost native booster
      aegis_processor.json         scaler means/scales + KMeans centroids
      aegis_profile_resolver.json  cluster → investor profile
      aegis_artifact_meta.json     metrics, decision threshold, feature names
    plus CSV reference data for the DriftMonitor.
  - The transformer now returns a named DataFrame, so the booster stores real
    feature names and SHAP explanations are labelled without an fN lookup table.
  - Raw age and balance are model inputs too (they are the strongest signals);
    the cluster one-hot adds the segment context on top.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    ARTIFACT_DIR,
    ARTIFACT_META_PATH,
    CHURN_FEATURES,
    CLUSTER_FEATURES,
    EVALUATION_PATH,
    FEATURE_GROUPS,
    HOLDOUT_PATH,
    MONOTONE_CONSTRAINTS,
    N_CLUSTERS,
    PROCESSOR_PATH,
    PROFILE_RESOLVER_PATH,
    RANDOM_STATE,
    REFERENCE_DATA_PATH,
    XGB_MODEL_PATH,
    XGB_PARAMS,
)

log = logging.getLogger("aegis")

CLUSTER_COLUMNS: list[str] = [f"cluster_{i}" for i in range(N_CLUSTERS)]
OUTPUT_FEATURES: list[str] = CHURN_FEATURES + CLUSTER_COLUMNS


def group_contributions(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Sum per-feature SHAP values into business concepts (config.FEATURE_GROUPS);
    the one-hot cluster columns become "segment". Valid because SHAP values are
    additive: the grouped values still add up to the model margin.
    """
    values = np.atleast_2d(values)
    index = {name: i for i, name in enumerate(feature_names)}
    groups = {g: [index[f] for f in members] for g, members in FEATURE_GROUPS.items()}
    groups["segment"] = [index[c] for c in CLUSTER_COLUMNS]
    return pd.DataFrame({g: values[:, cols].sum(axis=1) for g, cols in groups.items()})


def _one_hot(cluster_ids: np.ndarray) -> np.ndarray:
    one_hot = np.zeros((len(cluster_ids), N_CLUSTERS), dtype=np.float64)
    one_hot[np.arange(len(cluster_ids)), cluster_ids] = 1.0
    return one_hot


# ═════════════════════════════════════════════════════════════════════════════
# ClusterInjector — sklearn transformer
# ═════════════════════════════════════════════════════════════════════════════
class ClusterInjector(BaseEstimator, TransformerMixin):
    """
    Stage 1 of the cross-pollination pipeline.

    Fit:       StandardScaler + KMeans(N_CLUSTERS) on CLUSTER_FEATURES.
    Transform: [CHURN_FEATURES | one-hot(cluster)] as a named DataFrame.
    """

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "ClusterInjector":
        self.scaler_ = StandardScaler().fit(X[CLUSTER_FEATURES])
        self.kmeans_ = KMeans(n_clusters=N_CLUSTERS, random_state=self.random_state, n_init=15, max_iter=500)
        self.kmeans_.fit(self.scaler_.transform(X[CLUSTER_FEATURES]))
        return self

    def predict_cluster(self, X: pd.DataFrame) -> np.ndarray:
        return self.kmeans_.predict(self.scaler_.transform(X[CLUSTER_FEATURES]))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        one_hot = _one_hot(self.predict_cluster(X))
        return pd.DataFrame(np.hstack([X[CHURN_FEATURES].to_numpy(np.float64), one_hot]),
                            columns=OUTPUT_FEATURES, index=X.index)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return OUTPUT_FEATURES


# ═════════════════════════════════════════════════════════════════════════════
# XGBWithValidation — thin XGBClassifier subclass
# ═════════════════════════════════════════════════════════════════════════════
class XGBWithValidation(XGBClassifier):
    """
    Carves an internal stratified 80/20 split of whatever it is fitted on and
    uses the 20 % for early stopping. The final hold-out test set never reaches
    this class (churn_model splits it off before any fitting).
    """

    def fit(self, X, y, **kwargs):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )
        return super().fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline builder
# ═════════════════════════════════════════════════════════════════════════════
def build_pipeline() -> Pipeline:
    """Assemble the cross-pollination training pipeline (training only)."""
    constraints = tuple(MONOTONE_CONSTRAINTS.get(f, 0) for f in OUTPUT_FEATURES)
    xgb = XGBWithValidation(
        **XGB_PARAMS,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        monotone_constraints=constraints,
    )
    return Pipeline([
        ("cluster_features", ClusterInjector(random_state=RANDOM_STATE)),
        ("xgb_churn", xgb),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# ProcessorBundle — pickle-free serving copy of the fitted ClusterInjector
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ProcessorBundle:
    """Scaler statistics and KMeans centroids as plain arrays (JSON-serialisable)."""
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    centroids: np.ndarray        # (N_CLUSTERS, len(CLUSTER_FEATURES)) in scaled space

    @classmethod
    def from_injector(cls, injector: ClusterInjector) -> "ProcessorBundle":
        return cls(
            scaler_mean=injector.scaler_.mean_.copy(),
            scaler_scale=injector.scaler_.scale_.copy(),
            centroids=injector.kmeans_.cluster_centers_.copy(),
        )

    def predict_cluster(self, X: pd.DataFrame) -> np.ndarray:
        """Nearest centroid in scaled space — identical to KMeans.predict."""
        z = (X[CLUSTER_FEATURES].to_numpy(np.float64) - self.scaler_mean) / self.scaler_scale
        d2 = ((z[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replicates ClusterInjector.transform() for serving."""
        one_hot = _one_hot(self.predict_cluster(X))
        return pd.DataFrame(np.hstack([X[CHURN_FEATURES].to_numpy(np.float64), one_hot]),
                            columns=OUTPUT_FEATURES, index=X.index)

    @property
    def output_feature_names(self) -> list[str]:
        return OUTPUT_FEATURES

    def save(self, path: str | Path) -> None:
        payload = {
            "cluster_features": CLUSTER_FEATURES,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "centroids": self.centroids.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProcessorBundle":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["cluster_features"] != CLUSTER_FEATURES:
            raise ValueError(
                f"{path} was trained on {payload['cluster_features']}, "
                f"config expects {CLUSTER_FEATURES}. Retrain with churn_model.py."
            )
        return cls(
            scaler_mean=np.asarray(payload["scaler_mean"]),
            scaler_scale=np.asarray(payload["scaler_scale"]),
            centroids=np.asarray(payload["centroids"]),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Serialisation helpers
# ═════════════════════════════════════════════════════════════════════════════
def save_artifacts(
    pipeline: Pipeline,
    profile_resolver,              # DynamicProfileResolver — import avoided (circular dep)
    meta: dict,
    evaluation: dict,
    reference_data: pd.DataFrame,
    holdout: pd.DataFrame,
) -> None:
    """Persist every artifact as JSON / CSV under ARTIFACT_DIR."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline.named_steps["xgb_churn"].get_booster().save_model(str(XGB_MODEL_PATH))
    ProcessorBundle.from_injector(pipeline.named_steps["cluster_features"]).save(PROCESSOR_PATH)
    profile_resolver.save(PROFILE_RESOLVER_PATH)
    ARTIFACT_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    reference_data.to_csv(REFERENCE_DATA_PATH, index=False)
    holdout.to_csv(HOLDOUT_PATH, index=False)
    log.info(f"  Artifacts saved → {ARTIFACT_DIR}")


def load_processor() -> ProcessorBundle:
    if not PROCESSOR_PATH.exists():
        raise FileNotFoundError(f"'{PROCESSOR_PATH}' not found. Run `python churn_model.py` first.")
    return ProcessorBundle.load(PROCESSOR_PATH)


def load_xgb_model() -> XGBClassifier:
    """Load the booster from native JSON into a fresh XGBClassifier shell (no pickle)."""
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(f"'{XGB_MODEL_PATH}' not found. Run `python churn_model.py` first.")
    xgb = XGBClassifier()
    xgb.load_model(str(XGB_MODEL_PATH))
    return xgb


def load_artifact_meta() -> dict:
    try:
        return json.loads(ARTIFACT_META_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.warning(f"  Metadata file '{ARTIFACT_META_PATH}' not found.")
        return {}


def load_evaluation() -> dict:
    try:
        return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
