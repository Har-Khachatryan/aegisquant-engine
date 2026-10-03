"""
CoinStats Portfolio Intelligence — unsupervised investor segmentation.

KMeans (k=4) on six standardised behavioural features, followed by a
deterministic mapping from raw cluster IDs to business archetypes.

Why a JSON model instead of a pickled sklearn object?
─────────────────────────────────────────────────────
A fitted StandardScaler + KMeans is fully described by a few arrays
(means, scales, centroids). Persisting those as JSON means:
  • no pickle → no arbitrary-code-execution risk when the API loads the model;
  • no sklearn version lock between the training box and the serving image;
  • the model file is human-readable and diff-able in code review.
Inference (standardise → nearest centroid) is three lines of NumPy and gives
exactly the labels KMeans.predict() would.

Why Hungarian matching for archetype labels?
────────────────────────────────────────────
KMeans cluster numbering is arbitrary and can change between retrains. Each
archetype in settings declares a "signature" (which standardised features
should be high or low). We score every (cluster, archetype) pair and solve the
one-to-one assignment that maximises the total score, so "Whale" always lands
on the high-value / blue-chip cluster no matter what number KMeans gave it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import settings as cfg

_TRANSFORMS = {
    "identity": lambda x: x,
    "log10": lambda x: np.log10(np.maximum(x, 1e-9)),
    "log1p": np.log1p,
}


def build_design_matrix(features: pd.DataFrame, inputs: dict[str, str] = cfg.CLUSTER_INPUTS) -> np.ndarray:
    """Apply the per-feature transform (log10 / log1p / identity) → raw design matrix."""
    missing = [c for c in inputs if c not in features.columns]
    if missing:
        raise ValueError(f"Features missing clustering inputs: {missing}")
    cols = [_TRANSFORMS[t](features[c].to_numpy(dtype=float)) for c, t in inputs.items()]
    return np.column_stack(cols)


@dataclass
class SegmentationModel:
    """Serializable segmentation model: scaler params + centroids + archetype map."""

    inputs: dict[str, str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    centroids: list[list[float]]               # in standardised space
    cluster_to_archetype: dict[int, int]
    silhouette: float
    n_samples: int
    trained_at: str
    model_version: str = cfg.MODEL_VERSION
    archetype_scores: list[list[float]] = field(default_factory=list)
    data_fingerprint: str = ""                  # sha256 of the training CSV (lineage)

    # ── inference ────────────────────────────────────────────────────────────
    def standardise(self, features: pd.DataFrame) -> np.ndarray:
        X = build_design_matrix(features, self.inputs)
        return (X - np.asarray(self.scaler_mean)) / np.asarray(self.scaler_scale)

    def predict_cluster(self, features: pd.DataFrame) -> np.ndarray:
        Z = self.standardise(features)
        C = np.asarray(self.centroids)
        d2 = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1)

    def predict_archetype(self, features: pd.DataFrame) -> np.ndarray:
        clusters = self.predict_cluster(features)
        return np.array([self.cluster_to_archetype[int(c)] for c in clusters], dtype=int)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: Path | str = cfg.MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["cluster_to_archetype"] = {str(k): v for k, v in self.cluster_to_archetype.items()}
        payload["archetype_names"] = {str(k): a.name for k, a in cfg.ARCHETYPES.items()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str = cfg.MODEL_PATH) -> "SegmentationModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}. Run `python pipeline.py` first.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("archetype_names", None)
        payload["cluster_to_archetype"] = {int(k): int(v) for k, v in payload["cluster_to_archetype"].items()}
        return cls(**payload)


def match_archetypes(centroids: np.ndarray, inputs: dict[str, str] = cfg.CLUSTER_INPUTS) -> tuple[dict[int, int], np.ndarray]:
    """
    One-to-one cluster → archetype assignment via the Hungarian algorithm.
    Returns (mapping, score_matrix[cluster, archetype]).
    """
    names = list(inputs)
    arch_ids = sorted(cfg.ARCHETYPES)
    scores = np.zeros((centroids.shape[0], len(arch_ids)))
    for j, aid in enumerate(arch_ids):
        for feat, weight in cfg.ARCHETYPES[aid].signature.items():
            scores[:, j] += weight * centroids[:, names.index(feat)]
    rows, cols = linear_sum_assignment(scores, maximize=True)
    return {int(r): int(arch_ids[c]) for r, c in zip(rows, cols)}, scores


def fit_segmentation(features: pd.DataFrame) -> SegmentationModel:
    """Fit scaler + KMeans on portfolio features and build the archetype map."""
    X = build_design_matrix(features)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    km = KMeans(
        n_clusters=cfg.N_CLUSTERS, n_init=cfg.KMEANS_N_INIT, random_state=cfg.KMEANS_SEED
    ).fit(Z)
    mapping, scores = match_archetypes(km.cluster_centers_)
    return SegmentationModel(
        inputs=dict(cfg.CLUSTER_INPUTS),
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        centroids=km.cluster_centers_.tolist(),
        cluster_to_archetype=mapping,
        silhouette=float(silhouette_score(Z, km.labels_)),
        n_samples=int(len(features)),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        archetype_scores=scores.round(4).tolist(),
    )


def attach_segments(features: pd.DataFrame, model: SegmentationModel) -> pd.DataFrame:
    """Add cluster_id, archetype_id, archetype, archetype_short columns."""
    out = features.copy()
    out["cluster_id"] = model.predict_cluster(out)
    out["archetype_id"] = out["cluster_id"].map(model.cluster_to_archetype).astype(int)
    out["archetype"] = out["archetype_id"].map({k: a.name for k, a in cfg.ARCHETYPES.items()})
    out["archetype_short"] = out["archetype_id"].map({k: a.short for k, a in cfg.ARCHETYPES.items()})
    return out
