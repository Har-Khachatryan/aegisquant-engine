"""
AegisQuant — Data loading, feature engineering and investor profile resolution (v4.0).

Upgrade v3.2 → v4.0
────────────────────
  - generate_synthetic_data() is retired: the engine now trains on the Kaggle
    "Churn Modelling" bank dataset (data/Churn_Modelling.csv).
  - load_churn_dataset() validates the schema, renames columns to snake_case,
    drops identifiers that must never reach a model (RowNumber, Surname) and
    fails loudly on missing values or duplicate customers instead of silently
    imputing them.
  - engineer_features() is the ONE place where derived features are computed.
    Training and online inference (API, dashboard) both call it, so the two
    paths cannot drift apart.
  - DynamicProfileResolver labels KMeans clusters by life-stage risk capacity
    (younger and wealthier → more capacity for risk) instead of synthetic
    holding periods, and serialises to JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CHURN_FEATURES, CLUSTER_FEATURES, DATA_PATH, GEOGRAPHIES, ClientFeatures

log = logging.getLogger("aegis")

# Kaggle header → internal snake_case name
COLUMN_MAP: dict[str, str] = {
    "CustomerId":      "customer_id",
    "CreditScore":     "credit_score",
    "Geography":       "geography",
    "Gender":          "gender",
    "Age":             "age",
    "Tenure":          "tenure",
    "Balance":         "balance",
    "NumOfProducts":   "num_products",
    "HasCrCard":       "has_cr_card",
    "IsActiveMember":  "is_active_member",
    "EstimatedSalary": "estimated_salary",
    "Exited":          "churn",
}
# Identifiers with no predictive meaning (and, for Surname, personal data)
DROPPED_COLUMNS: list[str] = ["RowNumber", "Surname"]

PROFILES: tuple[str, str, str] = ("conservative", "balanced", "aggressive")


def load_churn_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """
    Load and validate the bank churn dataset.

    Returns one row per customer with snake_case columns: customer_id, the
    model inputs, `gender` (audit only, never a model input) and `churn` (0/1).
    """
    raw = pd.read_csv(path)
    missing = sorted(set(COLUMN_MAP) - set(raw.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    df = raw.drop(columns=[c for c in DROPPED_COLUMNS if c in raw.columns]).rename(columns=COLUMN_MAP)
    df = df[list(COLUMN_MAP.values())]

    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"{path}: missing values in {bad}")
    if df["customer_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate customer_id values")
    unknown_geo = set(df["geography"]) - set(GEOGRAPHIES)
    if unknown_geo:
        raise ValueError(f"{path}: unexpected geography values {sorted(unknown_geo)}")
    if not set(df["churn"].unique()) <= {0, 1}:
        raise ValueError(f"{path}: target `Exited` must be 0/1")

    log.info(f"  Loaded {len(df):,} customers | churn rate = {df['churn'].mean():.1%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the derived model features to a frame holding the INPUT_FEATURES columns.

      geo_germany / geo_spain — one-hot market (France is the baseline)
      balance_to_salary       — liquid wealth relative to income
      zero_balance            — 36 % of customers keep no balance at all
      tenure_to_age           — share of adult life spent as a customer (loyalty)

    Booleans are cast to int so the matrix fed to XGBoost is purely numeric.
    """
    out = df.copy()
    for col in ("has_cr_card", "is_active_member"):
        out[col] = out[col].astype(int)
    out["geo_germany"] = (out["geography"] == "Germany").astype(int)
    out["geo_spain"] = (out["geography"] == "Spain").astype(int)
    out["balance_to_salary"] = out["balance"] / out["estimated_salary"].clip(lower=1.0)
    out["zero_balance"] = (out["balance"] == 0).astype(int)
    out["tenure_to_age"] = out["tenure"] / out["age"]
    return out


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features in the exact column order the pipeline expects."""
    feats = engineer_features(df)
    cols = list(dict.fromkeys(CHURN_FEATURES + CLUSTER_FEATURES))
    return feats[cols].astype(float)


def client_to_frame(client: ClientFeatures) -> pd.DataFrame:
    """One-row model matrix for a single customer (API / dashboard path)."""
    row = client.model_dump(exclude={"customer_id", "description"})
    return model_matrix(pd.DataFrame([row]))


class DynamicProfileResolver:
    """
    Maps KMeans cluster IDs → investor archetypes by life-stage risk capacity.

    For each cluster, capacity = z(mean balance) − z(mean age), where z uses the
    population mean and standard deviation. A long investment horizon (younger)
    and a larger cushion (higher balance) both raise the capacity to take risk.
    Lowest capacity → "conservative", highest → "aggressive".

    The labels are derived from the fitted centroids, not hard-coded to cluster
    numbers, so they stay correct when KMeans renumbers clusters on retraining.
    """

    def __init__(self) -> None:
        self.cluster_to_profile_: dict[int, str] = {}
        self.capacity_: dict[int, float] = {}

    def fit(self, X: pd.DataFrame) -> "DynamicProfileResolver":
        """X must contain 'cluster_id', 'age' and 'balance'."""
        means = X.groupby("cluster_id")[["age", "balance"]].mean()
        if len(means) != len(PROFILES):
            raise ValueError(f"Expected {len(PROFILES)} clusters, got {len(means)}. Adjust N_CLUSTERS.")
        z = (means - X[["age", "balance"]].mean()) / X[["age", "balance"]].std(ddof=0)
        capacity = (z["balance"] - z["age"]).sort_values()
        self.capacity_ = {int(c): round(float(v), 4) for c, v in capacity.items()}
        self.cluster_to_profile_ = {int(c): p for c, p in zip(capacity.index, PROFILES)}
        log.info(f"  Profile mapping: {self.cluster_to_profile_}")
        return self

    def transform(self, X: pd.DataFrame) -> list[str]:
        if not self.cluster_to_profile_:
            raise RuntimeError("DynamicProfileResolver must be fit before transform.")
        return [self.cluster_to_profile_.get(int(c), "balanced") for c in X["cluster_id"]]

    def resolve(self, cluster_id: int) -> str:
        return self.transform(pd.DataFrame({"cluster_id": [cluster_id]}))[0]

    # ── JSON serialisation (no pickle) ────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        payload = {
            "cluster_to_profile": {str(k): v for k, v in self.cluster_to_profile_.items()},
            "capacity": {str(k): v for k, v in self.capacity_.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DynamicProfileResolver":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.cluster_to_profile_ = {int(k): v for k, v in payload["cluster_to_profile"].items()}
        obj.capacity_ = {int(k): float(v) for k, v in payload.get("capacity", {}).items()}
        return obj


def summarise_segments(df: pd.DataFrame, cluster_ids: np.ndarray, resolver: DynamicProfileResolver) -> list[dict]:
    """Per-segment size, churn rate and life-stage averages for reports and the dashboard."""
    seg = df.assign(cluster_id=cluster_ids, profile=resolver.transform(pd.DataFrame({"cluster_id": cluster_ids})))
    rows = []
    for (cid, profile), g in seg.groupby(["cluster_id", "profile"]):
        rows.append({
            "cluster_id": int(cid),
            "profile": profile,
            "customers": int(len(g)),
            "share": round(len(g) / len(seg), 4),
            "churn_rate": round(float(g["churn"].mean()), 4),
            "avg_age": round(float(g["age"].mean()), 1),
            "avg_balance": round(float(g["balance"].mean()), 0),
            "avg_salary": round(float(g["estimated_salary"].mean()), 0),
            "zero_balance_share": round(float((g["balance"] == 0).mean()), 4),
        })
    order = {p: i for i, p in enumerate(PROFILES)}
    return sorted(rows, key=lambda r: order[r["profile"]])
