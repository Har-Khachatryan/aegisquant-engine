"""
AegisQuant — Data generation and investor profile resolution.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config import CLUSTER_FEATURES

log = logging.getLogger("aegis")


def generate_synthetic_data(n: int = 3_000) -> pd.DataFrame:
    """
    Generate a synthetic investor dataset using a financially motivated logistic model.

    Gaussian noise (σ=0.12) is added to the latent probability and then clipped
    to [0, 1] before Bernoulli sampling.
    """
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "avg_holding_days":   rng.integers(2, 200, n),
        "crypto_ratio":       rng.uniform(0.0, 0.9, n),
        "tech_stocks_ratio":  rng.uniform(0.1, 0.8, n),
        "account_balance":    rng.integers(5_000, 150_000, n).astype(float),
        "balance_velocity":   rng.uniform(0.1, 1.5, n),
        "market_pain_index":  rng.uniform(0.0, 1.0, n),
        "login_freq_drop":    rng.uniform(0.0, 1.5, n),
    })

    z = (
        - 4.0 * (df["balance_velocity"]  - 0.70)
        + 4.5 * (df["market_pain_index"] - 0.50)
        + 3.0 * (df["login_freq_drop"]   - 0.60)
        - 0.5 * (df["account_balance"] / 50_000 - 1.5)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    prob = np.clip(prob + rng.normal(0.0, 0.12, n), 0.0, 1.0)
    df["churn"] = (rng.uniform(0.0, 1.0, n) < prob).astype(int)

    log.info(
        f"  Synthetic data: {n} samples  |  "
        f"churn rate = {df['churn'].mean():.1%}"
    )
    return df


class DynamicProfileResolver(BaseEstimator, TransformerMixin):
    """
    Maps KMeans cluster IDs → semantic investor archetypes.

    Cluster with LOWEST mean avg_holding_days → "aggressive",
    HIGHEST → "conservative", middle → "balanced".
    """

    def __init__(self) -> None:
        self.cluster_to_profile_: dict[int, str] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "DynamicProfileResolver":
        """
        X must contain 'cluster_id' and 'avg_holding_days'.
        """
        mean_hold = (
            X.groupby("cluster_id")["avg_holding_days"]
            .mean()
            .sort_values()
        )
        sorted_clusters = mean_hold.index.tolist()

        if len(sorted_clusters) != 3:
            raise ValueError(
                f"Expected exactly 3 clusters, got {len(sorted_clusters)}. "
                "Adjust KMeans n_clusters."
            )

        labels = ["aggressive", "balanced", "conservative"]
        self.cluster_to_profile_ = {
            int(cid): lbl for cid, lbl in zip(sorted_clusters, labels)
        }
        log.info(f"  Profile mapping: {self.cluster_to_profile_}")
        return self

    def transform(self, X: pd.DataFrame) -> list[str]:
        if not self.cluster_to_profile_:
            raise RuntimeError("DynamicProfileResolver must be fit before transform.")
        return [
            self.cluster_to_profile_.get(int(c), "balanced")
            for c in X["cluster_id"]
        ]

    def get_params(self, deep: bool = True) -> dict:
        return {}

    def set_params(self, **params) -> "DynamicProfileResolver":
        return self