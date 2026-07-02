"""
AegisQuant — Online inference engine using unified cross‑pollination pipeline.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from scipy.optimize import minimize, OptimizeResult
from sklearn.covariance import ledoit_wolf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

from config import (
    ARTIFACT_PATH,
    ASSETS,
    TICKER_MAP,
    CRYPTO_ASSETS,
    TECH_ASSETS,
    CLUSTER_FEATURES,
    CHURN_FEATURES,
    RISK_AVERSION,
    WEIGHT_MAX,
    MARKET_CACHE_TTL,
    CHURN_THRESHOLD,
    DIVERSIFICATION_LAMBDA,
    MIN_WEIGHT_BY_PROFILE,
    GAMMA_CHURN_SCALE,
    SIGMA_JITTER,
    SLSQP_FTOL,
    SLSQP_MAXITER,
    ClientPayload,
)
from data_pipeline import DynamicProfileResolver
from churn_model import run_training_pipeline

log = logging.getLogger("aegis")


def _ensure_positive_definite(matrix: np.ndarray, jitter: float = SIGMA_JITTER) -> np.ndarray:
    """PD check with jitter."""
    scale = jitter
    for _ in range(20):
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            matrix = matrix + scale * np.eye(matrix.shape[0])
            scale *= 10.0
    raise ValueError("Covariance matrix could not be made PD after 20 jitter iterations.")


class AegisQuantEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        if not os.path.exists(ARTIFACT_PATH):
            log.warning("⚠️  Artifacts not found — running training pipeline first...")
            run_training_pipeline()

        artifacts: dict = joblib.load(ARTIFACT_PATH)
        # New unified pipeline
        self.pipeline = artifacts["pipeline"]
        self.profile_resolver: DynamicProfileResolver = artifacts["profile_resolver"]

        log.info(
            f"🧠  [ENGINE] Cross‑pollination pipeline loaded | "
            f"Val AUC: {artifacts.get('val_auc', float('nan')):.4f} | "
            f"Trained: {artifacts.get('trained_at', 'N/A')}"
        )

        self._market_cache: tuple[pd.Series, pd.DataFrame] | None = None
        self._cache_timestamp: datetime | None = None

    # ── Market data methods (unchanged) ──────────────────────────────────────
    @staticmethod
    def _normalise_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                return raw["Close"]
            if "Close" in raw.columns.get_level_values(1):
                return raw.xs("Close", axis=1, level=1)
            raise ValueError("MultiIndex columns found but 'Close' not in any level.")
        if "Close" in raw.columns:
            return raw[["Close"]].rename(columns={"Close": raw.columns[0]})
        return raw

    def _fetch_market_data(self, retries=3, backoff=2.0):
        for attempt in range(1, retries + 1):
            try:
                tickers = list(TICKER_MAP.values())
                raw_download = yf.download(tickers, period="1y", timeout=15,
                                           auto_adjust=True, progress=False)
                close_df = self._normalise_multiindex(raw_download)
                close_df = close_df.rename(columns={v: k for k, v in TICKER_MAP.items()})
                close_df = close_df[[a for a in ASSETS if a in close_df.columns]]
                if close_df.empty or close_df.shape[0] < 50:
                    raise ValueError(f"Insufficient data: {close_df.shape[0]} rows")
                returns = close_df.ffill().pct_change().dropna()
                mean_ret = returns.mean() * 252
                lw_cov_raw, shrinkage = ledoit_wolf(returns.values)
                cov_ann = pd.DataFrame(lw_cov_raw * 252, index=close_df.columns, columns=close_df.columns)
                cov_ann = cov_ann.reindex(index=ASSETS, columns=ASSETS).fillna(0.0)
                mean_ret = mean_ret.reindex(ASSETS).fillna(0.0)
                log.info(f"  📈 Market data: {len(returns)} days, shrinkage={shrinkage:.4f}")
                return mean_ret, cov_ann
            except Exception as exc:
                log.warning(f"  ⚠️  Fetch attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    time.sleep(backoff ** attempt + np.random.uniform(0, 0.5))
        log.error("🚨  All market fetches failed — using synthetic fallback.")
        rng = np.random.default_rng(0)
        fake_ret = pd.DataFrame(rng.normal(0.0006, 0.012, (252, len(ASSETS))), columns=ASSETS)
        lw_cov_raw, _ = ledoit_wolf(fake_ret.values)
        mean_ret = fake_ret.mean() * 252
        cov_ann = pd.DataFrame(lw_cov_raw * 252, index=ASSETS, columns=ASSETS)
        return mean_ret, cov_ann

    def _compute_gamma(self, profile: str, churn_prob: float) -> float:
        base = RISK_AVERSION.get(profile, 4.0)
        if churn_prob <= CHURN_THRESHOLD:
            return base
        norm_excess = (churn_prob - CHURN_THRESHOLD) / (1.0 - CHURN_THRESHOLD)
        return base * np.exp(norm_excess * GAMMA_CHURN_SCALE)

    @staticmethod
    def _build_asset_bounds(profile: str, crypto_ratio: float, tech_ratio: float):
        n = len(ASSETS)
        min_w = MIN_WEIGHT_BY_PROFILE.get(profile, 0.02)
        if n * min_w > 1.0:
            min_w = 1.0 / n
        crypto_per_asset = crypto_ratio / max(len(CRYPTO_ASSETS), 1)
        tech_per_asset = tech_ratio / max(len(TECH_ASSETS), 1)
        raw_ubs = []
        for asset in ASSETS:
            if asset in CRYPTO_ASSETS:
                raw_ubs.append(crypto_per_asset)
            elif asset in TECH_ASSETS:
                raw_ubs.append(tech_per_asset)
            else:
                raw_ubs.append(WEIGHT_MAX)
        clamped_ubs = [max(min_w, min(WEIGHT_MAX, u)) for u in raw_ubs]
        ub_sum = sum(clamped_ubs)
        if ub_sum < 1.0 - 1e-9:
            deficit = 1.0 - ub_sum
            headroom = [(i, WEIGHT_MAX - clamped_ubs[i]) for i in range(n)]
            total_room = sum(h for _, h in headroom)
            if total_room > 1e-12:
                for i, room in headroom:
                    clamped_ubs[i] += (room / total_room) * deficit
                    clamped_ubs[i] = min(WEIGHT_MAX, clamped_ubs[i])
            else:
                clamped_ubs = [WEIGHT_MAX] * n
        return [(min_w, ub) for ub in clamped_ubs]

    def get_market_context(self):
        with self._lock:
            now = datetime.now(timezone.utc)
            cache_stale = (self._market_cache is None or self._cache_timestamp is None
                           or (now - self._cache_timestamp) > MARKET_CACHE_TTL)
            if cache_stale:
                log.info("  🔄  Refreshing market data cache...")
                self._market_cache = self._fetch_market_data()
                self._cache_timestamp = now
            return self._market_cache

    def optimize_portfolio(self, profile, mean_ret, cov, churn_prob, payload):
        n = len(ASSETS)
        gamma = self._compute_gamma(profile, churn_prob)
        w0 = np.full(n, 1.0 / n)
        mu = mean_ret.values.astype(float)
        Sigma = _ensure_positive_definite(cov.values.astype(float), jitter=SIGMA_JITTER)
        bounds = self._build_asset_bounds(profile, payload.crypto_ratio, payload.tech_stocks_ratio)

        def neg_utility(w):
            ret = float(np.dot(w, mu))
            var = float(np.dot(w, np.dot(Sigma, w)))
            pen = DIVERSIFICATION_LAMBDA * float(np.dot(w - w0, w - w0))
            return -(ret - 0.5 * gamma * var - pen)

        def neg_utility_grad(w):
            return -mu + gamma * np.dot(Sigma, w) + 2.0 * DIVERSIFICATION_LAMBDA * (w - w0)

        lb_arr = np.array([b[0] for b in bounds])
        ub_arr = np.array([b[1] for b in bounds])
        x0 = np.clip(w0, lb_arr, ub_arr)
        if x0.sum() <= 0:
            x0 = lb_arr.copy()
        x0 = x0 / x0.sum()
        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0,
                        "jac": lambda w: np.ones(n)}]
        result: OptimizeResult = minimize(
            neg_utility, x0, jac=neg_utility_grad, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER}
        )
        if not result.success:
            log.warning(f"  ⚠️  SLSQP failed for {profile}: {result.message}")
            return x0
        weights = np.clip(result.x, lb_arr, ub_arr)
        total = weights.sum()
        if total <= 0:
            return x0
        return weights / total

    def predict_client(self, payload: ClientPayload) -> tuple[str, float]:
        """
        Uses the unified pipeline to get both cluster profile and churn probability.
        """
        # Build a DataFrame with all 7 features in correct order
        feat = pd.DataFrame([[
            payload.avg_holding_days,
            payload.crypto_ratio,
            payload.tech_stocks_ratio,
            payload.account_balance,
            payload.balance_velocity,
            payload.market_pain_index,
            payload.login_freq_drop,
        ]], columns=CLUSTER_FEATURES + CHURN_FEATURES)

        # Extract cluster ID from the pipeline’s first step
        cluster_injector = self.pipeline.named_steps["cluster_features"]
        scaled_cluster = cluster_injector.scaler.transform(feat[CLUSTER_FEATURES])
        cluster_id = int(cluster_injector.kmeans.predict(scaled_cluster)[0])

        # Profile label using the stored resolver
        profile_type = self.profile_resolver.transform(
            pd.DataFrame({"cluster_id": [cluster_id]})
        )[0]

        # Churn probability from the full pipeline
        churn_prob = float(self.pipeline.predict_proba(feat)[0][1])

        return profile_type, churn_prob

    def warm_up(self) -> None:
        log.info("  🔥  Engine warm‑up: pre‑fetching market data...")
        self.get_market_context()
        log.info("  ✅  Market cache populated.")