"""
AegisQuant — Online inference & retention-portfolio engine (v4.0)

Upgrade v3.2 → v4.0
────────────────────
[1] SHARED PREDICTOR
    Churn inference is delegated to inference.ChurnPredictor (also used by the
    REST API), so the engine and the API can no longer disagree.

[2] PROFILE-DRIVEN BOUNDS
    v3.x read crypto/tech appetite from synthetic client ratios. Bank customers
    have no such fields, so thematic exposure caps now come from the investor
    profile resolved by KMeans (config.PROFILE_EXPOSURE). The universe gained a
    defensive core (SPY, BND, GLD next to KO) so a conservative allocation is
    feasible without forcing money into tech or crypto. Minimum weights apply
    to core assets only; tech and crypto may go to zero.

[3] UNCHANGED FROM v3.2
    Background market-data worker (yfinance → Ledoit-Wolf covariance, synthetic
    fallback when offline), positive-definite check, churn-scaled risk aversion
    and the SLSQP solver with analytical Jacobian.

Business logic
──────────────
For a customer whose churn probability crosses the decision threshold, the
engine proposes a personalised, risk-adjusted investment allocation as a
retention offer. Higher churn risk → higher risk aversion γ → a calmer
portfolio (customers in distress should not be offered volatility).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from sklearn.covariance import ledoit_wolf

from config import (
    ASSETS,
    CRYPTO_ASSETS,
    DIVERSIFICATION_LAMBDA,
    GAMMA_CHURN_SCALE,
    MARKET_CACHE_TTL,
    MIN_WEIGHT_BY_PROFILE,
    PROFILE_EXPOSURE,
    RISK_AVERSION,
    SIGMA_JITTER,
    SLSQP_FTOL,
    SLSQP_MAXITER,
    TECH_ASSETS,
    TICKER_MAP,
    WEIGHT_MAX,
    ClientFeatures,
)
from inference import ChurnPredictor

log = logging.getLogger("aegis")

# ── Market-data type alias ────────────────────────────────────────────────────
MarketSnapshot = tuple[pd.Series, pd.DataFrame]   # (mean_ret, cov_ann)


# ═════════════════════════════════════════════════════════════════════════════
# PD helper
# ═════════════════════════════════════════════════════════════════════════════
def _ensure_positive_definite(matrix: np.ndarray, jitter: float = SIGMA_JITTER) -> np.ndarray:
    """
    Validate positive definiteness via Cholesky; add growing diagonal jitter
    until it passes. A non-PD covariance makes the quadratic objective
    non-convex and SLSQP's behaviour undefined.
    """
    scale = jitter
    for _ in range(20):
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            matrix = matrix + scale * np.eye(matrix.shape[0])
            scale *= 10.0
    raise ValueError("Covariance matrix could not be made PD after 20 jitter iterations.")


# ═════════════════════════════════════════════════════════════════════════════
# _MarketDataWorker — background daemon thread
# ═════════════════════════════════════════════════════════════════════════════
class _MarketDataWorker:
    """
    Refreshes market data on a fixed TTL in a daemon thread.

    The cache reference is swapped under a lock held for microseconds; readers
    take the reference without locking (atomic under CPython's GIL). A
    threading.Event signals the first successful (or fallback) snapshot.
    """

    def __init__(self) -> None:
        self._cache: MarketSnapshot | None = None
        self._swap_lock = threading.Lock()
        self._ready = threading.Event()
        self.source = "pending"
        self._thread = threading.Thread(target=self._run, name="aegis-market-worker", daemon=True)
        self._thread.start()

    @property
    def cache(self) -> MarketSnapshot | None:
        return self._cache

    def wait_for_first_fetch(self, timeout: float = 60.0) -> None:
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError(f"Market data worker did not complete initial fetch within {timeout}s.")

    def _run(self) -> None:
        while True:
            snapshot = self._fetch_with_retry()
            if snapshot is not None:
                with self._swap_lock:
                    self._cache = snapshot
                self.source = "yfinance"
                self._ready.set()
            elif not self._ready.is_set():
                # Never leave the engine without data: synthetic snapshot until the next refresh.
                self._cache = self._synthetic_fallback()
                self.source = "synthetic fallback (market data unavailable)"
                self._ready.set()
            time.sleep(MARKET_CACHE_TTL.total_seconds())

    @staticmethod
    def _normalise_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise yfinance MultiIndex / flat column structures → close prices."""
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                return raw["Close"]
            if "Close" in raw.columns.get_level_values(1):
                return raw.xs("Close", axis=1, level=1)
            raise ValueError("MultiIndex columns present but 'Close' not found in any level.")
        if "Close" in raw.columns:
            return raw[["Close"]].rename(columns={"Close": raw.columns[0]})
        return raw

    def _fetch_with_retry(self, retries: int = 3, backoff: float = 2.0) -> MarketSnapshot | None:
        """1-year daily closes → annualised mean returns + Ledoit-Wolf covariance."""
        import yfinance as yf   # imported lazily: only the dashboard needs live market data

        for attempt in range(1, retries + 1):
            try:
                raw = yf.download(list(TICKER_MAP.values()), period="1y", timeout=15,
                                  auto_adjust=True, progress=False)
                close = self._normalise_multiindex(raw).rename(columns={v: k for k, v in TICKER_MAP.items()})
                close = close[[a for a in ASSETS if a in close.columns]]
                missing = set(ASSETS) - set(close.columns)
                if missing:
                    raise ValueError(f"missing tickers {sorted(missing)}")
                if close.shape[0] < 50:
                    raise ValueError(f"insufficient market data: {close.shape[0]} rows")

                returns = close.ffill().pct_change().dropna()
                lw_cov, shrinkage = ledoit_wolf(returns.to_numpy())
                cov_ann = pd.DataFrame(lw_cov * 252, index=close.columns, columns=close.columns).loc[ASSETS, ASSETS]
                mean_ret = (returns.mean() * 252).reindex(ASSETS)
                log.info(f"  [worker] Market refresh OK — {len(returns)} days, LW shrinkage={shrinkage:.4f}")
                return mean_ret, cov_ann
            except Exception as exc:
                log.warning(f"  [worker] Fetch attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    time.sleep(backoff ** attempt + np.random.uniform(0.0, 0.5))
        log.error("  [worker] All market fetch attempts failed — using synthetic fallback.")
        return None

    @staticmethod
    def _synthetic_fallback() -> MarketSnapshot:
        """Emergency fallback: Gaussian synthetic returns (clearly labelled in the UI)."""
        rng = np.random.default_rng(0)
        fake = pd.DataFrame(rng.normal(0.0006, 0.012, (252, len(ASSETS))), columns=ASSETS)
        lw_cov, _ = ledoit_wolf(fake.to_numpy())
        return fake.mean() * 252, pd.DataFrame(lw_cov * 252, index=ASSETS, columns=ASSETS)


# ═════════════════════════════════════════════════════════════════════════════
# AegisQuantEngine
# ═════════════════════════════════════════════════════════════════════════════
class AegisQuantEngine:
    """
    Churn inference (ChurnPredictor) + Dynamic Markowitz retention portfolio.

    Objective (maximise):
        U(w) = μᵀw − (γ/2)·wᵀΣw − λ‖w − w₀‖²₂
    Constraints:
        Σwᵢ = 1;  core assets in [min_w[profile], WEIGHT_MAX];
        tech / crypto assets in [0, profile cap / number of assets in the theme]
    """

    def __init__(self, start_market_worker: bool = True) -> None:
        self.predictor = ChurnPredictor()
        self.threshold = self.predictor.threshold
        self._worker = _MarketDataWorker() if start_market_worker else None

    # ── Churn inference ───────────────────────────────────────────────────────
    def predict_client(self, payload: ClientFeatures) -> tuple[str, float]:
        """(investor_profile, churn_probability)."""
        return self.predictor.predict_client(payload)

    def explain_client(self, payload: ClientFeatures) -> dict:
        return self.predictor.explain(payload)

    # ── Market data ───────────────────────────────────────────────────────────
    def get_market_context(self) -> MarketSnapshot:
        if self._worker is None:
            return _MarketDataWorker._synthetic_fallback()
        if self._worker.cache is None:
            self._worker.wait_for_first_fetch(timeout=60.0)
        return self._worker.cache   # type: ignore[return-value]

    @property
    def market_source(self) -> str:
        return self._worker.source if self._worker else "synthetic (worker disabled)"

    def warm_up(self) -> None:
        if self._worker is not None:
            self._worker.wait_for_first_fetch(timeout=60.0)

    # ── Gamma scaling ─────────────────────────────────────────────────────────
    def _compute_gamma(self, profile: str, churn_prob: float) -> float:
        """
        Churn-scaled risk aversion. Below the decision threshold γ = γ_base;
        above it γ = γ_base · exp(excess · GAMMA_CHURN_SCALE) with
        excess = (p − threshold) / (1 − threshold) ∈ [0, 1] → multiplier ≤ e.
        """
        base = RISK_AVERSION.get(profile, 4.0)
        if churn_prob <= self.threshold:
            return base
        excess = (churn_prob - self.threshold) / (1.0 - self.threshold)
        return base * float(np.exp(excess * GAMMA_CHURN_SCALE))

    # ── Bounds ────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_asset_bounds(profile: str) -> list[tuple[float, float]]:
        """
        Per-asset (lb, ub), feasible by construction:
          1. Thematic caps from the profile: crypto / tech share split evenly.
          2. Core assets: [min_w, WEIGHT_MAX]; thematic assets: [0, cap].
          3. Repair: if Σub < 1, raise core ubs (never thematic caps) to fit.
        """
        exposure = PROFILE_EXPOSURE.get(profile, PROFILE_EXPOSURE["balanced"])
        crypto_cap = exposure["crypto"] / max(len(CRYPTO_ASSETS), 1)
        tech_cap = exposure["tech"] / max(len(TECH_ASSETS), 1)
        core = [a for a in ASSETS if a not in CRYPTO_ASSETS and a not in TECH_ASSETS]
        min_w = min(MIN_WEIGHT_BY_PROFILE.get(profile, 0.02), 1.0 / len(core))

        bounds: dict[str, list[float]] = {}
        for asset in ASSETS:
            if asset in CRYPTO_ASSETS:
                bounds[asset] = [0.0, min(WEIGHT_MAX, crypto_cap)]
            elif asset in TECH_ASSETS:
                bounds[asset] = [0.0, min(WEIGHT_MAX, tech_cap)]
            else:
                bounds[asset] = [min_w, WEIGHT_MAX]

        deficit = 1.0 - sum(ub for _, ub in bounds.values())
        if deficit > 1e-9:
            for asset in core:
                bounds[asset][1] += deficit / len(core)
        return [tuple(bounds[a]) for a in ASSETS]

    # ── Markowitz SLSQP ───────────────────────────────────────────────────────
    def optimize_portfolio(
        self,
        profile: str,
        mean_ret: pd.Series,
        cov: pd.DataFrame,
        churn_prob: float,
        payload: ClientFeatures | None = None,   # kept for the v3.x call signature
    ) -> np.ndarray:
        """Solve the Dynamic Markowitz problem; returns weights summing to 1 (ASSETS order)."""
        n = len(ASSETS)
        gamma = self._compute_gamma(profile, churn_prob)
        mu = mean_ret.reindex(ASSETS).to_numpy(float)
        Sigma = _ensure_positive_definite(cov.loc[ASSETS, ASSETS].to_numpy(float), jitter=SIGMA_JITTER)
        bounds = self._build_asset_bounds(profile)
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        w0 = np.clip(np.full(n, 1.0 / n), lb, ub)
        w0 = w0 / w0.sum()

        def neg_utility(w: np.ndarray) -> float:
            return -(float(w @ mu) - 0.5 * gamma * float(w @ Sigma @ w)
                     - DIVERSIFICATION_LAMBDA * float((w - w0) @ (w - w0)))

        def neg_utility_grad(w: np.ndarray) -> np.ndarray:
            # ∂/∂w = −μ + γΣw + 2λ(w − w₀)
            return -mu + gamma * (Sigma @ w) + 2.0 * DIVERSIFICATION_LAMBDA * (w - w0)

        result: OptimizeResult = minimize(
            neg_utility, w0, jac=neg_utility_grad, method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0, "jac": lambda w: np.ones(n)}],
            options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER},
        )
        if not result.success:
            log.warning(f"  SLSQP did not converge for profile={profile}: {result.message}. Returning feasible start.")
            return w0
        weights = np.clip(result.x, lb, ub)
        return weights / weights.sum()

    def portfolio_stats(self, weights: np.ndarray, mean_ret: pd.Series, cov: pd.DataFrame) -> dict:
        mu = mean_ret.reindex(ASSETS).to_numpy(float)
        vol = float(np.sqrt(weights @ cov.loc[ASSETS, ASSETS].to_numpy(float) @ weights))
        return {"expected_return": float(weights @ mu), "volatility": vol}
