"""
AegisQuant — Online inference engine (v3.2)

Upgrade v3.1 → v3.2
────────────────────
[1] SECURE & NATIVE SERIALISATION
    AegisQuantEngine no longer loads the combined joblib pickle.  Instead it
    calls the three dedicated loaders from feature_cross_pollination:
      • load_xgb_model()   → XGBClassifier from aegis_xgb.json
      • load_processor()   → ProcessorBundle (scaler + kmeans) from joblib
      • joblib.load(PROFILE_RESOLVER_PATH) → DynamicProfileResolver
    This eliminates the arbitrary-code-execution risk of pickle deserialization
    on the XGBoost model.

[2] ASYNC / NON-BLOCKING MARKET DATA FETCHING
    The v3.1 design called yf.download() synchronously inside a threading.Lock,
    meaning every incoming request that found a stale cache had to wait for a
    full network round-trip (~2-8 s) while holding the lock — blocking ALL
    concurrent requests.

    v3.2 pattern: GIL-safe atomic background worker
    ─────────────────────────────────────────────────
    • A single daemon thread (_MarketDataWorker) runs an infinite loop:
        sleep(TTL) → fetch → atomically replace _cache via threading.Event.
    • The cache variable is replaced atomically using a threading.Lock only
      around the pointer swap (microseconds), never around network I/O.
    • get_market_context() reads the current cache snapshot WITHOUT acquiring
      any lock — Python's GIL guarantees that reading an object reference is
      atomic on CPython.  The pattern is:
        cache_ref = self._worker.cache   # atomic read under GIL
        return cache_ref
    • If the worker has not yet completed its first fetch (startup), a
      threading.Event blocks get_market_context() until data is available.
      Subsequent calls return instantly from the pre-populated cache.
    • This design ensures zero lock contention on the hot path (inference loop).

    Security note: _MarketDataWorker is a daemon thread — it is automatically
    killed when the main process exits, preventing zombie threads.

[3] MARKOWITZ SLSQP — unchanged from v3.1
    All mathematical constraints, the feasibility-repair layer
    (_build_asset_bounds), PD jitter, and the analytical Jacobian are
    preserved exactly.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, OptimizeResult
from sklearn.covariance import ledoit_wolf

from config import (
    ASSETS,
    TICKER_MAP,
    CRYPTO_ASSETS,
    TECH_ASSETS,
    CLUSTER_FEATURES,
    CHURN_FEATURES,
    ALL_FEATURES,
    N_CLUSTERS,
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
    PROFILE_RESOLVER_PATH,
    ClientFeatures,
)
from data_pipeline import DynamicProfileResolver
from feature_cross_pollination import (
    ProcessorBundle,
    load_xgb_model,
    load_processor,
    load_artifact_meta,
)

log = logging.getLogger("aegis")

# ── Market-data type alias ────────────────────────────────────────────────────
MarketSnapshot = tuple[pd.Series, pd.DataFrame]   # (mean_ret, cov_ann)


# ═════════════════════════════════════════════════════════════════════════════
# PD helper (unchanged from v3.1)
# ═════════════════════════════════════════════════════════════════════════════
def _ensure_positive_definite(
    matrix: np.ndarray,
    jitter: float = SIGMA_JITTER,
) -> np.ndarray:
    """
    Validate that `matrix` is positive definite via Cholesky decomposition.
    Adds progressively larger diagonal jitter until the check passes.
    Critical before SLSQP: a non-PD covariance matrix makes the quadratic
    objective non-convex, causing undefined solver behaviour.
    """
    scale = jitter
    for _ in range(20):
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            matrix = matrix + scale * np.eye(matrix.shape[0])
            scale  *= 10.0
    raise ValueError(
        "Covariance matrix could not be made PD after 20 jitter iterations. "
        "Check your market data for degenerate return series."
    )


# ═════════════════════════════════════════════════════════════════════════════
# _MarketDataWorker — background daemon thread
# ═════════════════════════════════════════════════════════════════════════════
class _MarketDataWorker:
    """
    Background daemon thread that refreshes market data on a fixed TTL schedule.

    Design
    ──────
    • __init__ starts the daemon thread immediately.
    • The thread sleeps for MARKET_CACHE_TTL.total_seconds() between refreshes.
    • After each successful fetch, the cache reference is replaced atomically
      via a threading.Lock that is held for microseconds (pointer swap only).
    • A threading.Event (_ready) signals the first successful fetch so that
      callers blocked in get_market_context() can proceed.

    Why not asyncio?
    ────────────────
    The optimizer is used from both FastAPI (ASGI async) and Streamlit (sync).
    A daemon thread avoids the need to inject an event loop and works correctly
    in both contexts.  The network I/O inside yf.download() releases the GIL,
    so the worker does not meaningfully block other threads during the fetch.
    """

    def __init__(self) -> None:
        self._cache: MarketSnapshot | None = None
        self._swap_lock = threading.Lock()   # held for pointer swap only
        self._ready     = threading.Event()  # set after first successful fetch

        self._thread = threading.Thread(
            target=self._run,
            name="aegis-market-worker",
            daemon=True,   # killed automatically when the main process exits
        )
        self._thread.start()

    # ── Public interface ──────────────────────────────────────────────────────
    @property
    def cache(self) -> MarketSnapshot | None:
        """
        Atomic read of the current market snapshot.
        Under CPython's GIL, reading an object reference is atomic.
        No explicit locking required for reads.
        """
        return self._cache

    def wait_for_first_fetch(self, timeout: float = 60.0) -> None:
        """
        Block until the first successful market fetch completes.
        Called once from AegisQuantEngine.__init__ / warm_up() so that the
        first inference request is never delayed by a cold cache.
        Raises RuntimeError if the timeout expires.
        """
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError(
                "Market data worker did not complete initial fetch within "
                f"{timeout}s. Check yfinance connectivity."
            )

    # ── Internal fetch + refresh loop ────────────────────────────────────────
    def _run(self) -> None:
        """Infinite loop: fetch → swap cache → sleep → repeat."""
        while True:
            snapshot = self._fetch_with_retry()
            if snapshot is not None:
                with self._swap_lock:       # pointer swap — microseconds
                    self._cache = snapshot
                if not self._ready.is_set():
                    self._ready.set()       # unblock any waiting callers
            else:
                # If even the fallback failed, use synthetic data so the engine
                # is never in a permanently broken state.
                if not self._ready.is_set():
                    self._cache = self._synthetic_fallback()
                    self._ready.set()

            # Sleep for the full TTL before the next refresh.
            time.sleep(MARKET_CACHE_TTL.total_seconds())

    @staticmethod
    def _normalise_multiindex(raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise yfinance MultiIndex / flat column structures → close prices."""
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                return raw["Close"]
            if "Close" in raw.columns.get_level_values(1):
                return raw.xs("Close", axis=1, level=1)
            raise ValueError(
                "MultiIndex columns present but 'Close' not found in any level. "
                f"Levels: {[list(raw.columns.get_level_values(i)) for i in range(raw.columns.nlevels)]}"
            )
        if "Close" in raw.columns:
            return raw[["Close"]].rename(columns={"Close": raw.columns[0]})
        return raw

    def _fetch_with_retry(
        self,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> MarketSnapshot | None:
        """
        Download 1-year OHLCV from yfinance, compute annualised mean returns
        and Ledoit-Wolf covariance.  Returns None if all retries fail
        (caller will fall back to synthetic data).
        """
        for attempt in range(1, retries + 1):
            try:
                tickers      = list(TICKER_MAP.values())
                raw_download = yf.download(
                    tickers,
                    period="1y",
                    timeout=15,
                    auto_adjust=True,
                    progress=False,
                )
                close_df = self._normalise_multiindex(raw_download)
                close_df = close_df.rename(columns={v: k for k, v in TICKER_MAP.items()})
                close_df = close_df[[a for a in ASSETS if a in close_df.columns]]

                missing = set(ASSETS) - set(close_df.columns)
                if missing:
                    log.warning(f"  [worker] Missing tickers: {missing}")

                if close_df.empty or close_df.shape[0] < 50:
                    raise ValueError(
                        f"Insufficient market data: {close_df.shape[0]} rows returned."
                    )

                returns  = close_df.ffill().pct_change().dropna()
                mean_ret = returns.mean() * 252

                lw_cov_raw, shrinkage = ledoit_wolf(returns.values)
                cov_ann = pd.DataFrame(
                    lw_cov_raw * 252,
                    index=close_df.columns,
                    columns=close_df.columns,
                )
                cov_ann  = cov_ann.reindex(index=ASSETS, columns=ASSETS).fillna(0.0)
                mean_ret = mean_ret.reindex(ASSETS).fillna(0.0)

                log.info(
                    f"  [worker] Market refresh OK — "
                    f"{len(returns)} days, LW shrinkage={shrinkage:.4f}"
                )
                return mean_ret, cov_ann

            except Exception as exc:
                log.warning(f"  [worker] Fetch attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    sleep_s = backoff ** attempt + np.random.uniform(0.0, 0.5)
                    time.sleep(sleep_s)

        log.error("  [worker] All market fetch attempts failed — using synthetic fallback.")
        return None

    @staticmethod
    def _synthetic_fallback() -> MarketSnapshot:
        """Emergency fallback: Gaussian synthetic returns."""
        rng      = np.random.default_rng(0)
        fake_ret = pd.DataFrame(
            rng.normal(0.0006, 0.012, (252, len(ASSETS))),
            columns=ASSETS,
        )
        lw_cov_raw, _ = ledoit_wolf(fake_ret.values)
        mean_ret       = fake_ret.mean() * 252
        cov_ann        = pd.DataFrame(lw_cov_raw * 252, index=ASSETS, columns=ASSETS)
        return mean_ret, cov_ann


# ═════════════════════════════════════════════════════════════════════════════
# AegisQuantEngine
# ═════════════════════════════════════════════════════════════════════════════
class AegisQuantEngine:
    """
    Online inference engine.

    Component loading (v3.2)
    ────────────────────────
    Loads three independently-serialised artifacts:
      1. XGBoost native JSON → XGBClassifier  (no pickle)
      2. ProcessorBundle joblib → ClusterInjector logic (scaler + kmeans)
      3. DynamicProfileResolver joblib → cluster-to-archetype mapping

    Market data (v3.2)
    ──────────────────
    A _MarketDataWorker daemon thread pre-fetches and periodically refreshes
    the market snapshot in the background.  get_market_context() reads the
    cached value without holding any lock, providing zero-latency access on
    the hot inference path.

    Markowitz optimisation (unchanged from v3.1)
    ────────────────────────────────────────────
    Objective (maximise):
        U(w) = μᵀw − (γ/2)·wᵀΣw − λ‖w − w₀‖²₂

    Constraints:
        Σwᵢ = 1            (budget)
        wᵢ ≥ min_w[profile]
        wᵢ ≤ per-asset ub  (feasibility-repair layer)
    """

    def __init__(self) -> None:
        self._bootstrap_if_needed()
        self._load_artifacts()

        # Start background market-data worker immediately
        self._worker = _MarketDataWorker()

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    def _bootstrap_if_needed(self) -> None:
        """Run training pipeline if native artifacts are absent."""
        from config import XGB_MODEL_PATH, PROCESSOR_PATH, PROFILE_RESOLVER_PATH
        missing = [
            p for p in (XGB_MODEL_PATH, PROCESSOR_PATH, PROFILE_RESOLVER_PATH)
            if not os.path.exists(p)
        ]
        if missing:
            log.warning(
                f"  Artifacts missing ({missing}) — running training pipeline..."
            )
            from churn_model import run_training_pipeline
            run_training_pipeline()

    # ── Artifact loading ──────────────────────────────────────────────────────
    def _load_artifacts(self) -> None:
        """Load all three modular artifacts with clear error messages."""
        # 1. XGBoost model (native JSON)
        self.xgb_model = load_xgb_model()

        # 2. sklearn ProcessorBundle (scaler + kmeans)
        self.processor: ProcessorBundle = load_processor()

        # 3. DynamicProfileResolver
        try:
            self.profile_resolver: DynamicProfileResolver = joblib.load(
                PROFILE_RESOLVER_PATH
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Profile resolver '{PROFILE_RESOLVER_PATH}' not found. "
                "Run churn_model.run_training_pipeline() first."
            )

        # 4. Metadata (non-critical — used for logging only)
        meta = load_artifact_meta()
        log.info(
            f"  AegisQuantEngine ready | "
            f"Val AUC: {meta.get('val_auc', 'N/A')} | "
            f"Trained: {meta.get('trained_at', 'N/A')}"
        )

    # ── Market data access (zero-latency hot path) ────────────────────────────
    def get_market_context(self) -> MarketSnapshot:
        """
        Return the current market snapshot from the background worker cache.

        This method is lock-free on the hot path: Python's GIL guarantees
        that reading a reference is atomic on CPython.  The background worker
        uses a lock only during the pointer-swap step (microseconds).

        Blocks once (up to 60 s) on the very first call if the worker has
        not yet completed its initial fetch — this is handled by warm_up().
        """
        snapshot = self._worker.cache
        if snapshot is None:
            # Should only happen if get_market_context() is called before warm_up()
            log.warning("  Market cache empty — waiting for background worker...")
            self._worker.wait_for_first_fetch(timeout=60.0)
            snapshot = self._worker.cache
        return snapshot   # type: ignore[return-value]

    def warm_up(self) -> None:
        """
        Block until the background worker's first market fetch completes.
        Call this once after engine construction (e.g. in Streamlit's
        @st.cache_resource factory) so that the first inference request
        never experiences network I/O latency.
        """
        log.info("  [warm-up] Waiting for initial market data fetch...")
        self._worker.wait_for_first_fetch(timeout=60.0)
        log.info("  [warm-up] Market cache populated — engine ready.")

    # ── Gamma scaling (v3.1 logic preserved exactly) ──────────────────────────
    def _compute_gamma(self, profile: str, churn_prob: float) -> float:
        """
        Churn-scaled risk-aversion coefficient γ.

        Below threshold: γ = γ_base (no scaling).
        Above threshold: γ = γ_base × exp(norm_excess × GAMMA_CHURN_SCALE)
          where norm_excess = (churn_prob − 0.5) / 0.5 ∈ [0, 1].
        Multiplier range: [1.0, e^1 ≈ 2.72].
        """
        base = RISK_AVERSION.get(profile, 4.0)
        if churn_prob <= CHURN_THRESHOLD:
            return base
        norm_excess = (churn_prob - CHURN_THRESHOLD) / (1.0 - CHURN_THRESHOLD)
        return base * np.exp(norm_excess * GAMMA_CHURN_SCALE)

    # ── Feasibility-repair layer (v3.1 logic preserved exactly) ──────────────
    @staticmethod
    def _build_asset_bounds(
        profile: str,
        crypto_ratio: float,
        tech_ratio: float,
    ) -> list[tuple[float, float]]:
        """
        Construct per-asset (lb, ub) box constraints guaranteed to be feasible:
          Sum(UBs) >= 1.0  and  lb <= ub  for all assets.

        4-step pipeline:
          1. Compute raw preference UBs from client crypto/tech ratios.
          2. Clamp each raw UB to [min_w, WEIGHT_MAX].
          3. If Sum(clamped_UBs) < 1.0: distribute the deficit proportionally
             to assets with headroom (ub < WEIGHT_MAX).
          4. Return (min_w, ub) pairs — all feasible by construction.
        """
        n     = len(ASSETS)
        min_w = MIN_WEIGHT_BY_PROFILE.get(profile, 0.02)
        if n * min_w > 1.0:
            min_w = 1.0 / n

        crypto_per = crypto_ratio / max(len(CRYPTO_ASSETS), 1)
        tech_per   = tech_ratio   / max(len(TECH_ASSETS),   1)

        # Step 1: raw preference UBs
        raw_ubs: list[float] = []
        for asset in ASSETS:
            if asset in CRYPTO_ASSETS:
                raw_ubs.append(crypto_per)
            elif asset in TECH_ASSETS:
                raw_ubs.append(tech_per)
            else:
                raw_ubs.append(WEIGHT_MAX)

        # Step 2: clamp to [min_w, WEIGHT_MAX]
        clamped: list[float] = [max(min_w, min(WEIGHT_MAX, u)) for u in raw_ubs]

        # Step 3: feasibility repair
        ub_sum = sum(clamped)
        if ub_sum < 1.0 - 1e-9:
            deficit    = 1.0 - ub_sum
            headroom   = [(i, WEIGHT_MAX - clamped[i]) for i in range(n)]
            total_room = sum(h for _, h in headroom)
            if total_room > 1e-12:
                for i, room in headroom:
                    clamped[i] = min(WEIGHT_MAX, clamped[i] + (room / total_room) * deficit)
            else:
                clamped = [WEIGHT_MAX] * n

        # Step 4: final (lb, ub) pairs
        return [(min_w, ub) for ub in clamped]

    # ── Markowitz SLSQP optimizer (v3.1 math unchanged) ──────────────────────
    def optimize_portfolio(
        self,
        profile:    str,
        mean_ret:   pd.Series,
        cov:        pd.DataFrame,
        churn_prob: float,
        payload:    ClientFeatures,
    ) -> np.ndarray:
        """
        Solve the Dynamic Markowitz problem.

        Objective (maximise):
            U(w) = μᵀw − (γ/2)·wᵀΣw − λ‖w − w₀‖²₂

        γ is churn-scaled. Σ is PD-guaranteed. Bounds include the feasibility-
        repair layer. Analytical Jacobian plugged directly into SLSQP.

        Returns
        ───────
        np.ndarray of shape (n_assets,) — normalised weights summing to 1.
        """
        n     = len(ASSETS)
        gamma = self._compute_gamma(profile, churn_prob)
        w0    = np.full(n, 1.0 / n)
        mu    = mean_ret.values.astype(float)
        Sigma = _ensure_positive_definite(cov.values.astype(float), jitter=SIGMA_JITTER)
        bounds = self._build_asset_bounds(
            profile, payload.crypto_ratio, payload.tech_stocks_ratio
        )

        lb_arr = np.array([b[0] for b in bounds])
        ub_arr = np.array([b[1] for b in bounds])

        def neg_utility(w: np.ndarray) -> float:
            ret = float(np.dot(w, mu))
            var = float(np.dot(w, np.dot(Sigma, w)))
            pen = DIVERSIFICATION_LAMBDA * float(np.dot(w - w0, w - w0))
            return -(ret - 0.5 * gamma * var - pen)

        def neg_utility_grad(w: np.ndarray) -> np.ndarray:
            """
            Analytical gradient:
              ∂/∂w [ -(μᵀw − γ/2·wᵀΣw − λ‖w−w₀‖²) ]
              = −μ + γ·Σw + 2λ(w − w₀)
            """
            return -mu + gamma * np.dot(Sigma, w) + 2.0 * DIVERSIFICATION_LAMBDA * (w - w0)

        x0 = np.clip(w0, lb_arr, ub_arr)
        if x0.sum() <= 0:
            x0 = lb_arr.copy()
        x0 = x0 / x0.sum()

        constraints = [
            {
                "type": "eq",
                "fun":  lambda w: float(np.sum(w)) - 1.0,
                "jac":  lambda w: np.ones(n),
            }
        ]

        result: OptimizeResult = minimize(
            neg_utility,
            x0,
            jac=neg_utility_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER},
        )

        if not result.success:
            log.warning(
                f"  SLSQP did not converge for profile={profile}: "
                f"{result.message}. Returning feasible fallback point."
            )
            return x0

        weights = np.clip(result.x, lb_arr, ub_arr)
        total   = weights.sum()
        return weights / total if total > 0 else x0

    # ── Client inference ──────────────────────────────────────────────────────
    def predict_client(self, payload: ClientFeatures) -> tuple[str, float]:
        """
        Two-stage inference:

        Stage 1 — KMeans investor segmentation (via ProcessorBundle):
          Transform (avg_holding_days, crypto_ratio, tech_stocks_ratio) through
          the fitted StandardScaler and predict cluster ID.  Resolve to archetype
          string via DynamicProfileResolver.

        Stage 2 — Churn prediction (via XGBoost native model):
          Build the full transformed feature vector through ProcessorBundle.transform()
          (replicating ClusterInjector.transform() without the sklearn Pipeline),
          then call xgb_model.predict_proba().

        Returns
        ───────
        (profile_type: str, churn_prob: float)
        """
        # Build full 7-feature input DataFrame
        feat = pd.DataFrame(
            [[
                payload.avg_holding_days,
                payload.crypto_ratio,
                payload.tech_stocks_ratio,
                payload.account_balance,
                payload.balance_velocity,
                payload.market_pain_index,
                payload.login_freq_drop,
            ]],
            columns=ALL_FEATURES,
        )

        # Stage 1: cluster → profile
        cluster_arr = feat[CLUSTER_FEATURES].copy()
        cluster_id  = int(self.processor.predict_cluster(cluster_arr)[0])
        profile_type = self.profile_resolver.transform(
            pd.DataFrame({"cluster_id": [cluster_id]})
        )[0]

        # Stage 2: transform → XGBoost churn probability
        X_transformed = self.processor.transform(feat)
        churn_prob    = float(self.xgb_model.predict_proba(X_transformed)[0][1])

        return profile_type, churn_prob
