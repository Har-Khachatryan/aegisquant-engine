"""
AegisQuant — AI Portfolio Shield & Risk Engine
Global configuration, hyperparameters, and data contracts.

Upgrade v3.1 → v3.2
────────────────────
[DRY / SINGLE SOURCE OF TRUTH]
  - ClientFeatures (api.py) and ClientPayload (config.py) were two separate
    Pydantic models describing the same domain object.  Any field-level change
    had to be applied in two places — a maintenance hazard.
  - Resolution: both are replaced by the single canonical model ClientFeatures,
    defined here and imported everywhere else.
  - ClientFeatures is a strict superset of the old ClientPayload:
      • All financial bounds preserved (Field ge/le/gt metadata).
      • @model_validator cross-field check (crypto + tech ≤ 1.0) preserved.
      • Optional UI-facing fields (client_id, description) carry defaults so
        the FastAPI endpoint, which does NOT supply them, passes validation
        without modification.
  - api.py, optimizer.py, and app.py all import ClientFeatures from config;
    no other Pydantic model for client data exists in the codebase.

Artifact paths
────────────────
  ARTIFACT_PATH         — legacy single-pickle path (kept for monitor_and_retrain)
  XGB_MODEL_PATH        — native XGBoost JSON (secure, version-stable)
  PROCESSOR_PATH        — joblib file containing sklearn components only
  PROFILE_RESOLVER_PATH — joblib file for DynamicProfileResolver mapping
  ARTIFACT_META_PATH    — JSON sidecar: val_auc, trained_at, feature_names
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from pydantic import BaseModel, Field, model_validator

# ── Artifact storage ──────────────────────────────────────────────────────────
# Legacy combined pickle (kept for backward-compat with DriftMonitor/monitor_and_retrain)
ARTIFACT_PATH: str = "aegis_quant_artifacts.pkl"

# v3.2 modular artifact paths (secure native serialisation)
XGB_MODEL_PATH:        str = "aegis_xgb.json"
PROCESSOR_PATH:        str = "aegis_processor.joblib"
PROFILE_RESOLVER_PATH: str = "aegis_profile_resolver.joblib"
ARTIFACT_META_PATH:    str = "aegis_artifact_meta.json"

# Reference dataset for drift monitoring
REFERENCE_DATA_PATH: str = "reference_data.pkl"

# ── Asset universe ────────────────────────────────────────────────────────────
ASSETS: list[str] = ["AAPL", "MSFT", "KO", "NVDA", "TSLA", "BTC", "ETH"]

TICKER_MAP: dict[str, str] = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "KO":   "KO",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
}

CRYPTO_ASSETS: list[str] = ["BTC", "ETH"]
TECH_ASSETS:   list[str] = ["AAPL", "MSFT", "NVDA", "TSLA"]

# ── Feature column sets ───────────────────────────────────────────────────────
CLUSTER_FEATURES: list[str] = ["avg_holding_days", "crypto_ratio", "tech_stocks_ratio"]

CHURN_FEATURES: list[str] = [
    "account_balance",
    "balance_velocity",
    "market_pain_index",
    "login_freq_drop",
]

# Canonical ordered feature list fed into the unified pipeline
ALL_FEATURES: list[str] = CLUSTER_FEATURES + CHURN_FEATURES

# Number of KMeans clusters — must match ClusterInjector.N_CLUSTERS
N_CLUSTERS: int = 3

# ── Risk aversion base values (γ) per investor archetype ─────────────────────
RISK_AVERSION: dict[str, float] = {
    "conservative": 8.0,
    "balanced":     4.0,
    "aggressive":   1.5,
}

# ── Optimisation constraints ──────────────────────────────────────────────────
WEIGHT_MAX: float = 0.40
MARKET_CACHE_TTL: timedelta = timedelta(hours=1)
CHURN_THRESHOLD: float = 0.50
DIVERSIFICATION_LAMBDA: float = 0.20
MIN_WEIGHT_BY_PROFILE: dict[str, float] = {
    "aggressive":   0.02,
    "balanced":     0.05,
    "conservative": 0.08,
}
GAMMA_CHURN_SCALE: float = 1.0
SIGMA_JITTER: float = 1e-8
SLSQP_FTOL: float = 1e-10
SLSQP_MAXITER: int = 2_000


# ── Unified client data contract (Pydantic v2) ────────────────────────────────
class ClientFeatures(BaseModel):
    """
    Single canonical Pydantic model for all client data flowing through
    AegisQuant.  Replaces both the old ClientPayload (config.py) and the
    old ClientFeatures (api.py).

    Field semantics
    ───────────────
    client_id         — internal identifier; optional for the REST API endpoint
    description       — human-readable label; optional, UI/logging only
    avg_holding_days  — mean position holding period in calendar days [1, 3650]
    crypto_ratio      — fraction of AUM the client allocates to crypto [0, 1]
    tech_stocks_ratio — fraction of AUM the client allocates to tech equities [0, 1]
    account_balance   — total investable AUM in USD (strict positive)
    balance_velocity  — normalised net-flow rate; >1 = inflows, <1 = outflows [0, 5]
    market_pain_index — proprietary drawdown / stress composite score [0, 1]
    login_freq_drop   — ratio of login frequency decline; 0 = stable, 5 = gone dark [0, 5]

    Cross-field invariant
    ─────────────────────
    crypto_ratio + tech_stocks_ratio <= 1.0
    Ensures the two declared thematic exposures do not exceed 100 % of AUM.
    Validated after individual field validators via @model_validator(mode="after").

    Compatibility notes
    ───────────────────
    - FastAPI endpoint: does not supply client_id or description -> defaults apply.
    - Streamlit dashboard: supplies all fields from DEMO_CLIENTS dicts.
    - optimizer.predict_client: accesses all seven numeric fields.
    """
    # UI / tracking fields — optional so the REST API never has to supply them
    client_id:         Optional[int] = None
    description:       str           = ""

    # Investor behaviour features (used by ClusterInjector)
    avg_holding_days:  int   = Field(ge=1,   le=3_650)
    crypto_ratio:      float = Field(ge=0.0, le=1.0)
    tech_stocks_ratio: float = Field(ge=0.0, le=1.0)

    # Financial health features (used by XGBoost churn model)
    account_balance:   float = Field(gt=0.0)
    balance_velocity:  float = Field(ge=0.0, le=5.0)
    market_pain_index: float = Field(ge=0.0, le=1.0)
    login_freq_drop:   float = Field(ge=0.0, le=5.0)

    @model_validator(mode="after")
    def ratios_must_not_exceed_one(self) -> "ClientFeatures":
        """Combined thematic exposure cannot exceed total investable AUM."""
        total = self.crypto_ratio + self.tech_stocks_ratio
        if total > 1.0:
            raise ValueError(
                f"crypto_ratio ({self.crypto_ratio:.2f}) + "
                f"tech_stocks_ratio ({self.tech_stocks_ratio:.2f}) "
                f"= {total:.2f} > 1.0"
            )
        return self


# ── Backward-compatibility alias ─────────────────────────────────────────────
# app.py imports ClientPayload from config — keep this alias so the Streamlit
# dashboard continues to work without any changes.
ClientPayload = ClientFeatures


# ── SHAP feature-name mapping ─────────────────────────────────────────────────
# Order matches ALL_FEATURES + one-hot cluster columns produced by ClusterInjector:
#   f0=avg_holding_days, f1=crypto_ratio, f2=tech_stocks_ratio,
#   f3=account_balance,  f4=balance_velocity, f5=market_pain_index,
#   f6=login_freq_drop,  f7=cluster_0, f8=cluster_1, f9=cluster_2
SHAP_FEATURE_REASON_MAP: dict[str, str] = {
    "avg_holding_days":  "Short average holding period (high turnover)",
    "crypto_ratio":      "High crypto exposure (elevated volatility risk)",
    "tech_stocks_ratio": "High tech concentration (sector drawdown sensitivity)",
    "account_balance":   "Low account balance (limited loss cushion)",
    "balance_velocity":  "Negative balance velocity (net outflows detected)",
    "market_pain_index": "Elevated market pain index (drawdown stress)",
    "login_freq_drop":   "Sudden drop in login frequency (disengagement signal)",
    "cluster_0":         "Cluster 0 membership (investor segmentation signal)",
    "cluster_1":         "Cluster 1 membership (investor segmentation signal)",
    "cluster_2":         "Cluster 2 membership (investor segmentation signal)",
    # Fallback for raw fN booster names
    "f0": "Short average holding period (high turnover)",
    "f1": "High crypto exposure (elevated volatility risk)",
    "f2": "High tech concentration (sector drawdown sensitivity)",
    "f3": "Low account balance (limited loss cushion)",
    "f4": "Negative balance velocity (net outflows detected)",
    "f5": "Elevated market pain index (drawdown stress)",
    "f6": "Sudden drop in login frequency (disengagement signal)",
    "f7": "Cluster 0 membership (investor segmentation signal)",
    "f8": "Cluster 1 membership (investor segmentation signal)",
    "f9": "Cluster 2 membership (investor segmentation signal)",
}

# ── Demo batch for the Streamlit dashboard ───────────────────────────────────
DEMO_CLIENTS: list[dict] = [
    {
        "client_id": 9001, "description": "Crypto Enthusiast — Aggressive",
        "avg_holding_days": 6,   "crypto_ratio": 0.75, "tech_stocks_ratio": 0.20,
        "account_balance": 42_000,  "balance_velocity": 0.62,
        "market_pain_index": 0.65,  "login_freq_drop": 0.45,
    },
    {
        "client_id": 9002, "description": "Institutional HNW — Solid Inflows",
        "avg_holding_days": 185, "crypto_ratio": 0.00, "tech_stocks_ratio": 0.15,
        "account_balance": 450_000, "balance_velocity": 1.20,
        "market_pain_index": 0.05,  "login_freq_drop": 1.40,
    },
    {
        "client_id": 9003, "description": "Retail Balanced — Drawdown Frustration",
        "avg_holding_days": 55,  "crypto_ratio": 0.25, "tech_stocks_ratio": 0.35,
        "account_balance": 18_000,  "balance_velocity": 0.42,
        "market_pain_index": 0.88,  "login_freq_drop": 0.28,
    },
    {
        "client_id": 9004, "description": "Active Growth Trader — Marginal Decay",
        "avg_holding_days": 15,  "crypto_ratio": 0.50, "tech_stocks_ratio": 0.40,
        "account_balance": 85_000,  "balance_velocity": 0.58,
        "market_pain_index": 0.55,  "login_freq_drop": 0.75,
    },
    {
        "client_id": 9005, "description": "Conservative Senior — Sudden Outflow",
        "avg_holding_days": 145, "crypto_ratio": 0.05, "tech_stocks_ratio": 0.10,
        "account_balance": 110_000, "balance_velocity": 0.22,
        "market_pain_index": 0.40,  "login_freq_drop": 0.80,
    },
    {
        "client_id": 9006, "description": "Standard Mid-Tier — Status Quo",
        "avg_holding_days": 45,  "crypto_ratio": 0.15, "tech_stocks_ratio": 0.45,
        "account_balance": 35_000,  "balance_velocity": 0.98,
        "market_pain_index": 0.35,  "login_freq_drop": 1.05,
    },
    {
        "client_id": 9007, "description": "Dormant Account — Extreme Aggressive",
        "avg_holding_days": 4,   "crypto_ratio": 0.80, "tech_stocks_ratio": 0.10,
        "account_balance": 50_000,  "balance_velocity": 0.05,
        "market_pain_index": 0.98,  "login_freq_drop": 0.02,
    },
]
