"""
AegisQuant — AI Portfolio Shield & Risk Engine
Global configuration, hyperparameters, and data contracts.
"""

from __future__ import annotations

from datetime import timedelta
from pydantic import BaseModel, Field, model_validator

# ── Artifact storage ──────────────────────────────────────────────────────────
ARTIFACT_PATH: str = "aegis_quant_artifacts.pkl"

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

# ── Risk aversion base values (γ) per investor archetype ────────────────────
RISK_AVERSION: dict[str, float] = {
    "conservative": 8.0,
    "balanced":     4.0,
    "aggressive":   1.5,
}

# ── Optimisation constraints ──────────────────────────────────────────────────
WEIGHT_MAX: float = 0.40                     # Hard ceiling per asset
MARKET_CACHE_TTL: timedelta = timedelta(hours=1)
CHURN_THRESHOLD: float = 0.50                # Retention trigger threshold
DIVERSIFICATION_LAMBDA: float = 0.20         # L2 penalty weight (distance from EW)
MIN_WEIGHT_BY_PROFILE: dict[str, float] = {
    "aggressive":   0.02,
    "balanced":     0.05,
    "conservative": 0.08,
}
GAMMA_CHURN_SCALE: float = 1.0               # Exponent normaliser for churn‑scaled γ
SIGMA_JITTER: float = 1e-8                   # PD regularisation jitter
SLSQP_FTOL: float = 1e-10                    # SLSQP convergence tolerance
SLSQP_MAXITER: int = 2_000                   # Max SLSQP iterations

# ── Client payload contract (Pydantic v2) ────────────────────────────────────
class ClientPayload(BaseModel):
    """
    Strict input contract.  All fields are validated before inference.

    The cross‑field validator ensures crypto_ratio + tech_stocks_ratio ≤ 1.0.
    """
    client_id:         int
    description:       str   = ""
    avg_holding_days:  int   = Field(ge=1,   le=3_650)
    crypto_ratio:      float = Field(ge=0.0, le=1.0)
    tech_stocks_ratio: float = Field(ge=0.0, le=1.0)
    account_balance:   float = Field(gt=0.0)
    balance_velocity:  float = Field(ge=0.0, le=5.0)
    market_pain_index: float = Field(ge=0.0, le=1.0)
    login_freq_drop:   float = Field(ge=0.0, le=5.0)

    @model_validator(mode="after")
    def ratios_must_not_exceed_one(self) -> "ClientPayload":
        total = self.crypto_ratio + self.tech_stocks_ratio
        if total > 1.0:
            raise ValueError(
                f"crypto_ratio ({self.crypto_ratio:.2f}) + "
                f"tech_stocks_ratio ({self.tech_stocks_ratio:.2f}) "
                f"= {total:.2f} > 1.0"
            )
        return self

# ── Demo batch for dashboard ─────────────────────────────────────────────────
DEMO_CLIENTS: list[dict] = [
    {
        "client_id": 9001, "description": "Crypto Enthusiast — Aggressive",
        "avg_holding_days": 6,   "crypto_ratio": 0.75, "tech_stocks_ratio": 0.20,
        "account_balance": 42_000, "balance_velocity": 0.62,
        "market_pain_index": 0.65, "login_freq_drop": 0.45,
    },
    {
        "client_id": 9002, "description": "Institutional HNW — Solid Inflows",
        "avg_holding_days": 185, "crypto_ratio": 0.00, "tech_stocks_ratio": 0.15,
        "account_balance": 450_000, "balance_velocity": 1.20,
        "market_pain_index": 0.05, "login_freq_drop": 1.40,
    },
    {
        "client_id": 9003, "description": "Retail Balanced — Drawdown Frustration",
        "avg_holding_days": 55,  "crypto_ratio": 0.25, "tech_stocks_ratio": 0.35,
        "account_balance": 18_000, "balance_velocity": 0.42,
        "market_pain_index": 0.88, "login_freq_drop": 0.28,
    },
    {
        "client_id": 9004, "description": "Active Growth Trader — Marginal Decay",
        "avg_holding_days": 15,  "crypto_ratio": 0.50, "tech_stocks_ratio": 0.40,
        "account_balance": 85_000, "balance_velocity": 0.58,
        "market_pain_index": 0.55, "login_freq_drop": 0.75,
    },
    {
        "client_id": 9005, "description": "Conservative Senior — Sudden Outflow",
        "avg_holding_days": 145, "crypto_ratio": 0.05, "tech_stocks_ratio": 0.10,
        "account_balance": 110_000, "balance_velocity": 0.22,
        "market_pain_index": 0.40, "login_freq_drop": 0.80,
    },
    {
        "client_id": 9006, "description": "Standard Mid-Tier — Status Quo",
        "avg_holding_days": 45,  "crypto_ratio": 0.15, "tech_stocks_ratio": 0.45,
        "account_balance": 35_000, "balance_velocity": 0.98,
        "market_pain_index": 0.35, "login_freq_drop": 1.05,
    },
    {
        "client_id": 9007, "description": "Dormant Account — Extreme Aggressive",
        "avg_holding_days": 4,   "crypto_ratio": 0.80, "tech_stocks_ratio": 0.10,
        "account_balance": 50_000, "balance_velocity": 0.05,
        "market_pain_index": 0.98, "login_freq_drop": 0.02,
    },
]