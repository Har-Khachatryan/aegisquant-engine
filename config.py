"""
AegisQuant — AI Portfolio Shield & Risk Engine
Global configuration, hyperparameters, and data contracts.

Upgrade v3.2 → v4.0
────────────────────
[REAL DATA]
  - v3.x trained on a synthetic generator whose churn label was a known
    logistic formula of its own features. v4.0 trains on the public Kaggle
    "Churn Modelling" bank dataset (10,000 customers, real `Exited` label).
  - The feature contract (ClientFeatures) now describes a bank customer instead
    of synthetic behavioural ratios. Column names are snake_case internally;
    data_pipeline.COLUMN_MAP translates the Kaggle headers.

[FAIRNESS]
  - `Gender` is deliberately NOT a model input. It is kept in the data only to
    audit predictions per group (churn_model writes a fairness report).

[PICKLE-FREE ARTIFACTS]
  - Every artifact is JSON or CSV in ARTIFACT_DIR (the v3.2 legacy pickle and
    joblib files are gone): XGBoost native JSON, scaler + KMeans centroids as
    JSON, profile mapping as JSON, reference data as CSV.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_VERSION: str = "aegis_quant_v4.0"

# ── Data & artifact storage ───────────────────────────────────────────────────
DATA_PATH: Path = Path(os.getenv("AEGIS_DATA_PATH", ROOT / "data" / "Churn_Modelling.csv"))
ARTIFACT_DIR: Path = Path(os.getenv("AEGIS_ARTIFACT_DIR", ROOT / "artifacts"))

XGB_MODEL_PATH:        Path = ARTIFACT_DIR / "aegis_xgb.json"
PROCESSOR_PATH:        Path = ARTIFACT_DIR / "aegis_processor.json"
PROFILE_RESOLVER_PATH: Path = ARTIFACT_DIR / "aegis_profile_resolver.json"
ARTIFACT_META_PATH:    Path = ARTIFACT_DIR / "aegis_artifact_meta.json"
EVALUATION_PATH:       Path = ARTIFACT_DIR / "evaluation.json"
HOLDOUT_PATH:          Path = ARTIFACT_DIR / "holdout_predictions.csv"
REFERENCE_DATA_PATH:   Path = ARTIFACT_DIR / "reference_data.csv"   # DriftMonitor baseline
MODEL_LOG_PATH:        Path = ROOT / "models_log.json"

# ── Training protocol ─────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE:    float = 0.20     # untouched hold-out, used once for final metrics
CV_FOLDS:     int = 5          # out-of-fold predictions on train → threshold choice

# ── Feature column sets ───────────────────────────────────────────────────────
GEOGRAPHIES: tuple[str, ...] = ("France", "Germany", "Spain")   # France = baseline

# Life-stage features used by KMeans to segment customers into investor profiles
CLUSTER_FEATURES: list[str] = ["age", "balance", "estimated_salary"]

# Features fed to XGBoost (raw + engineered), followed by the one-hot cluster columns
CHURN_FEATURES: list[str] = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "num_products",
    "has_cr_card",
    "is_active_member",
    "estimated_salary",
    "geo_germany",
    "geo_spain",
    "balance_to_salary",
    "zero_balance",
    "tenure_to_age",
    "products_one",
    "products_many",
    "inactive_senior",
    "age_x_active",
    "germany_balance",
    "credit_per_age",
]

# Explanations are reported per business concept: the SHAP values of a raw input
# and the features engineered from it are summed (SHAP values are additive).
FEATURE_GROUPS: dict[str, list[str]] = {
    "num_products":     ["num_products", "products_one", "products_many"],
    "age":              ["age"],
    "is_active_member": ["is_active_member", "inactive_senior", "age_x_active"],
    "geography":        ["geo_germany", "geo_spain", "germany_balance"],
    "balance":          ["balance", "zero_balance", "balance_to_salary"],
    "credit_score":     ["credit_score", "credit_per_age"],
    "tenure":           ["tenure", "tenure_to_age"],
    "estimated_salary": ["estimated_salary"],
    "has_cr_card":      ["has_cr_card"],
}

# Columns a caller must supply (everything else is derived in data_pipeline.engineer_features)
INPUT_FEATURES: list[str] = [
    "credit_score", "geography", "age", "tenure", "balance",
    "num_products", "has_cr_card", "is_active_member", "estimated_salary",
]

# Number of KMeans clusters — must match the three investor profiles
N_CLUSTERS: int = 3

# Monotonic constraints (+1 raises churn, −1 lowers it, 0 free). Only relationships
# that are both intuitive and visible in the data are constrained: active members
# churn less (14 % vs 27 %). Age and product count are clearly non-monotonic.
MONOTONE_CONSTRAINTS: dict[str, int] = {"is_active_member": -1}

# XGBoost hyperparameters: best of a 30-trial random search scored by 5-fold CV
# AUC on the training split only (reproduce with `python benchmark.py --search`).
XGB_PARAMS: dict = {
    "max_depth": 3,
    "learning_rate": 0.0184,
    "min_child_weight": 1.24,
    "subsample": 0.89,
    "colsample_bytree": 0.54,
    "reg_lambda": 0.81,
    "n_estimators": 2_000,          # upper bound — early stopping picks the actual count
    "early_stopping_rounds": 60,
}

# ── Decision policy ───────────────────────────────────────────────────────────
# Fallback only: training picks the F1-optimal threshold on out-of-fold
# predictions and stores it in aegis_artifact_meta.json.
CHURN_THRESHOLD: float = 0.50
RISK_TIER_BOUNDS: tuple[float, float] = (0.30, 0.60)   # Low < 0.30 ≤ Elevated < 0.60 ≤ High

# ── Asset universe for the retention portfolio ────────────────────────────────
ASSETS: list[str] = ["SPY", "BND", "GLD", "KO", "AAPL", "MSFT", "NVDA", "TSLA", "BTC", "ETH"]

TICKER_MAP: dict[str, str] = {
    "SPY": "SPY", "BND": "BND", "GLD": "GLD", "KO": "KO",
    "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "TSLA": "TSLA",
    "BTC": "BTC-USD", "ETH": "ETH-USD",
}

CRYPTO_ASSETS: list[str] = ["BTC", "ETH"]
TECH_ASSETS:   list[str] = ["AAPL", "MSFT", "NVDA", "TSLA"]
# Everything else (broad equity, bonds, gold, defensive consumer) is the "core".

ASSET_CLASS: dict[str, str] = {
    a: ("Crypto" if a in CRYPTO_ASSETS else "Tech equity" if a in TECH_ASSETS else "Core")
    for a in ASSETS
}

# Maximum thematic exposure per investor profile (shares of the whole portfolio)
PROFILE_EXPOSURE: dict[str, dict[str, float]] = {
    "conservative": {"crypto": 0.04, "tech": 0.30},
    "balanced":     {"crypto": 0.10, "tech": 0.50},
    "aggressive":   {"crypto": 0.25, "tech": 0.70},
}

# ── Risk aversion base values (γ) per investor archetype ─────────────────────
RISK_AVERSION: dict[str, float] = {
    "conservative": 8.0,
    "balanced":     4.0,
    "aggressive":   1.5,
}

# ── Optimisation constraints ──────────────────────────────────────────────────
WEIGHT_MAX: float = 0.40
MARKET_CACHE_TTL: timedelta = timedelta(hours=1)
DIVERSIFICATION_LAMBDA: float = 0.20
# Minimum weight per CORE asset (thematic assets may go to zero)
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
    Single canonical Pydantic model for a bank customer flowing through
    AegisQuant (REST API body, dashboard input, engine inference).

    Field semantics follow the Kaggle "Churn Modelling" dataset. Bounds cover
    the training data with headroom; values outside them are rejected (422)
    rather than silently extrapolated.
    """
    # UI / tracking fields — optional so the REST API never has to supply them
    customer_id: Optional[int] = None
    description: str = ""

    credit_score:     int   = Field(ge=300, le=900, description="Credit score")
    geography:        Literal["France", "Germany", "Spain"]
    age:              int   = Field(ge=18, le=100)
    tenure:           int   = Field(ge=0, le=60, description="Years as a customer")
    balance:          float = Field(ge=0.0, description="Account balance (EUR)")
    num_products:     int   = Field(ge=1, le=4, description="Number of bank products held")
    has_cr_card:      bool
    is_active_member: bool
    estimated_salary: float = Field(gt=0.0, description="Estimated yearly salary (EUR)")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "credit_score": 619, "geography": "Germany", "age": 52, "tenure": 2,
                "balance": 118_000.0, "num_products": 3, "has_cr_card": True,
                "is_active_member": False, "estimated_salary": 101_348.88,
            }]
        }
    }


# ── Human-readable labels for the explanation groups ──────────────────────────
GROUP_LABELS: dict[str, str] = {
    "num_products":     "Products held",
    "age":              "Age",
    "is_active_member": "Active member",
    "geography":        "Market",
    "balance":          "Balance",
    "credit_score":     "Credit score",
    "tenure":           "Tenure",
    "estimated_salary": "Salary",
    "has_cr_card":      "Credit card",
    "segment":          "Life-stage segment",
}

# ── Demo customers for the dashboard (hand-written, not rows of the dataset) ──
DEMO_CLIENTS: list[dict] = [
    {
        "customer_id": 9001, "description": "Mid-career, 3 products, inactive — Germany",
        "credit_score": 610, "geography": "Germany", "age": 52, "tenure": 2,
        "balance": 118_000.0, "num_products": 3, "has_cr_card": True,
        "is_active_member": False, "estimated_salary": 95_000.0,
    },
    {
        "customer_id": 9002, "description": "Young professional, 2 products, active — France",
        "credit_score": 720, "geography": "France", "age": 29, "tenure": 6,
        "balance": 0.0, "num_products": 2, "has_cr_card": True,
        "is_active_member": True, "estimated_salary": 61_000.0,
    },
    {
        "customer_id": 9003, "description": "Affluent saver, 1 product, inactive — Spain",
        "credit_score": 680, "geography": "Spain", "age": 45, "tenure": 8,
        "balance": 152_000.0, "num_products": 1, "has_cr_card": False,
        "is_active_member": False, "estimated_salary": 140_000.0,
    },
    {
        "customer_id": 9004, "description": "Retiree, 2 products, active — France",
        "credit_score": 790, "geography": "France", "age": 67, "tenure": 9,
        "balance": 96_000.0, "num_products": 2, "has_cr_card": True,
        "is_active_member": True, "estimated_salary": 38_000.0,
    },
]
