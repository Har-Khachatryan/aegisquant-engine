"""
CoinStats Portfolio Intelligence — central configuration.

Every threshold, weight, and label used by the pipeline, the API, and the
dashboard lives here, so a product/risk owner can tune the system in one place
and all three surfaces stay consistent.

Paths can be overridden with environment variables (used by the Docker image):
    COINSTATS_DATA_PATH     — raw holdings CSV
    COINSTATS_ARTIFACT_DIR  — where pipeline.py writes model.json + features CSV
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent

DATA_PATH = Path(
    os.getenv(
        "COINSTATS_DATA_PATH",
        PACKAGE_DIR.parent / "data" / "coinstats_manual_portfolios_holdings.csv",
    )
)
ARTIFACT_DIR = Path(os.getenv("COINSTATS_ARTIFACT_DIR", PACKAGE_DIR / "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "model.json"                    # scaler + centroids, no pickle
FEATURES_PATH = ARTIFACT_DIR / "portfolio_features.csv"      # one row per portfolio

MODEL_VERSION = "coinstats-intel-1.0.0"

# ── Market-tier definitions (by CoinStats market-cap rank) ────────────────────
BLUECHIP_MAX_RANK = 10        # rank <= 10  → blue-chip
SPECULATIVE_MIN_RANK = 100    # rank  > 100 → speculative / meme tail

# Stablecoins sit inside the top-10 by rank (USDT, USDC), so they inflate the
# blue-chip ratio. They are tracked separately as "dry powder" for swap campaigns.
STABLECOIN_SYMBOLS: frozenset[str] = frozenset({
    "USDT", "USDC", "DAI", "PYUSD", "RLUSD", "GHO", "USD1", "USDP", "USDD",
    "USDE", "FDUSD", "TUSD", "USDC.E", "USDT.C", "HYUSD", "UXD", "APXUSD", "SUSDAT",
})

# ── Cost-basis sanitisation ───────────────────────────────────────────────────
# Manual portfolios contain data-entry errors (e.g. total cost typed as unit
# price: SHIB "bought" at $2,017). Without a guard, ~130 such rows fabricate
# over $1 trillion of losses against an ~$89M book.
# A holding's cost basis is trusted only if avg_buy / price lies inside a
# plausibility band. Top-100 coins cannot realistically sit >100x below a real
# entry price; long-tail tokens can genuinely lose 99.9%, so they get 1000x.
MAX_LOSS_RATIO_TOP_COINS = 100.0     # rank <= TOP_COIN_RANK
MAX_LOSS_RATIO_TAIL_COINS = 1_000.0  # rank  > TOP_COIN_RANK
TOP_COIN_RANK = 100
MAX_GAIN_RATIO = 1_000.0             # avg_buy < price / 1000 → treated as unknown basis
# Portfolio PnL is reported as unreliable when trusted cost basis covers less
# than this share of portfolio value.
MIN_COST_BASIS_COVERAGE = 0.50

# ── Capitulation (attrition) vulnerability score ──────────────────────────────
# Rules-based composite in [0, 100]: three smooth "pressure" components, each a
# logistic curve centred on the business threshold from the brief.
LOSS_CENTER_PCT = -20.0       # PnL below −20 % = deep loss
LOSS_SOFTNESS = 8.0           # width of the logistic transition, in PnL points
HHI_CENTER = 0.50             # HHI above 0.5 = concentrated
HHI_SOFTNESS = 0.08
SPECULATIVE_CENTER = 0.35     # >35 % of value in rank >100 coins = speculative-heavy
SPECULATIVE_SOFTNESS = 0.10

SCORE_WEIGHTS: dict[str, float] = {
    "loss": 0.45,             # drawdown pain is the strongest capitulation trigger
    "concentration": 0.25,    # single-asset bets amplify that pain
    "speculation": 0.30,      # meme/long-tail exposure → highest wipe-out risk
}

HIGH_RISK_THRESHOLD = 60.0    # score >= 60 → High
ELEVATED_RISK_THRESHOLD = 40.0  # 40–60 → Elevated, below → Low

# "Triple threat" = all three hard conditions from the brief at once.
TRIPLE_THREAT_PNL_PCT = -20.0
TRIPLE_THREAT_HHI = 0.50
TRIPLE_THREAT_SPECULATIVE = 0.30

# ── Segmentation (KMeans) ─────────────────────────────────────────────────────
N_CLUSTERS = 4
KMEANS_SEED = 42
KMEANS_N_INIT = 50

# Clustering inputs and the transform applied before standardisation.
# num_assets and weighted rank are heavy-tailed (one portfolio holds 504 coins,
# ranks reach 25,000); without log compression a handful of outliers would
# claim an entire cluster.
CLUSTER_INPUTS: dict[str, str] = {
    "total_portfolio_value": "log10",
    "num_assets": "log1p",
    "hhi_index": "identity",
    "top10_bluechip_ratio": "identity",
    "speculative_meme_ratio": "identity",
    "weighted_market_rank": "log10",
}


@dataclass(frozen=True)
class Archetype:
    archetype_id: int
    name: str
    short: str
    description: str
    # How to recognise this persona from a cluster centroid (z-scores of the
    # CLUSTER_INPUTS). Clusters are matched to archetypes one-to-one by
    # maximising the total signature score (Hungarian assignment), so the
    # labels stay correct even when KMeans renumbers its clusters.
    signature: dict[str, float]


ARCHETYPES: dict[int, Archetype] = {
    0: Archetype(
        0, "Blue-Chip Whale / Bitcoin Hodler", "Whale",
        "Large balances parked in top-10 coins, often one dominant asset.",
        {"total_portfolio_value": 1.0, "top10_bluechip_ratio": 1.0, "speculative_meme_ratio": -1.0},
    ),
    1: Archetype(
        1, "Diversified Altcoin Explorer", "Explorer",
        "Many positions across the mid-cap market, low concentration.",
        {"num_assets": 1.0, "hhi_index": -1.0},
    ),
    2: Archetype(
        2, "High-Risk Speculative Degen", "Degen",
        "Majority of value in long-tail / meme coins ranked beyond 100.",
        {"speculative_meme_ratio": 1.0, "weighted_market_rank": 1.0},
    ),
    3: Archetype(
        3, "Micro Retail / Concentrated Novice", "Novice",
        "Small balances spread over few coins; early in the investing journey.",
        {"total_portfolio_value": -1.0, "hhi_index": 1.0, "num_assets": -1.0},
    ),
}

# ── Monetisation playbook defaults (dashboard sliders start here) ─────────────
DEFAULT_SWAP_FEE_RATE = 0.0075      # 0.75 % blended swap fee — placeholder assumption
DEFAULT_ADOPTION_RATE = 0.05        # share of targeted notional converted by a push
CONCENTRATION_CAP = 0.40            # target max weight of any single non-stable coin
SPECULATIVE_TARGET = 0.50           # Degen rotation: trim speculative share down to 50 %
DUST_WEIGHT = 0.01                  # positions under 1 % of a portfolio = "dust"
MIN_SWAP_USD = 25.0                 # ignore campaigns with less notional than this
