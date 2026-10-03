"""
CoinStats Portfolio Intelligence — portfolio feature engineering.

Single source of truth for turning holding rows into portfolio-level features.
The batch pipeline (500 portfolios) and the REST API (one portfolio per
request) call exactly the same functions, so a portfolio analysed live gets
the same numbers it would get in the nightly batch.

Pipeline
────────
    raw holdings ──prepare_holdings()──► clean holdings (one row per coin)
                 ──engineer_portfolio_features()──► one row per portfolio
                 ──score_vulnerability()──► + capitulation score, tier, drivers
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import settings as cfg

REQUIRED_COLUMNS = ["portfolio_id", "coin_id", "rank", "amount", "price_usd"]
OPTIONAL_COLUMNS = ["symbol", "avg_buy_price_usd"]


# ═════════════════════════════════════════════════════════════════════════════
# Loading & holding-level cleaning
# ═════════════════════════════════════════════════════════════════════════════
def load_holdings(path: Path | str = cfg.DATA_PATH) -> pd.DataFrame:
    """Read the CoinStats holdings CSV with explicit dtypes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Holdings file not found: {path}. "
            "Set COINSTATS_DATA_PATH or place the CSV in ./data/."
        )
    return pd.read_csv(
        path,
        dtype={"portfolio_id": "string", "coin_id": "string", "symbol": "string"},
    )


def prepare_holdings(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalise holding rows. Returns one row per (portfolio, coin)
    with value, weight, and sanitised cost-basis columns.

    • value_usd is recomputed as amount × price (the dataset's own definition),
      so API callers never have to send it and can't send an inconsistent one.
    • Duplicate coins inside a portfolio are merged (amount-weighted buy price).
    • Cost basis is trusted only inside the plausibility band from settings;
      untrusted rows still count toward value and concentration, but not PnL.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Holdings are missing required columns: {missing}")

    h = raw.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in h.columns:
            h[col] = np.nan if col == "avg_buy_price_usd" else ""
    h["symbol"] = h["symbol"].fillna("").astype("string").str.upper()
    h["avg_buy_price_usd"] = pd.to_numeric(h["avg_buy_price_usd"], errors="coerce")

    # Merge duplicate coins: total amount, amount-weighted average buy price.
    has_basis = h["avg_buy_price_usd"].notna()
    h["_basis_amount"] = h["amount"].where(has_basis, 0.0)
    h["_basis_cost"] = (h["amount"] * h["avg_buy_price_usd"]).where(has_basis, 0.0)
    h = (
        h.groupby(["portfolio_id", "coin_id"], as_index=False, sort=False)
        .agg(
            symbol=("symbol", "first"),
            rank=("rank", "min"),
            amount=("amount", "sum"),
            price_usd=("price_usd", "first"),
            _basis_amount=("_basis_amount", "sum"),
            _basis_cost=("_basis_cost", "sum"),
        )
    )
    h["avg_buy_price_usd"] = np.where(
        h["_basis_amount"] > 0, h["_basis_cost"] / h["_basis_amount"].where(h["_basis_amount"] > 0), np.nan
    )
    h = h.drop(columns=["_basis_amount", "_basis_cost"])

    h["value_usd"] = h["amount"] * h["price_usd"]
    totals = h.groupby("portfolio_id")["value_usd"].transform("sum")
    h["weight"] = np.where(totals > 0, h["value_usd"] / totals.where(totals > 0), 0.0)

    # Cost-basis plausibility band (rank-aware).
    ratio = h["avg_buy_price_usd"] / h["price_usd"].where(h["price_usd"] > 0)
    max_loss_ratio = np.where(
        h["rank"] <= cfg.TOP_COIN_RANK, cfg.MAX_LOSS_RATIO_TOP_COINS, cfg.MAX_LOSS_RATIO_TAIL_COINS
    )
    h["cost_basis_valid"] = (
        (h["avg_buy_price_usd"] > 0)
        & (ratio <= max_loss_ratio)
        & (ratio >= 1.0 / cfg.MAX_GAIN_RATIO)
    ).fillna(False).astype(bool)
    h["cost_basis_flagged"] = h["avg_buy_price_usd"].notna() & ~h["cost_basis_valid"]

    valid = h["cost_basis_valid"]
    h["cost_basis_usd"] = (h["amount"] * h["avg_buy_price_usd"]).where(valid, 0.0)
    h["unrealized_pnl_usd"] = ((h["price_usd"] - h["avg_buy_price_usd"]) * h["amount"]).where(valid, 0.0)
    h["covered_value_usd"] = h["value_usd"].where(valid, 0.0)

    h["is_stablecoin"] = h["symbol"].isin(list(cfg.STABLECOIN_SYMBOLS))
    h["is_bluechip"] = h["rank"] <= cfg.BLUECHIP_MAX_RANK
    h["is_speculative"] = h["rank"] > cfg.SPECULATIVE_MIN_RANK
    return h.sort_values(["portfolio_id", "weight"], ascending=[True, False], ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# Portfolio-level aggregation
# ═════════════════════════════════════════════════════════════════════════════
def engineer_portfolio_features(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate prepared holdings into one row per portfolio_id.

    Core metrics (from the brief)
    ─────────────────────────────
    total_portfolio_value    Σ value_usd
    num_assets               distinct coin_id
    unrealized_pnl_usd       Σ (price − avg_buy) × amount   [trusted basis only]
    unrealized_pnl_pct       unrealized_pnl_usd / total_cost_basis × 100
    hhi_index                Σ weight²  (1/n = perfectly spread, 1 = single coin)
    top1_weight, top3_weight share of the largest 1 / 3 holdings
    top10_bluechip_ratio     share in coins with rank ≤ 10
    speculative_meme_ratio   share in coins with rank > 100
    weighted_market_rank     Σ weight × rank
    capitulation_vulnerability_score   see score_vulnerability()

    Supporting metrics
    ──────────────────
    cost_basis_coverage, pnl_reliable, stablecoin_ratio, effective_num_assets,
    dust_positions — used for data-quality flags and the swap playbook.
    """
    h = holdings if "cost_basis_valid" in holdings.columns else prepare_holdings(holdings)
    pid = h["portfolio_id"]
    w = h["weight"]

    def share(mask: pd.Series) -> pd.Series:
        return w.where(mask, 0.0).groupby(pid).sum()

    g = h.groupby("portfolio_id", sort=True)
    top3 = (
        h.sort_values(["portfolio_id", "weight"], ascending=[True, False])
        .groupby("portfolio_id", sort=True).head(3)
        .groupby("portfolio_id")["weight"].sum()
    )

    f = pd.DataFrame({
        "total_portfolio_value": g["value_usd"].sum(),
        "num_assets": g["coin_id"].nunique(),
        "total_cost_basis": g["cost_basis_usd"].sum(),
        "unrealized_pnl_usd": g["unrealized_pnl_usd"].sum(),
        "hhi_index": (w ** 2).groupby(pid).sum(),
        "top1_weight": g["weight"].max(),
        "top3_weight": top3,
        "top10_bluechip_ratio": share(h["is_bluechip"]),
        "speculative_meme_ratio": share(h["is_speculative"]),
        "weighted_market_rank": (w * h["rank"]).groupby(pid).sum(),
        "stablecoin_ratio": share(h["is_stablecoin"]),
        "dust_positions": (w < cfg.DUST_WEIGHT).groupby(pid).sum().astype(int),
        "cost_basis_coverage": g["covered_value_usd"].sum() / g["value_usd"].sum(),
        "flagged_cost_basis_rows": g["cost_basis_flagged"].sum().astype(int),
    })
    f.index.name = "portfolio_id"

    f["effective_num_assets"] = 1.0 / f["hhi_index"]
    f["pnl_reliable"] = (f["cost_basis_coverage"] >= cfg.MIN_COST_BASIS_COVERAGE) & (f["total_cost_basis"] > 0)
    f["unrealized_pnl_pct"] = np.where(
        f["pnl_reliable"],
        f["unrealized_pnl_usd"] / f["total_cost_basis"].where(f["total_cost_basis"] > 0) * 100.0,
        np.nan,
    )
    return score_vulnerability(f).reset_index()


# ═════════════════════════════════════════════════════════════════════════════
# Capitulation vulnerability score
# ═════════════════════════════════════════════════════════════════════════════
def score_vulnerability(f: pd.DataFrame) -> pd.DataFrame:
    """
    Add the capitulation (attrition) vulnerability score in [0, 100].

        score = 100 × (0.45·L + 0.25·C + 0.30·S)

    L  loss pressure          logistic in PnL %, 0.5 at −20 %   (0 if PnL unreliable)
    C  concentration pressure logistic in HHI,   0.5 at 0.50
    S  speculation pressure   logistic in share of rank>100 coins, 0.5 at 35 %

    Logistic curves (instead of hard if/else cut-offs) keep the score smooth:
    a portfolio at −19 % and one at −21 % get almost the same pressure, while
    −45 % is clearly worse than −25 %. The hard "triple threat" flag is kept
    alongside for the crisp rule from the brief.

    This is a transparent, rules-based proxy. With CoinStats retention labels
    (did the user go inactive within 30/60/90 days?) the same features can
    train a supervised model and these weights become learned coefficients.
    """
    f = f.copy()
    pnl = f["unrealized_pnl_pct"]
    f["loss_pressure"] = np.where(
        pnl.notna(), expit((cfg.LOSS_CENTER_PCT - pnl.fillna(0.0)) / cfg.LOSS_SOFTNESS), 0.0
    )
    f["concentration_pressure"] = expit((f["hhi_index"] - cfg.HHI_CENTER) / cfg.HHI_SOFTNESS)
    f["speculation_pressure"] = expit(
        (f["speculative_meme_ratio"] - cfg.SPECULATIVE_CENTER) / cfg.SPECULATIVE_SOFTNESS
    )
    wts = cfg.SCORE_WEIGHTS
    f["capitulation_vulnerability_score"] = 100.0 * (
        wts["loss"] * f["loss_pressure"]
        + wts["concentration"] * f["concentration_pressure"]
        + wts["speculation"] * f["speculation_pressure"]
    )
    f["risk_tier"] = risk_tier(f["capitulation_vulnerability_score"])
    f["triple_threat"] = (
        (pnl < cfg.TRIPLE_THREAT_PNL_PCT)
        & (f["hhi_index"] > cfg.TRIPLE_THREAT_HHI)
        & (f["speculative_meme_ratio"] > cfg.TRIPLE_THREAT_SPECULATIVE)
    ).fillna(False).astype(bool)
    return f


def risk_tier(score: pd.Series | float) -> pd.Series | str:
    """Map score → 'High' / 'Elevated' / 'Low'. Works on scalars and Series."""
    def tier(s: float) -> str:
        if s >= cfg.HIGH_RISK_THRESHOLD:
            return "High"
        if s >= cfg.ELEVATED_RISK_THRESHOLD:
            return "Elevated"
        return "Low"

    if isinstance(score, pd.Series):
        return score.map(tier)
    return tier(float(score))


def risk_drivers(row: pd.Series | dict) -> list[str]:
    """
    Human-readable explanation of a single portfolio's score: every pressure
    component that is at least half-engaged, strongest contribution first.
    """
    row = dict(row)
    candidates = []
    pnl = row.get("unrealized_pnl_pct")
    if row.get("loss_pressure", 0) >= 0.5 and pnl is not None and not pd.isna(pnl):
        candidates.append((
            cfg.SCORE_WEIGHTS["loss"] * row["loss_pressure"],
            f"Deep unrealised loss ({pnl:+.1f}%) — classic capitulation trigger",
        ))
    if row.get("concentration_pressure", 0) >= 0.5:
        candidates.append((
            cfg.SCORE_WEIGHTS["concentration"] * row["concentration_pressure"],
            f"Concentrated book (HHI {row['hhi_index']:.2f}, top holding "
            f"{row['top1_weight']:.0%}) — one bad coin sinks the portfolio",
        ))
    if row.get("speculation_pressure", 0) >= 0.5:
        candidates.append((
            cfg.SCORE_WEIGHTS["speculation"] * row["speculation_pressure"],
            f"{row['speculative_meme_ratio']:.0%} of value in coins ranked >"
            f"{cfg.SPECULATIVE_MIN_RANK} — high wipe-out risk",
        ))
    candidates.sort(key=lambda x: x[0], reverse=True)
    drivers = [text for _, text in candidates]
    if not row.get("pnl_reliable", True):
        drivers.append("Cost basis missing or implausible for most holdings — PnL excluded from score")
    return drivers or ["No acute risk factors detected"]
