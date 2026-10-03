"""Unit tests for portfolio feature engineering and the vulnerability score."""

import math

import numpy as np
import pandas as pd
import pytest

import settings as cfg
from features import engineer_portfolio_features, load_holdings, prepare_holdings, risk_tier


def make(rows):
    cols = ["portfolio_id", "coin_id", "symbol", "rank", "amount", "price_usd", "avg_buy_price_usd"]
    return pd.DataFrame(rows, columns=cols)


def one(rows) -> pd.Series:
    return engineer_portfolio_features(make(rows)).iloc[0]


def test_concentration_metrics_on_toy_portfolio():
    f = one([
        ("p", "bitcoin", "BTC", 1, 3.0, 100.0, 100.0),   # $300 → 75 %
        ("p", "pepe", "PEPE", 150, 100.0, 1.0, 1.0),     # $100 → 25 %
    ])
    assert f["total_portfolio_value"] == pytest.approx(400.0)
    assert f["num_assets"] == 2
    assert f["hhi_index"] == pytest.approx(0.75**2 + 0.25**2)
    assert f["top1_weight"] == pytest.approx(0.75)
    assert f["top3_weight"] == pytest.approx(1.0)
    assert f["top10_bluechip_ratio"] == pytest.approx(0.75)
    assert f["speculative_meme_ratio"] == pytest.approx(0.25)
    assert f["weighted_market_rank"] == pytest.approx(0.75 * 1 + 0.25 * 150)
    assert f["effective_num_assets"] == pytest.approx(1 / 0.625)


def test_pnl_uses_only_trusted_cost_basis():
    f = one([
        ("p", "eth", "ETH", 2, 1.0, 150.0, 100.0),          # +$50, trusted
        ("p", "xrp", "XRP", 5, 10.0, 1.0, 75_000.0),        # total cost typed as unit price → flagged
        ("p", "bonk", "BONK", 212, 10.0, 1.0, None),        # unknown basis → excluded
    ])
    assert f["unrealized_pnl_usd"] == pytest.approx(50.0)
    assert f["total_cost_basis"] == pytest.approx(100.0)
    assert f["unrealized_pnl_pct"] == pytest.approx(50.0)
    assert f["flagged_cost_basis_rows"] == 1
    assert f["cost_basis_coverage"] == pytest.approx(150 / 170)
    assert bool(f["pnl_reliable"])


def test_long_tail_coins_may_genuinely_lose_99_percent():
    # 500x below entry on a rank-2000 token is a real wipe-out, not a typo.
    f = one([("p", "dead", "DEAD", 2000, 1_000.0, 0.001, 0.5), ("p", "x", "X", 50, 1.0, 1.0, 1.0)])
    assert f["flagged_cost_basis_rows"] == 0
    assert f["unrealized_pnl_pct"] < -99


def test_pnl_marked_unreliable_when_coverage_is_low():
    f = one([
        ("p", "eth", "ETH", 2, 100.0, 2659.0, 75_000_000.0),  # absurd → flagged, ~100 % of value
        ("p", "op", "OP", 210, 10.0, 0.13, 0.20),
    ])
    assert not bool(f["pnl_reliable"])
    assert math.isnan(f["unrealized_pnl_pct"])
    assert f["loss_pressure"] == 0.0


def test_duplicate_coins_are_merged_with_amount_weighted_basis():
    f = one([
        ("p", "sol", "SOL", 7, 1.0, 200.0, 100.0),
        ("p", "sol", "SOL", 7, 3.0, 200.0, 200.0),
        ("p", "btc", "BTC", 1, 1.0, 200.0, 200.0),
    ])
    assert f["num_assets"] == 2
    # SOL basis = (1·100 + 3·200) / 4 = 175 → pnl = (200−175)·4 = 100
    assert f["unrealized_pnl_usd"] == pytest.approx(100.0)


def test_score_rises_with_loss_concentration_and_speculation():
    base = [("p", "btc", "BTC", 1, 1.0, 100.0, 100.0), ("p", "eth", "ETH", 2, 1.0, 100.0, 100.0),
            ("p", "sol", "SOL", 7, 1.0, 100.0, 100.0)]
    calm = one(base)["capitulation_vulnerability_score"]
    lossy = one([(p, c, s, r, a, pr, 200.0) for p, c, s, r, a, pr, _ in base])["capitulation_vulnerability_score"]
    degen = one([("p", "meme", "MEME", 900, 1.0, 100.0, 300.0), ("p", "btc", "BTC", 1, 0.05, 100.0, 100.0)])
    assert calm < lossy < degen["capitulation_vulnerability_score"]
    assert degen["risk_tier"] == "High"
    assert bool(degen["triple_threat"])


def test_risk_tier_thresholds():
    assert risk_tier(cfg.HIGH_RISK_THRESHOLD) == "High"
    assert risk_tier(cfg.HIGH_RISK_THRESHOLD - 0.01) == "Elevated"
    assert risk_tier(cfg.ELEVATED_RISK_THRESHOLD - 0.01) == "Low"


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_holdings(pd.DataFrame({"portfolio_id": ["p"], "coin_id": ["btc"]}))


@pytest.mark.skipif(not cfg.DATA_PATH.exists(), reason="CoinStats CSV not available")
def test_real_dataset_invariants():
    h = prepare_holdings(load_holdings())
    f = engineer_portfolio_features(h)
    assert len(f) == 500
    assert np.allclose(h.groupby("portfolio_id")["weight"].sum(), 1.0)
    assert (f["hhi_index"] >= 1 / f["num_assets"] - 1e-9).all() and (f["hhi_index"] <= 1 + 1e-9).all()
    assert f["capitulation_vulnerability_score"].between(0, 100).all()
    assert (f.loc[f["pnl_reliable"], "unrealized_pnl_pct"] >= -100).all()
    # The sanitiser must neutralise the known data-entry errors (e.g. p_193, p_454).
    assert f.set_index("portfolio_id").loc["p_193", "flagged_cost_basis_rows"] >= 10
    rel = f[f["pnl_reliable"]]
    book_pct = rel["unrealized_pnl_usd"].sum() / rel["total_cost_basis"].sum() * 100
    assert -100 < book_pct < 100
