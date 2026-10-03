"""Markowitz retention portfolio: feasibility, profile caps and churn-scaled risk aversion."""

import numpy as np
import pytest

from config import ASSETS, CRYPTO_ASSETS, MIN_WEIGHT_BY_PROFILE, PROFILE_EXPOSURE, RISK_AVERSION, TECH_ASSETS


@pytest.fixture(scope="module")
def engine(trained):
    from optimizer import AegisQuantEngine
    return AegisQuantEngine(start_market_worker=False)


@pytest.mark.parametrize("profile", ["conservative", "balanced", "aggressive"])
def test_weights_respect_budget_and_profile_caps(engine, profile):
    mean_ret, cov = engine.get_market_context()
    for churn_prob in (0.05, 0.6, 0.99):
        w = dict(zip(ASSETS, engine.optimize_portfolio(profile, mean_ret, cov, churn_prob)))
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
        assert sum(w[a] for a in CRYPTO_ASSETS) <= PROFILE_EXPOSURE[profile]["crypto"] + 1e-9
        assert sum(w[a] for a in TECH_ASSETS) <= PROFILE_EXPOSURE[profile]["tech"] + 1e-9
        core = [a for a in ASSETS if a not in CRYPTO_ASSETS + TECH_ASSETS]
        assert min(w[a] for a in core) >= MIN_WEIGHT_BY_PROFILE[profile] - 1e-9


@pytest.mark.parametrize("profile", ["conservative", "balanced", "aggressive"])
def test_bounds_are_feasible(engine, profile):
    bounds = engine._build_asset_bounds(profile)
    assert sum(lb for lb, _ in bounds) <= 1.0 <= sum(ub for _, ub in bounds)
    assert all(lb <= ub for lb, ub in bounds)


def test_gamma_rises_only_above_the_threshold(engine):
    base = RISK_AVERSION["balanced"]
    t = engine.threshold
    assert engine._compute_gamma("balanced", t - 0.01) == base
    gammas = [engine._compute_gamma("balanced", p) for p in np.linspace(t + 0.01, 1.0, 5)]
    assert all(a < b for a, b in zip(gammas, gammas[1:]))
    assert gammas[-1] == pytest.approx(base * np.e)


def test_higher_churn_risk_lowers_portfolio_volatility(engine):
    mean_ret, cov = engine.get_market_context()
    calm = engine.optimize_portfolio("aggressive", mean_ret, cov, 0.99)
    normal = engine.optimize_portfolio("aggressive", mean_ret, cov, 0.05)
    vol = lambda w: engine.portfolio_stats(w, mean_ret, cov)["volatility"]
    assert vol(calm) <= vol(normal) + 1e-9
