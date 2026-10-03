"""End-to-end tests for the FastAPI service (uses the real artifacts)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import settings as cfg
from api import PortfolioRequest, app

pytestmark = pytest.mark.skipif(not cfg.DATA_PATH.exists(), reason="CoinStats CSV not available")

DEGEN = PortfolioRequest.model_config["json_schema_extra"]["examples"][0]
WHALE = {
    "portfolio_id": "demo_whale",
    "holdings": [
        {"coin_id": "bitcoin", "symbol": "BTC", "rank": 1, "amount": 2.0, "price_usd": 82898.66, "avg_buy_price_usd": 60000},
        {"coin_id": "ethereum", "symbol": "ETH", "rank": 2, "amount": 5.0, "price_usd": 2659.72, "avg_buy_price_usd": 2000},
        {"coin_id": "tether", "symbol": "USDT", "rank": 3, "amount": 20000, "price_usd": 0.9998, "avg_buy_price_usd": 1.0},
    ],
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs the lifespan (loads artifacts)
        yield c


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy" and body["reference_portfolios"] == 500


def test_degen_portfolio_is_flagged(client):
    r = client.post("/analyze-portfolio", json=DEGEN)
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["archetype_id"] == 2
    assert a["risk_tier"] == "High"
    assert a["unrealized_pnl_pct"] < -20
    assert a["diagnosis"]["headline"].startswith("High Risk of Attrition")
    assert a["diagnosis"]["campaign"] == "Risk-Off Rotation"


def test_whale_portfolio_is_healthy(client):
    a = client.post("/analyze-portfolio", json=WHALE).json()
    assert a["archetype_id"] == 0
    assert a["risk_tier"] == "Low"
    assert a["stablecoin_ratio"] > 0.05


@pytest.mark.parametrize("payload", [
    {"holdings": []},
    {"holdings": [{"coin_id": "x", "rank": 5, "amount": -1, "price_usd": 1}]},
    {"holdings": [{"coin_id": "x", "rank": 5, "amount": 1, "price_usd": 0}]},
])
def test_invalid_payloads_are_rejected(client, payload):
    assert client.post("/analyze-portfolio", json=payload).status_code == 422


def test_cluster_summary(client):
    s = client.get("/cluster-summary").json()
    assert s["book"]["portfolios"] == 500
    assert len(s["segments"]) == 4
    assert sum(seg["portfolios"] for seg in s["segments"]) == 500


def test_reference_lookup_matches_batch_pipeline(client):
    batch = pd.read_csv(cfg.FEATURES_PATH, dtype={"portfolio_id": "string"}).set_index("portfolio_id")
    for pid in ["p_001", "p_193", "p_488"]:
        a = client.get(f"/portfolios/{pid}").json()
        assert a["attrition_vulnerability_score"] == pytest.approx(
            batch.loc[pid, "capitulation_vulnerability_score"], abs=0.01
        )
        assert a["archetype_id"] == batch.loc[pid, "archetype_id"]
    assert client.get("/portfolios/p_999").status_code == 404
