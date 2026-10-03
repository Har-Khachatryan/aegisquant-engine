"""End-to-end REST API tests (replaces the v3.x manual requests script)."""

import pytest
from fastapi.testclient import TestClient

from config import ClientFeatures

EXAMPLE = ClientFeatures.model_config["json_schema_extra"]["examples"][0]


@pytest.fixture(scope="module")
def client(trained):
    from api import app
    with TestClient(app) as c:      # runs the lifespan → loads the predictor
        yield c


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy" and body["test_auc"] > 0.8


def test_predict_returns_explained_decision(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["retention_action"] == (body["churn_probability"] >= body["decision_threshold"])
    assert body["risk_tier"] in {"Low", "Elevated", "High"}
    assert len(body["risk_drivers"]) == 2 and len(body["drivers"]) == 5


@pytest.mark.parametrize("patch", [
    {"credit_score": 100},
    {"geography": "Italy"},
    {"num_products": 5},
    {"estimated_salary": 0},
])
def test_invalid_payloads_are_rejected(client, patch):
    assert client.post("/predict", json={**EXAMPLE, **patch}).status_code == 422


def test_model_card(client):
    card = client.get("/model-card").json()
    assert card["test_metrics"]["roc_auc"] == card["meta"]["test_auc"]
    assert {s["profile"] for s in card["segments"]} == {"conservative", "balanced", "aggressive"}
