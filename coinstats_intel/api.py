"""
CoinStats Portfolio Intelligence — REST API.

    uvicorn api:app --reload --port 8000        # then open http://localhost:8000/docs

Endpoints
─────────
GET  /health                      service + model status
POST /analyze-portfolio           holdings in → HHI, PnL %, archetype, attrition score, playbook
GET  /cluster-summary             book KPIs + per-archetype stats over the 500 reference portfolios
GET  /portfolios/{portfolio_id}   same analysis for a portfolio from the reference sample (demo helper)

Every endpoint uses the same feature code as the offline pipeline
(features.py → segmentation.py → insights.py), so live and batch numbers match.
Handlers are plain `def` so FastAPI runs the pandas work in its threadpool
instead of blocking the event loop.
"""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, model_validator

import settings as cfg
from features import engineer_portfolio_features, load_holdings, prepare_holdings
from insights import diagnose, swap_levers
from pipeline import cluster_summary, load_artifacts
from segmentation import SegmentationModel, attach_segments

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("coinstats_intel.api")


# ═════════════════════════════════════════════════════════════════════════════
# Schemas
# ═════════════════════════════════════════════════════════════════════════════
class Holding(BaseModel):
    coin_id: str = Field(min_length=1, max_length=200, examples=["bitcoin"])
    symbol: str = Field(default="", max_length=40, examples=["BTC"])
    rank: int = Field(ge=1, le=1_000_000, description="CoinStats market-cap rank", examples=[1])
    amount: float = Field(gt=0, examples=[0.25])
    price_usd: float = Field(ge=0, examples=[82898.66])
    avg_buy_price_usd: float | None = Field(
        default=None, ge=0, description="Average cost per unit; omit if unknown", examples=[63000.0]
    )


class PortfolioRequest(BaseModel):
    portfolio_id: str | None = Field(default=None, max_length=64)
    holdings: list[Holding] = Field(min_length=1, max_length=5_000)

    @model_validator(mode="after")
    def portfolio_must_have_value(self) -> "PortfolioRequest":
        if sum(h.amount * h.price_usd for h in self.holdings) <= 0:
            raise ValueError("Portfolio has zero total value — at least one holding needs a positive price.")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "portfolio_id": "demo_degen",
                "holdings": [
                    {"coin_id": "bonk", "symbol": "BONK", "rank": 212, "amount": 120_000_000,
                     "price_usd": 0.000003405, "avg_buy_price_usd": 0.000021},
                    {"coin_id": "dogwifcoin", "symbol": "WIF", "rank": 252, "amount": 1_500,
                     "price_usd": 0.2247, "avg_buy_price_usd": 1.90},
                    {"coin_id": "popcat", "symbol": "POPCAT", "rank": 656, "amount": 4_000,
                     "price_usd": 0.051, "avg_buy_price_usd": 0.45},
                    {"coin_id": "solana", "symbol": "SOL", "rank": 7, "amount": 0.8,
                     "price_usd": 117.73, "avg_buy_price_usd": 190.0},
                ],
            }]
        }
    }


class Diagnosis(BaseModel):
    headline: str
    drivers: list[str]
    campaign: str
    recommended_action: str
    push_message: str
    swap_notional_usd: float
    eligible_for_campaign: bool


class PortfolioAnalysis(BaseModel):
    portfolio_id: str | None
    total_portfolio_value: float
    num_assets: int
    hhi_index: float
    effective_num_assets: float
    top1_weight: float
    top3_weight: float
    top10_bluechip_ratio: float
    speculative_meme_ratio: float
    stablecoin_ratio: float
    weighted_market_rank: float
    unrealized_pnl_usd: float | None = Field(description="Null when cost basis is unreliable")
    unrealized_pnl_pct: float | None = Field(description="Null when cost basis is unreliable")
    pnl_reliable: bool
    cost_basis_coverage: float
    archetype_id: int
    archetype: str
    attrition_vulnerability_score: float = Field(ge=0, le=100)
    risk_tier: str
    triple_threat: bool
    diagnosis: Diagnosis
    model_version: str


# ═════════════════════════════════════════════════════════════════════════════
# Core analysis (shared by POST and GET-by-id)
# ═════════════════════════════════════════════════════════════════════════════
def _clean(x: float | None, ndigits: int = 4) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), ndigits)


def analyze_holdings(prepared: pd.DataFrame, model: SegmentationModel, portfolio_id: str | None) -> PortfolioAnalysis:
    """prepared holdings for ONE portfolio → full analysis response."""
    feats = attach_segments(engineer_portfolio_features(prepared), model)
    row = feats.iloc[0]
    levers = swap_levers(prepared).iloc[0]
    dx = diagnose(row, levers)
    reliable = bool(row["pnl_reliable"])
    return PortfolioAnalysis(
        portfolio_id=portfolio_id,
        total_portfolio_value=round(float(row["total_portfolio_value"]), 2),
        num_assets=int(row["num_assets"]),
        hhi_index=_clean(row["hhi_index"]),
        effective_num_assets=_clean(row["effective_num_assets"], 2),
        top1_weight=_clean(row["top1_weight"]),
        top3_weight=_clean(row["top3_weight"]),
        top10_bluechip_ratio=_clean(row["top10_bluechip_ratio"]),
        speculative_meme_ratio=_clean(row["speculative_meme_ratio"]),
        stablecoin_ratio=_clean(row["stablecoin_ratio"]),
        weighted_market_rank=_clean(row["weighted_market_rank"], 1),
        unrealized_pnl_usd=_clean(row["unrealized_pnl_usd"], 2) if reliable else None,
        unrealized_pnl_pct=_clean(row["unrealized_pnl_pct"], 2) if reliable else None,
        pnl_reliable=reliable,
        cost_basis_coverage=_clean(row["cost_basis_coverage"]),
        archetype_id=int(row["archetype_id"]),
        archetype=str(row["archetype"]),
        attrition_vulnerability_score=_clean(row["capitulation_vulnerability_score"], 2),
        risk_tier=str(row["risk_tier"]),
        triple_threat=bool(row["triple_threat"]),
        diagnosis=Diagnosis(**{k: v for k, v in dx.items() if k != "risk_tier"}),
        model_version=model.model_version,
    )


# ═════════════════════════════════════════════════════════════════════════════
# App
# ═════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    model, segmented = load_artifacts()
    app.state.model = model
    app.state.summary = cluster_summary(segmented)
    try:
        app.state.reference_holdings = prepare_holdings(load_holdings())
    except FileNotFoundError:
        app.state.reference_holdings = None
        log.warning("Reference CSV not found — /portfolios/{id} disabled; other endpoints unaffected.")
    log.info("Model %s loaded (trained %s, silhouette %.3f)", model.model_version, model.trained_at, model.silhouette)
    yield


app = FastAPI(
    title="CoinStats Portfolio Intelligence API",
    version=cfg.MODEL_VERSION,
    description=(
        "Portfolio health analytics for CoinStats: concentration (HHI), unrealised PnL, "
        "KMeans investor archetypes, capitulation/attrition vulnerability score, and a "
        "risk-reducing In-App Swap recommendation."
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send browsers that open the bare URL to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(request: Request) -> dict:
    model: SegmentationModel = request.app.state.model
    return {
        "status": "healthy",
        "model_version": model.model_version,
        "trained_at": model.trained_at,
        "reference_portfolios": model.n_samples,
        "silhouette": round(model.silhouette, 3),
        "reference_data_loaded": request.app.state.reference_holdings is not None,
    }


@app.post("/analyze-portfolio", response_model=PortfolioAnalysis)
def analyze_portfolio(payload: PortfolioRequest, request: Request) -> PortfolioAnalysis:
    pid = payload.portfolio_id or "request"
    raw = pd.DataFrame([h.model_dump() for h in payload.holdings])
    raw.insert(0, "portfolio_id", pid)
    try:
        prepared = prepare_holdings(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return analyze_holdings(prepared, request.app.state.model, payload.portfolio_id)


@app.get("/cluster-summary")
def get_cluster_summary(request: Request) -> dict:
    return request.app.state.summary


@app.get("/portfolios/{portfolio_id}", response_model=PortfolioAnalysis)
def analyze_reference_portfolio(portfolio_id: str, request: Request) -> PortfolioAnalysis:
    ref = request.app.state.reference_holdings
    if ref is None:
        raise HTTPException(status_code=503, detail="Reference dataset not loaded on this instance.")
    prepared = ref[ref["portfolio_id"] == portfolio_id]
    if prepared.empty:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio_id '{portfolio_id}'.")
    return analyze_holdings(prepared, request.app.state.model, portfolio_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
