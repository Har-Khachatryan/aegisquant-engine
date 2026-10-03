"""
CoinStats Portfolio Intelligence — offline pipeline.

    python pipeline.py            # build features, fit KMeans, write artifacts

Artifacts (no pickle anywhere):
    artifacts/model.json               scaler params + centroids + archetype map + model card
    artifacts/portfolio_features.csv   one row per portfolio: features, score, tier, archetype

The API and dashboard call load_artifacts(), which bootstraps the pipeline
automatically if the artifacts are missing or older than the source CSV.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

import settings as cfg
from features import engineer_portfolio_features, load_holdings, prepare_holdings
from segmentation import SegmentationModel, attach_segments, fit_segmentation

log = logging.getLogger("coinstats_intel")


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pipeline(data_path: Path = cfg.DATA_PATH) -> tuple[SegmentationModel, pd.DataFrame]:
    """Holdings CSV → features → KMeans → archetypes → persisted artifacts."""
    raw = load_holdings(data_path)
    holdings = prepare_holdings(raw)
    features = engineer_portfolio_features(holdings)

    model = fit_segmentation(features)
    model.data_fingerprint = _fingerprint(Path(data_path))
    segmented = attach_segments(features, model)

    cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(cfg.MODEL_PATH)
    segmented.to_csv(cfg.FEATURES_PATH, index=False)

    log.info(
        "Pipeline complete | %d holdings -> %d portfolios | silhouette=%.3f | %d cost-basis rows flagged",
        len(raw), len(segmented), model.silhouette, int(segmented["flagged_cost_basis_rows"].sum()),
    )
    return model, segmented


def load_artifacts() -> tuple[SegmentationModel, pd.DataFrame]:
    """Load model + features; (re)build them if missing or stale vs. the CSV."""
    stale = (
        not cfg.MODEL_PATH.exists()
        or not cfg.FEATURES_PATH.exists()
        or (cfg.DATA_PATH.exists() and cfg.DATA_PATH.stat().st_mtime > cfg.MODEL_PATH.stat().st_mtime)
    )
    if stale:
        log.warning("Artifacts missing or stale — running pipeline.")
        return run_pipeline()
    features = pd.read_csv(cfg.FEATURES_PATH, dtype={"portfolio_id": "string"})
    return SegmentationModel.load(cfg.MODEL_PATH), features


def cluster_summary(segmented: pd.DataFrame) -> dict:
    """Book-level KPIs + per-archetype statistics (feeds /cluster-summary and the dashboard)."""
    df = segmented
    reliable = df[df["pnl_reliable"]]
    book = {
        "portfolios": int(len(df)),
        "total_aum_usd": round(float(df["total_portfolio_value"].sum()), 2),
        "avg_hhi_index": round(float(df["hhi_index"].mean()), 4),
        "median_unrealized_pnl_pct": round(float(reliable["unrealized_pnl_pct"].median()), 2),
        "mean_unrealized_pnl_pct": round(float(reliable["unrealized_pnl_pct"].mean()), 2),
        "book_unrealized_pnl_pct": round(
            float(reliable["unrealized_pnl_usd"].sum() / reliable["total_cost_basis"].sum() * 100), 2
        ),
        "high_risk_portfolios": int((df["risk_tier"] == "High").sum()),
        "elevated_risk_portfolios": int((df["risk_tier"] == "Elevated").sum()),
        "triple_threat_portfolios": int(df["triple_threat"].sum()),
        "portfolios_with_unreliable_pnl": int((~df["pnl_reliable"]).sum()),
        "note": "AUM is approximate: the dataset scales each portfolio's amounts by a random 0.5–2x factor. "
                "Weights, HHI and PnL % are exact.",
    }
    segments = []
    for aid, seg in df.groupby("archetype_id"):
        rel = seg[seg["pnl_reliable"]]
        segments.append({
            "archetype_id": int(aid),
            "archetype": cfg.ARCHETYPES[int(aid)].name,
            "portfolios": int(len(seg)),
            "share_of_portfolios": round(len(seg) / len(df), 4),
            "aum_usd": round(float(seg["total_portfolio_value"].sum()), 2),
            "share_of_aum": round(float(seg["total_portfolio_value"].sum() / df["total_portfolio_value"].sum()), 4),
            "median_portfolio_value_usd": round(float(seg["total_portfolio_value"].median()), 2),
            "median_num_assets": float(seg["num_assets"].median()),
            "avg_hhi_index": round(float(seg["hhi_index"].mean()), 4),
            "avg_top10_bluechip_ratio": round(float(seg["top10_bluechip_ratio"].mean()), 4),
            "avg_speculative_meme_ratio": round(float(seg["speculative_meme_ratio"].mean()), 4),
            "median_weighted_market_rank": round(float(seg["weighted_market_rank"].median()), 1),
            "median_unrealized_pnl_pct": round(float(rel["unrealized_pnl_pct"].median()), 2) if len(rel) else None,
            "avg_vulnerability_score": round(float(seg["capitulation_vulnerability_score"].mean()), 2),
            "high_risk_portfolios": int((seg["risk_tier"] == "High").sum()),
        })
    return {"book": book, "segments": segments}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    model, segmented = run_pipeline()
    summary = cluster_summary(segmented)
    b = summary["book"]
    print(f"\nPortfolios: {b['portfolios']}  |  AUM ~ ${b['total_aum_usd']:,.0f}  |  "
          f"avg HHI {b['avg_hhi_index']:.3f}  |  median PnL {b['median_unrealized_pnl_pct']:+.1f}%")
    print(f"High risk: {b['high_risk_portfolios']}  |  Elevated: {b['elevated_risk_portfolios']}  |  "
          f"Triple-threat: {b['triple_threat_portfolios']}  |  silhouette {model.silhouette:.3f}\n")
    cols = ["archetype", "portfolios", "share_of_aum", "median_portfolio_value_usd", "median_num_assets",
            "avg_hhi_index", "avg_speculative_meme_ratio", "median_unrealized_pnl_pct",
            "avg_vulnerability_score", "high_risk_portfolios"]
    print(pd.DataFrame(summary["segments"])[cols].to_string(index=False))
