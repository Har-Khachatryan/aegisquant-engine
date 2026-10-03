"""
AegisQuant — Online churn inference with local SHAP explanations (v4.0)

ChurnPredictor is the single serving component shared by the REST API, the
dashboard and the optimisation engine (v3.2 had two diverging copies of this
logic in api.py and optimizer.py).

Per request:
  1. ClientFeatures → one-row model matrix (data_pipeline.client_to_frame,
     the same feature code used in training).
  2. ProcessorBundle → nearest-centroid segment + one-hot (NumPy, no pickle).
  3. XGBoost (native JSON) → churn probability; decision threshold and risk
     tier come from the training metadata.
  4. shap.TreeExplainer → exact local contributions; the top positive ones are
     returned as plain-language risk drivers with the customer's own values.
"""

from __future__ import annotations

import logging

import numpy as np
import shap

from config import CHURN_THRESHOLD, FEATURE_LABELS, MODEL_VERSION, RISK_TIER_BOUNDS, ClientFeatures
from data_pipeline import DynamicProfileResolver, client_to_frame
from feature_cross_pollination import load_artifact_meta, load_processor, load_xgb_model

log = logging.getLogger("aegis")


def risk_tier(prob: float) -> str:
    low, high = RISK_TIER_BOUNDS
    return "High" if prob >= high else "Elevated" if prob >= low else "Low"


def _format_value(feature: str, value: float) -> str:
    if feature in {"balance", "estimated_salary"}:
        return f"€{value:,.0f}"
    if feature in {"has_cr_card", "is_active_member", "geo_germany", "geo_spain", "zero_balance"} or feature.startswith("cluster_"):
        return "yes" if value >= 0.5 else "no"
    if feature in {"balance_to_salary", "tenure_to_age"}:
        return f"{value:.2f}"
    return f"{value:.0f}"


def _context(feature: str, value: float) -> str:
    """Dataset facts that make a churn-raising driver actionable (Churn_Modelling.csv)."""
    if feature == "num_products":
        if value >= 3:
            return "customers with 3–4 products churn 83–100 %"
        if value <= 1:
            return "single-product customers churn 28 % vs 8 % with two products"
    if feature == "is_active_member" and value < 0.5:
        return "inactive members churn about twice as often (27 % vs 14 %)"
    if feature == "geo_germany" and value >= 0.5:
        return "German customers churn 32 % vs 16–17 % in France and Spain"
    if feature == "age" and 45 <= value <= 65:
        return "churn peaks between 45 and 60, reaching 56 % in the 50s"
    return ""


def describe_driver(feature: str, value: float, contribution: float) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    effect = "raises" if contribution > 0 else "lowers"
    text = f"{label}: {_format_value(feature, value)} — {effect} churn risk"
    ctx = _context(feature, value) if contribution > 0 else ""
    return f"{text} ({ctx})" if ctx else text


class ChurnPredictor:
    """Loads the JSON artifacts once; thread-safe for concurrent reads."""

    def __init__(self) -> None:
        from churn_model import ensure_trained   # local import: training deps only when needed
        ensure_trained()

        self.xgb_model = load_xgb_model()
        self.booster = self.xgb_model.get_booster()
        self.processor = load_processor()
        from config import PROFILE_RESOLVER_PATH
        self.profile_resolver = DynamicProfileResolver.load(PROFILE_RESOLVER_PATH)
        self.meta = load_artifact_meta()
        self.threshold = float(self.meta.get("decision_threshold", CHURN_THRESHOLD))
        self.feature_names = list(self.booster.feature_names or self.processor.output_feature_names)
        self.explainer = shap.TreeExplainer(self.booster)
        self.model_version = f"{self.meta.get('model_version', MODEL_VERSION)} (trained {self.meta.get('trained_at', 'unknown')[:10]})"
        log.info(f"  ChurnPredictor ready | test AUC {self.meta.get('test_auc', 'N/A')} | threshold {self.threshold:.2f}")

    def _transform(self, client: ClientFeatures):
        frame = client_to_frame(client)
        return frame, self.processor.transform(frame)

    def predict_client(self, client: ClientFeatures) -> tuple[str, float]:
        """(investor_profile, churn_probability) — the v3.x engine contract."""
        frame, Xt = self._transform(client)
        cluster_id = int(self.processor.predict_cluster(frame)[0])
        prob = float(self.xgb_model.predict_proba(Xt)[0, 1])
        return self.profile_resolver.resolve(cluster_id), prob

    def explain(self, client: ClientFeatures, top_k: int = 5) -> dict:
        """Full prediction with local SHAP drivers."""
        frame, Xt = self._transform(client)
        cluster_id = int(self.processor.predict_cluster(frame)[0])
        prob = float(self.xgb_model.predict_proba(Xt)[0, 1])

        shap_vec = np.asarray(self.explainer.shap_values(Xt)).reshape(-1)
        values = Xt.iloc[0].to_numpy()
        drivers = [
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "value": float(val),
                "contribution": round(float(c), 4),
                "text": describe_driver(name, float(val), float(c)),
            }
            for name, val, c in zip(self.feature_names, values, shap_vec)
            if not name.startswith("cluster_")      # segment effect is reported as the profile
        ]
        drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        raising = [d for d in drivers if d["contribution"] > 0]
        top_two = (raising + [d for d in drivers if d["contribution"] <= 0])[:2]

        return {
            "churn_probability": round(prob, 4),
            "risk_tier": risk_tier(prob),
            "retention_action": prob >= self.threshold,
            "decision_threshold": self.threshold,
            "investor_profile": self.profile_resolver.resolve(cluster_id),
            "segment_id": cluster_id,
            "risk_drivers": [d["text"] for d in top_two],
            "drivers": drivers[:top_k],
            "base_value": round(float(np.asarray(self.explainer.expected_value).reshape(-1)[0]), 4),
            "model_version": self.model_version,
        }
