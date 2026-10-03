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
  4. shap.TreeExplainer → exact local contributions, summed per business
     concept (a raw input plus the features engineered from it) and returned as
     plain-language drivers with the customer's own values.
"""

from __future__ import annotations

import logging

import numpy as np
import shap

from config import CHURN_THRESHOLD, GROUP_LABELS, MODEL_VERSION, PROFILE_RESOLVER_PATH, RISK_TIER_BOUNDS, ClientFeatures
from data_pipeline import DynamicProfileResolver, client_to_frame
from feature_cross_pollination import group_contributions, load_artifact_meta, load_processor, load_xgb_model

log = logging.getLogger("aegis")


def risk_tier(prob: float) -> str:
    low, high = RISK_TIER_BOUNDS
    return "High" if prob >= high else "Elevated" if prob >= low else "Low"


def _display_value(group: str, client: ClientFeatures, profile: str) -> str:
    return {
        "num_products": str(client.num_products),
        "age": str(client.age),
        "is_active_member": "yes" if client.is_active_member else "no",
        "geography": client.geography,
        "balance": f"€{client.balance:,.0f}",
        "credit_score": str(client.credit_score),
        "tenure": f"{client.tenure} years",
        "estimated_salary": f"€{client.estimated_salary:,.0f}",
        "has_cr_card": "yes" if client.has_cr_card else "no",
        "segment": profile,
    }[group]


def _context(group: str, client: ClientFeatures) -> str:
    """Dataset facts (Churn_Modelling.csv) that make a churn-raising driver actionable."""
    if group == "num_products":
        if client.num_products >= 3:
            return "customers with 3–4 products churn 83–100 %"
        if client.num_products == 1:
            return "single-product customers churn 28 % vs 8 % with two products"
    if group == "is_active_member" and not client.is_active_member:
        if client.age >= 45:
            return "inactive customers aged 45+ churn 67 %"
        return "inactive members churn about twice as often (27 % vs 14 %)"
    if group == "geography" and client.geography == "Germany":
        return "German customers churn 32 % vs 16–17 % in France and Spain"
    if group == "age" and 45 <= client.age <= 65:
        return "churn peaks between 45 and 60, reaching 56 % in the 50s"
    if group == "balance" and client.balance > 0:
        return "customers holding a balance churn 24 % vs 14 % at zero"
    return ""


def describe_driver(group: str, value: str, contribution: float, context: str = "") -> str:
    effect = "raises" if contribution > 0 else "lowers"
    text = f"{GROUP_LABELS.get(group, group)}: {value} — {effect} churn risk"
    return f"{text} ({context})" if context and contribution > 0 else text


class ChurnPredictor:
    """Loads the JSON artifacts once; thread-safe for concurrent reads."""

    def __init__(self) -> None:
        from churn_model import ensure_trained   # local import: training deps only when needed
        ensure_trained()

        self.xgb_model = load_xgb_model()
        self.booster = self.xgb_model.get_booster()
        self.processor = load_processor()
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
        """Full prediction with grouped local SHAP drivers (largest effect first)."""
        frame, Xt = self._transform(client)
        cluster_id = int(self.processor.predict_cluster(frame)[0])
        profile = self.profile_resolver.resolve(cluster_id)
        prob = float(self.xgb_model.predict_proba(Xt)[0, 1])

        grouped = group_contributions(np.asarray(self.explainer.shap_values(Xt)), self.feature_names).iloc[0]
        drivers = []
        for group, contribution in grouped.items():
            value = _display_value(group, client, profile)
            context = _context(group, client)
            drivers.append({
                "feature": group,
                "label": GROUP_LABELS.get(group, group),
                "value": value,
                "contribution": round(float(contribution), 4),
                "text": describe_driver(group, value, float(contribution), context),
            })
        drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        raising = [d for d in drivers if d["contribution"] > 0]
        top_two = (raising + [d for d in drivers if d["contribution"] <= 0])[:2]

        return {
            "churn_probability": round(prob, 4),
            "risk_tier": risk_tier(prob),
            "retention_action": prob >= self.threshold,
            "decision_threshold": self.threshold,
            "investor_profile": profile,
            "segment_id": cluster_id,
            "risk_drivers": [d["text"] for d in top_two],
            "drivers": drivers[:top_k],
            "base_value": round(float(np.asarray(self.explainer.expected_value).reshape(-1)[0]), 4),
            "model_version": self.model_version,
        }
