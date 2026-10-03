"""
AegisQuant — Offline training pipeline (v4.0)

Upgrade v3.2 → v4.0
────────────────────
[EVALUATION LEAK FIXED]
  v3.2 fitted the pipeline on ALL rows and then reported AUC on a 20 % split of
  those same rows, so the "held-out" AUC was measured on training data. v4.0
  uses a strict protocol:

    1. Stratified 80/20 split. The 20 % test set is touched exactly once, at
       the end, for the reported metrics.
    2. 5-fold out-of-fold predictions on the 80 % → cross-validated AUC and the
       F1-optimal decision threshold (chosen without seeing the test set).
    3. Final pipeline fitted on the 80 % (XGBWithValidation carves its own
       early-stopping split inside it).
    4. Test-set report: ROC-AUC, PR-AUC, Brier score, precision/recall/F1 at
       the chosen threshold, capture rate of the riskiest 10 % / 20 %, a
       logistic-regression baseline, SHAP global importance and a fairness
       audit by gender (not a model input) and geography.

Run:  python churn_model.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACT_META_PATH,
    CHURN_FEATURES,
    CV_FOLDS,
    DATA_PATH,
    MODEL_LOG_PATH,
    MODEL_VERSION,
    PROCESSOR_PATH,
    PROFILE_RESOLVER_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    XGB_MODEL_PATH,
)
from data_pipeline import DynamicProfileResolver, load_churn_dataset, model_matrix, summarise_segments
from feature_cross_pollination import OUTPUT_FEATURES, build_pipeline, save_artifacts

log = logging.getLogger("aegis")

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def log_model_metadata(version: str, auc_roc: float, status: str) -> None:
    """Ավտոմատ կերպով գրանցում է մոդելի մետատվյալները models_log.json ֆայլում:"""
    new_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
        "auc_roc": round(auc_roc, 4),
        "status": status,
    }
    data = []
    if MODEL_LOG_PATH.exists():
        try:
            data = json.loads(MODEL_LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    data.append(new_log)
    MODEL_LOG_PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def _file_sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _f1_optimal_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    scores = [f1_score(y, (prob >= t).astype(int), zero_division=0) for t in THRESHOLD_GRID]
    best = int(np.argmax(scores))
    return float(THRESHOLD_GRID[best]), float(scores[best])


def _curve(x: np.ndarray, y: np.ndarray, points: int = 120) -> dict:
    """Down-sample a curve for the dashboard (keeps the end points)."""
    idx = np.unique(np.linspace(0, len(x) - 1, min(points, len(x))).round().astype(int))
    return {"x": np.round(x[idx], 4).tolist(), "y": np.round(y[idx], 4).tolist()}


def _capture_rate(y: np.ndarray, prob: np.ndarray, top: float) -> float:
    """Share of all churners found in the `top` fraction of customers ranked by risk."""
    k = int(round(top * len(prob)))
    order = np.argsort(-prob)[:k]
    return float(y[order].sum() / y.sum())


def _classification_block(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "pr_auc": round(float(average_precision_score(y, prob)), 4),
        "brier": round(float(brier_score_loss(y, prob)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "flagged_share": round(float(pred.mean()), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "capture_top10": round(_capture_rate(y, prob, 0.10), 4),
        "capture_top20": round(_capture_rate(y, prob, 0.20), 4),
    }


def _fairness_audit(test: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict:
    """Per-group calibration and equal-opportunity check (recall of churners)."""
    audit = {}
    frame = test.assign(prob=prob, flagged=(prob >= threshold).astype(int))
    for col in ("gender", "geography"):
        rows = []
        for group, g in frame.groupby(col):
            churners = g[g["churn"] == 1]
            rows.append({
                "group": str(group),
                "customers": int(len(g)),
                "actual_churn_rate": round(float(g["churn"].mean()), 4),
                "mean_predicted": round(float(g["prob"].mean()), 4),
                "flagged_share": round(float(g["flagged"].mean()), 4),
                "recall": round(float(churners["flagged"].mean()), 4) if len(churners) else None,
                "roc_auc": round(float(roc_auc_score(g["churn"], g["prob"])), 4) if g["churn"].nunique() == 2 else None,
            })
        audit[col] = rows
    return audit


def _shap_importance(pipeline, X: pd.DataFrame) -> list[dict]:
    """Mean |SHAP| per transformed feature on the given rows (exact TreeSHAP)."""
    Xt = pipeline.named_steps["cluster_features"].transform(X)
    explainer = shap.TreeExplainer(pipeline.named_steps["xgb_churn"].get_booster())
    values = np.asarray(explainer.shap_values(Xt))
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": OUTPUT_FEATURES[i], "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in order]


def run_training_pipeline() -> dict:
    """
    Train, evaluate and persist the AegisQuant churn engine.
    Safe to call from any module that needs to bootstrap artifacts.
    Returns the artifact metadata dict.
    """
    log.info("=" * 70)
    log.info(f"AegisQuant {MODEL_VERSION}  |  [OFFLINE] Training pipeline — START")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    df = load_churn_dataset(DATA_PATH)
    X = model_matrix(df)
    y = df["churn"].to_numpy()

    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
    log.info(f"  [1/5] Split: {len(idx_tr):,} train / {len(idx_te):,} test (stratified)")

    # ── 2. Out-of-fold predictions → CV AUC + decision threshold ──────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(build_pipeline(), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    cv_auc = float(roc_auc_score(y_tr, oof))
    threshold, oof_f1 = _f1_optimal_threshold(y_tr, oof)
    log.info(f"  [2/5] {CV_FOLDS}-fold CV AUC = {cv_auc:.4f} | F1-optimal threshold = {threshold:.2f} (OOF F1 {oof_f1:.3f})")

    # ── 3. Final fit on the training split ────────────────────────────────────
    pipeline = build_pipeline().fit(X_tr, y_tr)
    injector = pipeline.named_steps["cluster_features"]
    resolver = DynamicProfileResolver().fit(
        df.iloc[idx_tr][["age", "balance"]].assign(cluster_id=injector.predict_cluster(X_tr))
    )
    log.info("  [3/5] Final pipeline fitted on the training split")

    # ── 4. One-time evaluation on the untouched test split ────────────────────
    prob_te = pipeline.predict_proba(X_te)[:, 1]
    test_metrics = _classification_block(y_te, prob_te, threshold)

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000))
    baseline.fit(X_tr[CHURN_FEATURES], y_tr)
    prob_bl = baseline.predict_proba(X_te[CHURN_FEATURES])[:, 1]
    baseline_metrics = _classification_block(y_te, prob_bl, _f1_optimal_threshold(y_tr, baseline.predict_proba(X_tr[CHURN_FEATURES])[:, 1])[0])

    fpr, tpr, _ = roc_curve(y_te, prob_te)
    fpr_b, tpr_b, _ = roc_curve(y_te, prob_bl)
    prec, rec, _ = precision_recall_curve(y_te, prob_te)
    prec_b, rec_b, _ = precision_recall_curve(y_te, prob_bl)
    threshold_curve = {
        "threshold": THRESHOLD_GRID.tolist(),
        "precision": [round(float(precision_score(y_te, prob_te >= t, zero_division=0)), 4) for t in THRESHOLD_GRID],
        "recall": [round(float(recall_score(y_te, prob_te >= t, zero_division=0)), 4) for t in THRESHOLD_GRID],
    }
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(prob_te, bins) - 1, 0, 9)
    calibration = [
        {"predicted": round(float(prob_te[which == b].mean()), 4), "actual": round(float(y_te[which == b].mean()), 4),
         "customers": int((which == b).sum())}
        for b in range(10) if (which == b).any()
    ]
    log.info(
        f"  [4/5] TEST  AUC {test_metrics['roc_auc']:.4f} | PR-AUC {test_metrics['pr_auc']:.4f} | "
        f"recall {test_metrics['recall']:.3f} @ precision {test_metrics['precision']:.3f} | "
        f"baseline AUC {baseline_metrics['roc_auc']:.4f}"
    )

    test_rows = df.iloc[idx_te]
    evaluation = {
        "test": test_metrics,
        "baseline_logistic": baseline_metrics,
        "base_churn_rate": round(float(y.mean()), 4),
        "roc": {"xgb": _curve(fpr, tpr), "baseline": _curve(fpr_b, tpr_b)},
        "pr": {"xgb": _curve(rec[::-1], prec[::-1]), "baseline": _curve(rec_b[::-1], prec_b[::-1])},
        "threshold_curve": threshold_curve,
        "calibration": calibration,
        "shap_importance": _shap_importance(pipeline, X_te),
        "fairness": _fairness_audit(test_rows, prob_te, threshold),
        "segments": summarise_segments(df, injector.predict_cluster(X), resolver),
    }

    cluster_te = injector.predict_cluster(X_te)
    holdout = pd.DataFrame({
        "customer_id": test_rows["customer_id"].to_numpy(),
        "churn": y_te,
        "probability": np.round(prob_te, 5),
        "cluster_id": cluster_te,
        "profile": resolver.transform(pd.DataFrame({"cluster_id": cluster_te})),
    })

    # ── 5. Persist ────────────────────────────────────────────────────────────
    booster = pipeline.named_steps["xgb_churn"]
    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_path": DATA_PATH.name,
        "data_sha256": _file_sha256(DATA_PATH),
        "n_train": int(len(idx_tr)),
        "n_test": int(len(idx_te)),
        "cv_auc": round(cv_auc, 4),
        "val_auc": test_metrics["roc_auc"],          # kept for v3.x consumers
        "test_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "decision_threshold": threshold,
        "best_iteration": int(getattr(booster, "best_iteration", -1) or -1),
        "feature_names": OUTPUT_FEATURES,
        "n_features": len(OUTPUT_FEATURES),
        "profile_mapping": {str(k): v for k, v in resolver.cluster_to_profile_.items()},
    }
    save_artifacts(pipeline, resolver, meta, evaluation, X_tr, holdout)
    log_model_metadata(version=MODEL_VERSION, auc_roc=test_metrics["roc_auc"], status="Trained on Churn_Modelling.csv")
    log.info("  [5/5] Artifacts persisted. Training pipeline complete.")
    log.info("=" * 70)
    return meta


def ensure_trained() -> None:
    """Train if any artifact is missing or older than the dataset."""
    paths = [XGB_MODEL_PATH, PROCESSOR_PATH, PROFILE_RESOLVER_PATH, ARTIFACT_META_PATH]
    stale = not all(p.exists() for p in paths) or min(os.path.getmtime(p) for p in paths) < os.path.getmtime(DATA_PATH)
    if stale:
        log.warning("  Artifacts missing or older than the dataset — running training pipeline...")
        run_training_pipeline()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_training_pipeline()
