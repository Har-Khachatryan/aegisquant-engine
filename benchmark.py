"""
AegisQuant — reproducible model benchmark (5-fold CV on the TRAINING split only).

The hold-out test set is never touched here: model choice is made on
cross-validation, and churn_model.py evaluates the chosen model on the test set
exactly once.

    python benchmark.py            # compare candidate models → reports/model_benchmark.json
    python benchmark.py --search   # also run the 30-trial XGBoost random search behind config.XGB_PARAMS

LightGBM and CatBoost are included when installed (they are not runtime
dependencies of AegisQuant).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import CHURN_FEATURES, CV_FOLDS, RANDOM_STATE, ROOT, TEST_SIZE, XGB_PARAMS
from data_pipeline import load_churn_dataset, model_matrix
from feature_cross_pollination import build_pipeline

REPORT_PATH: Path = ROOT / "reports" / "model_benchmark.json"
BASE_FEATURES = CHURN_FEATURES[:13]          # the v4.0 feature set before the six engineered additions
V32_PARAMS = dict(max_depth=4, learning_rate=0.03, n_estimators=600, subsample=0.85, colsample_bytree=0.85,
                  min_child_weight=5, reg_lambda=1.5, early_stopping_rounds=40)


class _EarlyStopXGB(XGBClassifier):
    """XGBoost with early stopping on an inner stratified 20 % split (as in the pipeline)."""

    def fit(self, X, y, **kwargs):
        a, b, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        return super().fit(a, ya, eval_set=[(b, yb)], verbose=False)


def _xgb(params: dict) -> XGBClassifier:
    return _EarlyStopXGB(**params, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)


def _candidates() -> list[tuple[str, callable, list[str] | None]]:
    """(name, factory, feature subset — None means the full AegisQuant pipeline input)."""
    models = [
        ("Logistic regression", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000)), CHURN_FEATURES),
        ("Random forest", lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1,
                                                         random_state=RANDOM_STATE), CHURN_FEATURES),
        ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                                                                        l2_regularization=1.0, random_state=RANDOM_STATE), CHURN_FEATURES),
        ("XGBoost — v3.2 settings, base features", lambda: _xgb(V32_PARAMS), BASE_FEATURES),
        ("XGBoost — v3.2 settings, + engineered features", lambda: _xgb(V32_PARAMS), CHURN_FEATURES),
    ]
    try:
        import lightgbm as lgb
        models.append(("LightGBM", lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=15,
                                                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                                                              colsample_bytree=0.8, reg_lambda=2.0,
                                                              random_state=RANDOM_STATE, verbose=-1), CHURN_FEATURES))
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        models.append(("CatBoost", lambda: CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                                                              random_seed=RANDOM_STATE, verbose=0), CHURN_FEATURES))
    except ImportError:
        pass
    models.append(("AegisQuant — tuned XGBoost + segments", build_pipeline, None))
    return models


def _cv(factory, X: pd.DataFrame, y: np.ndarray, folds) -> dict:
    oof = np.zeros(len(y))
    start = time.perf_counter()
    for tr, va in folds:
        model = factory().fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    per_fold = [roc_auc_score(y[va], oof[va]) for _, va in folds]
    return {
        "cv_auc": round(float(roc_auc_score(y, oof)), 4),
        "cv_auc_std": round(float(np.std(per_fold)), 4),
        "cv_pr_auc": round(float(average_precision_score(y, oof)), 4),
        "seconds": round(time.perf_counter() - start, 1),
    }


def random_search(X: pd.DataFrame, y: np.ndarray, folds, trials: int = 30) -> dict:
    rng = np.random.default_rng(0)
    best = None
    for _ in range(trials):
        params = {
            "max_depth": int(rng.integers(2, 7)),
            "learning_rate": float(10 ** rng.uniform(-2.2, -1)),
            "min_child_weight": float(10 ** rng.uniform(0, 1.5)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "reg_lambda": float(10 ** rng.uniform(-1, 1.3)),
            "gamma": float(rng.choice([0, 0, 0.1, 0.5, 1.0])),
            "n_estimators": 2_000,
            "early_stopping_rounds": 60,
        }
        score = _cv(lambda: _xgb(params), X[CHURN_FEATURES], y, folds)["cv_auc"]
        if best is None or score > best[0]:
            best = (score, params)
        print(f"  trial AUC {score:.4f} | best {best[0]:.4f}")
    return {"cv_auc": best[0], "params": best[1]}


def main(search: bool = False) -> dict:
    df = load_churn_dataset()
    y_all = df["churn"].to_numpy()
    idx_tr, _ = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE)
    X, y = model_matrix(df.iloc[idx_tr]).reset_index(drop=True), y_all[idx_tr]
    folds = list(StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))

    rows = []
    for name, factory, feats in _candidates():
        result = _cv(factory, X if feats is None else X[feats], y, folds)
        rows.append({"model": name, "features": len(feats) if feats else len(CHURN_FEATURES) + 3,
                     "aegisquant": feats is None, **result})
        print(f"{name:48s} CV AUC {result['cv_auc']:.4f} ± {result['cv_auc_std']:.4f} | PR-AUC {result['cv_pr_auc']:.4f}")

    report = {"protocol": f"{CV_FOLDS}-fold stratified CV on the {len(y):,}-customer training split", "results": rows,
              "xgb_params": XGB_PARAMS}
    if search:
        report["random_search"] = random_search(X, y, folds)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport → {REPORT_PATH}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search", action="store_true", help="run the XGBoost random search too")
    main(search=parser.parse_args().search)
