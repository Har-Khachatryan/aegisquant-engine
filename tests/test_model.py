"""Model quality, evaluation protocol, pickle-free processor and SHAP correctness."""

import json

import numpy as np
import pandas as pd
import pytest

from config import ARTIFACT_META_PATH, DEMO_CLIENTS, EVALUATION_PATH, HOLDOUT_PATH, ClientFeatures
from data_pipeline import load_churn_dataset, model_matrix
from feature_cross_pollination import ClusterInjector, ProcessorBundle


@pytest.fixture(scope="module")
def meta(trained):
    return json.loads(ARTIFACT_META_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def evaluation(trained):
    return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))


def test_holdout_quality_and_no_overfitting(meta):
    assert meta["test_auc"] > 0.85
    assert abs(meta["cv_auc"] - meta["test_auc"]) < 0.03       # CV estimate ≈ untouched test
    assert 0.1 < meta["decision_threshold"] < 0.9


def test_xgboost_beats_logistic_baseline(evaluation):
    # The baseline sees the same engineered features, so the margin is the value of non-linearity.
    assert evaluation["test"]["roc_auc"] > evaluation["baseline_logistic"]["roc_auc"] + 0.01
    assert evaluation["test"]["pr_auc"] > evaluation["baseline_logistic"]["pr_auc"]


def test_grouped_explanations_add_up(predictor):
    from feature_cross_pollination import group_contributions
    client = ClientFeatures(**DEMO_CLIENTS[0])
    _, Xt = predictor._transform(client)
    raw = np.asarray(predictor.explainer.shap_values(Xt))
    grouped = group_contributions(raw, predictor.feature_names)
    assert grouped.to_numpy().sum() == pytest.approx(raw.sum(), abs=1e-6)   # every feature counted once


def test_test_split_is_disjoint_and_complete(meta):
    holdout = pd.read_csv(HOLDOUT_PATH)
    assert meta["n_train"] + meta["n_test"] == 10_000
    assert len(holdout) == meta["n_test"] and holdout["customer_id"].is_unique


def test_processor_bundle_matches_sklearn_kmeans():
    X = model_matrix(load_churn_dataset().sample(2_000, random_state=0))
    injector = ClusterInjector().fit(X)
    bundle = ProcessorBundle.from_injector(injector)
    np.testing.assert_array_equal(bundle.predict_cluster(X), injector.predict_cluster(X))
    pd.testing.assert_frame_equal(bundle.transform(X), injector.transform(X))


def test_shap_values_add_up_to_the_model_margin(predictor):
    client = ClientFeatures(**DEMO_CLIENTS[0])
    _, Xt = predictor._transform(client)
    shap_vec = np.asarray(predictor.explainer.shap_values(Xt)).reshape(-1)
    base = float(np.asarray(predictor.explainer.expected_value).reshape(-1)[0])
    prob = predictor.xgb_model.predict_proba(Xt)[0, 1]
    assert base + shap_vec.sum() == pytest.approx(np.log(prob / (1 - prob)), abs=1e-3)


def test_explanations_are_sensible(predictor):
    risky = predictor.explain(ClientFeatures(**DEMO_CLIENTS[0]))     # 52, Germany, 3 products, inactive
    safe = predictor.explain(ClientFeatures(**DEMO_CLIENTS[1]))      # 29, France, 2 products, active
    assert risky["retention_action"] and risky["risk_tier"] == "High"
    assert not safe["retention_action"] and safe["risk_tier"] == "Low"
    assert any("Products held" in d for d in risky["risk_drivers"])
    assert risky["investor_profile"] in {"conservative", "balanced", "aggressive"}
