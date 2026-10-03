"""Tests for the KMeans segmentation, JSON model persistence, and archetype mapping."""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import settings as cfg
from features import engineer_portfolio_features, load_holdings
from segmentation import SegmentationModel, attach_segments, build_design_matrix, fit_segmentation

pytestmark = pytest.mark.skipif(not cfg.DATA_PATH.exists(), reason="CoinStats CSV not available")


@pytest.fixture(scope="module")
def features():
    return engineer_portfolio_features(load_holdings())


@pytest.fixture(scope="module")
def model(features):
    return fit_segmentation(features)


def test_numpy_inference_matches_sklearn_kmeans(features, model):
    Z = StandardScaler().fit_transform(build_design_matrix(features))
    km = KMeans(cfg.N_CLUSTERS, n_init=cfg.KMEANS_N_INIT, random_state=cfg.KMEANS_SEED).fit(Z)
    assert np.array_equal(model.predict_cluster(features), km.labels_)


def test_archetype_mapping_is_one_to_one(model):
    assert sorted(model.cluster_to_archetype.values()) == sorted(cfg.ARCHETYPES)


def test_archetypes_are_semantically_correct(features, model):
    seg = attach_segments(features, model)
    by = seg.groupby("archetype_id")
    assert by["total_portfolio_value"].median().idxmax() == 0     # Whale: largest balances
    assert by["top10_bluechip_ratio"].mean().idxmax() == 0        # ... in blue chips
    assert by["num_assets"].median().idxmax() == 1                # Explorer: most coins
    assert by["hhi_index"].mean().idxmin() == 1                   # ... least concentrated
    assert by["speculative_meme_ratio"].mean().idxmax() == 2      # Degen: long-tail heavy


def test_json_roundtrip_preserves_predictions(tmp_path, features, model):
    path = tmp_path / "model.json"
    model.save(path)
    loaded = SegmentationModel.load(path)
    assert np.array_equal(loaded.predict_archetype(features), model.predict_archetype(features))
    assert "pickle" not in path.read_text()  # plain JSON
