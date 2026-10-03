"""Data loading, feature engineering, train/serve parity and profile resolution."""

import numpy as np
import pandas as pd
import pytest

from config import CHURN_FEATURES, CLUSTER_FEATURES, DATA_PATH, ClientFeatures
from data_pipeline import DynamicProfileResolver, client_to_frame, load_churn_dataset, model_matrix


def test_dataset_is_clean_and_drops_identifiers():
    df = load_churn_dataset()
    assert len(df) == 10_000
    assert "Surname" not in df and "RowNumber" not in df
    assert set(df["churn"].unique()) == {0, 1}
    assert df["churn"].mean() == pytest.approx(0.2037, abs=1e-4)


def test_missing_column_is_rejected(tmp_path):
    raw = pd.read_csv(DATA_PATH).drop(columns=["Exited"])
    path = tmp_path / "broken.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Exited"):
        load_churn_dataset(path)


def test_engineered_features():
    row = pd.DataFrame([{
        "credit_score": 600, "geography": "Germany", "age": 40, "tenure": 4, "balance": 0.0,
        "num_products": 2, "has_cr_card": True, "is_active_member": False, "estimated_salary": 50_000.0,
    }])
    X = model_matrix(row).iloc[0]
    assert list(model_matrix(row).columns) == list(dict.fromkeys(CHURN_FEATURES + CLUSTER_FEATURES))
    assert X["geo_germany"] == 1 and X["geo_spain"] == 0
    assert X["zero_balance"] == 1 and X["balance_to_salary"] == 0
    assert X["tenure_to_age"] == pytest.approx(0.1)
    assert X["has_cr_card"] == 1 and X["is_active_member"] == 0


def test_serving_features_match_training_features():
    """The API path (ClientFeatures) must build exactly the training matrix."""
    df = load_churn_dataset().head(50)
    train_side = model_matrix(df)
    for i, r in df.reset_index(drop=True).iterrows():
        client = ClientFeatures(
            credit_score=r.credit_score, geography=r.geography, age=r.age, tenure=r.tenure, balance=r.balance,
            num_products=r.num_products, has_cr_card=bool(r.has_cr_card),
            is_active_member=bool(r.is_active_member), estimated_salary=r.estimated_salary,
        )
        np.testing.assert_allclose(client_to_frame(client).iloc[0].to_numpy(), train_side.iloc[i].to_numpy())


def test_profile_resolver_orders_by_risk_capacity(tmp_path):
    X = pd.DataFrame({
        "cluster_id": [0] * 3 + [1] * 3 + [2] * 3,
        "age":        [60, 62, 64, 30, 32, 34, 31, 33, 35],
        "balance":    [80e3, 90e3, 85e3, 0, 0, 1e3, 120e3, 130e3, 125e3],
    })
    resolver = DynamicProfileResolver().fit(X)
    assert resolver.cluster_to_profile_ == {0: "conservative", 1: "balanced", 2: "aggressive"}
    resolver.save(tmp_path / "r.json")
    assert DynamicProfileResolver.load(tmp_path / "r.json").cluster_to_profile_ == resolver.cluster_to_profile_
