"""Shared fixtures: train once (if needed) and keep the market worker offline."""

import os

import pytest

os.environ.setdefault("AEGIS_OFFLINE", "1")   # never hit yfinance from the test suite


@pytest.fixture(scope="session")
def trained():
    from churn_model import ensure_trained
    ensure_trained()


@pytest.fixture(scope="session")
def predictor(trained):
    from inference import ChurnPredictor
    return ChurnPredictor()
