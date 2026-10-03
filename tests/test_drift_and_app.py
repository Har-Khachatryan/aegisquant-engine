"""Drift monitor on real data and a headless smoke test of the dashboard."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from config import REFERENCE_DATA_PATH
from data_pipeline import load_churn_dataset
from DriftMonitor import DriftMonitor, simulate_shifted_batch

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_drift_monitor_flags_only_the_shifted_batch(trained):
    monitor = DriftMonitor(pd.read_csv(REFERENCE_DATA_PATH))
    batch = load_churn_dataset().sample(2_000, random_state=7)
    assert not monitor.should_retrain(monitor.compute_drift(batch))
    drifted = {r.feature_name for r in monitor.compute_drift(simulate_shifted_batch(batch)) if r.drift_detected}
    assert {"age", "balance"} <= drifted


def test_dashboard_renders_all_tabs(trained):
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"ROC-AUC", "Churners caught", "Offer hit-rate", "Churn probability", "Investor profile",
            "Churners reached"} <= labels


def test_dashboard_demo_and_custom_modes(trained):
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio[0].set_value("Demo persona").run()
    assert not at.exception
    at.radio[0].set_value("Custom").run()
    assert not at.exception
    at.slider[0].set_value(40).run()          # Business impact: retention budget
    assert not at.exception
