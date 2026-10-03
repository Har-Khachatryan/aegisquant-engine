"""Headless smoke test for the Streamlit dashboard (all four tabs execute on every run)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import settings as cfg

pytestmark = pytest.mark.skipif(not cfg.DATA_PATH.exists(), reason="CoinStats CSV not available")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_dashboard_renders_without_exceptions():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Total AUM (approx.)", "Average HHI", "High-risk portfolios", "Median unrealised PnL"} <= labels


def test_filters_and_inspector_interactions():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.multiselect[0].set_value([2]).run()          # Cluster Explorer: Degens only
    at.toggle[1].set_value(True).run()              # Inspector: High-risk only
    assert not at.exception
    assert any("Showing **113**" in c.value for c in at.caption)
