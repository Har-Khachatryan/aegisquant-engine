"""
AegisQuant — AI Portfolio Shield & Risk Engine (v3.1)
Enterprise‑grade Streamlit dashboard.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import logging
import sys
from io import StringIO

# Local modules
from config import DEMO_CLIENTS, CHURN_THRESHOLD, ASSETS
from data_pipeline import generate_synthetic_data
from churn_model import run_training_pipeline
from optimizer import AegisQuantEngine

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AegisQuant — AI Portfolio Shield & Risk Engine (v3.1)",
    page_icon="🛡️",
    layout="wide",
)

# ── Cached engine initialization ─────────────────────────────────────────────
@st.cache_resource(ttl=3600)
def get_engine() -> AegisQuantEngine:
    engine = AegisQuantEngine()
    engine.warm_up()
    return engine

engine = get_engine()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ AegisQuant v3.1")
st.sidebar.markdown("### Client Configuration")

client_mode = st.sidebar.radio(
    "Select data source",
    ["Demo Client (Batch)", "Custom Enterprise Client"],
)

if client_mode == "Demo Client (Batch)":
    client_options = {
        f"Client #{c['client_id']} — {c['description']}": c
        for c in DEMO_CLIENTS
    }
    selected_label = st.sidebar.selectbox("Choose a client", list(client_options.keys()))
    payload_dict = client_options[selected_label]
else:
    st.sidebar.markdown("#### Investor Behaviour")
    avg_holding_days = st.sidebar.slider("Avg Holding Days", 1, 365, 60)
    crypto_ratio = st.sidebar.slider("Crypto Ratio", 0.0, 1.0, 0.2, 0.05)
    tech_stocks_ratio = st.sidebar.slider("Tech Stocks Ratio", 0.0, 1.0 - crypto_ratio, 0.3, 0.05)
    st.sidebar.markdown("#### Financial Health")
    account_balance = st.sidebar.number_input("Account Balance ($)", 1000, 500000, 50000, 5000)
    balance_velocity = st.sidebar.slider("Balance Velocity", 0.0, 5.0, 1.0, 0.1)
    market_pain_index = st.sidebar.slider("Market Pain Index", 0.0, 1.0, 0.5, 0.05)
    login_freq_drop = st.sidebar.slider("Login Frequency Drop", 0.0, 5.0, 1.0, 0.1)
    payload_dict = {
        "client_id": 9999,
        "description": "Custom Enterprise Client",
        "avg_holding_days": avg_holding_days,
        "crypto_ratio": crypto_ratio,
        "tech_stocks_ratio": tech_stocks_ratio,
        "account_balance": account_balance,
        "balance_velocity": balance_velocity,
        "market_pain_index": market_pain_index,
        "login_freq_drop": login_freq_drop,
    }

# ── Dashboard header ─────────────────────────────────────────────────────────
st.title("AegisQuant — AI Portfolio Shield & Risk Engine (v3.1)")
st.markdown("#### Enterprise Risk Analytics & Dynamic Allocation")

# ── Run analysis ─────────────────────────────────────────────────────────────
from config import ClientPayload

try:
    payload = ClientPayload(**payload_dict)
except Exception as e:
    st.error(f"Payload validation error: {e}")
    st.stop()

profile, churn_prob = engine.predict_client(payload)
action_required = churn_prob > CHURN_THRESHOLD

# ── Layout: two asymmetric columns ───────────────────────────────────────────
left_col, right_col = st.columns([0.35, 0.65])

with left_col:
    st.subheader("Risk Analytics Metrics")
    # Color coded status
    if churn_prob < 0.4:
        color = "green"
        icon = "🟢"
    elif churn_prob < 0.7:
        color = "orange"
        icon = "🟡"
    else:
        color = "red"
        icon = "🔴"

    st.metric(
        label="Investor Profile",
        value=profile.upper(),
    )
    st.markdown(
        f"<h3 style='color:{color}; margin-top:0;'>{icon} Churn Attrition Risk: {churn_prob:.2%}</h3>",
        unsafe_allow_html=True,
    )
    if churn_prob < 0.4:
        st.success("Low risk — no immediate action required")
    elif churn_prob < 0.7:
        st.warning("Elevated risk — consider monitoring")
    else:
        st.error("High risk — retention action recommended")

    if action_required:
        st.markdown("**Status:** Retention intervention triggered")
    else:
        st.info("Client is currently stable")

with right_col:
    if action_required:
        st.subheader("Dynamic Asset Allocation")
        market_ctx = engine.get_market_context()
        weights = engine.optimize_portfolio(
            profile, market_ctx[0], market_ctx[1], churn_prob, payload
        )
        # Prepare data for chart
        assets = []
        percentages = []
        dollar_amounts = []
        for asset, w in zip(ASSETS, weights):
            if w > 0.005:
                assets.append(asset)
                percentages.append(w * 100)
                dollar_amounts.append(payload.account_balance * w)

        # Plotly donut chart
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=assets,
                    values=percentages,
                    hole=0.4,
                    textinfo="label+percent",
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Allocation: %{value:.2f}%<br>"
                        "Amount: $%{customdata:,.2f}"
                    ),
                    customdata=dollar_amounts,
                )
            ]
        )
        fig.update_layout(
            title_text="Optimal Portfolio Weights",
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table summary
        st.markdown("#### Detailed Breakdown")
        summary_df = {
            "Asset": assets,
            "Weight (%)": [f"{p:.2f}%" for p in percentages],
            "Capital Allocated ($)": [f"${d:,.2f}" for d in dollar_amounts],
        }
        st.dataframe(summary_df, hide_index=True)
    else:
        st.info("Churn risk is below the 50% threshold — no portfolio rebalancing required.")

# ── Log expander ─────────────────────────────────────────────────────────────
with st.expander("📋 AegisQuant v3.1 Execution Log (Terminal ANSI Box‑Art)"):
    box_art = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AEGISQUANT — AI PORTFOLIO SHIELD & RISK ENGINE v3.1                 ║
║          Stack  : XGBoost · KMeans · Ledoit‑Wolf · Dynamic Markowitz         ║
║          Status : All v3.1 bug fixes applied (feasibility repair, PD check)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    st.code(box_art, language=None)