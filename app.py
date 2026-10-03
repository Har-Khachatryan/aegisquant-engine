"""
AegisQuant — AI Portfolio Shield & Risk Engine (v4.0)
Streamlit dashboard on the Kaggle "Churn Modelling" bank dataset.

Tabs
────
  Customer            — risk meter, grouped SHAP drivers, churn-adjusted retention portfolio
  Business impact     — how many churners a retention budget reaches; what drives churn
  Model quality       — benchmark of 8 models, ROC / PR vs baseline, calibration, errors
  Segments & fairness — life-stage segments and the per-group audit
  Monitoring          — KS drift monitor and the retraining trigger

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (ASSET_CLASS, ASSETS, DEMO_CLIENTS, GROUP_LABELS, HOLDOUT_PATH, REFERENCE_DATA_PATH,
                    RISK_TIER_BOUNDS, ROOT, ClientFeatures)
from data_pipeline import load_churn_dataset
from DriftMonitor import (DRIFT_P_VALUE_THRESHOLD, MAX_DRIFTED_FEATURES_BEFORE_RETRAIN, DriftMonitor,
                          results_frame, simulate_shifted_batch)
from feature_cross_pollination import load_evaluation
from optimizer import AegisQuantEngine

st.set_page_config(page_title="AegisQuant — Churn Risk Engine", page_icon="🛡️", layout="wide")

# ── Visual system ─────────────────────────────────────────────────────────────
# Categorical hues in fixed order (validated for colour-vision deficiency on white).
BLUE, AQUA, ORANGE, VIOLET = "#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"
PROFILE_COLOR = {"conservative": BLUE, "balanced": AQUA, "aggressive": ORANGE}
CLASS_COLOR = {"Core": BLUE, "Tech equity": AQUA, "Crypto": VIOLET}
RAISES, LOWERS = ORANGE, BLUE
# Risk tier = state → reserved status colours, always paired with an icon and a label.
TIER_COLOR = {"High": "#d03b3b", "Elevated": "#fab219", "Low": "#0ca30c"}
TIER_ICON = {"High": "🔴", "Elevated": "🟠", "Low": "🟢"}
GRID, AXIS, MUTED, INK = "#e1e0d9", "#c3c2b7", "#8a8a80", "#3d3d38"
PERCENT = "%.1f%%"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1400px;}
      [data-testid="stMetricValue"] {font-size: 1.9rem;}
      .lede {color: #6b6b63; font-size: 1.02rem; margin: -0.6rem 0 1.2rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def chart(fig: go.Figure, title: str, height: int = 360, caption: str | None = None) -> None:
    """Title as text above the plot, so it never collides with a top legend."""
    st.markdown(f"**{title}**")
    st.plotly_chart(style(fig, height), width="stretch", config={"displayModeBar": False})
    if caption:
        st.caption(caption)


def pct(x: float, digits: int = 0) -> str:
    return f"{x:.{digits}%}"


def as_percent(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Shares → 0–100 so tables use a locale-independent "%.1f%%" format."""
    return df.assign(**{c: df[c] * 100 for c in cols if c in df})


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AegisQuant engine…")
def get_engine() -> AegisQuantEngine:
    # Trains on first run; market data loads in the background. AEGIS_OFFLINE=1 skips
    # yfinance entirely (tests, demos without internet) and uses the labelled fallback.
    return AegisQuantEngine(start_market_worker=os.getenv("AEGIS_OFFLINE") != "1")


@st.cache_data(show_spinner=False)
def get_customers() -> pd.DataFrame:
    return load_churn_dataset()


@st.cache_data(show_spinner=False)
def get_holdout() -> pd.DataFrame:
    holdout = pd.read_csv(HOLDOUT_PATH)
    return holdout.merge(get_customers().drop(columns=["churn"]), on="customer_id").sort_values("probability", ascending=False)


@st.cache_data(show_spinner=False)
def get_benchmark() -> dict:
    path = ROOT / "reports" / "model_benchmark.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


engine = get_engine()
evaluation = load_evaluation()
customers, holdout, benchmark = get_customers(), get_holdout(), get_benchmark()
meta, threshold = engine.predictor.meta, engine.threshold
test, baseline, base_rate = evaluation["test"], evaluation["baseline_logistic"], evaluation["base_churn_rate"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ AegisQuant")
    st.caption("v4.0 · churn risk & retention engine")
    st.markdown("### Score a customer")
    mode = st.radio("Source", ["Hold-out customer (real)", "Demo persona", "Custom"], label_visibility="collapsed")
    actual = None
    if mode.startswith("Hold-out"):
        only_flagged = st.toggle("Only customers above the threshold", value=True)
        pool = holdout[holdout["probability"] >= threshold] if only_flagged else holdout
        labels = {f"#{r.customer_id} · {r.probability:.0%} · {r.geography}, {r.age}": r for r in pool.itertuples()}
        choice = labels[st.selectbox(f"{len(labels):,} customers, riskiest first", list(labels))]
        actual = "Churned" if choice.churn == 1 else "Stayed"
        payload = ClientFeatures(
            customer_id=int(choice.customer_id), credit_score=int(choice.credit_score), geography=choice.geography,
            age=int(choice.age), tenure=int(choice.tenure), balance=float(choice.balance),
            num_products=int(choice.num_products), has_cr_card=bool(choice.has_cr_card),
            is_active_member=bool(choice.is_active_member), estimated_salary=float(choice.estimated_salary),
        )
    elif mode == "Demo persona":
        demo = {c["description"]: c for c in DEMO_CLIENTS}
        payload = ClientFeatures(**demo[st.selectbox("Persona", list(demo))])
    else:
        with st.form("custom"):
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 18, 100, 48)
            tenure = c2.number_input("Tenure (yrs)", 0, 60, 3)
            credit = c1.number_input("Credit score", 300, 900, 650)
            products = c2.selectbox("Products", [1, 2, 3, 4], index=0)
            balance = c1.number_input("Balance (€)", 0, 1_000_000, 90_000, step=5_000)
            salary = c2.number_input("Salary (€)", 1_000, 1_000_000, 80_000, step=5_000)
            geo = st.selectbox("Market", ["France", "Germany", "Spain"])
            active = st.toggle("Active member", value=False)
            card = st.toggle("Has credit card", value=True)
            st.form_submit_button("Score customer", type="primary", width="stretch")
        payload = ClientFeatures(credit_score=credit, geography=geo, age=age, tenure=tenure, balance=balance,
                                 num_products=products, has_cr_card=card, is_active_member=active,
                                 estimated_salary=salary)
    st.divider()
    st.markdown(
        f"""
**Model card**
- Data: Kaggle *Churn Modelling* — 10,000 customers, {pct(base_rate, 1)} churned
- Pipeline: KMeans segments → tuned XGBoost → SHAP
- Trained {meta.get('trained_at', '')[:10]} on {meta.get('n_train', 0):,}; tested once on {meta.get('n_test', 0):,} unseen
- Decision threshold {threshold:.2f} (F1-optimal on cross-validation)
"""
    )
    st.caption("Gender is not a model input — it is used only to audit fairness.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Churn Risk & Retention Engine")
st.markdown(
    "<p class='lede'>Who is about to leave the bank, why, and what to offer them — scored on real customer "
    "data, measured on customers the model never saw.</p>",
    unsafe_allow_html=True,
)
top20_lift = test["capture_top20"] / 0.20
k = st.columns(4)
k[0].metric("ROC-AUC", f"{test['roc_auc']:.3f}", f"{test['roc_auc'] - baseline['roc_auc']:+.3f} vs logistic regression",
            border=True, help="How well the model ranks churners above stayers (0.5 = random, 1 = perfect). Hold-out set.")
k[1].metric("Churners caught", pct(test["recall"]), border=True,
            help=f"Share of customers who actually left that the model flags (threshold {threshold:.2f}).")
k[2].metric("Offer hit-rate", pct(test["precision"]), border=True,
            help="Share of flagged customers who really leave — retention offers that reach a real churner.")
k[3].metric("Riskiest 20 % holds", f"{pct(test['capture_top20'])} of churners", f"{top20_lift:.1f}× random", border=True,
            help="Contacting the 20 % highest-risk customers reaches this share of everyone who leaves.")

tab_customer, tab_impact, tab_model, tab_segments, tab_monitor = st.tabs(
    ["Customer", "Business impact", "Model quality", "Segments & fairness", "Monitoring"]
)

# ═════════════════════════════════════════════════════════════════════════════
# Customer
# ═════════════════════════════════════════════════════════════════════════════
with tab_customer:
    result = engine.explain_client(payload)
    prob, tier, profile = result["churn_probability"], result["risk_tier"], result["investor_profile"]
    action = result["retention_action"]

    verdict = "retention offer recommended" if action else "no action needed"
    {"High": st.error, "Elevated": st.warning, "Low": st.success}[tier](
        f"{TIER_ICON[tier]} **{tier} churn risk — {prob:.1%}** · {verdict}"
    )

    # Risk meter: tier zones, the customer's probability and the decision threshold
    low, high = RISK_TIER_BOUNDS
    meter = go.Figure()
    for x0, x1, name in ((0, low, "Low"), (low, high, "Elevated"), (high, 1, "High")):
        meter.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=TIER_COLOR[name], opacity=0.16, line_width=0)
        meter.add_annotation(x=(x0 + x1) / 2, y=1.32, text=name, showarrow=False, font=dict(color=INK, size=12))
    meter.add_shape(type="rect", x0=0, x1=prob, y0=0.3, y1=0.7, fillcolor=INK, line_width=0)
    meter.add_shape(type="line", x0=threshold, x1=threshold, y0=-0.05, y1=1.05, line=dict(color=INK, dash="dot", width=1.5))
    meter.add_annotation(x=threshold, y=-0.42, text=f"action threshold {threshold:.2f}", showarrow=False,
                         font=dict(color=MUTED, size=11))
    meter.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=False, linecolor=AXIS)
    meter.update_yaxes(visible=False, range=[-0.7, 1.6])
    meter.update_layout(height=120, margin=dict(l=8, r=8, t=4, b=4), plot_bgcolor="white", showlegend=False)
    st.plotly_chart(meter, width="stretch", config={"displayModeBar": False})

    m = st.columns(4)
    m[0].metric("Churn probability", f"{prob:.1%}", border=True)
    m[1].metric("Investor profile", profile.title(), border=True,
                help="Life-stage segment (KMeans on age, balance, salary) mapped by risk capacity.")
    m[2].metric("Products · active", f"{payload.num_products} · {'yes' if payload.is_active_member else 'no'}", border=True)
    m[3].metric("Actual outcome" if actual else "Market", actual or payload.geography, border=True,
                help="Known for hold-out customers only; the model never saw it." if actual else None)

    left, right = st.columns([0.46, 0.54], gap="large")
    with left:
        st.subheader("Why this score")
        drivers = pd.DataFrame(result["drivers"]).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=drivers["contribution"], y=drivers["label"] + "  ·  " + drivers["value"], orientation="h",
            marker=dict(color=[RAISES if c > 0 else LOWERS for c in drivers["contribution"]], cornerradius=4),
            customdata=drivers["text"], hovertemplate="%{customdata}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=AXIS)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="contribution to churn log-odds")
        chart(fig, "Top factors for this customer", 290,
              "Orange pushes risk up, blue pulls it down (exact SHAP values, grouped by input).")
        for d in result["drivers"][:3]:
            st.markdown(f"- {d['text']}")

    with right:
        st.subheader("Retention offer")
        if not action:
            st.info(f"Churn risk is below the {threshold:.2f} action threshold, so no offer is generated. "
                    "Try a hold-out customer above the threshold or the first demo persona.")
        else:
            mean_ret, cov = engine.get_market_context()
            weights = engine.optimize_portfolio(profile, mean_ret, cov, prob, payload)
            stats = engine.portfolio_stats(weights, mean_ret, cov)
            gamma = engine._compute_gamma(profile, prob)
            alloc = pd.DataFrame({"asset": ASSETS, "weight": weights, "class": [ASSET_CLASS[a] for a in ASSETS]})
            alloc = alloc[alloc["weight"] > 0.005].sort_values("weight")

            fig = go.Figure()
            for cls, color in CLASS_COLOR.items():
                part = alloc[alloc["class"] == cls]
                if len(part):
                    fig.add_bar(x=part["weight"], y=part["asset"], orientation="h", name=cls,
                                marker=dict(color=color, cornerradius=4),
                                text=[pct(w) for w in part["weight"]], textposition="outside",
                                hovertemplate="%{y}: %{x:.1%}<extra>" + cls + "</extra>")
            fig.update_xaxes(tickformat=".0%", range=[0, alloc["weight"].max() * 1.25])
            fig.update_yaxes(categoryorder="array", categoryarray=alloc["asset"].tolist())
            chart(fig, f"Churn-adjusted Markowitz portfolio — {profile} profile", 330)

            s = st.columns(3)
            s[0].metric("Expected return", pct(stats["expected_return"], 1), border=True,
                        help="Trailing one-year average — not a forecast.")
            s[1].metric("Volatility", pct(stats["volatility"], 1), border=True)
            s[2].metric("Risk aversion γ", f"{gamma:.1f}", border=True,
                        help="Profile base γ, raised as churn risk exceeds the threshold — a calmer offer for a customer in distress.")
            source = engine.market_source
            if "synthetic" in source:
                st.warning("Live market data unavailable — the allocation uses a synthetic covariance and is illustrative.")
            else:
                st.caption(f"Market data: {source}, one year of daily prices, Ledoit-Wolf covariance.")

# ═════════════════════════════════════════════════════════════════════════════
# Business impact
# ═════════════════════════════════════════════════════════════════════════════
with tab_impact:
    st.subheader("How much churn can a retention budget reach?")
    y_sorted = holdout.sort_values("probability", ascending=False)["churn"].to_numpy()
    n, total = len(y_sorted), y_sorted.sum()
    share = st.slider("Contact the riskiest … of customers", 5, 60, 20, 5, format="%d%%") / 100
    k_contact = int(round(share * n))
    caught = int(y_sorted[:k_contact].sum())
    recall_at, precision_at = caught / total, caught / k_contact

    b = st.columns(4)
    b[0].metric("Customers contacted", f"{k_contact:,} of {n:,}", border=True)
    b[1].metric("Churners reached", f"{caught} of {total}", f"{pct(recall_at)} of all churners", border=True)
    b[2].metric("Hit rate", pct(precision_at), f"{precision_at / base_rate:.1f}× the {pct(base_rate)} base rate", border=True)
    b[3].metric("Lift vs random list", f"{recall_at / share:.1f}×", border=True,
                help="A random list of the same size would reach the same share of churners as of customers.")

    c1, c2 = st.columns([0.58, 0.42], gap="large")
    with c1:
        xs = np.arange(n + 1) / n
        gains = np.concatenate([[0], np.cumsum(y_sorted)]) / total
        perfect = np.minimum(np.arange(n + 1) / total, 1)
        fig = go.Figure()
        fig.add_scatter(x=xs, y=perfect, mode="lines", name="Perfect model", line=dict(color=AXIS, width=1.5, dash="dash"))
        fig.add_scatter(x=xs, y=gains, mode="lines", name="AegisQuant", line=dict(color=BLUE, width=2.5),
                        hovertemplate="Contact %{x:.0%} → reach %{y:.0%} of churners<extra></extra>")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random list", line=dict(color=MUTED, width=1.5, dash="dot"))
        fig.add_scatter(x=[share], y=[recall_at], mode="markers", showlegend=False,
                        marker=dict(color=BLUE, size=12, line=dict(color="white", width=2)), hoverinfo="skip")
        fig.add_annotation(x=share, y=recall_at, text=f"{pct(share)} → {pct(recall_at)}", ax=40, ay=30,
                           font=dict(color=INK), arrowcolor=AXIS)
        fig.update_xaxes(title="Share of customers contacted (riskiest first)", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="Share of churners reached", tickformat=".0%", range=[0, 1.02])
        chart(fig, "Cumulative gains on the 2,000 hold-out customers", 400)
    with c2:
        imp = pd.DataFrame(evaluation["shap_importance"])
        imp = imp[imp["feature"] != "segment"].head(8).iloc[::-1]
        imp["label"] = imp["feature"].map(GROUP_LABELS)
        fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["label"], orientation="h",
                               marker=dict(color=BLUE, cornerradius=4), hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.update_xaxes(title="mean |SHAP| (log-odds)")
        chart(fig, "What drives churn overall", 400)

    full = customers.shape[0]
    st.info(
        f"Scaled to the whole bank ({full:,} customers, about {int(base_rate * full):,} leavers): calling the riskiest "
        f"{pct(share)} — {int(share * full):,} customers — would reach roughly **{int(recall_at * base_rate * full):,} "
        f"leavers**, versus {int(share * base_rate * full):,} with a random list."
    )

    st.markdown("**Who leaves — churn rate by the strongest factors** (all 10,000 customers)")
    views = {
        "Products held": customers.groupby("num_products")["churn"].mean().rename(index=str),
        "Age": customers.groupby(pd.cut(customers["age"], [17, 30, 40, 50, 60, 100],
                                        labels=["18–30", "31–40", "41–50", "51–60", "60+"]), observed=True)["churn"].mean(),
        "Active member": customers.groupby(customers["is_active_member"].map({1: "Active", 0: "Inactive"}))["churn"].mean(),
        "Market": customers.groupby("geography")["churn"].mean(),
    }
    cols = st.columns(4)
    for col, (title, series) in zip(cols, views.items()):
        with col:
            fig = go.Figure(go.Bar(x=series.index.astype(str), y=series.values,
                                   marker=dict(color=[ORANGE if v == series.max() else BLUE for v in series.values], cornerradius=4),
                                   text=[pct(v) for v in series.values], textposition="outside",
                                   hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
            fig.add_hline(y=base_rate, line=dict(color=MUTED, dash="dot", width=1))
            fig.update_yaxes(tickformat=".0%", range=[0, min(1.15, series.max() * 1.25)], showticklabels=False)
            fig.update_layout(showlegend=False)
            chart(fig, title, 230)
    st.caption(f"Orange marks the highest-churn group; the dotted line is the bank average ({pct(base_rate, 1)}).")

# ═════════════════════════════════════════════════════════════════════════════
# Model quality
# ═════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.subheader("Is this the best model the data allows?")
    if benchmark:
        bench = pd.DataFrame(benchmark["results"]).sort_values("cv_auc")
        best_other = bench[~bench["aegisquant"]]["cv_auc"].max()
        fig = go.Figure()
        fig.add_scatter(
            x=bench["cv_auc"], y=bench["model"], mode="markers",
            error_x=dict(type="data", array=bench["cv_auc_std"], color=AXIS, thickness=1.5, width=0),
            marker=dict(size=13, color=[ORANGE if a else BLUE for a in bench["aegisquant"]],
                        line=dict(color="white", width=2)),
            customdata=np.stack([bench["cv_auc_std"], bench["cv_pr_auc"]], axis=1),
            hovertemplate="%{y}<br>CV AUC %{x:.4f} ± %{customdata[0]:.4f}<br>PR-AUC %{customdata[1]:.4f}<extra></extra>",
        )
        fig.update_xaxes(title="5-fold cross-validated ROC-AUC (± one standard deviation across folds)")
        chart(fig, "Eight approaches, same training data, same folds", 380,
              "Every gradient-boosted model lands at ≈ 0.86 — the information ceiling of these ten inputs. "
              "AegisQuant (orange) is at that ceiling with the best PR-AUC and the most stable folds, and keeps exact "
              "SHAP explanations and pickle-free JSON serving. Reproduce with `python benchmark.py`.")
    st.markdown(
        f"Cross-validated AUC **{meta['cv_auc']:.3f}** vs one-time hold-out AUC **{test['roc_auc']:.3f}** — the "
        f"near-identical numbers show no overfitting. Logistic regression with the same engineered features reaches "
        f"{baseline['roc_auc']:.3f}; feature engineering carries most of the signal, gradient boosting adds the interactions."
    )

    c1, c2 = st.columns(2)
    names = {"xgb": ("AegisQuant", BLUE, test), "baseline": ("Logistic regression", ORANGE, baseline)}
    with c1:
        fig = go.Figure()
        for key, (name, color, metrics) in names.items():
            curve = evaluation["roc"][key]
            fig.add_scatter(x=curve["x"], y=curve["y"], mode="lines", name=f"{name} · AUC {metrics['roc_auc']:.3f}",
                            line=dict(color=color, width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=MUTED, dash="dot", width=1))
        fig.update_xaxes(title="False-positive rate", range=[0, 1])
        fig.update_yaxes(title="True-positive rate", range=[0, 1.02])
        chart(fig, "ROC curve — hold-out")
    with c2:
        fig = go.Figure()
        for key, (name, color, metrics) in names.items():
            curve = evaluation["pr"][key]
            fig.add_scatter(x=curve["x"], y=curve["y"], mode="lines", name=f"{name} · PR-AUC {metrics['pr_auc']:.3f}",
                            line=dict(color=color, width=2))
        fig.add_hline(y=base_rate, line=dict(color=MUTED, dash="dot", width=1),
                      annotation_text="random = churn rate", annotation_position="bottom right")
        fig.update_xaxes(title="Recall", range=[0, 1])
        fig.update_yaxes(title="Precision", range=[0, 1.02])
        chart(fig, "Precision–recall curve — hold-out")

    c3, c4, c5 = st.columns([0.37, 0.37, 0.26], gap="medium")
    with c3:
        cal = pd.DataFrame(evaluation["calibration"])
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(color=MUTED, dash="dot", width=1))
        fig.add_scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers", name="AegisQuant",
                        line=dict(color=BLUE, width=2), marker=dict(size=9, line=dict(color="white", width=2)),
                        customdata=cal["customers"],
                        hovertemplate="Predicted %{x:.0%} → actual %{y:.0%}<br>%{customdata} customers<extra></extra>")
        fig.update_xaxes(title="Predicted churn probability", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="Actual churn rate", tickformat=".0%", range=[0, 1.02])
        chart(fig, "Calibration — a 40 % score means 40 % leave", 340)
    with c4:
        tc = evaluation["threshold_curve"]
        fig = go.Figure()
        fig.add_scatter(x=tc["threshold"], y=tc["recall"], mode="lines", name="Churners caught", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=tc["threshold"], y=tc["precision"], mode="lines", name="Offer hit-rate", line=dict(color=ORANGE, width=2))
        fig.add_vline(x=threshold, line=dict(color=AXIS, dash="dot"), annotation_text=f"chosen {threshold:.2f}",
                      annotation_position="top right")
        fig.update_xaxes(title="Decision threshold", range=[0, 1])
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
        chart(fig, "Threshold trade-off", 340)
    with c5:
        cm = test["confusion"]
        # Colour encodes right (blue tint) vs wrong (orange tint), not the count.
        text = [[f"<b>{cm['tp']:,}</b><br>caught", f"<b>{cm['fn']:,}</b><br>missed"],
                [f"<b>{cm['fp']:,}</b><br>false alarm", f"<b>{cm['tn']:,}</b><br>left alone"]]
        fig = go.Figure(go.Heatmap(
            z=[[1, 0], [0, 1]], x=["Flagged", "Not flagged"], y=["Churned", "Stayed"],
            colorscale=[[0, "#fde3d6"], [1, "#d9e7f9"]], showscale=False, text=text, texttemplate="%{text}",
            textfont=dict(size=15, color=INK), hoverinfo="skip", xgap=4, ygap=4,
        ))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False, side="top")
        chart(fig, f"Errors at {threshold:.2f}", 340,
              f"{cm['tp']} of {cm['tp'] + cm['fn']} churners flagged; {cm['fp']} offers go to customers who would stay.")

# ═════════════════════════════════════════════════════════════════════════════
# Segments & fairness
# ═════════════════════════════════════════════════════════════════════════════
with tab_segments:
    seg = pd.DataFrame(evaluation["segments"])
    st.subheader("Life-stage segments")
    st.markdown("KMeans groups customers by age, balance and salary; each segment maps to an investor profile by "
                "**risk capacity** (younger, larger balance → more capacity). The mapping is derived from the "
                "centroids, so it survives retraining.")
    c1, c2 = st.columns([0.6, 0.4], gap="large")
    with c1:
        table = as_percent(seg.drop(columns=["cluster_id", "avg_salary"]), ["share", "churn_rate", "zero_balance_share"])
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "profile": "Investor profile",
            "customers": st.column_config.NumberColumn("Customers", format="%,d"),
            "share": st.column_config.ProgressColumn("Share", format="%.0f%%", min_value=0, max_value=100, width="small"),
            "churn_rate": st.column_config.NumberColumn("Churn rate", format=PERCENT),
            "avg_age": st.column_config.NumberColumn("Avg age", format="%.1f"),
            "avg_balance": st.column_config.NumberColumn("Avg balance", format="€%,.0f"),
            "zero_balance_share": st.column_config.NumberColumn("Zero balance", format="%.0f%%"),
        })
        worst = seg.loc[seg["churn_rate"].idxmax()]
        st.info(f"The **{worst['profile']}** segment (average age {worst['avg_age']:.0f}) churns at "
                f"**{pct(worst['churn_rate'])}** — {worst['churn_rate'] / base_rate:.1f}× the bank average — and is "
                f"{pct(worst['share'])} of customers: the first place to spend retention budget.")
    with c2:
        fig = go.Figure(go.Bar(
            x=seg["profile"].str.title(), y=seg["churn_rate"],
            marker=dict(color=[PROFILE_COLOR[p] for p in seg["profile"]], cornerradius=4),
            text=[pct(v) for v in seg["churn_rate"]], textposition="inside", insidetextanchor="end",
            textfont=dict(color="white"), hovertemplate="%{x}: %{y:.1%} churn<extra></extra>"))
        fig.add_hline(y=base_rate, line=dict(color=MUTED, dash="dot", width=1),
                      annotation_text=f"bank average {pct(base_rate)}", annotation_position="top right")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickformat=".0%", range=[0, seg["churn_rate"].max() * 1.25])
        chart(fig, "Churn rate by segment", 300)

    st.divider()
    st.subheader("Fairness audit — hold-out set")
    st.markdown("Gender is **excluded** from the model (removing it cost no accuracy in cross-validation). The audit "
                "checks *equal opportunity* — are churners caught equally often in every group? — and whether "
                "predicted risk tracks actual churn.")
    share_cols = ["actual_churn_rate", "mean_predicted", "flagged_share", "recall"]
    fair_cfg = {
        "group": "Group",
        "customers": st.column_config.NumberColumn("Customers", format="%,d"),
        "actual_churn_rate": st.column_config.NumberColumn("Actual churn", format=PERCENT),
        "mean_predicted": st.column_config.NumberColumn("Mean predicted", format=PERCENT),
        "flagged_share": st.column_config.NumberColumn("Flagged", format=PERCENT),
        "recall": st.column_config.NumberColumn("Churners caught", format=PERCENT),
        "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
    }
    c1, c2 = st.columns(2)
    for col, (title, key) in zip((c1, c2), (("By gender", "gender"), ("By market", "geography"))):
        with col:
            st.markdown(f"**{title}**")
            table = as_percent(pd.DataFrame(evaluation["fairness"][key]), share_cols).drop(columns=["flagged_share"])
            st.dataframe(table, hide_index=True, width="stretch", column_config=fair_cfg)
    g = {r["group"]: r for r in evaluation["fairness"]["gender"]}
    st.caption(f"Churners caught: {pct(g['Female']['recall'])} of women, {pct(g['Male']['recall'])} of men. Without gender as "
               f"an input the model under-predicts women's churn on average ({pct(g['Female']['mean_predicted'])} predicted vs "
               f"{pct(g['Female']['actual_churn_rate'])} actual). Germany's higher recall reflects its genuinely higher churn.")

# ═════════════════════════════════════════════════════════════════════════════
# Monitoring
# ═════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.subheader("Data drift monitor")
    st.markdown(f"Each feature is compared with the training data by a two-sample Kolmogorov–Smirnov test "
                f"(p < {DRIFT_P_VALUE_THRESHOLD}). When {MAX_DRIFTED_FEATURES_BEFORE_RETRAIN} or more features drift, "
                f"`monitor_and_retrain.py` retrains the model and refreshes the reference.")
    monitor = DriftMonitor(reference_data=pd.read_csv(REFERENCE_DATA_PATH))
    d1, d2 = st.columns(2, gap="large")
    for col, (title, current) in zip((d1, d2), (("Hold-out customers — no drift expected", holdout),
                                                ("Simulated shift — customers 6 years older, balances −25 %",
                                                 simulate_shifted_batch(holdout)))):
        with col:
            results = monitor.compute_drift(current)
            retrain = monitor.should_retrain(results)
            st.markdown(f"**{title}**")
            st.dataframe(results_frame(results).drop(columns=["stat_test"]), hide_index=True, width="stretch", column_config={
                "feature_name": "Feature",
                "drift_detected": st.column_config.CheckboxColumn("Drift"),
                "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
                "ks_statistic": st.column_config.NumberColumn("KS statistic", format="%.3f"),
            })
            (st.error if retrain else st.success)("🔁 Retraining triggered" if retrain else "✅ No retraining needed")
