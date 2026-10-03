"""
AegisQuant — AI Portfolio Shield & Risk Engine (v4.0)
Streamlit dashboard on the Kaggle "Churn Modelling" bank dataset.

Tabs
────
  Customer Risk & Retention — churn probability, local SHAP drivers, investor
                              profile and the churn-adjusted Markowitz portfolio
  Model Performance         — hold-out ROC / PR vs a logistic baseline,
                              threshold trade-off, calibration, SHAP importance
  Segments                  — KMeans life-stage segments → investor profiles
  Fairness & Drift          — per-group audit and the KS drift monitor

Run:  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ASSET_CLASS, ASSETS, DEMO_CLIENTS, HOLDOUT_PATH, REFERENCE_DATA_PATH, ClientFeatures
from data_pipeline import load_churn_dataset
from DriftMonitor import (DRIFT_P_VALUE_THRESHOLD, MAX_DRIFTED_FEATURES_BEFORE_RETRAIN, DriftMonitor,
                          results_frame, simulate_shifted_batch)
from feature_cross_pollination import load_evaluation
from optimizer import AegisQuantEngine

st.set_page_config(page_title="AegisQuant v4.0", page_icon="🛡️", layout="wide")

# ── Visual system ─────────────────────────────────────────────────────────────
# Categorical hues in fixed order (validated for colour-vision deficiency on white).
BLUE, AQUA, ORANGE, VIOLET = "#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"
PROFILE_COLOR = {"conservative": BLUE, "balanced": AQUA, "aggressive": ORANGE}
CLASS_COLOR = {"Core": BLUE, "Tech equity": AQUA, "Crypto": VIOLET}
SERIES = {"XGBoost (AegisQuant)": BLUE, "Logistic baseline": ORANGE}
RAISES, LOWERS = ORANGE, BLUE
# Risk tier = state → reserved status colours, always paired with an icon and a label.
TIER_COLOR = {"High": "#d03b3b", "Elevated": "#fab219", "Low": "#0ca30c"}
TIER_ICON = {"High": "🔴", "Elevated": "🟠", "Low": "🟢"}
GRID, AXIS, MUTED = "#e1e0d9", "#c3c2b7", "#8a8a80"


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=48, b=8), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


def chart(fig: go.Figure, title: str, height: int = 380) -> None:
    """Title as text above the plot, so it never collides with a top legend."""
    st.markdown(f"**{title}**")
    st.plotly_chart(style(fig, height), width="stretch")


def as_percent(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Shares → 0–100 so tables use a locale-independent "%.1f%%" format."""
    return df.assign(**{c: df[c] * 100 for c in cols if c in df})


PERCENT = "%.1f%%"


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AegisQuant engine…")
def get_engine() -> AegisQuantEngine:
    # Trains on first run; market data loads in the background. AEGIS_OFFLINE=1 skips
    # yfinance entirely (tests, demos without internet) and uses the labelled fallback.
    return AegisQuantEngine(start_market_worker=os.getenv("AEGIS_OFFLINE") != "1")


@st.cache_data(show_spinner=False)
def get_evaluation() -> dict:
    return load_evaluation()


@st.cache_data(show_spinner=False)
def get_holdout() -> pd.DataFrame:
    customers = load_churn_dataset()
    holdout = pd.read_csv(HOLDOUT_PATH)
    return holdout.merge(customers.drop(columns=["churn"]), on="customer_id").sort_values("probability", ascending=False)


@st.cache_data(show_spinner=False)
def get_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DATA_PATH)


engine = get_engine()
evaluation = get_evaluation()
holdout = get_holdout()
meta = engine.predictor.meta
threshold = engine.threshold
test = evaluation["test"]
baseline = evaluation["baseline_logistic"]

# ── Sidebar: model card + customer selection ──────────────────────────────────
with st.sidebar:
    st.title("🛡️ AegisQuant v4.0")
    st.markdown(
        f"""
- **Data** Kaggle *Churn Modelling* — 10,000 bank customers, {pct(evaluation['base_churn_rate'])} churned
- **Model** KMeans segments → XGBoost, local SHAP
- **Trained** {meta.get('trained_at', '')[:10]} on {meta.get('n_train', 0):,} customers
- **Evaluated** once on {meta.get('n_test', 0):,} unseen customers
- **Decision threshold** {threshold:.2f} (F1-optimal on cross-validation)
"""
    )
    st.caption("Gender is not a model input; it is used only for the fairness audit.")
    st.divider()

    st.markdown("### Customer")
    mode = st.radio("Source", ["Hold-out customer (real data)", "Demo customer", "Custom customer"],
                    label_visibility="collapsed")
    actual = None
    if mode.startswith("Hold-out"):
        only_flagged = st.toggle("Only customers above the threshold", value=True)
        pool = holdout[holdout["probability"] >= threshold] if only_flagged else holdout
        labels = {
            f"#{r.customer_id} · {r.probability:.0%} risk · {r.geography}, {r.age}": r
            for r in pool.itertuples()
        }
        choice = labels[st.selectbox(f"{len(labels):,} customers, riskiest first", list(labels))]
        actual = "churned" if choice.churn == 1 else "stayed"
        payload = ClientFeatures(
            customer_id=int(choice.customer_id), credit_score=int(choice.credit_score), geography=choice.geography,
            age=int(choice.age), tenure=int(choice.tenure), balance=float(choice.balance),
            num_products=int(choice.num_products), has_cr_card=bool(choice.has_cr_card),
            is_active_member=bool(choice.is_active_member), estimated_salary=float(choice.estimated_salary),
        )
    elif mode == "Demo customer":
        demo = {c["description"]: c for c in DEMO_CLIENTS}
        payload = ClientFeatures(**demo[st.selectbox("Persona", list(demo))])
    else:
        with st.form("custom"):
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 18, 100, 45)
            tenure = c2.number_input("Tenure (years)", 0, 60, 3)
            credit = c1.number_input("Credit score", 300, 900, 650)
            products = c2.selectbox("Products", [1, 2, 3, 4], index=0)
            balance = c1.number_input("Balance (€)", 0, 1_000_000, 90_000, step=5_000)
            salary = c2.number_input("Salary (€)", 1_000, 1_000_000, 80_000, step=5_000)
            geo = st.selectbox("Market", ["France", "Germany", "Spain"])
            active = st.toggle("Active member", value=False)
            card = st.toggle("Has credit card", value=True)
            st.form_submit_button("Score customer", type="primary")
        payload = ClientFeatures(credit_score=credit, geography=geo, age=age, tenure=tenure, balance=balance,
                                 num_products=products, has_cr_card=card, is_active_member=active,
                                 estimated_salary=salary)

# ── Header & KPI strip ────────────────────────────────────────────────────────
st.title("AegisQuant — Churn Risk & Retention Portfolio Engine")
st.caption("Real bank-customer data · metrics measured on a hold-out set the model never saw during training")

k = st.columns(5)
k[0].metric("ROC-AUC (hold-out)", f"{test['roc_auc']:.3f}", f"{test['roc_auc'] - baseline['roc_auc']:+.3f} vs logistic",
            border=True, help="Probability that a random churner is ranked above a random stayer.")
k[1].metric("PR-AUC", f"{test['pr_auc']:.3f}", f"{test['pr_auc'] - baseline['pr_auc']:+.3f} vs logistic", border=True,
            help=f"Precision–recall area; a random model scores the churn rate ({pct(evaluation['base_churn_rate'])}).")
k[2].metric("Churners caught", pct(test["recall"], 0), border=True,
            help=f"Recall at the {threshold:.2f} threshold.")
k[3].metric("Precision", pct(test["precision"], 0), border=True,
            help="Share of flagged customers who actually churned.")
k[4].metric("Top-20 % capture", pct(test["capture_top20"], 0), border=True,
            help="Share of all churners found by contacting the riskiest 20 % of customers.")

tab_risk, tab_model, tab_seg, tab_fair = st.tabs(
    ["Customer Risk & Retention", "Model Performance", "Segments", "Fairness & Drift"]
)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 1 — customer risk & retention portfolio
# ═════════════════════════════════════════════════════════════════════════════
with tab_risk:
    result = engine.explain_client(payload)
    prob, tier, profile = result["churn_probability"], result["risk_tier"], result["investor_profile"]

    banner = f"{TIER_ICON[tier]} **{tier} churn risk — {prob:.1%}**"
    if result["retention_action"]:
        banner += f" · above the {threshold:.2f} threshold → retention offer recommended"
    else:
        banner += f" · below the {threshold:.2f} threshold → no action"
    {"High": st.error, "Elevated": st.warning, "Low": st.success}[tier](banner)

    m = st.columns(4)
    m[0].metric("Churn probability", pct(prob), border=True)
    m[1].metric("Investor profile", profile.title(), border=True,
                help="KMeans life-stage segment (age, balance, salary) mapped by risk capacity.")
    m[2].metric("Products held", payload.num_products, border=True)
    m[3].metric("Actual outcome" if actual else "Market", actual.title() if actual else payload.geography, border=True,
                help="Known for hold-out customers only — the model never saw it." if actual else None)

    left, right = st.columns([0.45, 0.55], gap="large")
    with left:
        st.subheader("Why this score")
        drivers = pd.DataFrame(result["drivers"]).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=drivers["contribution"], y=drivers["label"], orientation="h",
            marker=dict(color=[RAISES if c > 0 else LOWERS for c in drivers["contribution"]], cornerradius=4),
            customdata=drivers["text"], hovertemplate="%{customdata}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=AXIS)
        fig.update_layout(showlegend=False)
        chart(fig, "Local SHAP contribution (log-odds)", 300)
        st.caption(f"Orange pushes churn up, blue pulls it down. Starts from the average log-odds ({result['base_value']:+.2f}).")
        for d in result["drivers"]:
            st.markdown(f"- {d['text']}")

    with right:
        st.subheader("Retention portfolio offer")
        if not result["retention_action"]:
            st.info("Churn risk is below the decision threshold — no retention offer is generated for this customer.")
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
                                text=[pct(w, 0) for w in part["weight"]], textposition="outside",
                                hovertemplate="%{y}: %{x:.1%}<extra>" + cls + "</extra>")
            fig.update_layout(barmode="relative")
            fig.update_xaxes(tickformat=".0%", range=[0, alloc["weight"].max() * 1.25])
            fig.update_yaxes(categoryorder="array", categoryarray=alloc["asset"].tolist())
            chart(fig, f"Churn-adjusted Markowitz allocation — {profile} profile", 360)

            s = st.columns(3)
            s[0].metric("Expected return (1y hist.)", pct(stats["expected_return"]), border=True)
            s[1].metric("Volatility", pct(stats["volatility"]), border=True)
            s[2].metric("Risk aversion γ", f"{gamma:.2f}", border=True,
                        help="Base γ of the profile, raised as churn probability exceeds the threshold.")
            source = engine.market_source
            st.caption(f"Market data: {source}. Expected returns are trailing one-year averages, not forecasts.")
            if "synthetic" in source:
                st.warning("Live market data is unavailable — the allocation uses a synthetic covariance and is illustrative only.")
            if payload.balance > 0:
                table = alloc.sort_values("weight", ascending=False).assign(amount=lambda d: d["weight"] * payload.balance)
                st.dataframe(as_percent(table, ["weight"]), hide_index=True, width="stretch", column_config={
                    "asset": "Asset", "class": "Class",
                    "weight": st.column_config.NumberColumn("Weight", format=PERCENT),
                    "amount": st.column_config.NumberColumn("Amount of balance (€)", format="€%,.0f"),
                })

# ═════════════════════════════════════════════════════════════════════════════
# Tab 2 — model performance
# ═════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown(
        f"Trained on {meta['n_train']:,} customers, cross-validated AUC **{meta['cv_auc']:.3f}**; "
        f"measured once on {meta['n_test']:,} unseen customers: AUC **{test['roc_auc']:.3f}**. "
        f"The near-identical numbers show the model is not overfitting. A logistic regression on the same "
        f"features reaches {baseline['roc_auc']:.3f} — the gain comes from non-linear effects "
        f"(churn peaks in the 50s and explodes at 3–4 products)."
    )
    c1, c2 = st.columns(2)
    names = {"xgb": "XGBoost (AegisQuant)", "baseline": "Logistic baseline"}
    with c1:
        fig = go.Figure()
        for key, name in names.items():
            curve = evaluation["roc"][key]
            auc = test["roc_auc"] if key == "xgb" else baseline["roc_auc"]
            fig.add_scatter(x=curve["x"], y=curve["y"], mode="lines", name=f"{name} — AUC {auc:.3f}",
                            line=dict(color=SERIES[name], width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=MUTED, dash="dot", width=1))
        fig.update_xaxes(title="False-positive rate", range=[0, 1])
        fig.update_yaxes(title="True-positive rate (recall)", range=[0, 1.02])
        chart(fig, "ROC curve (hold-out)")
    with c2:
        fig = go.Figure()
        for key, name in names.items():
            curve = evaluation["pr"][key]
            ap = test["pr_auc"] if key == "xgb" else baseline["pr_auc"]
            fig.add_scatter(x=curve["x"], y=curve["y"], mode="lines", name=f"{name} — PR-AUC {ap:.3f}",
                            line=dict(color=SERIES[name], width=2))
        fig.add_hline(y=evaluation["base_churn_rate"], line=dict(color=MUTED, dash="dot", width=1),
                      annotation_text="Random = churn rate", annotation_position="bottom right")
        fig.update_xaxes(title="Recall", range=[0, 1])
        fig.update_yaxes(title="Precision", range=[0, 1.02])
        chart(fig, "Precision–recall curve (hold-out)")

    c3, c4 = st.columns(2)
    with c3:
        tc = evaluation["threshold_curve"]
        fig = go.Figure()
        fig.add_scatter(x=tc["threshold"], y=tc["recall"], mode="lines", name="Recall (churners caught)", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=tc["threshold"], y=tc["precision"], mode="lines", name="Precision (offers well spent)", line=dict(color=ORANGE, width=2))
        fig.add_vline(x=threshold, line=dict(color=AXIS, dash="dot"), annotation_text=f"chosen {threshold:.2f}",
                      annotation_position="top right")
        fig.update_xaxes(title="Decision threshold", range=[0, 1])
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
        chart(fig, "Threshold trade-off")
    with c4:
        cal = pd.DataFrame(evaluation["calibration"])
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color=MUTED, dash="dot", width=1))
        fig.add_scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers", name="AegisQuant",
                        line=dict(color=BLUE, width=2), marker=dict(size=9, line=dict(color="white", width=2)),
                        customdata=cal["customers"], hovertemplate="Predicted %{x:.0%} → actual %{y:.0%}<br>%{customdata} customers<extra></extra>")
        fig.update_xaxes(title="Predicted probability (decile bin)", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="Actual churn rate", tickformat=".0%", range=[0, 1.02])
        chart(fig, "Calibration: predicted vs actual churn")

    c5, c6 = st.columns([0.6, 0.4])
    with c5:
        imp = pd.DataFrame(evaluation["shap_importance"]).head(10).iloc[::-1]
        imp["label"] = imp["feature"].map(lambda f: f.replace("_", " "))
        fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["label"], orientation="h",
                               marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        chart(fig, "Global importance — mean |SHAP| on the hold-out set", 360)
    with c6:
        cm = test["confusion"]
        st.markdown(f"**Confusion matrix at threshold {threshold:.2f}** ({meta['n_test']:,} hold-out customers)")
        st.dataframe(pd.DataFrame(
            {"Predicted: stays": [cm["tn"], cm["fn"]], "Predicted: churns": [cm["fp"], cm["tp"]]},
            index=["Actually stayed", "Actually churned"]), width="stretch")
        st.markdown(
            f"- {cm['tp']} of {cm['tp'] + cm['fn']} churners are caught ({pct(test['recall'], 0)})\n"
            f"- {pct(test['precision'], 0)} of retention offers reach a real churner\n"
            f"- Brier score {test['brier']:.3f} (lower is better; always predicting the churn rate scores "
            f"{evaluation['base_churn_rate'] * (1 - evaluation['base_churn_rate']):.3f})"
        )

# ═════════════════════════════════════════════════════════════════════════════
# Tab 3 — segments
# ═════════════════════════════════════════════════════════════════════════════
with tab_seg:
    seg = pd.DataFrame(evaluation["segments"])
    st.markdown(
        "KMeans groups customers by life stage (age, balance, salary). Each segment is mapped to an investor "
        "profile by **risk capacity** — younger customers with a larger balance can carry more risk. The "
        "mapping is computed from the centroids, so it survives retraining."
    )
    c1, c2 = st.columns([0.62, 0.38])
    with c1:
        # Salary is ~€100K in every segment (uniform in this dataset), so it is not shown.
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
    with c2:
        fig = go.Figure(go.Bar(
            x=seg["profile"].str.title(), y=seg["churn_rate"],
            marker=dict(color=[PROFILE_COLOR[p] for p in seg["profile"]], cornerradius=4),
            text=[pct(v, 0) for v in seg["churn_rate"]], textposition="inside", insidetextanchor="end",
            textfont=dict(color="white"), hovertemplate="%{x}: %{y:.1%} churn<extra></extra>"))
        fig.add_hline(y=evaluation["base_churn_rate"], line=dict(color=MUTED, dash="dot", width=1),
                      annotation_text=f"bank average {pct(evaluation['base_churn_rate'], 0)}",
                      annotation_position="top right")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickformat=".0%", range=[0, seg["churn_rate"].max() * 1.25])
        chart(fig, "Churn rate by segment", 340)
    worst = seg.loc[seg["churn_rate"].idxmax()]
    st.info(f"The **{worst['profile']}** segment (average age {worst['avg_age']:.0f}) churns at "
            f"**{pct(worst['churn_rate'], 0)}** — {worst['churn_rate'] / evaluation['base_churn_rate']:.1f}× the "
            f"bank average. It is {pct(worst['share'], 0)} of customers: the first place to spend retention budget.")

# ═════════════════════════════════════════════════════════════════════════════
# Tab 4 — fairness & drift
# ═════════════════════════════════════════════════════════════════════════════
with tab_fair:
    st.subheader("Fairness audit (hold-out set)")
    st.markdown(
        "Gender is **excluded** from the model. The audit checks whether churners are caught equally often in "
        "every group (*equal opportunity*: similar recall) and whether predicted risk tracks actual churn."
    )
    fair_cfg = {
        "group": "Group",
        "customers": st.column_config.NumberColumn("Customers", format="%,d"),
        "actual_churn_rate": st.column_config.NumberColumn("Actual churn", format=PERCENT),
        "mean_predicted": st.column_config.NumberColumn("Mean predicted", format=PERCENT),
        "flagged_share": st.column_config.NumberColumn("Flagged", format=PERCENT),
        "recall": st.column_config.NumberColumn("Recall", format=PERCENT, help="Share of actual churners the model flags"),
        "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
    }
    share_cols = ["actual_churn_rate", "mean_predicted", "flagged_share", "recall"]
    c1, c2 = st.columns(2)
    for col, (title, key) in zip((c1, c2), (("By gender", "gender"), ("By geography", "geography"))):
        with col:
            st.markdown(f"**{title}**")
            st.dataframe(as_percent(pd.DataFrame(evaluation["fairness"][key]), share_cols),
                         hide_index=True, width="stretch", column_config=fair_cfg)
    g = {r["group"]: r for r in evaluation["fairness"]["gender"]}
    st.caption(
        f"Recall is {pct(g['Female']['recall'], 0)} for women and {pct(g['Male']['recall'], 0)} for men. Because gender is "
        f"not an input, the model slightly under-predicts women's churn on average "
        f"({pct(g['Female']['mean_predicted'], 0)} predicted vs {pct(g['Female']['actual_churn_rate'], 0)} actual); "
        f"Germany's higher recall reflects its genuinely higher churn rate."
    )

    st.divider()
    st.subheader("Data drift monitor")
    st.markdown(
        f"Two-sample Kolmogorov–Smirnov test of each feature against the training data (p < {DRIFT_P_VALUE_THRESHOLD}). "
        f"Retraining triggers when {MAX_DRIFTED_FEATURES_BEFORE_RETRAIN}+ features drift — scheduled by `monitor_and_retrain.py`."
    )
    monitor = DriftMonitor(reference_data=get_reference())
    batch = holdout.copy()
    d1, d2 = st.columns(2)
    for col, (title, current) in zip((d1, d2), (("Hold-out customers — no drift expected", batch),
                                                ("Simulated shift — customers 6 years older, balances −25 %",
                                                 simulate_shifted_batch(batch)))):
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
            (st.error if retrain else st.success)("Retraining triggered" if retrain else "No retraining needed")
