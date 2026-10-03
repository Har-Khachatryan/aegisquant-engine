"""
CoinStats Portfolio Intelligence — Executive Dashboard.

    streamlit run app.py

Sections
────────
  KPI strip                      book-level health at a glance (always visible)
  1. Executive Overview          segment economics + where the risk sits
  2. Cluster Explorer            Total Value × HHI, filterable by archetype / risk tier
  3. Portfolio Health Inspector  pick any of the 500 portfolios → diagnosis + push preview
  4. Monetization & In-App Swaps campaign sizing with adjustable fee / adoption assumptions

All numbers come from the same modules as the API (features → segmentation →
insights), so the dashboard and /analyze-portfolio can never disagree.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import settings as cfg
from features import load_holdings, prepare_holdings
from insights import PLAYBOOK, campaign_plan, diagnose, swap_levers
from pipeline import cluster_summary, load_artifacts

st.set_page_config(page_title="CoinStats Portfolio Intelligence", page_icon="📊", layout="wide")

# ── Visual encoding ───────────────────────────────────────────────────────────
# Archetype = identity → fixed categorical palette, validated all-pairs for a
# 4-series scatter (CVD ΔE ≥ 9.2, normal-vision ΔE ≥ 16.3). Colour follows the
# entity on every chart; marker symbols are a second, colour-free channel.
ARCH_COLOR = {0: "#2a78d6", 1: "#1baf7a", 2: "#eb6834", 3: "#4a3aa7"}
ARCH_SYMBOL = {0: "circle", 1: "square", 2: "diamond", 3: "triangle-up"}
ARCH_NAME = {k: a.name for k, a in cfg.ARCHETYPES.items()}
# Risk tier = state → reserved status colours, always shown with a text label.
TIER_COLOR = {"High": "#d03b3b", "Elevated": "#fab219", "Low": "#0ca30c"}
TIER_ORDER = ["High", "Elevated", "Low"]
TIER_ICON = {"High": "🔴", "Elevated": "🟠", "Low": "🟢"}
SINGLE_SERIES = "#2a78d6"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"


def usd(x: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"${x / div:,.1f}{suffix}"
    return f"${x:,.0f}"


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    """Recessive chrome: hairline solid grid, no plot border, generous padding."""
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=40, b=8), bargap=0.35,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


# ── Data (cached) ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building portfolio intelligence…")
def get_model_and_features():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_holdings() -> pd.DataFrame:
    return prepare_holdings(load_holdings())


@st.cache_data(show_spinner=False)
def get_levers() -> pd.DataFrame:
    return swap_levers(get_holdings())


model, seg = get_model_and_features()
holdings = get_holdings()
levers = get_levers()
summary = cluster_summary(seg)
book = summary["book"]

# ── Sidebar: model card & data caveats ───────────────────────────────────────
with st.sidebar:
    st.header("Model card")
    st.markdown(
        f"""
- **Version** `{model.model_version}`
- **Trained** {model.trained_at[:10]} on **{model.n_samples}** portfolios
- **Segmentation** KMeans, k = {cfg.N_CLUSTERS}, silhouette **{model.silhouette:.3f}**
- **Score** rules-based capitulation proxy (0–100)
"""
    )
    st.caption(
        "Silhouette ≈ 0.24 means segments overlap at the edges — expected for "
        "behavioural data. Archetypes are labelled by centroid signature, not by hand."
    )
    st.divider()
    st.subheader("Data notes")
    st.caption(
        f"• AUM is approximate: the anonymised dataset scales each portfolio by a random 0.5–2× factor. "
        f"Weights, HHI and PnL % are exact.\n\n"
        f"• {int(seg['flagged_cost_basis_rows'].sum())} holdings with implausible cost basis "
        f"(e.g. total cost typed as unit price) are excluded from PnL; "
        f"{book['portfolios_with_unreliable_pnl']} portfolios have too little trusted basis for a PnL figure.\n\n"
        "• Academic sample — not for redistribution. Deploy privately."
    )

# ── Header & KPI strip ────────────────────────────────────────────────────────
st.title("CoinStats Portfolio Intelligence")
st.caption("Retention & monetisation analytics on 500 manually tracked CoinStats portfolios (Sept 2026 snapshot)")

high = seg[seg["risk_tier"] == "High"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total AUM (approx.)", usd(book["total_aum_usd"]), border=True,
          help="Sum of portfolio values. Approximate — see data notes.")
k2.metric("Average HHI", f"{book['avg_hhi_index']:.2f}", border=True,
          help=f"Herfindahl concentration. ≈ {1 / book['avg_hhi_index']:.1f} effective coins per portfolio; "
               "above 0.50 = concentrated.")
k3.metric("High-risk portfolios", f"{book['high_risk_portfolios']}", border=True,
          help=f"{book['high_risk_portfolios'] / book['portfolios']:.0%} of the book; score ≥ {cfg.HIGH_RISK_THRESHOLD:.0f}. "
               f"{book['triple_threat_portfolios']} meet all three hard rules (PnL < −20%, HHI > 0.5, >30% speculative).")
k4.metric("Median unrealised PnL", f"{book['median_unrealized_pnl_pct']:+.1f}%", border=True,
          help=f"Median across portfolios with reliable cost basis. Cost-weighted book PnL: "
               f"{book['book_unrealized_pnl_pct']:+.1f}%. The mean ({book['mean_unrealized_pnl_pct']:+.0f}%) "
               "is distorted by a few 10–60× winners.")
k5.metric("AUM in high-risk portfolios", usd(high["total_portfolio_value"].sum()), border=True,
          help="Value held by High-tier portfolios — the balance most exposed to capitulation.")

tab_overview, tab_explorer, tab_inspector, tab_money = st.tabs(
    ["Executive Overview", "Cluster Explorer", "Portfolio Health Inspector", "Monetization & In-App Swaps"]
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. Executive overview
# ═════════════════════════════════════════════════════════════════════════════
with tab_overview:
    segs = pd.DataFrame(summary["segments"])
    whale = segs.set_index("archetype_id").loc[0]
    degen = segs.set_index("archetype_id").loc[2]
    explorer = segs.set_index("archetype_id").loc[1]
    st.subheader("What the 500 portfolios tell us")
    st.markdown(
        f"""
- **Whales are the revenue base.** {whale['share_of_portfolios']:.0%} of portfolios hold
  **{whale['share_of_aum']:.0%} of AUM**, are up a median **{whale['median_unrealized_pnl_pct']:+.0f}%**, and almost
  never score High-risk — the play is engagement (DCA, alerts), not rescue.
- **Degens are where churn lives.** {degen['share_of_portfolios']:.0%} of portfolios but
  **{degen['high_risk_portfolios']} of the {book['high_risk_portfolios']} High-risk** ones;
  {degen['avg_speculative_meme_ratio']:.0%} of their value sits in coins ranked beyond {cfg.SPECULATIVE_MIN_RANK}.
- **Explorers are the quiet risk.** They hold **{explorer['share_of_aum']:.0%} of AUM** across a median of
  {explorer['median_num_assets']:.0f} coins, yet sit at a median **{explorer['median_unrealized_pnl_pct']:+.0f}%** —
  diversified into a drawdown, with lots of dust positions to clean up.
"""
    )

    st.markdown("**Segment economics**")
    table = segs.assign(
        share_of_portfolios=segs["share_of_portfolios"] * 100,
        share_of_aum=segs["share_of_aum"] * 100,
        avg_speculative_meme_ratio=segs["avg_speculative_meme_ratio"] * 100,
    )[["archetype", "portfolios", "share_of_portfolios", "share_of_aum", "median_portfolio_value_usd",
       "median_num_assets", "avg_hhi_index", "avg_speculative_meme_ratio", "median_unrealized_pnl_pct",
       "avg_vulnerability_score", "high_risk_portfolios"]]
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "archetype": st.column_config.TextColumn("Archetype", width="medium"),
            "portfolios": st.column_config.NumberColumn("Portfolios"),
            "share_of_portfolios": st.column_config.ProgressColumn("% of users", format="%.0f%%", min_value=0,
                                                                   max_value=100, width="small"),
            "share_of_aum": st.column_config.ProgressColumn("% of AUM", format="%.0f%%", min_value=0,
                                                            max_value=100, width="small"),
            "median_portfolio_value_usd": st.column_config.NumberColumn("Median value", format="$%,.0f"),
            "median_num_assets": st.column_config.NumberColumn("Median coins", format="%.0f"),
            "avg_hhi_index": st.column_config.NumberColumn("Avg HHI", format="%.2f"),
            "avg_speculative_meme_ratio": st.column_config.NumberColumn("Speculative %", format="%.0f%%"),
            "median_unrealized_pnl_pct": st.column_config.NumberColumn("Median PnL %", format="%+.1f%%"),
            "avg_vulnerability_score": st.column_config.NumberColumn("Avg risk score", format="%.0f"),
            "high_risk_portfolios": st.column_config.NumberColumn("High-risk"),
        },
    )
    left, right = st.columns(2, gap="large")
    with left:
        counts = seg.pivot_table(index="archetype_id", columns="risk_tier", values="portfolio_id",
                                 aggfunc="count", fill_value=0).reindex(columns=TIER_ORDER, fill_value=0)
        names = [ARCH_NAME[i] for i in counts.index]
        fig = go.Figure([
            go.Bar(y=names, x=counts[tier], name=f"{tier} risk", orientation="h", marker_color=TIER_COLOR[tier],
                   marker_line_color=SURFACE, marker_line_width=2,
                   hovertemplate="%{y}<br>" + tier + " risk: <b>%{x}</b> portfolios<extra></extra>")
            for tier in TIER_ORDER
        ])
        fig = style(fig, 340)
        fig.update_layout(title="Risk tier by archetype", barmode="stack", barcornerradius=4,
                          legend_traceorder="normal")
        fig.update_xaxes(title="Portfolios")
        fig.update_yaxes(autorange="reversed")  # first archetype on top
        st.plotly_chart(fig, width="stretch", theme="streamlit")
        st.caption(f"High = score ≥ {cfg.HIGH_RISK_THRESHOLD:.0f} · Elevated = {cfg.ELEVATED_RISK_THRESHOLD:.0f}–"
                   f"{cfg.HIGH_RISK_THRESHOLD:.0f} · Low = below {cfg.ELEVATED_RISK_THRESHOLD:.0f}")
    with right:
        fig = go.Figure(go.Histogram(
            x=seg["capitulation_vulnerability_score"], xbins=dict(start=0, end=100, size=5),
            marker_color=SINGLE_SERIES, marker_line_color=SURFACE, marker_line_width=2,
            hovertemplate="Score %{x}: <b>%{y}</b> portfolios<extra></extra>",
        ))
        for thr, label in ((cfg.ELEVATED_RISK_THRESHOLD, "Elevated"), (cfg.HIGH_RISK_THRESHOLD, "High")):
            fig.add_vline(x=thr, line_width=1, line_dash="dot", line_color="#898781",
                          annotation_text=label, annotation_position="top right", annotation_font_color="#52514e")
        fig = style(fig, 340)
        fig.update_layout(title="Capitulation vulnerability score — distribution", showlegend=False, bargap=0)
        fig.update_xaxes(title="Score (0–100)", range=[0, 100])
        fig.update_yaxes(title="Portfolios")
        st.plotly_chart(fig, width="stretch", theme="streamlit")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Cluster explorer
# ═════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    f1, f2, f3 = st.columns([0.5, 0.3, 0.2], vertical_alignment="bottom")
    chosen = f1.multiselect("Archetypes", options=sorted(ARCH_NAME), default=sorted(ARCH_NAME),
                            format_func=ARCH_NAME.get)
    tiers = f2.multiselect("Risk tiers", options=TIER_ORDER, default=TIER_ORDER,
                           format_func=lambda t: f"{TIER_ICON[t]} {t}")
    log_x = f3.toggle("Log scale (value)", value=True)

    view = seg[seg["archetype_id"].isin(chosen) & seg["risk_tier"].isin(tiers)].copy()
    st.caption(f"Showing **{len(view)}** of {len(seg)} portfolios · {usd(view['total_portfolio_value'].sum())} AUM")

    if view.empty:
        st.info("No portfolios match the current filters.")
    else:
        view["pnl_label"] = view["unrealized_pnl_pct"].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}%")
        fig = go.Figure()
        for aid in sorted(view["archetype_id"].unique()):
            d = view[view["archetype_id"] == aid]
            fig.add_trace(go.Scatter(
                x=d["total_portfolio_value"], y=d["hhi_index"], mode="markers", name=ARCH_NAME[aid],
                marker=dict(color=ARCH_COLOR[aid], symbol=ARCH_SYMBOL[aid], size=10, opacity=0.85,
                            line=dict(color=SURFACE, width=1.5)),
                customdata=d[["portfolio_id", "num_assets", "pnl_label", "capitulation_vulnerability_score",
                              "risk_tier", "speculative_meme_ratio"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Value: $%{x:,.0f}<br>HHI: %{y:.2f}<br>"
                    "Coins: %{customdata[1]}<br>PnL: %{customdata[2]}<br>"
                    "Speculative: %{customdata[5]:.0%}<br>"
                    "Risk: %{customdata[3]:.0f} (%{customdata[4]})<extra>" + cfg.ARCHETYPES[aid].short + "</extra>"
                ),
            ))
        fig.add_hline(y=cfg.HHI_CENTER, line_width=1, line_dash="dot", line_color="#898781",
                      annotation_text="HHI 0.5 — concentration threshold", annotation_position="top left",
                      annotation_font_color="#52514e")
        fig.update_xaxes(type="log" if log_x else "linear", title="Total portfolio value (USD)")
        fig.update_yaxes(title="HHI (concentration)", range=[0, 1.05])
        fig.update_layout(title="Total value vs. concentration", hovermode="closest")
        st.plotly_chart(style(fig, 520), width="stretch", theme="streamlit")

        with st.expander("Table view of the filtered portfolios"):
            cols = ["portfolio_id", "archetype", "total_portfolio_value", "num_assets", "hhi_index", "top1_weight",
                    "top10_bluechip_ratio", "speculative_meme_ratio", "unrealized_pnl_pct",
                    "capitulation_vulnerability_score", "risk_tier"]
            shares = ["top1_weight", "top10_bluechip_ratio", "speculative_meme_ratio"]
            st.dataframe(
                view[cols].assign(**{c: view[c] * 100 for c in shares})
                .sort_values("capitulation_vulnerability_score", ascending=False),
                hide_index=True, width="stretch",
                column_config={
                    "total_portfolio_value": st.column_config.NumberColumn("Value", format="$%,.0f"),
                    "hhi_index": st.column_config.NumberColumn("HHI", format="%.2f"),
                    "top1_weight": st.column_config.NumberColumn("Top-1", format="%.0f%%"),
                    "top10_bluechip_ratio": st.column_config.NumberColumn("Blue-chip", format="%.0f%%"),
                    "speculative_meme_ratio": st.column_config.NumberColumn("Speculative", format="%.0f%%"),
                    "unrealized_pnl_pct": st.column_config.NumberColumn("PnL %", format="%+.1f%%"),
                    "capitulation_vulnerability_score": st.column_config.NumberColumn("Risk score", format="%.0f"),
                },
            )

# ═════════════════════════════════════════════════════════════════════════════
# 3. Portfolio health inspector
# ═════════════════════════════════════════════════════════════════════════════
with tab_inspector:
    st.markdown("Simulates what the CoinStats app would know the moment a user opens their portfolio.")
    c1, c2 = st.columns([0.7, 0.3], vertical_alignment="bottom")
    only_high = c2.toggle("High-risk only", value=False)
    pool = seg[seg["risk_tier"] == "High"] if only_high else seg
    pool = pool.sort_values("capitulation_vulnerability_score", ascending=False)
    labels = {
        r.portfolio_id: f"{r.portfolio_id} · {cfg.ARCHETYPES[r.archetype_id].short} · "
                        f"{usd(r.total_portfolio_value)} · risk {r.capitulation_vulnerability_score:.0f}"
        for r in pool.itertuples()
    }
    pid = c1.selectbox("Portfolio (sorted by risk, highest first)", options=list(labels), format_func=labels.get)

    row = seg.set_index("portfolio_id").loc[pid]
    row_d = row.to_dict() | {"portfolio_id": pid}
    dx = diagnose(row_d, levers.loc[pid])
    banner = {"High": st.error, "Elevated": st.warning, "Low": st.success}[dx["risk_tier"]]
    banner(f"**{dx['headline']}**", icon={"High": "🚨", "Elevated": "⚠️", "Low": "✅"}[dx["risk_tier"]])

    m = st.columns(6)
    m[0].metric("Archetype", cfg.ARCHETYPES[int(row["archetype_id"])].short, help=ARCH_NAME[int(row["archetype_id"])])
    m[1].metric("Risk score", f"{row['capitulation_vulnerability_score']:.0f} / 100",
                help=f"{TIER_ICON[dx['risk_tier']]} {dx['risk_tier']} tier")
    m[2].metric("Value", usd(row["total_portfolio_value"]))
    m[3].metric("Coins", f"{int(row['num_assets'])}", help=f"≈ {row['effective_num_assets']:.1f} effective coins")
    m[4].metric("HHI", f"{row['hhi_index']:.2f}", help=f"Top holding {row['top1_weight']:.0%} · top-3 {row['top3_weight']:.0%}")
    m[5].metric("Unrealised PnL", "n/a" if pd.isna(row["unrealized_pnl_pct"]) else f"{row['unrealized_pnl_pct']:+.1f}%",
                help=f"Cost basis covers {row['cost_basis_coverage']:.0%} of value")

    left, right = st.columns([0.55, 0.45], gap="large")
    with left:
        st.markdown("**Why this score**")
        for d in dx["drivers"]:
            st.markdown(f"- {d}")
        contrib = pd.DataFrame({
            "component": ["Loss pressure", "Concentration pressure", "Speculation pressure"],
            "points": [100 * cfg.SCORE_WEIGHTS[k] * row[f"{k}_pressure"] for k in ("loss", "concentration", "speculation")],
            "max": [100 * cfg.SCORE_WEIGHTS[k] for k in ("loss", "concentration", "speculation")],
        })
        fig = go.Figure(go.Bar(
            y=contrib["component"], x=contrib["points"], orientation="h", marker_color=SINGLE_SERIES,
            text=[f"{p:.0f} / {mx:.0f}" for p, mx in zip(contrib["points"], contrib["max"])],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: <b>%{x:.1f}</b> points<extra></extra>",
        ))
        fig.update_layout(title="Score composition (points out of each component's maximum)", barcornerradius=4,
                          showlegend=False)
        fig.update_xaxes(range=[0, 50], title="Points")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 260), width="stretch", theme="streamlit")

    with right:
        st.markdown("**Recommended in-app action**")
        with st.container(border=True):
            st.caption(f"📲 PUSH PREVIEW · campaign: {dx['campaign']}")
            st.markdown(f"**CoinStats** — {dx['push_message']}")
            st.caption(dx["recommended_action"])
            if dx["eligible_for_campaign"]:
                st.markdown(f"Swap notional in play: **{usd(dx['swap_notional_usd'])}**")
            else:
                st.markdown("_Below the minimum swap size — send education content instead of a swap CTA._")

    st.markdown("**Holdings**")
    h = holdings[holdings["portfolio_id"] == pid].copy()
    h["pnl_pct"] = (h["price_usd"] / h["avg_buy_price_usd"] - 1).where(h["cost_basis_valid"]) * 100
    h["tier"] = h["rank"].map(lambda r: "Blue-chip" if r <= cfg.BLUECHIP_MAX_RANK
                              else ("Speculative" if r > cfg.SPECULATIVE_MIN_RANK else "Mid-cap"))
    h["basis"] = h.apply(lambda r: "flagged — excluded" if r["cost_basis_flagged"]
                         else ("ok" if r["cost_basis_valid"] else "unknown"), axis=1)
    st.dataframe(
        h[["symbol", "rank", "tier", "value_usd", "weight", "price_usd", "avg_buy_price_usd", "pnl_pct", "basis"]]
        .assign(weight=lambda d: d["weight"] * 100),
        hide_index=True, width="stretch", height=min(38 * (len(h) + 1), 420),
        column_config={
            "symbol": "Coin", "rank": "Rank", "tier": "Market tier",
            "value_usd": st.column_config.NumberColumn("Value", format="$%,.2f"),
            "weight": st.column_config.ProgressColumn("Weight", format="%.1f%%", min_value=0, max_value=100),
            "price_usd": st.column_config.NumberColumn("Price", format="%.6g"),
            "avg_buy_price_usd": st.column_config.NumberColumn("Avg buy", format="%.6g"),
            "pnl_pct": st.column_config.NumberColumn("PnL %", format="%+.1f%%"),
            "basis": "Cost basis",
        },
    )

# ═════════════════════════════════════════════════════════════════════════════
# 4. Monetization & in-app swaps
# ═════════════════════════════════════════════════════════════════════════════
with tab_money:
    st.markdown(
        "Every campaign nudges a **risk-reducing** swap — diversify, trim long-tail exposure, consolidate dust, "
        "or deploy idle stablecoins on a schedule. Healthier portfolios churn less, and each swap earns a fee: "
        "retention and revenue pull in the same direction."
    )
    a1, a2, a3 = st.columns(3)
    fee = a1.slider("Blended swap fee", 0.10, 1.50, cfg.DEFAULT_SWAP_FEE_RATE * 100, 0.05, format="%.2f%%",
                    help="Placeholder assumption — replace with CoinStats' actual take rate.") / 100
    adoption = a2.slider("Push → swap conversion (of eligible notional)", 1.0, 20.0,
                         cfg.DEFAULT_ADOPTION_RATE * 100, 0.5, format="%.1f%%") / 100
    scale_to = a3.number_input("Extrapolate to N active manual portfolios", 1_000, 5_000_000, 100_000, 10_000,
                               help="Linear scale-up from this 500-portfolio sample — illustrative only.")

    plan = campaign_plan(seg, levers, fee_rate=fee, adoption_rate=adoption)
    factor = scale_to / len(seg)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Eligible swap notional", usd(plan["eligible_notional_usd"].sum()), border=True,
              help=f"Portfolios with ≥ ${cfg.MIN_SWAP_USD:.0f} of campaign-relevant value.")
    t2.metric("Expected swap volume / wave", usd(plan["expected_swap_volume_usd"].sum()), border=True)
    t3.metric("Fee revenue / wave (sample)", usd(plan["expected_fee_revenue_usd"].sum()), border=True)
    t4.metric(f"Fee revenue / wave @ {usd(scale_to).lstrip('$')}", usd(plan["expected_fee_revenue_usd"].sum() * factor),
              border=True, help=f"Scaled linearly to {scale_to:,} portfolios. Assumes the sample's mix of "
                                "archetypes and balances holds at scale.")

    left, right = st.columns([0.6, 0.4], gap="large")
    with left:
        p = plan.sort_values("expected_fee_revenue_usd")
        revenue = p["expected_fee_revenue_usd"] * factor
        fig = go.Figure(go.Bar(
            y=p["campaign"], x=revenue, orientation="h",
            marker_color=[ARCH_COLOR[a] for a in p["archetype_id"]],
            text=[usd(v) for v in revenue], textposition="outside", cliponaxis=False,
            customdata=p[["archetype", "targeted_portfolios"]].to_numpy(),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Revenue / wave: $%{x:,.0f}"
                          "<br>Targeted in sample: %{customdata[1]}<extra></extra>",
        ))
        fig.update_layout(title=f"Fee revenue per campaign wave @ {scale_to:,} portfolios", barcornerradius=4,
                          showlegend=False)
        fig.update_xaxes(title="USD", range=[0, max(revenue.max(), 1) * 1.2])  # headroom for end labels
        st.plotly_chart(style(fig, 330), width="stretch", theme="streamlit")
        st.caption("Bar colour = target archetype (same colours as the Cluster Explorer).")
    with right:
        st.markdown("**How the numbers are built**")
        st.markdown(
            f"""
1. **Eligible notional** — per portfolio, the USD a risk-reducing swap would move
   (idle stablecoins, sub-1% dust, speculative share above {cfg.SPECULATIVE_TARGET:.0%},
   or any coin above a {cfg.CONCENTRATION_CAP:.0%} cap). Only portfolios with ≥ ${cfg.MIN_SWAP_USD:.0f} are targeted.
2. **× conversion** — share of that notional actually swapped after the push ({adoption:.1%}).
3. **× blended fee** — CoinStats' take on swap volume ({fee:.2%}).
4. **× scale-up** — linear from 500 to {scale_to:,} portfolios.

Whales dominate the dollars (idle stablecoins), Degens dominate the High-risk users —
so the same engine serves both the revenue and the retention agenda.
"""
        )

    st.markdown("**Campaign plan (sample of 500)**")
    st.dataframe(
        plan[["campaign", "archetype", "targeted_portfolios", "high_risk_targeted", "eligible_notional_usd",
              "expected_swap_volume_usd", "expected_fee_revenue_usd"]],
        hide_index=True, width="stretch",
        column_config={
            "campaign": "Campaign", "archetype": "Target archetype",
            "targeted_portfolios": st.column_config.NumberColumn("Targeted"),
            "high_risk_targeted": st.column_config.NumberColumn("…of which High-risk",
                                                                help="Receive the protective (retention) copy variant."),
            "eligible_notional_usd": st.column_config.NumberColumn("Eligible notional", format="$%,.0f"),
            "expected_swap_volume_usd": st.column_config.NumberColumn("Expected volume", format="$%,.0f"),
            "expected_fee_revenue_usd": st.column_config.NumberColumn("Fee revenue", format="$%,.0f"),
        },
    )

    st.subheader("Push-notification playbook")
    cards = st.columns(4)
    for col, (aid, play) in zip(cards, PLAYBOOK.items()):
        with col.container(border=True):
            st.markdown(f"**{play.campaign}**")
            st.caption(f"{ARCH_NAME[aid]} · trigger: {play.trigger}")
            st.markdown("📲 _Growth copy_")
            st.markdown(play.push_growth.format(stable_usd="$4,200", dust_n=12, spec_pct="78%", top1_pct="65%"))
            st.markdown("🛟 _High-risk copy_")
            st.markdown(play.push_retention.format(stable_usd="$4,200", dust_n=12, spec_pct="78%", top1_pct="65%"))

    with st.expander("Guardrails & measurement plan"):
        st.markdown(
            f"""
- **Only risk-reducing swaps.** Campaigns never push users *into* more speculative assets; High-risk users get
  protective copy without promotional language.
- **Frequency caps.** Max one portfolio-health push per user per 7 days; suppress for 72 h after any swap.
- **Minimum size.** No swap CTA below ${cfg.MIN_SWAP_USD:.0f} of eligible notional — those users get education content.
- **Prove incrementality.** 90/10 randomised holdout per campaign; primary metrics: 30/60/90-day retention and
  swap fee revenue per user vs. control. Holdout results also become the training labels that upgrade the
  rules-based score into a supervised churn model.
"""
        )
