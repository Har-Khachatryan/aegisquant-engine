"""
CoinStats Portfolio Intelligence — diagnosis & monetisation playbook.

Turns features + segment into something a product team can ship:
  • a per-portfolio diagnosis (headline, drivers, recommended action), and
  • an In-App Swap campaign plan per archetype, sized in USD notional.

Design principle: every swap we nudge is RISK-REDUCING for the user
(diversify, trim long-tail exposure, consolidate dust, deploy idle cash on a
schedule). Revenue comes from making portfolios healthier, which is also what
keeps users from capitulating — monetisation and retention pull the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import settings as cfg
from features import risk_drivers


@dataclass(frozen=True)
class Play:
    campaign: str
    lever: str            # column in swap_levers() output that sizes the swap
    trigger: str
    push_growth: str      # copy for Low/Elevated-risk users
    push_retention: str   # copy for High-risk users (protective framing, no upsell)
    action: str


PLAYBOOK: dict[int, Play] = {
    0: Play(
        campaign="Dry-Powder DCA",
        lever="stablecoin_usd",
        trigger="Idle stablecoins in a blue-chip portfolio",
        push_growth="You have {stable_usd} sitting in stablecoins. Put it to work with a weekly BTC/ETH auto-buy — set up in one tap.",
        push_retention="Your portfolio is under pressure. Review your health check and set a staged DCA plan instead of selling at the bottom.",
        action="Offer recurring DCA from idle stablecoins into BTC/ETH; recurring swaps = recurring fees.",
    ),
    1: Play(
        campaign="Dust Consolidation",
        lever="dust_usd",
        trigger="Many positions under 1 % of portfolio value",
        push_growth="{dust_n} of your coins are each under 1% of your portfolio. Consolidate the dust into your top picks in one swap.",
        push_retention="Cut the noise: merge {dust_n} tiny positions into the coins you actually believe in — one swap, cleaner portfolio.",
        action="One-tap batch swap of sub-1 % positions into the user's top holdings.",
    ),
    2: Play(
        campaign="Risk-Off Rotation",
        lever="speculative_excess_usd",
        trigger=f"Speculative (rank > {cfg.SPECULATIVE_MIN_RANK}) share above {cfg.SPECULATIVE_TARGET:.0%}",
        push_growth="{spec_pct} of your portfolio is in high-volatility coins. Lock in some gains — rotate part into BTC, ETH or stablecoins.",
        push_retention="Protect what's left: {spec_pct} of your portfolio is in coins that can drop 90% in a week. Rotate part into BTC/ETH in one tap.",
        action=f"Suggest trimming long-tail exposure toward {cfg.SPECULATIVE_TARGET:.0%} via BTC/ETH/stablecoin swaps.",
    ),
    3: Play(
        campaign="Smart Rebalance",
        lever="concentration_excess_usd",
        trigger=f"A single non-stable coin above {cfg.CONCENTRATION_CAP:.0%} of the portfolio",
        push_growth="{top1_pct} of your portfolio is in one coin. Tap to spread risk across a diversified starter basket.",
        push_retention="One coin is {top1_pct} of your portfolio — that's what makes red days hurt. Rebalance into a starter basket in one tap.",
        action=f"One-tap rebalance of the excess above a {cfg.CONCENTRATION_CAP:.0%} single-coin cap into a diversified basket.",
    ),
}

DIAGNOSIS = {
    "High": "High Risk of Attrition — Smart Rebalancing Recommended",
    "Elevated": "Elevated Risk — Proactive Health Check Recommended",
    "Low": "Healthy — Growth & Engagement Opportunity",
}


# ═════════════════════════════════════════════════════════════════════════════
# Swap levers (USD notional a rebalancing nudge could move)
# ═════════════════════════════════════════════════════════════════════════════
def swap_levers(
    holdings: pd.DataFrame,
    concentration_cap: float = cfg.CONCENTRATION_CAP,
    speculative_target: float = cfg.SPECULATIVE_TARGET,
    dust_weight: float = cfg.DUST_WEIGHT,
) -> pd.DataFrame:
    """
    Per-portfolio swap notional for each campaign lever. Expects the output of
    features.prepare_holdings().

    stablecoin_usd            value held in stablecoins (dry powder)
    dust_usd                  value in positions below `dust_weight`
    speculative_excess_usd    value in rank>100 coins above `speculative_target`
    concentration_excess_usd  value of non-stable coins above `concentration_cap`
    """
    h = holdings
    pid = h["portfolio_id"]
    total = h.groupby(pid)["value_usd"].sum()
    excess_w = (h["weight"] - concentration_cap).clip(lower=0.0).where(~h["is_stablecoin"], 0.0)
    spec_share = h["weight"].where(h["is_speculative"], 0.0).groupby(pid).sum()

    out = pd.DataFrame({
        "stablecoin_usd": h["value_usd"].where(h["is_stablecoin"], 0.0).groupby(pid).sum(),
        "dust_usd": h["value_usd"].where(h["weight"] < dust_weight, 0.0).groupby(pid).sum(),
        "speculative_excess_usd": (spec_share - speculative_target).clip(lower=0.0) * total,
        "concentration_excess_usd": excess_w.groupby(pid).sum() * total,
    })
    out.index.name = "portfolio_id"
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Per-portfolio diagnosis
# ═════════════════════════════════════════════════════════════════════════════
def diagnose(row: pd.Series | dict, levers: pd.Series | dict) -> dict:
    """Headline, drivers, recommended action and push copy for one portfolio."""
    row, levers = dict(row), dict(levers)
    tier = row["risk_tier"]
    play = PLAYBOOK[int(row["archetype_id"])]
    template = play.push_retention if tier == "High" else play.push_growth
    push = template.format(
        stable_usd=f"${levers.get('stablecoin_usd', 0):,.0f}",
        dust_n=int(row.get("dust_positions", 0)),
        spec_pct=f"{row['speculative_meme_ratio']:.0%}",
        top1_pct=f"{row['top1_weight']:.0%}",
    )
    notional = float(levers.get(play.lever, 0.0))
    return {
        "headline": DIAGNOSIS[tier],
        "risk_tier": tier,
        "drivers": risk_drivers(row),
        "campaign": play.campaign,
        "recommended_action": play.action,
        "push_message": push,
        "swap_notional_usd": round(notional, 2),
        "eligible_for_campaign": notional >= cfg.MIN_SWAP_USD,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Campaign sizing across the book
# ═════════════════════════════════════════════════════════════════════════════
def campaign_plan(
    segmented: pd.DataFrame,
    levers: pd.DataFrame,
    fee_rate: float = cfg.DEFAULT_SWAP_FEE_RATE,
    adoption_rate: float = cfg.DEFAULT_ADOPTION_RATE,
) -> pd.DataFrame:
    """
    One row per archetype campaign:
      targeted portfolios (lever ≥ MIN_SWAP_USD), of which High-risk,
      eligible notional, expected swap volume (× adoption), fee revenue (× fee).
    """
    df = segmented.set_index("portfolio_id").join(levers, how="left")
    rows = []
    for aid, play in PLAYBOOK.items():
        seg = df[df["archetype_id"] == aid]
        notional = seg[play.lever].fillna(0.0)
        target = notional >= cfg.MIN_SWAP_USD
        eligible = float(notional[target].sum())
        volume = eligible * adoption_rate
        rows.append({
            "archetype_id": aid,
            "archetype": cfg.ARCHETYPES[aid].name,
            "campaign": play.campaign,
            "trigger": play.trigger,
            "portfolios_in_segment": int(len(seg)),
            "targeted_portfolios": int(target.sum()),
            "high_risk_targeted": int((target & (seg["risk_tier"] == "High")).sum()),
            "eligible_notional_usd": eligible,
            "expected_swap_volume_usd": volume,
            "expected_fee_revenue_usd": volume * fee_rate,
            "sample_push": play.push_growth,
        })
    return pd.DataFrame(rows)
