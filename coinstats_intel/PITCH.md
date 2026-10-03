# 2-minute pitch — CoinStats CEO / CTO

*~300 words ≈ 2 minutes at a calm pace. Numbers are from the 500-portfolio snapshot.*
*Armenian version: [PITCH.hy.md](PITCH.hy.md).*

---

**The problem.** CoinStats' most valuable asset is the user who keeps opening the app. In 500 active
manual portfolios, the median holder is barely green — plus 5%. But the median hides the danger:
140 portfolios are down more than 20%, and 78 — roughly one in six — show the full capitulation
pattern: deep losses, concentrated in one or two coins, heavy in long-tail meme tokens. That is the
user who deletes the app after the next red week.

**What I built.** A portfolio-intelligence engine that runs on data CoinStats already has. It cleans
the data first — 131 typo'd cost bases alone would have invented a trillion dollars of losses — then
computes ten health metrics, segments users into four archetypes with KMeans, and scores every
portfolio 0 to 100 for capitulation risk, with a plain-English reason. It's live as a FastAPI service and this dashboard, running privately on Google Cloud Run.

**What it tells us.** Whales — a quarter of users — hold 70% of assets and are up 19%: engage them,
don't rescue them. Degens are under a quarter of users but 49 of the 78 high-risk portfolios.
Explorers hold a fifth of assets yet sit at minus 16%. And high-risk users hold just 2% of assets —
this is a *user*-retention problem, measured in engagement, not AUM.

**How it pays.** Every segment gets one risk-reducing, one-tap swap: put idle stablecoins on a DCA
plan, consolidate dust, rotate out of meme exposure, or rebalance a single-coin bet. That's
$9.7 million of swap-eligible value in just 500 portfolios — at an assumed 5% conversion and 0.75%
fee, about $730 thousand per campaign wave across 100,000 portfolios. Each swap makes the portfolio
healthier, so revenue and retention move together.

**The ask.** Give me historical retention labels and a 90/10 holdout test, and within a quarter we
turn this score into a validated churn model — and measure exactly how much lifetime value it protects.

---

## Demo flow (run alongside the pitch)

| Pitch beat | Show |
|---|---|
| The problem | KPI strip → *High-risk portfolios: 78*, *Median PnL +4.9%* |
| What I built | *Cluster Explorer* → four colours separating on value × concentration |
| What it tells us | *Executive Overview* → segment table + risk-tier bars |
| (live moment) | *Portfolio Health Inspector* → top of the list (p_488): 🚨 diagnosis + push preview |
| How it pays | *Monetization* → drag the conversion slider; revenue updates live |
| The ask | *Guardrails & measurement plan* expander |

## Likely questions — short answers

- **"Is the score validated?"** Not yet — there are no churn labels in the sample. It's a transparent
  rules-based proxy built on three behaviours that precede capitulation; the holdout test produces the
  labels that turn it into a supervised model.
- **"Why trust the PnL?"** Cost bases outside a rank-aware plausibility band are excluded and every
  portfolio reports what share of its value has a trusted basis; 7 portfolios get no PnL figure at all.
- **"Are the dollar figures real?"** Allocation, HHI and PnL % are exact. Absolute balances carry the
  dataset's 0.5–2× anonymisation noise, and fee/conversion are stated assumptions on a slider.
- **"Why KMeans with a silhouette of 0.24?"** Behavioural segments overlap at the edges; what matters
  is that the four centroids are clearly distinct and actionable — and labels are assigned by centroid
  signature, so they survive retraining.
- **"Can we open it ourselves?"** It runs on Cloud Run with no public URL — the dataset licence forbids
  redistribution. Access is granted per Google account through IAM — I can grant yours today, and
  browser sign-in through Identity-Aware Proxy is a small next step.
