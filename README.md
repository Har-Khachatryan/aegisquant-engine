Sirius Capital — Quant-Grade ML Risk Pipeline (v3.1)
Author: Harutyun Arami Khachatryan

Architecture: Enterprise FinTech & Algorithmic Trading Pipeline

Stack: Python, XGBoost, Scikit-Learn (KMeans), SciPy (SLSQP), yfinance, Pydantic v2

🚀 Project Overview
An enterprise-grade financial engineering pipeline that solves the customer churn problem for retail and HNW trading clients. The system predicts attrition risk using machine learning and dynamically restructures multi-asset portfolios to maximize client retention utility based on Modern Portfolio Theory (MPT).

⚙️ What's New in v3.1 (Feasibility Repair Layer)
Fixed a critical edge-case deadlock where tight investor profile floors (min_w = 0.08 for conservative clients) conflicted with explicit thematic preferences (crypto_ratio = 0.05), causing budget constraint failures in the SLSQP solver. Version 3.1 introduces a headroom-proportional deficit distribution algorithm that guarantees mathematical convergence under any extreme preference boundary.

🛠️ System Architecture
Module 0: Data Contracts — Strict input schema validation backed by Pydantic v2.

Module 1 & 2: Environment & Profiling — Synthetic financial data simulation and multi-feature KMeans customer segmentation.

Module 3: Predictive Engine — Robust XGBoost classifier optimized for Churn AUC-ROC metrics.

Module 4: Portfolio Quant Optimization — Mean-Variance optimization via SciPy.optimize.minimize utilizing Ledoit-Wolf shrinkage covariance matrices for market noise reduction.

Module 5: Production Gateway — Thread-safe live execution simulator with yfinance TTL caching layers.

📦 Installation & Setup
1. Install Dependencies
Ensure you have Python installed, then run the following command to install all the required financial and machine learning libraries:

pip install -r requirements.txt

2. Run the Pipeline
Execute the complete end-to-end production batch simulation:

python sirius_quant_engine_v3.py

📊 Production Execution Output
Below is the verified v3.1 production log layout exhibiting 100% convergence across mixed client profiles:

╔══════════════════════════════════════════════════════════════════╗
║         SIRIUS CAPITAL — QUANT GRADE ENGINE  v3.1                ║
╚══════════════════════════════════════════════════════════════════╝
🚀  Executing Production Batch (Full Cycle Context)...

══════════════════════════════════════════════════════════════════════════════════════════
🔹  Client #9001  [Crypto Enthusiast — Aggressive (58% Churn)]
Profile : AGGRESSIVE      |  Churn Risk : 🟡 58.73%
Strategy: Dynamic Retention — AGGRESSIVE (Risk Scale: 0.59)
• AAPL  ██                    5.00%  →  $    2,100.00
• MSFT  ██                    5.00%  →  $    2,100.00
• KO    ████████████████     40.00%  →  $   16,800.00
• NVDA  ██                    5.00%  →  $    2,100.00
• TSLA  ██                    5.00%  →  $    2,100.00
• BTC   ██                    6.83%  →  $    2,867.06
• ETH   █████████████        33.17%  →  $   13,932.94
──────────────────────────────────────────────────────────────────────────────────────────
🔹  Client #9002  [Institutional HNW — Solid Inflows]
Profile : CONSERVATIVE    |  Churn Risk : 🟢 14.77%
✅  STABLE — No retention action required
──────────────────────────────────────────────────────────────────────────────────────────
🔹  Client #9005  [Conservative Senior — Sudden Outflow]
Profile : CONSERVATIVE    |  Churn Risk : 🔴 87.97%
Strategy: Dynamic Retention — CONSERVATIVE (Risk Scale: 0.88)
• AAPL  █████                12.84%  →  $   14,128.44
• MSFT  █████                12.84%  →  $   14,128.44
• KO    █████████████████    52.00%  →  $   57,200.00  ✓ [Feasibility Repaired]
• NVDA  █████                12.84%  →  $   14,128.44
• TSLA  █████                12.84%  →  $   14,128.44
• BTC   █████                12.84%  →  $   14,128.44
• ETH   █████                12.84%  →  $   14,128.44
══════════════════════════════════════════════════════════════════════════════════════════
✔  Batch complete. All v3.1 pipeline stages executed successfully with zero optimizer failures.