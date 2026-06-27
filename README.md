# Sirius Capital — Quant-Grade ML Risk Pipeline (v3.1)

**Author:** Harutyun Arami Khachatryan  
**Stack:** Python, XGBoost, Scikit-Learn (KMeans), SciPy (SLSQP), yfinance, Pydantic v2

An enterprise-grade financial engineering pipeline that solves the customer churn problem for retail/HNW trading clients and dynamically restructures asset allocation portfolios to maximize retention utility.

---

## 🚀 Key Features

* **Investor Segmentation:** Multi-feature `KMeans` clustering mapping clients to dynamic risk profiles (*Aggressive*, *Balanced*, *Conservative*).
* **Churn Predictive Engine:** `XGBoost` classifier predicting client attrition probability using behavioral and balance velocity features.
* **Dynamic Markowitz Optimization:** Custom asset allocation engine based on Modern Portfolio Theory (MPT) using `SciPy.optimize.minimize` (SLSQP).
* **Feasibility Repair Layer (v3.1):** Advanced headroom-proportional math layer ensuring portfolio box constraints and asset ceiling boundaries (`WEIGHT_MAX = 40%`) never conflict with budget constraints ($\sum w_i = 1$).
* **Robust Covariance:** Implements `Ledoit-Wolf` shrinkage for stable annualized covariance matrices against noisy financial data.

---

## 🛠️ Architecture Blueprint

1. **Module 0:** Input Contract via strict `Pydantic v2` models.
2. **Module 1 & 2:** Synthetic financial environment generator & Dynamic profile resolver.
3. **Module 3:** Offline training pipeline targeting Churn AUC-ROC maximization.
4. **Module 4 & 5:** Quant portfolio optimization engine backed by thread-safe `yfinance` TTL caching & production API gateway simulator.

---

## 📦 Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/Har-Khachatryan/sirius-capital-quant-engine.git](https://github.com/YOUR_USERNAME/sirius-capital-quant-engine.git)
   cd sirius-capital-quant-engine