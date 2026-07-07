# AegisQuant v3.2 — AI-Driven Portfolio Risk & Optimization Engine

<img width="1920" height="1037" alt="dashboard" src="https://github.com/user-attachments/assets/bc00242c-e49d-4c14-a0e9-e2726f18b969" />

## 🚀 Evolution: From v3.1 to v3.2 (Recent Upgrades)

The project was recently upgraded from v3.1 to v3.2 to address critical architectural and deployment limitations:

* **Better Explainability (SHAP Integration):** Shifted from global feature importances to local explanations using `shap.TreeExplainer`. Now, the API provides the top-2 specific risk drivers for *each individual client* based on their unique input data, instead of returning the same static global features for everyone.
* **Secure Model Serialization:** Replaced legacy `.pkl` (Pickle) monolith files with native XGBoost JSON serialization (`xgb_model.save_model("aegis_xgb.json")`). Preprocessing pipelines are separated to improve security and prevent potential code execution vulnerabilities.
* **Asynchronous Market Worker:** Implemented a background daemon thread (`_MarketDataWorker`) that runs in an infinite loop to fetch data from `yfinance`. This prevents the FastAPI endpoints from blocking or freezing when fetching live asset prices.
* **Single Source of Truth:** Consolidated internal data payloads and FastAPI structures into a single unified `ClientFeatures` Pydantic v2 model.

# 🛡️ AegisQuant — AI Portfolio Shield & Risk Engine v3.1

**Enterprise-grade churn prediction and dynamic portfolio optimization engine**  
Built with XGBoost, KMeans clustering, Ledoit‑Wolf shrinkage, and Dynamic Markowitz optimization.

---

## 📦 Project Structure
aegisquant/
├── config.py # Global constants, asset mappings, Pydantic contract
├── data_pipeline.py # Synthetic data generation & DynamicProfileResolver
├── feature_cross_pollination.py # Unified pipeline (ClusterInjector + XGBoost)
├── churn_model.py # Offline training pipeline
├── optimizer.py # AegisQuantEngine: online inference, Markowitz optimization
├── app.py # Streamlit dashboard (interactive UI)
├── api.py # FastAPI REST API with risk drivers explanation
├── DriftMonitor.py # Data drift detector (KS‑test, no external deps)
├── monitor_and_retrain.py # Automatic retraining trigger based on drift
├── test_api.py # Quick API test script
├── requirements.txt # Python dependencies
└── README.md

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Har-Khachatryan/aegisquant-engine.git
cd aegisquant-engine
```
### 2. Create and activate a virtual environment
```
python -m venv aegis_env
aegis_env\Scripts\activate      # Windows
# source aegis_env/bin/activate   # Linux/macOS
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Run the Streamlit Dashboard
```
streamlit run app.py
The dashboard will open in your browser at http://localhost:8501.
```
### 5. (Optional) Start the REST API
```
uvicorn api:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000/docs for interactive Swagger documentation.
```

🧠 Key Features
Unified ML pipeline – KMeans investor segmentation feeds directly into XGBoost churn classifier with monotonic constraints.

Dynamic Portfolio Optimization – Ledoit‑Wolf covariance, feasibility‑repair layer for box constraints, SLSQP solver with analytical Jacobian.

Risk Explainability – API returns top‑2 risk drivers based on global feature importance (ready to plug SHAP for per‑sample explanations).

Data Drift Monitoring – KS‑test‑based drift detector with automatic retraining trigger.

Production‑ready – Thread‑safe market data cache, configurable thresholds, full type hints, Pydantic data contracts.

🔧 Configuration
All hyperparameters and asset mappings are in config.py.
Key settings:

ASSETS / TICKER_MAP – investment universe

RISK_AVERSION – base risk aversion per investor profile

CHURN_THRESHOLD – probability threshold for retention action

MAX_DRIFTED_FEATURES_BEFORE_RETRAIN – retraining trigger

MARKET_CACHE_TTL – market data cache duration

📊 Drift Monitoring & Retraining
To enable automatic model retraining when data drift is detected:

Ensure a reference dataset exists (reference_data.pkl).
It is automatically saved after the first training run.

Schedule monitor_and_retrain.py daily (Windows Task Scheduler / cron):

```
python monitor_and_retrain.py
```
The script compares current production features against the reference and triggers run_training_pipeline() if drift exceeds the threshold.


📈 Example API Response
POST /predict

```json
{
  "account_balance": 45000,
  "balance_velocity": 0.6,
  "market_pain_index": 0.7,
  "login_freq_drop": 1.2,
  "avg_holding_days": 30,
  "crypto_ratio": 0.4,
  "tech_stocks_ratio": 0.5
}
```
Response

```json
{
  "churn_probability": 0.9369,
  "risk_drivers": [
    "Negative balance velocity (outflows)",
    "Sudden drop in login frequency"
  ],
  "model_version": "aegis_quant_v3.1_cross_pollination"
}
```
🛠️ Technologies
XGBoost with monotonic constraints

scikit‑learn (KMeans, Ledoit‑Wolf, Pipeline)

SciPy (SLSQP optimizer, KS‑test)

FastAPI + Uvicorn

Streamlit + Plotly

Pydantic (data contracts)

yfinance (market data)

📝 Version History

v3.1 – Unified cross‑pollination pipeline, feasibility repair layer, stable API, drift monitor.

v3.0 – Initial modular architecture, bug fixes, production hardening.

📬 Contact
For questions or contributions, please open an issue or contact the maintainer.

AegisQuant – protecting portfolios with AI. 🛡️
