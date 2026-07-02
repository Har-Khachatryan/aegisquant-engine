# test_api.py
import requests

url = "http://localhost:8000/predict"
payload = {
    "account_balance": 45000,
    "balance_velocity": 0.6,
    "market_pain_index": 0.7,
    "login_freq_drop": 1.2,
    "avg_holding_days": 30,
    "crypto_ratio": 0.4,
    "tech_stocks_ratio": 0.5
}

response = requests.post(url, json=payload)
print(f"Status Code: {response.status_code}")
print("Response:")
print(response.json())