FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE=api \
    PORT=8080

# XGBoost-ին անհրաժեշտ է OpenMP (libgomp1)։ build-essential-ը պետք չէ՝ բոլոր փաթեթներն ունեն պատրաստի wheel
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Եթե artifacts/-ը կա (նույն մոդելը, որի թվերը README-ում են), օգտագործվում է այն. հակառակ դեպքում
# (օրինակ՝ GitHub-ից նոր clone) մոդելը սովորեցվում է build-ի ժամանակ։ Աշխատում է ոչ-root օգտատիրոջից
RUN python -c "from churn_model import ensure_trained; ensure_trained()" \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser /app
USER appuser

# Cloud Run-ը ավտոմատ տրամադրում է PORT environment variable (լռելյայն 8080)
EXPOSE 8080

# SERVICE=api → FastAPI, SERVICE=dashboard → Streamlit (offline market data: AEGIS_OFFLINE=1)
CMD ["sh", "-c", "if [ \"$SERVICE\" = dashboard ]; then exec streamlit run app.py --server.port \"$PORT\" --server.address 0.0.0.0; else exec uvicorn api:app --host 0.0.0.0 --port \"$PORT\"; fi"]
