# CoinStats Portfolio Intelligence — one image, two services (API or dashboard).
#
# Build from the REPO ROOT (the CSV lives in ./data):
#   docker build -f coinstats_intel/Dockerfile -t coinstats-intel .
#
# Run:
#   docker run --rm -p 8501:8080 coinstats-intel                    # dashboard → http://localhost:8501
#   docker run --rm -p 8000:8080 -e SERVICE=api coinstats-intel     # API       → http://localhost:8000/docs
#
# The build context is restricted by coinstats_intel/Dockerfile.dockerignore
# (BuildKit picks it up automatically), so the AegisQuant venv and other repo
# files never leave your machine.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COINSTATS_DATA_PATH=/app/data/coinstats_manual_portfolios_holdings.csv \
    COINSTATS_ARTIFACT_DIR=/app/artifacts \
    SERVICE=dashboard \
    PORT=8080

WORKDIR /app

# Dependencies first → cached layer unless requirements change.
COPY coinstats_intel/requirements.txt .
RUN pip install -r requirements.txt

COPY coinstats_intel/ ./
COPY data/coinstats_manual_portfolios_holdings.csv ./data/

# Bake the model at build time: every image carries a fixed, reproducible model,
# cold starts skip training, and a broken pipeline fails the build — not a request.
RUN python pipeline.py \
 && useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run injects $PORT; SERVICE selects which app this container serves.
CMD ["sh", "-c", "if [ \"$SERVICE\" = \"api\" ]; then exec uvicorn api:app --host 0.0.0.0 --port \"$PORT\"; else exec streamlit run app.py --server.port=\"$PORT\" --server.address=0.0.0.0; fi"]
