"""
AegisQuant – Scheduled drift check and automatic retraining.
Run daily from cron / Windows Task Scheduler:  python monitor_and_retrain.py

v4.0:
  - The reference is the training split saved by churn_model.py
    (artifacts/reference_data.csv) — no more synthetic stand-in.
  - The production batch is read from data/latest_production_data.csv in the
    same raw format as Churn_Modelling.csv. Without that file the script runs a
    clearly labelled dry run on a random sample of the dataset.
  - v3.2 defined load_production_features() twice (the second silently
    replaced the first); there is now one loader.
  - Retraining refreshes the reference automatically (churn_model writes it).
"""

from __future__ import annotations

import logging

import pandas as pd

from config import REFERENCE_DATA_PATH, ROOT
from DriftMonitor import DriftMonitor, results_frame

log = logging.getLogger("aegis_retrain")

PRODUCTION_BATCH_PATH = ROOT / "data" / "latest_production_data.csv"


def load_reference() -> pd.DataFrame:
    """Training features saved by the last training run (trains first if absent)."""
    if not REFERENCE_DATA_PATH.exists():
        from churn_model import run_training_pipeline
        log.warning("Reference data not found — training the model first.")
        run_training_pipeline()
    return pd.read_csv(REFERENCE_DATA_PATH)


def load_production_features() -> pd.DataFrame:
    """Latest production customers in the raw Churn_Modelling format, cleaned the same way."""
    from data_pipeline import load_churn_dataset, model_matrix

    if PRODUCTION_BATCH_PATH.exists():
        log.info("Production batch: %s", PRODUCTION_BATCH_PATH)
        return model_matrix(load_churn_dataset(PRODUCTION_BATCH_PATH))
    log.warning("No %s — DRY RUN on a random sample of the training dataset.", PRODUCTION_BATCH_PATH.name)
    return model_matrix(load_churn_dataset().sample(2_000, random_state=7))


def check_and_retrain() -> bool:
    """Returns True if a retrain was triggered."""
    log.info("=" * 60)
    log.info("Running data drift check...")
    monitor = DriftMonitor(reference_data=load_reference())
    results = monitor.compute_drift(load_production_features())
    log.info("\n%s", results_frame(results).round(4).to_string(index=False))

    if monitor.should_retrain(results):
        from churn_model import run_training_pipeline
        log.warning("Drift detected — retraining AegisQuant (reference data is refreshed by the run).")
        run_training_pipeline()
        return True
    log.info("No drift detected. Retraining not required.")
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_and_retrain()
