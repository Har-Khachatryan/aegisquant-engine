"""
AegisQuant – Data Drift Monitor (scipy.stats – no external deps)
=================================================================
Uses the two-sample Kolmogorov–Smirnov test to detect distribution shifts in
the customer features the churn model relies on, plus a lightweight
retraining trigger.

v4.0: monitors the bank-customer features of the Churn Modelling dataset and
the demo below runs on real data (training reference vs. hold-out customers,
then vs. a deliberately shifted batch) instead of synthetic distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

log = logging.getLogger("aegis_drift")

# ═══════════════════════════════════════════════════════════════════
# Configuration – adjust thresholds to your operational needs
# ═══════════════════════════════════════════════════════════════════
MONITORED_FEATURES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "num_products",
    "estimated_salary",
]

# If at least this many monitored features drift → trigger retrain
MAX_DRIFTED_FEATURES_BEFORE_RETRAIN = 2

# p-value threshold for the KS test (lower = stricter)
DRIFT_P_VALUE_THRESHOLD = 0.01


@dataclass
class DriftResult:
    """Container for a single feature drift test result."""
    feature_name: str
    drift_detected: bool
    p_value: float
    stat_test: str
    ks_statistic: float


class DriftMonitor:
    """
    Monitors data drift between a reference dataset (the training features)
    and a current production batch using the two-sample KS test.
    """

    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = DRIFT_P_VALUE_THRESHOLD) -> None:
        self.reference_data = reference_data[MONITORED_FEATURES].copy()
        self.drift_threshold = drift_threshold
        log.info("DriftMonitor initialised with reference data of shape %s", self.reference_data.shape)

    def compute_drift(self, current_data: pd.DataFrame) -> List[DriftResult]:
        """One DriftResult per monitored feature."""
        current = current_data[MONITORED_FEATURES]
        results: List[DriftResult] = []
        for feature in MONITORED_FEATURES:
            ref_vals = self.reference_data[feature].dropna().to_numpy()
            cur_vals = current[feature].dropna().to_numpy()
            if len(ref_vals) < 5 or len(cur_vals) < 5:
                log.warning("Not enough data for feature '%s' – skipping", feature)
                results.append(DriftResult(feature, False, np.nan, "ks_2samp", np.nan))
                continue
            ks_stat, p_value = ks_2samp(ref_vals, cur_vals)
            results.append(DriftResult(feature, bool(p_value < self.drift_threshold), float(p_value), "ks_2samp", float(ks_stat)))
        log.info("Drift computation complete. Drifted features: %d", sum(r.drift_detected for r in results))
        return results

    def should_retrain(self, drift_results: List[DriftResult]) -> bool:
        """True if the number of drifted features reaches MAX_DRIFTED_FEATURES_BEFORE_RETRAIN."""
        num_drifted = sum(r.drift_detected for r in drift_results)
        trigger = num_drifted >= MAX_DRIFTED_FEATURES_BEFORE_RETRAIN
        if trigger:
            log.warning("Retraining triggered: %d/%d monitored features drifted.", num_drifted, len(drift_results))
        else:
            log.info("No retraining needed: %d drifted features, threshold is %d.", num_drifted, MAX_DRIFTED_FEATURES_BEFORE_RETRAIN)
        return trigger


def simulate_shifted_batch(batch: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    A plausible production shift for demos: the customer base ages by ~6 years
    and balances fall 25 % (e.g. a rate-cut driven outflow).
    """
    rng = np.random.default_rng(seed)
    shifted = batch.copy()
    shifted["age"] = (shifted["age"] + rng.normal(6, 2, len(shifted))).clip(18, 100).round()
    shifted["balance"] = shifted["balance"] * 0.75
    return shifted


def results_frame(results: List[DriftResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


if __name__ == "__main__":
    from config import REFERENCE_DATA_PATH
    from data_pipeline import load_churn_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 70)
    print("AegisQuant – Drift Monitor Demo (real Churn Modelling data)")
    print("=" * 70)
    reference = pd.read_csv(REFERENCE_DATA_PATH)          # written by churn_model.py
    customers = load_churn_dataset()
    monitor = DriftMonitor(reference_data=reference)

    batch = customers.sample(2_000, random_state=7)
    for name, current in [("random customers (no drift expected)", batch),
                          ("simulated shift: older, lower balances", simulate_shifted_batch(batch))]:
        results = monitor.compute_drift(current)
        print(f"\n--- {name} ---")
        print(results_frame(results).round(4).to_string(index=False))
        print(f"  >>> Retrain needed? {monitor.should_retrain(results)}")
