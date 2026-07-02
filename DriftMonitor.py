"""
AegisQuant – Data Drift Monitor (scipy.stats – no external deps)
=================================================================
Uses the two‑sample Kolmogorov–Smirnov test to detect distribution
shifts in behavioural / market features.  Includes a lightweight
retraining trigger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aegis_drift")

# ═══════════════════════════════════════════════════════════════════
# Configuration – adjust thresholds to your operational needs
# ═══════════════════════════════════════════════════════════════════
MONITORED_FEATURES = [
    "account_balance",
    "balance_velocity",
    "market_pain_index",
    "login_freq_drop",
]

# If more than this number of monitored features drift → trigger retrain
MAX_DRIFTED_FEATURES_BEFORE_RETRAIN = 1   # conservative: trigger even on 1 drift

# p‑value threshold for the KS test (lower = stricter)
DRIFT_P_VALUE_THRESHOLD = 0.05


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
    Monitors data drift between a reference dataset (usually the training set)
    and a current production batch using the two‑sample KS test.

    Parameters
    ----------
    reference_data : pd.DataFrame
        Baseline dataset (e.g. training features). Must contain all
        ``MONITORED_FEATURES`` columns.
    drift_threshold : float, optional
        p‑value threshold for the statistical test. Lower → stricter.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        drift_threshold: float = DRIFT_P_VALUE_THRESHOLD,
    ) -> None:
        self.reference_data = reference_data[MONITORED_FEATURES].copy()
        self.drift_threshold = drift_threshold
        log.info(
            "DriftMonitor initialised with reference data of shape %s",
            self.reference_data.shape,
        )

    def compute_drift(self, current_data: pd.DataFrame) -> List[DriftResult]:
        """
        Compare the current production batch against the reference data.

        Returns
        -------
        list[DriftResult]
            One DriftResult per monitored feature.
        """
        current = current_data[MONITORED_FEATURES].copy()
        drift_results: List[DriftResult] = []

        for feature in MONITORED_FEATURES:
            ref_vals = self.reference_data[feature].dropna().values
            cur_vals = current[feature].dropna().values

            # Edge case: not enough data
            if len(ref_vals) < 5 or len(cur_vals) < 5:
                log.warning("Not enough data for feature '%s' – skipping", feature)
                drift_results.append(
                    DriftResult(
                        feature_name=feature,
                        drift_detected=False,
                        p_value=np.nan,
                        stat_test="ks_2samp",
                        ks_statistic=np.nan,
                    )
                )
                continue

            # Perform KS test
            ks_stat, p_value = ks_2samp(ref_vals, cur_vals)
            drift_detected = p_value < self.drift_threshold

            drift_results.append(
                DriftResult(
                    feature_name=feature,
                    drift_detected=drift_detected,
                    p_value=p_value,
                    stat_test="ks_2samp",
                    ks_statistic=ks_stat,
                )
            )

        log.info(
            "Drift computation complete. Drifted features: %d",
            sum(1 for r in drift_results if r.drift_detected),
        )
        return drift_results

    def should_retrain(self, drift_results: List[DriftResult]) -> bool:
        """
        Decision rule: if the number of drifted features equals or exceeds
        ``MAX_DRIFTED_FEATURES_BEFORE_RETRAIN``, return True.
        """
        num_drifted = sum(1 for r in drift_results if r.drift_detected)
        trigger = num_drifted >= MAX_DRIFTED_FEATURES_BEFORE_RETRAIN

        if trigger:
            log.warning(
                "Retraining triggered: %d/%d monitored features drifted.",
                num_drifted,
                len(drift_results),
            )
        else:
            log.info(
                "No retraining needed: %d drifted features, threshold is %d.",
                num_drifted,
                MAX_DRIFTED_FEATURES_BEFORE_RETRAIN,
            )
        return trigger


# ────────────────────────────────────────────────────────────────
# Example usage (run this file directly to see a demo)
# ────────────────────────────────────────────────────────────────
def _generate_demo_data(
    n_ref: int = 1000, n_curr: int = 500, drift: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic reference and current datasets.
    When ``drift=True``, the current data is sampled from a different
    distribution to simulate a real shift.
    """
    rng = np.random.default_rng(42)

    ref = pd.DataFrame({
        "account_balance": rng.lognormal(mean=10, sigma=0.8, size=n_ref),
        "balance_velocity": rng.beta(a=2, b=5, size=n_ref),
        "market_pain_index": rng.uniform(0.1, 0.6, n_ref),
        "login_freq_drop": rng.exponential(scale=0.3, size=n_ref),
    })

    if drift:
        curr = pd.DataFrame({
            "account_balance": rng.lognormal(mean=10.5, sigma=1.0, size=n_curr),
            "balance_velocity": rng.beta(a=3, b=4, size=n_curr),
            "market_pain_index": rng.uniform(0.4, 0.9, n_curr),
            "login_freq_drop": rng.exponential(scale=0.5, size=n_curr),
        })
    else:
        curr = pd.DataFrame({
            "account_balance": rng.lognormal(mean=10, sigma=0.8, size=n_curr),
            "balance_velocity": rng.beta(a=2, b=5, size=n_curr),
            "market_pain_index": rng.uniform(0.1, 0.6, n_curr),
            "login_freq_drop": rng.exponential(scale=0.3, size=n_curr),
        })

    return ref, curr


if __name__ == "__main__":
    print("=" * 70)
    print("AegisQuant – Drift Monitor Demo (scipy.stats)")
    print("=" * 70)

    # 1. Reference (training) data
    ref_data, _ = _generate_demo_data(n_ref=1000, n_curr=100, drift=False)

    # 2. Initialise the monitor
    monitor = DriftMonitor(reference_data=ref_data)

    # 3. Batch WITHOUT drift
    _, curr_no_drift = _generate_demo_data(n_ref=1000, n_curr=500, drift=False)
    results_no = monitor.compute_drift(curr_no_drift)
    retrain_no = monitor.should_retrain(results_no)

    print("\n--- Batch 1 (no drift) ---")
    for res in results_no:
        print(f"  {res.feature_name:<20} drift={res.drift_detected!s:<5}  "
              f"p={res.p_value:.4f}  KS_stat={res.ks_statistic:.3f}")
    print(f"  >>> Retrain needed? {retrain_no}")

    # 4. Batch WITH drift
    _, curr_drift = _generate_demo_data(n_ref=1000, n_curr=500, drift=True)
    results_drift = monitor.compute_drift(curr_drift)
    retrain_drift = monitor.should_retrain(results_drift)

    print("\n--- Batch 2 (with drift) ---")
    for res in results_drift:
        print(f"  {res.feature_name:<20} drift={res.drift_detected!s:<5}  "
              f"p={res.p_value:.4f}  KS_stat={res.ks_statistic:.3f}")
    print(f"  >>> Retrain needed? {retrain_drift}")
    print("\nDone.")