"""
baseline_calculator.py
-----------------------
PHASE 1 PLACEHOLDER
Computes the patient-specific physiological baseline from their own historical data.

Approaches:
  1. Simple mean over first N observations (initial baseline)
  2. Exponentially Weighted Moving Average (EWMA) updated each timestep
  3. Percentile-based baseline (median / IQR)

The baseline is stored in PatientState and used by feature_engineer.py
for personalised deviation features.

NOT IMPLEMENTED.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class PatientBaselineCalculator:
    """
    Computes and maintains patient-specific physiological baselines.

    Conceptual formula:
        Baseline_t = alpha * X_t + (1 - alpha) * Baseline_{t-1}
    where alpha is the EWMA decay factor from config.
    """

    def __init__(self, ewma_alpha: float = 0.05, min_obs: int = 48):
        raise NotImplementedError("PatientBaselineCalculator: Phase 1 TODO")

    def compute_initial_baseline(self, df: pd.DataFrame) -> dict:
        """Compute initial baseline from first min_obs observations."""
        raise NotImplementedError

    def update_baseline(self, current_baseline: dict, new_observation: dict) -> dict:
        """Update EWMA baseline with a new observation timestep."""
        raise NotImplementedError

    def is_baseline_valid(self, num_observations: int) -> bool:
        """Return True if enough observations exist for a reliable baseline."""
        raise NotImplementedError
