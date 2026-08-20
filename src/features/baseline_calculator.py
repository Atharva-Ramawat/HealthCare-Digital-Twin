"""
baseline_calculator.py
-----------------------
PHASE 2 PLACEHOLDER
Computes the patient-specific physiological baseline from their own historical data.

ENGINEERING INITIAL APPROACH: EWMA (Exponentially Weighted Moving Average)
This is used as a starting point for the software pipeline.
It is NOT the final research methodology.

The final baseline methodology is a student team research decision (Phase 5).
It may be EWMA, percentile-based, regression-based, or another justified approach.

CRITICAL CAUSAL CONSTRAINT (non-negotiable):
  The baseline at time t MUST only use observations from time <= t.
  Future observations MUST NEVER influence the baseline used for an earlier prediction.
  This constraint applies to ALL baseline methodologies chosen by the student team.

NOT IMPLEMENTED.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class PatientBaselineCalculator:
    """
    Computes and maintains patient-specific physiological baselines.

    Engineering initial approach: EWMA
        B_t = alpha * X_t + (1 - alpha) * B_{t-1}
    where alpha is the EWMA decay factor from configs/settings.yaml.

    IMPORTANT: This class is an engineering starting point.
    The student team may replace or extend the baseline methodology in Phase 5.

    CAUSAL CONSTRAINT: All methods must be implemented such that no
    future observation influences the baseline at any earlier timestep.
    """

    def __init__(self, ewma_alpha: float = 0.05, min_obs: int = 48):
        """
        Args:
            ewma_alpha: EWMA decay factor (engineering default, see settings.yaml).
            min_obs: Minimum number of observations before baseline is considered valid.
        """
        raise NotImplementedError("PatientBaselineCalculator: Phase 2 TODO")

    def compute_initial_baseline(self, df: pd.DataFrame) -> dict:
        """
        Compute initial baseline from the first min_obs observations.
        Uses simple mean over the initial window.
        CAUSAL: Uses only the first N timesteps.
        """
        raise NotImplementedError

    def update_baseline(self, current_baseline: dict, new_observation: dict) -> dict:
        """
        Update EWMA baseline with a single new observation.
        CAUSAL: Uses only current_baseline (computed from past) and new_observation.
        """
        raise NotImplementedError

    def is_baseline_valid(self, num_observations: int) -> bool:
        """Return True if enough observations exist for a reliable baseline estimate."""
        raise NotImplementedError

    def compute_deviations(self, current_vitals: dict, baseline: dict) -> dict:
        """
        Compute per-vital deviation from patient baseline.
        Deviation_t = X_t - B_t
        """
        raise NotImplementedError
