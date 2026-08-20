"""
feature_engineer.py
--------------------
PHASE 1 PLACEHOLDER
Computes temporal and personalised features from the clean patient timeline.

Features to be computed:
  - Delta (first-order difference): ΔX_t = X_t - X_{t-1}
  - Rate of change: dX/dt approximation over window
  - Rolling statistics: mean, std, min, max over configurable windows
  - Deviation from patient-specific baseline: X_t - PatientBaseline
  - NEWS2 score (per timestep)
  - Shock Index: HR / SBP
  - Pulse Pressure: SBP - DBP (when DBP available)
  - Mean Arterial Pressure: (SBP + 2*DBP) / 3

NOT IMPLEMENTED.
"""
from __future__ import annotations
import pandas as pd


class ClinicalFeatureEngineer:
    """Computes derived clinical features from aligned vital-sign data."""

    def __init__(self, config: dict):
        raise NotImplementedError("ClinicalFeatureEngineer: Phase 1 TODO")

    def compute_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        """ΔX_t = X_t - X_{t-1} for each vital."""
        raise NotImplementedError

    def compute_rolling_stats(self, df: pd.DataFrame, windows: list[int] = [5, 15, 30]) -> pd.DataFrame:
        raise NotImplementedError

    def compute_baseline_deviation(self, df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
        """Deviation_t = X_t - PatientBaseline for each vital."""
        raise NotImplementedError

    def compute_news2_score(self, row: pd.Series) -> int:
        """Compute scalar NEWS2 score for a single observation."""
        raise NotImplementedError

    def compute_derived_vitals(self, df: pd.DataFrame) -> pd.DataFrame:
        """MAP, Shock Index, Pulse Pressure."""
        raise NotImplementedError

    def build_feature_matrix(self, df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
        """Full pipeline: compute all features and return unified feature DataFrame."""
        raise NotImplementedError
