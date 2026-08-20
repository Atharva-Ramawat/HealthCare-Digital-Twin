"""
normalizer.py
-------------
PHASE 1 PLACEHOLDER
Normalises vital-sign values for model consumption.

Strategies:
  - Min-Max scaling (based on physiological bounds from config)
  - Z-score normalisation (per-patient or population-level)
  - Patient-specific adaptive normalisation (relative to personal baseline)

NOT IMPLEMENTED.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class VitalsNormalizer:
    """Normalises clinical vitals for ML model consumption."""

    def __init__(self, strategy: str = "minmax", config: dict = None):
        raise NotImplementedError("VitalsNormalizer: Phase 1 TODO")

    def fit(self, df: pd.DataFrame) -> "VitalsNormalizer":
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
