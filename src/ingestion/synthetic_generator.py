"""
synthetic_generator.py (NEW LOCATION)
--------------------------------------
PHASE 0 / ACTIVE
Generates synthetic physiological vital-sign time series for development,
testing, and demo mode when real clinical data is unavailable.

Migrated and enhanced from legacy prototype.
Trajectories: normal, septic_shock, ards
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
# TODO Phase 1: load vital metadata from configs/settings.yaml instead of hardcoding
# See legacy prototype at docs/legacy_prototype/synthetic_generator.py

class SyntheticClinicalGenerator:
    """
    Generates realistic ICU patient vital-sign time series.
    ACTIVE - used as fallback in simulation layer.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_patient_series(
        self,
        patient_id: str = "PAT-001",
        trajectory: str = "normal",
        num_steps: int = 200,
        time_offset_min: int = 0,
    ) -> pd.DataFrame:
        """
        Generate a time-indexed DataFrame of vital signs.
        TODO Phase 1: pull vital bounds from config loader, not hardcoded values.
        """
        raise NotImplementedError("Phase 1: port from legacy prototype with config integration")

    def generate_ward_dataset(self, patient_profiles: list[dict], num_steps: int = 300) -> pd.DataFrame:
        """Generate multi-patient ward dataset."""
        raise NotImplementedError("Phase 1: port from legacy prototype with config integration")
