"""
temporal_aligner.py
--------------------
PHASE 2 PLACEHOLDER
Aligns irregularly-sampled clinical observations to a configurable uniform time grid.

Key responsibilities:
  - Resample to a configurable time resolution (NOT hardcoded to 1 minute)
  - The resolution is set in configs/settings.yaml -> preprocessing.temporal_resolution_minutes
  - This value is TBD -- it will be determined after Phase 1 dataset profiling
  - Handle multi-source timestamp reconciliation
  - Produce a unified patient timeline DataFrame indexed by (patient_id, timestamp)

IMPORTANT: Different clinical datasets have very different native sampling rates:
  - MIMIC-IV chartevents: typically hourly for many vital parameters
  - MIMIC-III waveform: up to 125 Hz
  - VitalDB: 1-500 Hz depending on the track
The correct resolution to use is a dataset-dependent decision, not a fixed assumption.

NOT IMPLEMENTED.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd


class TemporalAligner:
    """
    Aligns multi-frequency clinical observations to a configurable uniform time grid.

    resolution_minutes: configurable -- to be set after Phase 1 EDA.
    Do NOT default to 1 minute without confirming against the actual dataset.
    """

    def __init__(self, resolution_minutes: Optional[int] = None):
        """
        Args:
            resolution_minutes: Target resampling resolution in minutes.
                                 None means not yet configured (Phase 1 TBD).
        """
        if resolution_minutes is None:
            import warnings
            warnings.warn(
                "TemporalAligner: resolution_minutes not set. "
                "Set configs/settings.yaml -> preprocessing.temporal_resolution_minutes "
                "after Phase 1 dataset profiling.",
                UserWarning
            )
        self.resolution_minutes = resolution_minutes
        raise NotImplementedError("TemporalAligner: Phase 2 TODO")

    def align(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample observations to uniform time grid.
        Uses self.resolution_minutes -- must not be None at call time.
        """
        raise NotImplementedError
