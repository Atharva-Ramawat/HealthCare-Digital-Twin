"""
mimic_iv_loader.py
------------------
PHASE 1 PLACEHOLDER
Loads MIMIC-IV clinical tables (patients, admissions, chartevents, labevents,
icustays) and returns a standardised PatientRecord list.

Key responsibilities:
  - Read MIMIC-IV CSV/Parquet files from data/raw/mimic_iv/
  - Filter to ICU stays
  - Return tidy pandas DataFrames with standardised column names
  - Support subject_id / hadm_id / stay_id selection

NOT IMPLEMENTED. Do not call in production.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd


class MIMICIVLoader:
    """Loader for MIMIC-IV ICU clinical tables."""

    def __init__(self, data_root: str):
        # TODO Phase 1: validate path, check for required CSVs
        self.data_root = data_root
        raise NotImplementedError("MIMICIVLoader: Phase 1 TODO")

    def load_icu_stays(self) -> pd.DataFrame:
        """Return icustays table filtered to adults."""
        raise NotImplementedError

    def load_chartevents(self, stay_ids: list[int]) -> pd.DataFrame:
        """Return chartevents for specified ICU stay IDs."""
        raise NotImplementedError

    def load_labevents(self, hadm_ids: list[int]) -> pd.DataFrame:
        """Return labevents for specified hospital admission IDs."""
        raise NotImplementedError
