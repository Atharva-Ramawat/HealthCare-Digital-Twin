"""
mimic_iv_loader.py
------------------
Phase 1 — MIMIC-IV clinical table loader.

DESIGN CONSTRAINTS
  - Preserves ALL source identifiers (subject_id, hadm_id, stay_id)
  - Preserves raw timestamps WITHOUT modification (remain as strings)
  - Applies NO normalisation, resampling, or feature engineering
  - Raises DataNotAvailableError with clear instructions when files absent
  - Supports both .csv.gz and uncompressed .csv

This module is Phase 1 infrastructure only.
Final cleaning and temporal alignment are Phase 2 concerns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.data_availability import DataNotAvailableError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate item IDs — from public MIMIC-IV documentation (not fabricated).
# Source: https://mimic.mit.edu/docs/iv/
# Status: CANDIDATE — verified in Phase 1 EDA against actual d_items table.
# ---------------------------------------------------------------------------
CHARTEVENTS_VITAL_ITEM_IDS: list[int] = [
    220045,  # Heart Rate
    220050,  # Arterial BP systolic
    220179,  # Non-invasive BP systolic
    220051,  # Arterial BP diastolic
    220180,  # Non-invasive BP diastolic
    220052,  # Arterial BP mean
    220181,  # Non-invasive BP mean
    220277,  # O2 Saturation (SpO2)
    220210,  # Respiratory Rate
    223762,  # Temperature Celsius
    223761,  # Temperature Fahrenheit
    220621,  # Glucose (bedside/chartevents)
]

LABEVENTS_ITEM_IDS: list[int] = [
    50931,   # Glucose
    50813,   # Lactate
    50912,   # Creatinine
    51301,   # White Blood Cells
]

REQUIRED_TABLES: list[str] = [
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/d_items.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/d_labitems.csv.gz",
]

_PHYSIONET_MSG = """
MIMIC-IV data not found at: {path}

To obtain MIMIC-IV Demo (100 patients, free PhysioNet account, no credentialing):
  URL  : https://physionet.org/content/mimic-iv-demo/2.2/
  Place: data/raw/mimic_iv_demo/

To obtain full MIMIC-IV (requires CITI credentialing):
  URL  : https://physionet.org/content/mimiciv/
  Place: data/raw/mimic_iv/
"""

# Expected columns used to warn if a table looks malformed after loading
_EXPECTED_COLUMNS: dict[str, list[str]] = {
    "patients":    ["subject_id", "gender", "anchor_age"],
    "admissions":  ["subject_id", "hadm_id", "admittime", "dischtime"],
    "icustays":    ["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    "chartevents": ["subject_id", "hadm_id", "stay_id", "itemid",
                    "charttime", "value", "valuenum", "valueuom"],
    "labevents":   ["subject_id", "hadm_id", "itemid",
                    "charttime", "value", "valuenum", "valueuom"],
    "d_items":     ["itemid", "label"],
    "d_labitems":  ["itemid", "label"],
}


class MIMICIVLoader:
    """
    Loader for MIMIC-IV ICU clinical tables.

    Parameters
    ----------
    data_root : str
        Path to the MIMIC-IV root directory (contains hosp/ and icu/).
    chunksize : int, optional
        For large tables, read in chunks. None = load all at once.
        Recommended for full MIMIC-IV chartevents (~30 GB).
        Not needed for the Demo.
    """

    def __init__(self, data_root: str, chunksize: Optional[int] = None):
        self.data_root = Path(data_root).resolve()
        self.chunksize = chunksize
        if not self.data_root.exists():
            raise DataNotAvailableError(
                _PHYSIONET_MSG.format(path=self.data_root)
            )

    def _find(self, rel: str) -> Path:
        """Locate a table; supports .csv.gz and uncompressed .csv."""
        for candidate in [
            self.data_root / rel,
            self.data_root / rel.replace(".gz", ""),
        ]:
            if candidate.exists():
                return candidate
        raise DataNotAvailableError(
            f"Table not found: {self.data_root / rel}\n"
            + _PHYSIONET_MSG.format(path=self.data_root)
        )

    def _read(self, rel: str, **kwargs) -> pd.DataFrame:
        path = self._find(rel)
        logger.info("Loading %s", path.name)
        df = pd.read_csv(path, low_memory=False, **kwargs)
        # Warn if expected columns are absent (schema drift guard)
        stem = Path(rel).stem.replace(".csv", "")
        if stem in _EXPECTED_COLUMNS:
            missing = [c for c in _EXPECTED_COLUMNS[stem] if c not in df.columns]
            if missing:
                logger.warning("Table '%s' missing expected columns: %s", stem, missing)
        return df

    def check_availability(self) -> dict[str, bool]:
        """Return {rel_path: exists} for every required table."""
        result = {}
        for t in REQUIRED_TABLES:
            full = self.data_root / t
            alt = self.data_root / t.replace(".gz", "")
            result[t] = full.exists() or alt.exists()
        return result

    # ------------------------------------------------------------------
    # Core table loaders — timestamps preserved as raw strings
    # ------------------------------------------------------------------

    def load_patients(self) -> pd.DataFrame:
        """
        Load hosp/patients.
        Columns: subject_id, gender, anchor_age, anchor_year,
                 anchor_year_group, dod
        Timestamps kept as strings.
        """
        df = self._read("hosp/patients.csv.gz")
        logger.info("patients: %d rows", len(df))
        return df

    def load_admissions(self) -> pd.DataFrame:
        """
        Load hosp/admissions.
        Key columns: subject_id, hadm_id, admittime, dischtime,
                     deathtime, admission_type, insurance, ethnicity
        Timestamps kept as strings.
        """
        df = self._read("hosp/admissions.csv.gz")
        logger.info("admissions: %d rows", len(df))
        return df

    def load_icustays(self) -> pd.DataFrame:
        """
        Load icu/icustays.
        Key columns: subject_id, hadm_id, stay_id, first_careunit,
                     last_careunit, intime, outtime, los
        Timestamps kept as strings.
        """
        df = self._read("icu/icustays.csv.gz")
        logger.info("icustays: %d rows", len(df))
        return df

    def load_d_items(self) -> pd.DataFrame:
        """Load icu/d_items — item dictionary for chartevents."""
        df = self._read("icu/d_items.csv.gz")
        logger.info("d_items: %d items", len(df))
        return df

    def load_d_labitems(self) -> pd.DataFrame:
        """Load hosp/d_labitems — item dictionary for labevents."""
        df = self._read("hosp/d_labitems.csv.gz")
        logger.info("d_labitems: %d items", len(df))
        return df

    def load_chartevents(
        self,
        stay_ids: Optional[list[int]] = None,
        item_ids: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Load icu/chartevents.

        Key columns: subject_id, hadm_id, stay_id, itemid, charttime,
                     storetime, value, valuenum, valueuom, warning

        Timestamps returned as raw strings (object dtype).
        No normalisation. No resampling.

        Parameters
        ----------
        stay_ids : filter to these ICU stay IDs (optional).
        item_ids : filter to these item IDs.
                   Defaults to CHARTEVENTS_VITAL_ITEM_IDS.
        """
        if item_ids is None:
            item_ids = CHARTEVENTS_VITAL_ITEM_IDS

        path = self._find("icu/chartevents.csv.gz")
        logger.info("Loading chartevents from %s", path.name)

        if self.chunksize:
            chunks = []
            for chunk in pd.read_csv(path, chunksize=self.chunksize, low_memory=False):
                if item_ids:
                    chunk = chunk[chunk["itemid"].isin(item_ids)]
                if stay_ids:
                    chunk = chunk[chunk["stay_id"].isin(stay_ids)]
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path, low_memory=False)
            if item_ids:
                df = df[df["itemid"].isin(item_ids)]
            if stay_ids:
                df = df[df["stay_id"].isin(stay_ids)]

        logger.info("chartevents: %d rows after filter", len(df))
        return df

    def load_labevents(
        self,
        hadm_ids: Optional[list[int]] = None,
        item_ids: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Load hosp/labevents.

        Key columns: subject_id, hadm_id, itemid, charttime, storetime,
                     value, valuenum, valueuom, ref_range_lower,
                     ref_range_upper, flag

        Timestamps returned as raw strings.

        Parameters
        ----------
        hadm_ids : filter to these hospital admission IDs (optional).
        item_ids : filter to these item IDs.
                   Defaults to LABEVENTS_ITEM_IDS.
        """
        if item_ids is None:
            item_ids = LABEVENTS_ITEM_IDS

        path = self._find("hosp/labevents.csv.gz")
        logger.info("Loading labevents from %s", path.name)

        if self.chunksize:
            chunks = []
            for chunk in pd.read_csv(path, chunksize=self.chunksize, low_memory=False):
                if item_ids:
                    chunk = chunk[chunk["itemid"].isin(item_ids)]
                if hadm_ids:
                    chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path, low_memory=False)
            if item_ids:
                df = df[df["itemid"].isin(item_ids)]
            if hadm_ids:
                df = df[df["hadm_id"].isin(hadm_ids)]

        logger.info("labevents: %d rows after filter", len(df))
        return df

    def load_inputevents(
        self,
        stay_ids: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Load icu/inputevents — IV fluids and medications.
        Used for vasopressor-initiation label derivation.
        """
        path = self._find("icu/inputevents.csv.gz")
        df = pd.read_csv(path, low_memory=False)
        if stay_ids:
            df = df[df["stay_id"].isin(stay_ids)]
        logger.info("inputevents: %d rows", len(df))
        return df

    def load_procedureevents(
        self,
        stay_ids: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Load icu/procedureevents — mechanical ventilation events etc.
        Used for ventilation-onset label derivation.
        """
        path = self._find("icu/procedureevents.csv.gz")
        df = pd.read_csv(path, low_memory=False)
        if stay_ids:
            df = df[df["stay_id"].isin(stay_ids)]
        logger.info("procedureevents: %d rows", len(df))
        return df

    def load_diagnoses_icd(self) -> pd.DataFrame:
        """
        Load hosp/diagnoses_icd — ICD-9/10 DISCHARGE diagnoses.
        WARNING: Discharge diagnoses — do NOT use as features (leakage risk).
        Used for label construction only.
        """
        df = self._read("hosp/diagnoses_icd.csv.gz")
        logger.info("diagnoses_icd: %d rows", len(df))
        return df

    def load_core_tables(self) -> dict[str, pd.DataFrame]:
        """
        Load all core tables needed for Phase 1 EDA in one call.
        Returns dict: patients, admissions, icustays, d_items, d_labitems
        """
        return {
            "patients":   self.load_patients(),
            "admissions": self.load_admissions(),
            "icustays":   self.load_icustays(),
            "d_items":    self.load_d_items(),
            "d_labitems": self.load_d_labitems(),
        }
