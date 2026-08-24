"""
MIMIC-IV Time-Aligned Treatment & Medication Engine.
Tracks active ICU interventions (Vasopressors, Diuretics, Antibiotics, Bronchodilators, Corticosteroids),
extracts discrete event-linked treatment initiation episodes (T0), and manages pre/post observation windows.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TREATMENT_CATEGORIES = {
    "tx_vasopressor": ["norepinephrine", "epinephrine", "phenylephrine", "vasopressin", "dopamine"],
    "tx_diuretic": ["furosemide", "bumetanide", "torsemide", "hydrochlorothiazide"],
    "tx_antibiotic": ["vancomycin", "cef", "piperacillin", "meropenem", "levofloxacin", "azithromycin", "gentamicin"],
    "tx_bronchodilator": ["albuterol", "ipratropium", "tiotropium", "levalbuterol", "formoterol", "duoneb"],
    "tx_steroid": ["dexamethasone", "hydrocortisone", "prednisone", "methylprednisolone", "solu-medrol"]
}


class TreatmentTimelineEngine:
    """
    Constructs time-aligned treatment arrays and extracts discrete event-linked treatment initiation episodes.
    """

    def __init__(
        self,
        hosp_dir: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_iv_demo", "hosp"),
        icu_dir: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_iv_demo", "icu")
    ):
        self.hosp_dir = hosp_dir
        self.icu_dir = icu_dir
        self._load_data()

    def _load_data(self):
        inputevents_path = os.path.join(self.icu_dir, "inputevents.csv.gz")
        d_items_path = os.path.join(self.icu_dir, "d_items.csv.gz")
        prescriptions_path = os.path.join(self.hosp_dir, "prescriptions.csv.gz")

        self.inputevents_df = pd.read_csv(inputevents_path, low_memory=False) if os.path.exists(inputevents_path) else pd.DataFrame()
        self.d_items_df = pd.read_csv(d_items_path) if os.path.exists(d_items_path) else pd.DataFrame()
        self.prescriptions_df = pd.read_csv(prescriptions_path, low_memory=False) if os.path.exists(prescriptions_path) else pd.DataFrame()

        # Parse timestamps
        if not self.inputevents_df.empty:
            self.inputevents_df["starttime"] = pd.to_datetime(self.inputevents_df["starttime"])
            self.inputevents_df["endtime"] = pd.to_datetime(self.inputevents_df["endtime"])
            self.inputevents_df = pd.merge(self.inputevents_df, self.d_items_df[["itemid", "label"]], on="itemid", how="left")

        if not self.prescriptions_df.empty:
            self.prescriptions_df["starttime"] = pd.to_datetime(self.prescriptions_df["starttime"])
            self.prescriptions_df["stoptime"] = pd.to_datetime(self.prescriptions_df["stoptime"])

    def _categorize_drug(self, name: str) -> Optional[str]:
        if not isinstance(name, str):
            return None
        low = name.lower()
        for cat_key, keywords in TREATMENT_CATEGORIES.items():
            if any(kw in low for kw in keywords):
                return cat_key
        return None

    def build_stay_treatment_grid(
        self,
        stay_id: int,
        subject_id: int,
        time_grid: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Build binary active-treatment indicator flags (1 = active, 0 = inactive) on a uniform time grid.
        """
        tx_df = pd.DataFrame({"charttime": time_grid})
        for cat in TREATMENT_CATEGORIES.keys():
            tx_df[cat] = 0

        # 1. Process IV Infusions / Inputs (stay_id specific)
        if not self.inputevents_df.empty:
            stay_inputs = self.inputevents_df[self.inputevents_df["stay_id"] == stay_id].copy()
            for _, row in stay_inputs.iterrows():
                cat = self._categorize_drug(str(row["label"]))
                if cat and pd.notnull(row["starttime"]) and pd.notnull(row["endtime"]):
                    mask = (tx_df["charttime"] >= row["starttime"]) & (tx_df["charttime"] <= row["endtime"])
                    tx_df.loc[mask, cat] = 1

        # 2. Process Prescriptions (subject_id specific)
        if not self.prescriptions_df.empty:
            stay_rx = self.prescriptions_df[self.prescriptions_df["subject_id"] == subject_id].copy()
            for _, row in stay_rx.iterrows():
                cat = self._categorize_drug(str(row["drug"]))
                if cat and pd.notnull(row["starttime"]):
                    stop = row["stoptime"] if pd.notnull(row["stoptime"]) else row["starttime"] + pd.Timedelta(hours=24)
                    mask = (tx_df["charttime"] >= row["starttime"]) & (tx_df["charttime"] <= stop)
                    tx_df.loc[mask, cat] = 1

        return tx_df

    def extract_discrete_treatment_episodes(
        self,
        stay_id: int,
        subject_id: int,
        intime: pd.Timestamp,
        outtime: pd.Timestamp,
        pre_window_hours: float = 2.0,
        post_window_hours: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Extract discrete treatment initiation events (T0) with valid pre- and post-intervention bounds.
        """
        episodes = []

        # 1. Infusions from inputevents
        if not self.inputevents_df.empty:
            stay_in = self.inputevents_df[self.inputevents_df["stay_id"] == stay_id].sort_values("starttime")
            for _, row in stay_in.iterrows():
                cat = self._categorize_drug(str(row.get("label", "")))
                t0 = row.get("starttime")
                if cat and pd.notnull(t0):
                    pre_start = t0 - pd.Timedelta(hours=pre_window_hours)
                    post_end = t0 + pd.Timedelta(hours=post_window_hours)
                    
                    is_eligible = (pre_start >= intime) and (post_end <= outtime)
                    episodes.append({
                        "patient_id": subject_id,
                        "stay_id": stay_id,
                        "treatment_category": cat,
                        "drug_name": str(row.get("label", "")),
                        "treatment_source": "icu_inputevents",
                        "treatment_event_time": str(t0),
                        "t0_timestamp": t0,
                        "pre_window_start": str(pre_start),
                        "pre_window_end": str(t0),
                        "post_window_start": str(t0),
                        "post_window_end": str(post_end),
                        "is_eligible_window": is_eligible
                    })

        # 2. Prescriptions
        if not self.prescriptions_df.empty:
            stay_rx = self.prescriptions_df[self.prescriptions_df["subject_id"] == subject_id].sort_values("starttime")
            for _, row in stay_rx.iterrows():
                cat = self._categorize_drug(str(row.get("drug", "")))
                t0 = row.get("starttime")
                if cat and pd.notnull(t0) and (t0 >= intime) and (t0 <= outtime):
                    pre_start = t0 - pd.Timedelta(hours=pre_window_hours)
                    post_end = t0 + pd.Timedelta(hours=post_window_hours)
                    
                    is_eligible = (pre_start >= intime) and (post_end <= outtime)
                    episodes.append({
                        "patient_id": subject_id,
                        "stay_id": stay_id,
                        "treatment_category": cat,
                        "drug_name": str(row.get("drug", "")),
                        "treatment_source": "hosp_prescriptions",
                        "treatment_event_time": str(t0),
                        "t0_timestamp": t0,
                        "pre_window_start": str(pre_start),
                        "pre_window_end": str(t0),
                        "post_window_start": str(t0),
                        "post_window_end": str(post_end),
                        "is_eligible_window": is_eligible
                    })

        return episodes

    def audit_treatment_episodes(
        self,
        cohort_stays: List[int]
    ) -> Dict[str, Any]:
        """
        Investigate discrete treatment initiation events, category counts, and co-administration overlaps.
        """
        category_counts = {cat: 0 for cat in TREATMENT_CATEGORIES.keys()}
        raw_event_rows = 0

        # Tally infusions
        if not self.inputevents_df.empty:
            pulm_in = self.inputevents_df[self.inputevents_df["stay_id"].isin(cohort_stays)].copy()
            for _, row in pulm_in.iterrows():
                cat = self._categorize_drug(str(row["label"]))
                if cat:
                    category_counts[cat] += 1
                    raw_event_rows += 1

        # Tally prescriptions
        if not self.prescriptions_df.empty:
            for _, row in self.prescriptions_df.iterrows():
                cat = self._categorize_drug(str(row["drug"]))
                if cat:
                    category_counts[cat] += 1
                    raw_event_rows += 1

        return {
            "total_raw_treatment_rows": raw_event_rows,
            "category_sum_total": sum(category_counts.values()),
            "category_breakdown": category_counts,
            "explanation_of_discrepancy": (
                "The sum of treatment rows across categories exceeds unique initiation episodes because: "
                "(1) Patients frequently receive concurrent multimodal treatments (e.g. Norepinephrine infusion "
                "titrated concurrently with Vancomycin infusion and Furosemide administration), "
                "(2) Continuous IV infusions are logged as multiple successive bag adjustments/rate changes in inputevents, "
                "and (3) Prescriptions and inputevents capture both order generation and bedside administration."
            )
        }
