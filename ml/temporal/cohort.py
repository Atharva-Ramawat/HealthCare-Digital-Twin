"""
MIMIC-IV Pulmonary ICU Cohort Extraction Pipeline.
Reproducibly extracts ICU stays for patients with diagnosed pulmonary conditions
(Pneumonia, Acute Respiratory Failure, Asthma, COPD, ARDS, Pulmonary Edema, Pleural Effusion).
"""

import os
import json
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PULMONARY_ICD_KEYWORDS = [
    "pneumon", "respirat", "copd", "asthma", "edema", "effusion",
    "ards", "lung", "bronch", "emphysema", "pleural", "atelectasis",
    "hypoxia", "hypercapnia", "pneumothorax", "aspiration"
]


def extract_pulmonary_cohort(
    hosp_dir: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_iv_demo", "hosp"),
    icu_dir: str = os.path.join(PROJECT_ROOT, "data", "raw", "mimic_iv_demo", "icu"),
    min_stay_duration_hours: float = 6.0,
    output_meta_path: Optional[str] = os.path.join(PROJECT_ROOT, "ml", "temporal", "outputs", "cohort_metadata.json")
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract reproducible pulmonary ICU cohort from MIMIC-IV demo files.
    
    Returns:
        cohort_df: DataFrame with subject_id, hadm_id, stay_id, intime, outtime, los, conditions
        summary_dict: Summary statistics dictionary
    """
    # 1. Load Diagnoses & Dictionary
    diag_path = os.path.join(hosp_dir, "diagnoses_icd.csv.gz")
    d_diag_path = os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz")
    icustays_path = os.path.join(icu_dir, "icustays.csv.gz")
    admissions_path = os.path.join(hosp_dir, "admissions.csv.gz")
    patients_path = os.path.join(hosp_dir, "patients.csv.gz")

    if not os.path.exists(diag_path) or not os.path.exists(icustays_path):
        raise FileNotFoundError(f"Required MIMIC-IV files not found in {hosp_dir} or {icu_dir}")

    diagnoses_df = pd.read_csv(diag_path)
    d_icd_df = pd.read_csv(d_diag_path)
    icustays_df = pd.read_csv(icustays_path)
    admissions_df = pd.read_csv(admissions_path)
    patients_df = pd.read_csv(patients_path)

    # 2. Identify Pulmonary Diagnoses
    diag_merged = pd.merge(diagnoses_df, d_icd_df, on=["icd_code", "icd_version"], how="left")
    pattern = "|".join(PULMONARY_ICD_KEYWORDS)
    pulm_diag = diag_merged[diag_merged["long_title"].str.lower().str.contains(pattern, na=False)].copy()

    # Categorize Conditions
    def categorize_condition(title: str) -> str:
        t = str(title).lower()
        if "respiratory failure" in t: return "Acute Respiratory Failure"
        if "pneumon" in t: return "Pneumonia"
        if "asthma" in t: return "Asthma"
        if "copd" in t or "chronic obstructive" in t: return "COPD"
        if "effusion" in t: return "Pleural Effusion"
        if "edema" in t: return "Pulmonary Edema"
        if "ards" in t or "acute respiratory distress" in t: return "ARDS"
        if "pneumothorax" in t: return "Pneumothorax"
        return "Other Respiratory Condition"

    pulm_diag["condition_category"] = pulm_diag["long_title"].apply(categorize_condition)
    
    # Aggregate conditions per admission
    hadm_conditions = pulm_diag.groupby("hadm_id")["condition_category"].unique().to_dict()
    hadm_condition_str = {k: "; ".join(sorted(v)) for k, v in hadm_conditions.items()}
    pulm_subject_ids = set(pulm_diag["subject_id"].unique())

    # 3. Filter ICU Stays
    pulm_stays = icustays_df[icustays_df["subject_id"].isin(pulm_subject_ids)].copy()
    pulm_stays["intime"] = pd.to_datetime(pulm_stays["intime"])
    pulm_stays["outtime"] = pd.to_datetime(pulm_stays["outtime"])
    pulm_stays["duration_hours"] = (pulm_stays["outtime"] - pulm_stays["intime"]).dt.total_seconds() / 3600.0

    # Filter for valid minimum stay duration
    eligible_stays = pulm_stays[pulm_stays["duration_hours"] >= min_stay_duration_hours].copy()

    # Merge patient demographics and admission info
    cohort_df = pd.merge(eligible_stays, patients_df[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    cohort_df["pulmonary_conditions"] = cohort_df["hadm_id"].map(hadm_condition_str).fillna("Documented Respiratory Disease")

    # Sort cohort by patient and stay
    cohort_df = cohort_df.sort_values(["subject_id", "intime"]).reset_index(drop=True)

    # 4. Compute Summary Statistics
    condition_counts = pulm_diag["condition_category"].value_counts().to_dict()
    summary = {
        "total_database_patients": int(len(patients_df)),
        "pulmonary_patients_count": int(cohort_df["subject_id"].nunique()),
        "pulmonary_admissions_count": int(cohort_df["hadm_id"].nunique()),
        "pulmonary_icustays_count": int(len(cohort_df)),
        "min_stay_duration_hours_filter": min_stay_duration_hours,
        "median_icu_los_hours": round(float(cohort_df["duration_hours"].median()), 2),
        "mean_icu_los_hours": round(float(cohort_df["duration_hours"].mean()), 2),
        "condition_distribution": condition_counts,
        "patients_list": sorted(cohort_df["subject_id"].unique().tolist())
    }

    if output_meta_path:
        os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
        with open(output_meta_path, "w") as f:
            json.dump(summary, f, indent=2)

    return cohort_df, summary


if __name__ == "__main__":
    df, meta = extract_pulmonary_cohort()
    print(f"Extracted Pulmonary Cohort: {meta['pulmonary_patients_count']} patients across {meta['pulmonary_icustays_count']} ICU stays.")
