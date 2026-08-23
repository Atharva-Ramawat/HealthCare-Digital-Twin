"""
MIMIC-IV Patient-Level Splitting & Data Leakage Prevention Engine.
Enforces strictly disjoint partitioning by subject_id (70% Train, 15% Validation, 15% Test)
and performs automated mathematical assertions against cross-split patient contamination.
"""

import os
import json
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def create_patient_level_split(
    cohort_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_json_path: Optional[str] = os.path.join(PROJECT_ROOT, "ml", "temporal", "outputs", "split_summary.json")
) -> Tuple[Dict[str, Set[int]], Dict[str, Any]]:
    """
    Split cohort strictly by subject_id.
    
    Returns:
        split_dict: Dict mapping 'train', 'validate', 'test' to sets of subject_ids.
        summary_dict: Complete split statistics and leakage validation status.
    """
    unique_subjects = sorted(cohort_df["subject_id"].unique().tolist())
    n_patients = len(unique_subjects)

    if n_patients == 0:
        raise ValueError("Cohort dataframe contains zero patients.")

    # Deterministic patient shuffle
    np.random.seed(random_seed)
    shuffled_subjects = np.random.permutation(unique_subjects).tolist()

    n_train = int(np.round(train_ratio * n_patients))
    n_val = int(np.round(val_ratio * n_patients))
    # Remaining assigned to test
    n_test = n_patients - n_train - n_val

    train_subjects = set(shuffled_subjects[:n_train])
    val_subjects = set(shuffled_subjects[n_train:n_train + n_val])
    test_subjects = set(shuffled_subjects[n_train + n_val:])

    # 1. Mandatory Zero-Leakage Assertions
    train_val_overlap = train_subjects.intersection(val_subjects)
    train_test_overlap = train_subjects.intersection(test_subjects)
    val_test_overlap = val_subjects.intersection(test_subjects)

    assert len(train_val_overlap) == 0, f"LEAKAGE DETECTED: Train & Validation share patients: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"LEAKAGE DETECTED: Train & Test share patients: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"LEAKAGE DETECTED: Validation & Test share patients: {val_test_overlap}"
    assert len(train_subjects) + len(val_subjects) + len(test_subjects) == n_patients, "Patient count mismatch across splits."

    # 2. Count ICU Stays per Split
    train_stays = set(cohort_df[cohort_df["subject_id"].isin(train_subjects)]["stay_id"].unique())
    val_stays = set(cohort_df[cohort_df["subject_id"].isin(val_subjects)]["stay_id"].unique())
    test_stays = set(cohort_df[cohort_df["subject_id"].isin(test_subjects)]["stay_id"].unique())

    split_dict = {
        "train": train_subjects,
        "validate": val_subjects,
        "test": test_subjects
    }

    summary_dict = {
        "total_patients": n_patients,
        "total_icustays": int(len(cohort_df)),
        "random_seed": random_seed,
        "leakage_audit": {
            "passed": True,
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap)
        },
        "splits": {
            "train": {
                "patients_count": len(train_subjects),
                "patients_percentage": round(len(train_subjects) / n_patients * 100.0, 2),
                "icustays_count": len(train_stays),
                "patient_ids": sorted(list(train_subjects))
            },
            "validate": {
                "patients_count": len(val_subjects),
                "patients_percentage": round(len(val_subjects) / n_patients * 100.0, 2),
                "icustays_count": len(val_stays),
                "patient_ids": sorted(list(val_subjects))
            },
            "test": {
                "patients_count": len(test_subjects),
                "patients_percentage": round(len(test_subjects) / n_patients * 100.0, 2),
                "icustays_count": len(test_stays),
                "patient_ids": sorted(list(test_subjects))
            }
        }
    }

    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(summary_dict, f, indent=2)

    return split_dict, summary_dict


if __name__ == "__main__":
    from ml.temporal.cohort import extract_pulmonary_cohort
    cohort, _ = extract_pulmonary_cohort()
    splits, summary = create_patient_level_split(cohort)
    print(f"Patient Split Summary: Train={summary['splits']['train']['patients_count']}, "
          f"Val={summary['splits']['validate']['patients_count']}, "
          f"Test={summary['splits']['test']['patients_count']}. Zero leakage: {summary['leakage_audit']['passed']}")
