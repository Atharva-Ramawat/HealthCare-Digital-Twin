"""
MIMIC-CXR / CheXpert Label Mapping, Uncertainty Policies, and Patient-Level Split Verification.
Defines official thoracic findings, uncertainty resolution, and positive class imbalance weighting.
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd


# Official CheXpert 14 Findings in MIMIC-CXR-JPG
MIMIC_CHEXPERT_14_FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices"
]

# Target 8 Pulmonary / Thoracic Findings for Digital Twin Clinical Focus
TARGET_PULMONARY_CLASSES = [
    "Pneumonia",
    "Pleural Effusion",
    "Atelectasis",
    "Consolidation",
    "Edema",
    "Pneumothorax",
    "Cardiomegaly",
    "No Finding"
]

# Baseline Classification Thresholds per Pathology (optimized for sensitivity / F1)
DEFAULT_PATHOLOGY_THRESHOLDS = {
    "Pneumonia": 0.35,
    "Pleural Effusion": 0.40,
    "Atelectasis": 0.38,
    "Consolidation": 0.35,
    "Edema": 0.38,
    "Pneumothorax": 0.28,
    "Cardiomegaly": 0.42,
    "No Finding": 0.50
}


class UncertaintyPolicy(str, Enum):
    """
    Policy for handling uncertain (-1.0) CheXpert / MIMIC-CXR radiology labels.
    - U_ZERO: Treat all uncertain labels (-1) as negative (0.0). Conservative approach.
    - U_ONES: Treat uncertain labels (-1) as positive (1.0). High-sensitivity approach.
    - U_CHEXPERT: CheXpert competition policy (1 for Atelectasis, Edema, Effusion; 0 for others).
    """
    U_ZERO = "u_zero"
    U_ONES = "u_ones"
    U_CHEXPERT = "u_chexpert"


def map_chexpert_labels(
    raw_df: pd.DataFrame,
    target_classes: List[str] = TARGET_PULMONARY_CLASSES,
    policy: UncertaintyPolicy = UncertaintyPolicy.U_ZERO
) -> pd.DataFrame:
    """
    Standardize raw CheXpert / MIMIC-CXR label table to binary target matrix [0.0, 1.0].
    
    Raw Values in MIMIC-CXR:
      1.0  -> Positive
      0.0  -> Explicitly Negative
     -1.0  -> Uncertain / Equivocal
      NaN  -> Unmentioned / Negative
    """
    mapped_df = raw_df.copy()

    for col in target_classes:
        if col not in mapped_df.columns:
            # If a target column is missing, initialize as 0.0
            mapped_df[col] = 0.0
            continue

        series = mapped_df[col].fillna(0.0)

        if policy == UncertaintyPolicy.U_ZERO:
            mapped_df[col] = series.apply(lambda x: 1.0 if x == 1.0 else 0.0)
        elif policy == UncertaintyPolicy.U_ONES:
            mapped_df[col] = series.apply(lambda x: 1.0 if x in (1.0, -1.0) else 0.0)
        elif policy == UncertaintyPolicy.U_CHEXPERT:
            if col in ["Atelectasis", "Edema", "Pleural Effusion"]:
                mapped_df[col] = series.apply(lambda x: 1.0 if x in (1.0, -1.0) else 0.0)
            else:
                mapped_df[col] = series.apply(lambda x: 1.0 if x == 1.0 else 0.0)

    # Ensure No Finding consistency: if any pathology is 1.0, No Finding must be 0.0
    pathology_cols = [c for c in target_classes if c != "No Finding"]
    if "No Finding" in target_classes:
        has_pathology = (mapped_df[pathology_cols] == 1.0).any(axis=1)
        mapped_df.loc[has_pathology, "No Finding"] = 0.0
        # If no pathology is present and No Finding wasn't explicitly 0, set to 1.0
        no_pathology = ~has_pathology
        mapped_df.loc[no_pathology, "No Finding"] = 1.0

    return mapped_df


def verify_patient_level_split(df: pd.DataFrame, subject_col: str = "subject_id", split_col: str = "split") -> Dict[str, any]:
    """
    Verify strictly zero data leakage across patient subject IDs.
    Returns split statistics and asserts disjoint patient sets between train, validate, and test.
    """
    if split_col not in df.columns or subject_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{subject_col}' and '{split_col}' columns for split verification.")

    splits = df[split_col].unique()
    patients_by_split: Dict[str, Set[str]] = {}
    samples_by_split: Dict[str, int] = {}

    for s in splits:
        pats = set(df[df[split_col] == s][subject_col].astype(str).unique())
        patients_by_split[s] = pats
        samples_by_split[s] = len(df[df[split_col] == s])

    # Check pair-wise intersections
    overlaps = {}
    split_names = list(patients_by_split.keys())
    has_leakage = False
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            intersection = patients_by_split[s1].intersection(patients_by_split[s2])
            if len(intersection) > 0:
                overlaps[f"{s1}_vs_{s2}"] = len(intersection)
                has_leakage = True

    return {
        "is_valid": not has_leakage,
        "splits": split_names,
        "patients_count": {s: len(p) for s, p in patients_by_split.items()},
        "samples_count": samples_by_split,
        "leakage_overlaps": overlaps
    }


def calculate_positive_class_weights(
    labels_matrix: np.ndarray,
    clamp_min: float = 1.0,
    clamp_max: float = 25.0
) -> np.ndarray:
    """
    Calculate pos_weight for BCEWithLogitsLoss to counter medical class imbalance.
    pos_weight = (num_negative / num_positive)
    """
    num_samples = labels_matrix.shape[0]
    num_positives = np.sum(labels_matrix, axis=0)
    num_negatives = num_samples - num_positives

    # Avoid division by zero
    weights = np.where(num_positives > 0, num_negatives / (num_positives + 1e-6), 1.0)
    weights = np.clip(weights, clamp_min, clamp_max).astype(np.float32)
    return weights


if __name__ == "__main__":
    # Self-test mapping and verification
    dummy_data = {
        "subject_id": [101, 101, 102, 103, 104, 105],
        "study_id": [501, 502, 503, 504, 505, 506],
        "split": ["train", "train", "train", "validate", "test", "test"],
        "Pneumonia": [1.0, 1.0, -1.0, 0.0, 1.0, 0.0],
        "Pleural Effusion": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "Atelectasis": [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "No Finding": [np.nan, 0.0, 0.0, 0.0, 0.0, 1.0]
    }
    df = pd.DataFrame(dummy_data)
    mapped = map_chexpert_labels(df, TARGET_PULMONARY_CLASSES, UncertaintyPolicy.U_ZERO)
    print("Mapped DataFrame target classes:")
    print(mapped[TARGET_PULMONARY_CLASSES])
    
    split_info = verify_patient_level_split(mapped)
    print("\nSplit Verification:", split_info)
    assert split_info["is_valid"] is True, "Data leakage detected!"
    print("Label module self-test passed successfully!")
