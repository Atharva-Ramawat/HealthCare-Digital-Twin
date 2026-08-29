"""
MIMIC-CXR / CheXpert Label Mapping, Uncertainty Policies, and Patient-Level Split Verification.
Defines official thoracic findings, uncertainty resolution, and positive class imbalance weighting.
Supports both tabular CheXpert ground-truth and regex-based clinical NLP report parsing.
"""

import os
import re
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

# Clinical NLP Patterns for Report Finding Extraction
REPORT_NLP_PATTERNS = {
    "Pneumonia": (
        re.compile(r"\b(pneumonia|bronchopneumonia|infectious\s+(infiltrate|process)|lobar\s+infiltrate)\b", re.I),
        re.compile(r"\b(no\s+(evidence\s+of\s+)?pneumonia|without\s+pneumonia|free\s+of\s+pneumonia|no\s+acute\s+pneumonia|unlikely\s+pneumonia)\b", re.I)
    ),
    "Pleural Effusion": (
        re.compile(r"\b(pleural\s+effusion|effusions?|pleural\s+fluid|blunting\s+of\s+(the\s+)?(costophrenic|cp)\s+angle)\b", re.I),
        re.compile(r"\b(no\s+(evidence\s+of\s+)?(pleural\s+)?effusions?|without\s+(pleural\s+)?effusions?|no\s+pleural\s+fluid|clear\s+costophrenic\s+angles?)\b", re.I)
    ),
    "Atelectasis": (
        re.compile(r"\b(atelectas(is|es)|atelectatic|volume\s+loss|compressive\s+atelectasis|bibasilar\s+atelectasis)\b", re.I),
        re.compile(r"\b(no\s+(evidence\s+of\s+)?atelectas(is|es)|without\s+atelectasis|clear\s+of\s+atelectasis)\b", re.I)
    ),
    "Consolidation": (
        re.compile(r"\b(consolidation|airspace\s+disease|airspace\s+opacity|focal\s+opacity|dense\s+opacity)\b", re.I),
        re.compile(r"\b(no\s+(evidence\s+of\s+)?(focal\s+)?consolidation|without\s+consolidation|no\s+airspace\s+(disease|opacity))\b", re.I)
    ),
    "Edema": (
        re.compile(r"\b(pulmonary\s+edema|interstitial\s+edema|vascular\s+congestion|fluid\s+overload|congestive\s+heart\s+failure|chf|vascular\s+engorgement)\b", re.I),
        re.compile(r"\b(no\s+(evidence\s+of\s+)?(pulmonary\s+)?edema|without\s+edema|no\s+(overt\s+)?(vascular\s+)?congestion|no\s+vascular\s+engorgement)\b", re.I)
    ),
    "Pneumothorax": (
        re.compile(r"\b(pneumothorax|pneumothoraces|apical\s+pneumothorax)\b", re.I),
        re.compile(r"\b(no\s+(evidence\s+of\s+)?pneumothor(ax|aces)|without\s+pneumothorax|no\s+ptx|negative\s+for\s+pneumothorax)\b", re.I)
    ),
    "Cardiomegaly": (
        re.compile(r"\b(cardiomegaly|enlarged\s+heart|cardiac\s+(silhouette\s+)?is\s+(moderately\s+|severely\s+|mildly\s+)?enlarged|heart\s+is\s+(moderately\s+|severely\s+|mildly\s+)?enlarged|cardiothoracic\s+ratio\s+is\s+increased)\b", re.I),
        re.compile(r"\b(heart\s+size\s+is\s+normal|normal\s+heart\s+size|normal\s+cardiac\s+silhouette|cardiomediastinal\s+silhouette\s+is\s+normal|no\s+cardiomegaly)\b", re.I)
    ),
    "No Finding": (
        re.compile(r"\b(no\s+acute\s+(cardiopulmonary|intrathoracic)\s+(process|abnormality|disease)|normal\s+chest|clear\s+lungs?|lungs\s+are\s+clear|unremarkable|within\s+normal\s+limits)\b", re.I),
        re.compile(r"$^", re.I)
    )
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


def extract_labels_from_radiology_report(text: str, target_classes: List[str] = TARGET_PULMONARY_CLASSES) -> Dict[str, float]:
    """
    Extract multi-label pulmonary target vectors from radiology report text using clinical NLP rules.
    Accurately handles medical negation phrases ('no evidence of', 'without', 'clear of').
    """
    labels = {col: 0.0 for col in target_classes}
    if not text or not isinstance(text, str):
        if "No Finding" in labels:
            labels["No Finding"] = 1.0
        return labels

    for finding in target_classes:
        if finding not in REPORT_NLP_PATTERNS:
            continue
        pos_pat, neg_pat = REPORT_NLP_PATTERNS[finding]
        pos_matches = list(pos_pat.finditer(text))
        if not pos_matches:
            continue

        is_pos = False
        for m in pos_matches:
            # Find start of clause/sentence or up to 100 chars before match
            pre_text = text[:m.start()]
            sentence_start = max(0, pre_text.rfind("."), pre_text.rfind(";"), pre_text.rfind("\n"))
            start = max(sentence_start, m.start() - 100)
            window = text[start:m.end()]
            if neg_pat.search(window) or re.search(r"\b(no|without|negative\s+for|free\s+of|rules?\s+out|clear\s+of|denies)\b", window, re.I):
                continue
            is_pos = True
            break

        labels[finding] = 1.0 if is_pos else 0.0

    # Ensure No Finding consistency
    pathology_cols = [c for c in target_classes if c != "No Finding"]
    if "No Finding" in target_classes:
        has_pathology = any(labels[c] == 1.0 for c in pathology_cols)
        if has_pathology:
            labels["No Finding"] = 0.0
        else:
            labels["No Finding"] = 1.0

    return labels


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

    weights = np.where(num_positives > 0, num_negatives / (num_positives + 1e-6), 1.0)
    weights = np.clip(weights, clamp_min, clamp_max).astype(np.float32)
    return weights
