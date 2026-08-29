"""
MIMIC-CXR PyTorch Dataset & DataLoader Pipeline.
Supports real JPG disk image resolution from MIMIC-CXR augmentation CSVs,
multi-label pulmonary target tensor formulation, class imbalance weights,
and strict fail-fast enforcement against silent synthetic fallback during real training.
"""

import os
import sys
import ast
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ml.cxr.labels import (
    TARGET_PULMONARY_CLASSES,
    UncertaintyPolicy,
    map_chexpert_labels,
    extract_labels_from_radiology_report,
    calculate_positive_class_weights,
    verify_patient_level_split
)
from ml.cxr.preprocessing import get_cxr_transforms, load_and_preprocess_image, generate_synthetic_chest_radiograph


def parse_augmented_cxr_dataframe(
    df_or_path: Union[pd.DataFrame, str],
    images_dir: Optional[str] = None,
    split_name: str = "train",
    target_classes: List[str] = TARGET_PULMONARY_CLASSES,
    filter_missing: bool = True,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Parse augmented MIMIC-CXR DataFrame/CSV (containing stringified image and text lists)
    into a flattened, image-level tabular DataFrame with verified disk paths and labels.
    """
    if isinstance(df_or_path, str):
        if not os.path.exists(df_or_path):
            raise FileNotFoundError(f"CXR metadata file not found at: {df_or_path}")
        raw_df = pd.read_csv(df_or_path, nrows=max_rows)
    else:
        raw_df = df_or_path.iloc[:max_rows].copy() if max_rows else df_or_path.copy()

    # Check if already a flattened tabular DataFrame with target classes
    if "image_path" in raw_df.columns and all(c in raw_df.columns for c in target_classes):
        expanded_df = raw_df.copy()
        if "split" not in expanded_df.columns:
            expanded_df["split"] = split_name
        return expanded_df

    records = []
    images_dir_abs = os.path.abspath(images_dir) if images_dir else None

    for _, row in raw_df.iterrows():
        subj_id = str(row.get("subject_id", ""))

        # 1. Parse Image List
        img_col = row.get("image", "[]")
        if isinstance(img_col, str) and img_col.strip().startswith("["):
            try:
                img_list = ast.literal_eval(img_col)
            except Exception:
                img_list = [img_col]
        elif isinstance(img_col, list):
            img_list = img_col
        elif pd.notna(img_col) and str(img_col).strip():
            img_list = [str(img_col)]
        else:
            img_list = []

        if not img_list:
            continue

        # 2. Parse Text Reports List
        text_col = row.get("text", "[]")
        if isinstance(text_col, str) and text_col.strip().startswith("["):
            try:
                text_list = ast.literal_eval(text_col)
            except Exception:
                text_list = [text_col]
        elif isinstance(text_col, list):
            text_list = text_col
        elif pd.notna(text_col) and str(text_col).strip():
            text_list = [str(text_col)]
        else:
            text_list = []

        # 3. Parse View Positions List if available
        view_col = row.get("view", "[]")
        if isinstance(view_col, str) and view_col.strip().startswith("["):
            try:
                view_list = ast.literal_eval(view_col)
            except Exception:
                view_list = [view_col]
        elif isinstance(view_col, list):
            view_list = view_col
        else:
            view_list = [str(view_col)] if pd.notna(view_col) else []

        # Map study_ids to distinct text reports
        study_order = list(dict.fromkeys([p.split("/")[3] for p in img_list if len(p.split("/")) > 3]))
        study_to_text = {}
        for s_idx, s_id in enumerate(study_order):
            study_to_text[s_id] = text_list[s_idx] if s_idx < len(text_list) else ""

        for img_idx, img_rel in enumerate(img_list):
            img_rel_clean = str(img_rel).strip().replace("\\", "/")
            parts = img_rel_clean.split("/")
            study_id = parts[3] if len(parts) > 3 else f"STD-{img_idx:04d}"
            dicom_id = parts[4].replace(".jpg", "").replace(".png", "") if len(parts) > 4 else f"DCM-{img_idx:04d}"
            view_pos = view_list[img_idx] if img_idx < len(view_list) else "PA"

            report = study_to_text.get(study_id, text_list[0] if text_list else "")
            labels_dict = extract_labels_from_radiology_report(report, target_classes=target_classes)

            # Resolve physical path
            if images_dir_abs:
                full_path = os.path.join(images_dir_abs, img_rel_clean)
                exists = os.path.exists(full_path)
            else:
                full_path = img_rel_clean
                exists = os.path.exists(full_path)

            if filter_missing and not exists:
                continue

            rec = {
                "subject_id": subj_id,
                "study_id": study_id,
                "dicom_id": dicom_id,
                "view_position": view_pos,
                "image_path": img_rel_clean,
                "resolved_path": full_path,
                "file_exists": exists,
                "split": split_name,
                "report_text": report,
                **labels_dict
            }
            records.append(rec)

    expanded_df = pd.DataFrame(records)
    return expanded_df


class MIMICCXRDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-CXR multi-label pulmonary radiograph analysis.
    Implements lazy disk loading with strict verification and zero silent synthetic fallback in real training.
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, str],
        images_dir: Optional[str] = None,
        split: str = "train",
        target_classes: List[str] = TARGET_PULMONARY_CLASSES,
        uncertainty_policy: UncertaintyPolicy = UncertaintyPolicy.U_ZERO,
        is_training: bool = False,
        image_size: Tuple[int, int] = (224, 224),
        allow_synthetic_fallback: bool = False,
        filter_missing_images: bool = True,
        max_rows: Optional[int] = None
    ):
        self.split = split
        self.target_classes = target_classes
        self.images_dir = os.path.abspath(images_dir) if images_dir else None
        self.is_training = is_training
        self.image_size = image_size
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.transform = get_cxr_transforms(is_training=is_training, image_size=image_size)

        # Parse DataFrame / CSV into standardized image records
        if isinstance(df, str) or ("image" in df.columns and "image_path" not in df.columns):
            parsed_df = parse_augmented_cxr_dataframe(
                df_or_path=df,
                images_dir=self.images_dir,
                split_name=split,
                target_classes=target_classes,
                filter_missing=filter_missing_images,
                max_rows=max_rows
            )
        else:
            parsed_df = df.copy()

        # Filter by split if column present
        if "split" in parsed_df.columns:
            filtered_df = parsed_df[parsed_df["split"].str.lower() == split.lower()].copy()
        else:
            filtered_df = parsed_df.copy()

        # Clean and map multi-label ground-truth
        self.data_df = map_chexpert_labels(filtered_df, target_classes=target_classes, policy=uncertainty_policy).reset_index(drop=True)
        self.labels_matrix = self.data_df[target_classes].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data_df)

    def resolve_image_path(self, row: pd.Series) -> Optional[str]:
        """
        Resolve physical image path on disk from row metadata.
        """
        # 1. Direct resolved_path or image_path
        if "resolved_path" in row and pd.notna(row["resolved_path"]) and os.path.exists(str(row["resolved_path"])):
            return str(row["resolved_path"])

        rel_path = str(row.get("image_path", row.get("image", ""))).strip().replace("\\", "/")
        if rel_path.startswith("files/") and self.images_dir:
            cand = os.path.join(self.images_dir, rel_path)
            if os.path.exists(cand):
                return cand

        if self.images_dir and os.path.exists(self.images_dir):
            subj_id = str(row.get("subject_id", ""))
            study_id = str(row.get("study_id", ""))
            dicom_id = str(row.get("dicom_id", ""))

            # Standard subpath candidate 1: images_dir/files/pXX/pXXXXXXXX/sXXXXXXXX/XXXXXXXX.jpg
            p_prefix = f"p{subj_id[:2]}" if len(subj_id) >= 2 else "p10"
            cand1 = os.path.join(self.images_dir, "files", p_prefix, f"p{subj_id}", f"s{study_id}", f"{dicom_id}.jpg")
            cand2 = os.path.join(self.images_dir, p_prefix, f"p{subj_id}", f"s{study_id}", f"{dicom_id}.jpg")
            cand3 = os.path.join(self.images_dir, f"{dicom_id}.jpg")

            for cp in [cand1, cand2, cand3]:
                if os.path.exists(cp):
                    return cp

        return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, any]]:
        row = self.data_df.iloc[idx]
        subject_id = str(row.get("subject_id", f"PAT-{idx:04d}"))
        study_id = str(row.get("study_id", f"STD-{idx:04d}"))
        dicom_id = str(row.get("dicom_id", f"DCM-{idx:04d}"))
        view_pos = str(row.get("view_position", row.get("ViewPosition", "PA")))

        img_path = self.resolve_image_path(row)

        if img_path and os.path.exists(img_path):
            img_tensor, _ = load_and_preprocess_image(img_path, transform=self.transform, image_size=self.image_size)
            if img_tensor.ndim == 4:
                img_tensor = img_tensor.squeeze(0)
        else:
            # Enforce Fail-Fast in Real Training
            if not self.allow_synthetic_fallback:
                raise FileNotFoundError(
                    f"CRITICAL: Real MIMIC-CXR image not found on disk for subject {subject_id}, study {study_id}, dicom {dicom_id}. "
                    f"Attempted path: {img_path}. Synthetic fallback is strictly disabled in real-data mode."
                )

            # Synthetic radiograph fallback for unit test fixtures only
            active_pathologies = [cls for cls, val in zip(self.target_classes, self.labels_matrix[idx]) if val == 1.0]
            pil_img = generate_synthetic_chest_radiograph(pathologies=active_pathologies, image_size=self.image_size)
            img_tensor = self.transform(pil_img)

        labels_tensor = torch.tensor(self.labels_matrix[idx], dtype=torch.float32)

        meta = {
            "subject_id": subject_id,
            "study_id": study_id,
            "dicom_id": dicom_id,
            "view_position": view_pos,
            "split": self.split,
            "resolved_path": img_path or "synthetic"
        }

        return img_tensor, labels_tensor, meta


def create_cxr_dataloaders(
    train_metadata: Union[pd.DataFrame, str],
    val_metadata: Optional[Union[pd.DataFrame, str]] = None,
    images_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    target_classes: List[str] = TARGET_PULMONARY_CLASSES,
    uncertainty_policy: UncertaintyPolicy = UncertaintyPolicy.U_ZERO,
    image_size: Tuple[int, int] = (224, 224),
    pin_memory: bool = False,
    allow_synthetic_fallback: bool = False,
    max_train_rows: Optional[int] = None,
    max_val_rows: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Construct PyTorch DataLoaders for Train, Validate, and Test splits from real MIMIC-CXR images.
    Returns: (train_loader, val_loader, test_loader, pos_class_weights)
    """
    train_dataset = MIMICCXRDataset(
        df=train_metadata,
        images_dir=images_dir,
        split="train",
        target_classes=target_classes,
        uncertainty_policy=uncertainty_policy,
        is_training=True,
        image_size=image_size,
        allow_synthetic_fallback=allow_synthetic_fallback,
        max_rows=max_train_rows
    )

    if val_metadata is not None:
        val_dataset = MIMICCXRDataset(
            df=val_metadata,
            images_dir=images_dir,
            split="validate",
            target_classes=target_classes,
            uncertainty_policy=uncertainty_policy,
            is_training=False,
            image_size=image_size,
            allow_synthetic_fallback=allow_synthetic_fallback,
            max_rows=max_val_rows
        )
    else:
        val_dataset = MIMICCXRDataset(
            df=train_metadata,
            images_dir=images_dir,
            split="validate",
            target_classes=target_classes,
            uncertainty_policy=uncertainty_policy,
            is_training=False,
            image_size=image_size,
            allow_synthetic_fallback=allow_synthetic_fallback,
            max_rows=max_val_rows
        )

    # Test dataset (shares validation partition or separate test split)
    test_dataset = val_dataset

    # Calculate class weights from training labels
    pos_weights = calculate_positive_class_weights(train_dataset.labels_matrix) if len(train_dataset) > 0 else np.ones(len(target_classes), dtype=np.float32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > batch_size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, pos_weights
