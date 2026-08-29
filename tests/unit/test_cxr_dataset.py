"""
Unit Tests for MIMIC-CXR Real Dataset Integration and Path Resolution.
Tests:
- 'files/...' relative path resolution against CXR_IMAGE_ROOT
- Stringified list parsing for image, view, and text columns
- Train and validation metadata loading
- Fail-fast FileNotFoundError on missing images (zero synthetic fallback during real training)
- Radiology report NLP multi-label extraction
- Patient-level split disjointness
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ml.cxr.config import CXRConfig
from ml.cxr.labels import TARGET_PULMONARY_CLASSES, extract_labels_from_radiology_report, verify_patient_level_split
from ml.cxr.dataset import MIMICCXRDataset, parse_augmented_cxr_dataframe, create_cxr_dataloaders


def test_radiology_report_nlp_extraction():
    # 1. Positive Pneumonia & Effusion with Negated Pneumothorax
    report1 = "Findings: Focal consolidation in right base suggesting pneumonia. Moderate left pleural effusion. No evidence of pneumothorax. Impression: Pneumonia and effusion."
    lbls1 = extract_labels_from_radiology_report(report1)
    assert lbls1["Pneumonia"] == 1.0
    assert lbls1["Pleural Effusion"] == 1.0
    assert lbls1["Pneumothorax"] == 0.0
    assert lbls1["No Finding"] == 0.0

    # 2. Fully Normal / Clear Lungs
    report2 = "Findings: Lungs are clear without focal consolidation, pleural effusion, edema, or pneumothorax. Heart size is normal. Impression: No acute cardiopulmonary process."
    lbls2 = extract_labels_from_radiology_report(report2)
    assert lbls2["Pneumonia"] == 0.0
    assert lbls2["Pleural Effusion"] == 0.0
    assert lbls2["Edema"] == 0.0
    assert lbls2["Pneumothorax"] == 0.0
    assert lbls2["Cardiomegaly"] == 0.0
    assert lbls2["No Finding"] == 1.0


def test_augmented_dataframe_parsing(tmp_path):
    # Create mock augmented CSV
    mock_csv = tmp_path / "mock_augmented.csv"
    mock_df = pd.DataFrame({
        "subject_id": [10001],
        "image": ["['files/p10/p10001/s501/dcm01.jpg', 'files/p10/p10001/s502/dcm02.jpg']"],
        "view": ["['PA', 'LATERAL']"],
        "text": ["['Findings: Bilateral pulmonary edema and cardiomegaly.', 'Findings: Normal lungs.']"]
    })
    mock_df.to_csv(mock_csv, index=False)

    parsed = parse_augmented_cxr_dataframe(str(mock_csv), images_dir=str(tmp_path), split_name="validate", filter_missing=False)
    assert len(parsed) == 2
    assert parsed.iloc[0]["study_id"] == "s501"
    assert parsed.iloc[0]["dicom_id"] == "dcm01"
    assert parsed.iloc[0]["Edema"] == 1.0
    assert parsed.iloc[0]["Cardiomegaly"] == 1.0
    assert parsed.iloc[1]["study_id"] == "s502"
    assert parsed.iloc[1]["No Finding"] == 1.0


def test_fail_fast_missing_image_no_synthetic_fallback(tmp_path):
    # Test that missing images raise FileNotFoundError when allow_synthetic_fallback=False
    mock_df = pd.DataFrame({
        "subject_id": [10001],
        "study_id": ["s501"],
        "dicom_id": ["dcm01"],
        "image_path": ["files/p10/p10001/s501/nonexistent.jpg"],
        "resolved_path": [str(tmp_path / "files/p10/p10001/s501/nonexistent.jpg")],
        "Pneumonia": [1.0],
        "Pleural Effusion": [0.0],
        "Atelectasis": [0.0],
        "Consolidation": [0.0],
        "Edema": [0.0],
        "Pneumothorax": [0.0],
        "Cardiomegaly": [0.0],
        "No Finding": [0.0]
    })

    ds_real = MIMICCXRDataset(
        df=mock_df,
        images_dir=str(tmp_path),
        allow_synthetic_fallback=False,  # Strict real mode
        filter_missing_images=False
    )

    with pytest.raises(FileNotFoundError, match="Real MIMIC-CXR image not found"):
        _ = ds_real[0]

    # When synthetic fallback explicitly allowed (unit testing fixture mode), it returns tensor
    ds_syn = MIMICCXRDataset(
        df=mock_df,
        images_dir=str(tmp_path),
        allow_synthetic_fallback=True,
        filter_missing_images=False
    )
    img_t, lbl_t, meta = ds_syn[0]
    assert img_t.shape == (3, 224, 224)
    assert meta["resolved_path"] == "synthetic"


def test_real_validation_image_loading():
    config = CXRConfig(allow_synthetic_fallback=False)
    if not os.path.exists(config.val_metadata_path) or not os.path.exists(config.image_root):
        pytest.skip("Local MIMIC-CXR files not present")

    val_dataset = MIMICCXRDataset(
        df=config.val_metadata_path,
        images_dir=config.image_root,
        split="validate",
        allow_synthetic_fallback=False,
        filter_missing_images=True,
        max_rows=10
    )

    assert len(val_dataset) > 0
    img_tensor, label_tensor, meta = val_dataset[0]

    assert img_tensor.shape == (3, 224, 224)
    assert img_tensor.dtype == torch.float32
    assert not torch.isnan(img_tensor).any()
    assert not torch.isinf(img_tensor).any()
    assert os.path.exists(meta["resolved_path"])
    assert label_tensor.shape == (8,)


def test_train_val_patient_disjointness():
    config = CXRConfig()
    if not os.path.exists(config.train_metadata_path) or not os.path.exists(config.val_metadata_path):
        pytest.skip("Local MIMIC-CXR CSV files not present")

    df_train = pd.read_csv(config.train_metadata_path, usecols=["subject_id"], nrows=1000)
    df_val = pd.read_csv(config.val_metadata_path, usecols=["subject_id"])

    train_pts = set(df_train["subject_id"].astype(str))
    val_pts = set(df_val["subject_id"].astype(str))

    overlap = train_pts.intersection(val_pts)
    assert len(overlap) == 0, f"Patient leakage detected: {overlap}"
