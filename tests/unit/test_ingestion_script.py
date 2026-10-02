"""
Unit tests for MIMIC-CXR batch ingestion script (src/scripts/ingest_mimic.py).
Tests:
- Metadata parsing and path resolution
- Physical file existence check
- Fail-fast FileNotFoundError on missing files
- Audit violation logging
- Bulk database insertion with deduplication
"""

import os
import sys
import pytest
import pandas as pd
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.database.models import Base, Patient, CXRStudy
from src.scripts.ingest_mimic import process_csv_split, bulk_insert_records


@pytest.fixture
def mock_mimic_environment(tmp_path):
    """
    Set up mock metadata CSV and mock image filesystem.
    One image exists physically on disk; one image is missing.
    """
    image_root = tmp_path / "cxr_root"
    image_root.mkdir()

    # Create real physical JPG for study 1
    p_dir = image_root / "files" / "p10" / "p10001" / "s501"
    p_dir.mkdir(parents=True)
    real_jpg = p_dir / "valid_img.jpg"
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(real_jpg)

    # Missing image path: files/p10/p10001/s502/missing_img.jpg (not created)
    csv_file = tmp_path / "mock_metadata.csv"
    mock_df = pd.DataFrame({
        "subject_id": [10001],
        "image": ["['files/p10/p10001/s501/valid_img.jpg', 'files/p10/p10001/s502/missing_img.jpg']"],
        "view": ["['PA', 'AP']"],
        "text": ["['Findings: Right lower lobe pneumonia.', 'Findings: No acute cardiopulmonary abnormality.']"]
    })
    mock_df.to_csv(csv_file, index=False)

    return {
        "csv_path": str(csv_file),
        "image_root": str(image_root),
        "valid_path": str(real_jpg)
    }


def test_process_csv_split_with_audit_violations(mock_mimic_environment):
    """Test that missing files generate audit violations and valid files are parsed."""
    csv_path = mock_mimic_environment["csv_path"]
    img_root = mock_mimic_environment["image_root"]

    patients, studies, violations = process_csv_split(
        csv_path=csv_path,
        image_root=img_root,
        split_name="validate",
        fail_fast=False
    )

    assert len(patients) == 1
    assert patients[0]["subject_id"] == "10001"

    # Exactly 1 valid image should be parsed
    assert len(studies) == 1
    assert studies[0]["study_id"] == "s501"
    assert studies[0]["pneumonia"] == 1.0
    assert os.path.normpath(studies[0]["resolved_path"]) == os.path.normpath(mock_mimic_environment["valid_path"])

    # Exactly 1 audit violation should be recorded
    assert len(violations) == 1
    assert violations[0]["study_id"] == "s502"
    assert violations[0]["reason"] == "FILE_NOT_FOUND_ON_DISK"


def test_process_csv_split_fail_fast(mock_mimic_environment):
    """Test that fail_fast=True raises FileNotFoundError on missing physical image."""
    csv_path = mock_mimic_environment["csv_path"]
    img_root = mock_mimic_environment["image_root"]

    with pytest.raises(FileNotFoundError, match="CRITICAL AUDIT VIOLATION"):
        _ = process_csv_split(
            csv_path=csv_path,
            image_root=img_root,
            split_name="validate",
            fail_fast=True
        )


def test_bulk_insert_and_deduplication(tmp_path):
    """Test bulk insertion into SQLite/PostgreSQL with automatic deduplication."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    patients = [
        {"subject_id": "10001", "name": "Patient 10001", "split": "validate"},
        {"subject_id": "10002", "name": "Patient 10002", "split": "train"},
    ]
    studies = [
        {
            "study_id": "s501",
            "dicom_id": "dcm01",
            "subject_id": "10001",
            "view_position": "PA",
            "image_path": "files/p10/p10001/s501/dcm01.jpg",
            "resolved_path": "/path/to/dcm01.jpg",
            "split": "validate",
            "report_text": "Pneumonia",
            "pneumonia": 1.0,
            "pleural_effusion": 0.0,
            "atelectasis": 0.0,
            "consolidation": 0.0,
            "edema": 0.0,
            "pneumothorax": 0.0,
            "cardiomegaly": 0.0,
            "no_finding": 0.0,
        }
    ]

    with Session(engine) as session:
        ins_p, ins_s = bulk_insert_records(session, patients, studies, batch_size=10)
        assert ins_p == 2
        assert ins_s == 1

        # Check records exist
        db_patients = session.scalars(select(Patient)).all()
        assert len(db_patients) == 2

        db_studies = session.scalars(select(CXRStudy)).all()
        assert len(db_studies) == 1
        assert db_studies[0].study_id == "s501"
        assert db_studies[0].pneumonia == 1.0

        # Second insertion with same data should insert 0 new records
        ins_p2, ins_s2 = bulk_insert_records(session, patients, studies, batch_size=10)
        assert ins_p2 == 0
        assert ins_s2 == 0
