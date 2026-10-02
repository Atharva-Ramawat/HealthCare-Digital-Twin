"""
Unit tests for src/database/models.py and src/schemas/cxr_schema.py.
Verifies:
- SQLAlchemy 2.0 ORM models (Patient, CXRStudy) mapping to cohort metadata
- Foreign key relationships and cascade deletion
- Pydantic v2 schemas (StudyBase, StudyResponse, InferenceResult)
- Conversion from SQLAlchemy ORM instances to Pydantic schemas (from_attributes)
"""

import pytest
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.database.models import Base, Patient, CXRStudy
from src.schemas.cxr_schema import StudyBase, StudyResponse, InferenceResult, TARGET_PULMONARY_CLASSES


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for unit testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_patient_and_study_orm_creation(db_session: Session):
    """Test creating and querying Patient and CXRStudy with relationships."""
    patient = Patient(
        subject_id="10003502",
        name="John Doe",
        gender="M",
        age=65,
        split="validate"
    )
    db_session.add(patient)
    db_session.commit()

    study = CXRStudy(
        study_id="s50084553",
        dicom_id="70d7e600-373c1311-929f5ff9-23ee3621-ff551ff9",
        subject_id="10003502",
        view_position="AP",
        image_path="files/p10/p10003502/s50084553/70d7e600-373c1311-929f5ff9-23ee3621-ff551ff9.jpg",
        split="validate",
        report_text="Impression: Bilateral pleural effusions and atelectasis.",
        pneumonia=0.0,
        pleural_effusion=1.0,
        atelectasis=1.0,
        consolidation=0.0,
        edema=0.0,
        pneumothorax=0.0,
        cardiomegaly=0.0,
        no_finding=0.0,
    )
    db_session.add(study)
    db_session.commit()

    # Query back
    stmt = select(Patient).where(Patient.subject_id == "10003502")
    queried_patient = db_session.scalar(stmt)
    assert queried_patient is not None
    assert len(queried_patient.studies) == 1
    assert queried_patient.studies[0].study_id == "s50084553"
    assert queried_patient.studies[0].pleural_effusion == 1.0
    assert "Pleural Effusion" in queried_patient.studies[0].positive_findings
    assert "Atelectasis" in queried_patient.studies[0].positive_findings
    assert "Pneumonia" not in queried_patient.studies[0].positive_findings


def test_cascade_delete(db_session: Session):
    """Test that deleting a patient cascades to delete associated CXR studies."""
    patient = Patient(subject_id="10009999", split="train")
    db_session.add(patient)
    db_session.commit()

    study = CXRStudy(
        study_id="s99999999",
        subject_id="10009999",
        pneumonia=1.0
    )
    db_session.add(study)
    db_session.commit()

    # Verify study exists
    assert db_session.scalar(select(CXRStudy).where(CXRStudy.study_id == "s99999999")) is not None

    # Delete patient
    db_session.delete(patient)
    db_session.commit()

    # Study should now be deleted
    assert db_session.scalar(select(CXRStudy).where(CXRStudy.study_id == "s99999999")) is None


def test_study_base_and_response_pydantic():
    """Test Pydantic v2 schemas: validation, derived fields, and serialization."""
    base_data = {
        "study_id": "s50084553",
        "subject_id": "10003502",
        "dicom_id": "dcm-01",
        "view_position": "PA",
        "image_path": "files/p10/p10003502/s50084553/dcm-01.jpg",
        "split": "validate",
        "report_text": "Findings: Pneumonia and cardiomegaly.",
        "pneumonia": 1.0,
        "cardiomegaly": 1.0,
    }
    study_base = StudyBase(**base_data)
    assert study_base.pneumonia == 1.0
    assert study_base.edema == 0.0

    # Test StudyResponse with auto-derived fields
    resp_data = {**base_data, "id": 1}
    study_resp = StudyResponse(**resp_data)
    assert study_resp.id == 1
    assert study_resp.image_url == "/api/cxr/studies/s50084553/image"
    assert "Pneumonia" in study_resp.ground_truth_findings
    assert "Cardiomegaly" in study_resp.ground_truth_findings
    assert "Edema" not in study_resp.ground_truth_findings


def test_orm_to_pydantic_from_attributes(db_session: Session):
    """Test converting SQLAlchemy CXRStudy instance directly to StudyResponse."""
    patient = Patient(subject_id="10001234", split="validate")
    db_session.add(patient)
    db_session.commit()

    study = CXRStudy(
        study_id="s12345678",
        dicom_id="dcm1234",
        subject_id="10001234",
        view_position="PA",
        image_path="files/p10/p10001234/s12345678/dcm1234.jpg",
        split="validate",
        edema=1.0,
        consolidation=1.0,
    )
    db_session.add(study)
    db_session.commit()
    db_session.refresh(study)

    # Validate directly from ORM instance via Pydantic v2 model_validate
    study_pydantic = StudyResponse.model_validate(study)
    assert study_pydantic.id == study.id
    assert study_pydantic.study_id == "s12345678"
    assert study_pydantic.edema == 1.0
    assert study_pydantic.consolidation == 1.0
    assert "Edema" in study_pydantic.ground_truth_findings
    assert "Consolidation" in study_pydantic.ground_truth_findings


def test_inference_result_schema():
    """Test InferenceResult Pydantic v2 contract."""
    result = InferenceResult(
        study_id="s50084553",
        subject_id="10003502",
        model_version="DenseNet121-MIMICCXR-v1.0",
        probabilities={
            "Pneumonia": 0.72,
            "Pleural Effusion": 0.45,
            "Atelectasis": 0.20,
            "Consolidation": 0.15,
            "Edema": 0.10,
            "Pneumothorax": 0.02,
            "Cardiomegaly": 0.18,
            "No Finding": 0.05
        },
        predictions={
            "Pneumonia": True,
            "Pleural Effusion": True,
            "Atelectasis": False,
            "Consolidation": False,
            "Edema": False,
            "Pneumothorax": False,
            "Cardiomegaly": False,
            "No Finding": False
        },
        top_finding="Pneumonia",
        top_probability=0.72,
        thresholds={"Pneumonia": 0.35, "Pleural Effusion": 0.40},
        has_pathology=True,
        heatmap_available=True,
        heatmap_url="/api/cxr/studies/s50084553/heatmap?pathology=Pneumonia",
        execution_time_ms=45.2
    )

    assert result.study_id == "s50084553"
    assert result.top_finding == "Pneumonia"
    assert result.top_probability == 0.72
    assert result.has_pathology is True
    assert result.predictions["Pneumonia"] is True
    assert result.probabilities["Pneumonia"] == 0.72
    serialized = result.model_dump()
    assert serialized["top_finding"] == "Pneumonia"
    assert serialized["execution_time_ms"] == 45.2
