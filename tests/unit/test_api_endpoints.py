"""
Unit and integration tests for the FastAPI service in src/api/main.py.
Tests:
- GET /api/health
- GET /api/cxr/studies/{patient_id}
- POST /api/cxr/studies/{study_id}/infer
- GET /api/cxr/studies/{study_id}/heatmap
- 404 Not Found error handling
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import Base, Patient, CXRStudy
from src.database.connection import get_db
from src.api.main import app


@pytest.fixture
def test_db_session():
    """Create isolated in-memory SQLite database session with StaticPool."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Seed test data
    with TestingSessionLocal() as session:
        patient = Patient(
            subject_id="10003502",
            name="Test Patient 10003502",
            gender="M",
            age=64,
            split="validate"
        )
        session.add(patient)

        study = CXRStudy(
            study_id="s50084553",
            dicom_id="70d7e600-373c1311-929f5ff9-23ee3621-ff551ff9",
            subject_id="10003502",
            view_position="AP",
            image_path="files/p10/p10003502/s50084553/70d7e600-373c1311-929f5ff9-23ee3621-ff551ff9.jpg",
            resolved_path="C:/mock/files/p10/p10003502/s50084553/70d7e600-373c1311-929f5ff9-23ee3621-ff551ff9.jpg",
            split="validate",
            report_text="Impression: Bilateral pleural effusions and atelectasis.",
            pneumonia=0.0,
            pleural_effusion=1.0,
            atelectasis=1.0,
            consolidation=0.0,
            edema=0.0,
            pneumothorax=0.0,
            cardiomegaly=0.0,
            no_finding=0.0
        )
        session.add(study)
        session.commit()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(test_db_session):
    """FastAPI TestClient with overridden database session and lifespan management."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """Test healthcheck endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "mimic-cxr-fastapi"


def test_get_patient_studies(client):
    """Test retrieving studies for an existing patient."""
    response = client.get("/api/cxr/studies/10003502")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    study = data[0]
    assert study["study_id"] == "s50084553"
    assert study["subject_id"] == "10003502"
    assert study["pleural_effusion"] == 1.0
    assert study["atelectasis"] == 1.0
    assert "Pleural Effusion" in study["ground_truth_findings"]
    assert "Atelectasis" in study["ground_truth_findings"]
    assert study["image_url"] == "/api/cxr/studies/s50084553/image"


def test_get_nonexistent_patient_studies(client):
    """Test retrieving studies for a patient with no studies returns empty list."""
    response = client.get("/api/cxr/studies/99999999")
    assert response.status_code == 200
    data = response.json()
    assert data == []


def test_infer_study_success(client):
    """Test DenseNet-121 multi-label inference simulation on an existing study."""
    response = client.post("/api/cxr/studies/s50084553/infer")
    assert response.status_code == 200
    data = response.json()

    assert data["study_id"] == "s50084553"
    assert data["subject_id"] == "10003502"
    assert data["model_version"] == "DenseNet121-MIMICCXR-v1.0"
    assert "probabilities" in data
    assert "predictions" in data
    assert len(data["probabilities"]) == 8

    # All probabilities must be between 0.0 and 1.0
    for finding, prob in data["probabilities"].items():
        assert 0.0 <= prob <= 1.0

    assert data["top_finding"] in data["probabilities"]
    assert data["top_probability"] == data["probabilities"][data["top_finding"]]
    assert data["heatmap_available"] is True
    assert "heatmap_url" in data
    assert "latent_embedding" in data
    assert len(data["latent_embedding"]) == 1024
    assert data["execution_time_ms"] > 0


def test_infer_study_with_uploaded_file(client):
    """Test DenseNet-121 inference on an uploaded image file."""
    import io
    from PIL import Image

    # Create dummy PNG bytes
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/cxr/studies/s_upload_test/infer",
        files={"file": ("test.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["study_id"] == "s_upload_test"
    assert len(data["probabilities"]) == 8
    assert len(data["latent_embedding"]) == 1024


def test_infer_nonexistent_study(client):
    """Test inferring an unknown study with no uploaded file returns 404."""
    response = client.post("/api/cxr/studies/nonexistent_study/infer")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_study_heatmap_png_stream(client):
    """Test generating and streaming Grad-CAM heatmap directly as PNG image."""
    response = client.get("/api/cxr/studies/s50084553/heatmap?pathology=Pleural%20Effusion")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic header


def test_get_study_heatmap_json_metadata(client):
    """Test retrieving Grad-CAM heatmap metadata when format=json is requested."""
    response = client.get("/api/cxr/studies/s50084553/heatmap?pathology=Pleural%20Effusion&format=json")
    assert response.status_code == 200
    data = response.json()
    assert data["study_id"] == "s50084553"
    assert data["pathology"] == "Pleural Effusion"
    assert data["heatmap_available"] is True
    assert data["localization_score"] == 0.88


def test_get_heatmap_nonexistent_study(client):
    """Test retrieving heatmap for an unknown study returns 404."""
    response = client.get("/api/cxr/studies/nonexistent_study/heatmap?pathology=Pneumonia")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_heatmap_invalid_pathology(client):
    """Test requesting an invalid pathology returns 400."""
    response = client.get("/api/cxr/studies/s50084553/heatmap?pathology=InvalidCondition")
    assert response.status_code == 400
    assert "Invalid pathology" in response.json()["detail"]

