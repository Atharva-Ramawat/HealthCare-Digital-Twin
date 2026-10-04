"""
Unit and integration tests for Digital Twin Vitals Simulation, Multimodal Fusion, and API Router.
Tests:
- VitalsSimulator: 24h trajectory generation (96 15-min intervals), bounded physiological values, deterministic reproducibility.
- DigitalTwinFusion: 1024-dim embedding + vitals normalization, MLP forward pass, unified risk score (0-100%), risk tiering.
- GET /api/digital-twin/{patient_id}/{study_id}: Full multimodal fusion endpoint, response contracts, 404 error handling, trajectory query.
"""

from datetime import datetime
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import Base, Patient, CXRStudy
from src.database.connection import get_db
from src.api.main import app
from src.digital_twin.simulator import VitalsSimulator, VitalReading
from src.digital_twin.fusion import DigitalTwinFusion, FusionResult
from src.schemas.twin_schema import DigitalTwinResponse


# =============================================================================
# 1. VITALS SIMULATOR TESTS
# =============================================================================

def test_vitals_simulator_trajectory_shapes_and_intervals():
    """Verify 24-hour simulation produces exactly 96 steps spaced by 15-minute intervals."""
    simulator = VitalsSimulator()
    trajectory = simulator.generate_24h_trajectory(patient_id="test_patient_001")

    assert len(trajectory) == 96
    assert isinstance(trajectory[0], VitalReading)

    # Check 15-minute interval spacing
    for i in range(1, len(trajectory)):
        delta = trajectory[i].timestamp - trajectory[i - 1].timestamp
        assert delta.total_seconds() == 15 * 60, f"Step {i} interval is not 15 minutes"
        assert trajectory[i].step_index == i

    # Check physiological bounds
    for r in trajectory:
        assert 35.0 <= r.heart_rate <= 220.0
        assert 60.0 <= r.spo2 <= 100.0
        assert 50.0 <= r.sbp <= 240.0
        assert 6.0 <= r.respiratory_rate <= 60.0


def test_vitals_simulator_trajectory_profiles_and_determinism():
    """Verify trajectory dynamics for deteriorating and stable profiles and seed reproducibility."""
    simulator = VitalsSimulator()

    # Determinism check: same patient_id should produce identical trajectories
    t1 = simulator.generate_24h_trajectory(patient_id="patient_12345", trajectory_type="deteriorating")
    t2 = simulator.generate_24h_trajectory(patient_id="patient_12345", trajectory_type="deteriorating")

    assert len(t1) == len(t2)
    for r1, r2 in zip(t1, t2):
        assert r1.heart_rate == r2.heart_rate
        assert r1.spo2 == r2.spo2
        assert r1.sbp == r2.sbp
        assert r1.respiratory_rate == r2.respiratory_rate

    # Deteriorating profile check: final HR should be elevated, SpO2 lowered
    assert t1[-1].heart_rate > t1[0].heart_rate
    assert t1[-1].spo2 < t1[0].spo2

    # Latest vitals helper check
    latest = simulator.get_latest_vitals(patient_id="patient_12345", trajectory_type="deteriorating")
    assert latest["heart_rate"] == t1[-1].heart_rate
    assert latest["spo2"] == t1[-1].spo2
    assert "timestamp" in latest


# =============================================================================
# 2. DIGITAL TWIN FUSION MODEL & RISK SCORING TESTS
# =============================================================================

def test_digital_twin_fusion_normal_vitals_low_risk():
    """Verify fusion model computes a low or moderate risk score for normal vital signs."""
    fusion = DigitalTwinFusion()

    dummy_visual_embedding = np.random.randn(1024).astype(np.float32)
    normal_vitals = {
        "heart_rate": 72.0,
        "spo2": 98.5,
        "sbp": 120.0,
        "respiratory_rate": 15.0
    }

    result = fusion.compute_risk(
        visual_embedding=dummy_visual_embedding,
        vitals=normal_vitals,
        cxr_top_finding="No Finding",
        cxr_top_probability=0.85
    )

    assert isinstance(result, FusionResult)
    assert 0.0 <= result.deterioration_risk_score <= 100.0
    assert result.risk_tier in ["Low", "Moderate"]
    assert result.visual_risk_contribution + result.vitals_risk_contribution == pytest.approx(100.0, abs=1.0)
    assert len(result.clinical_recommendation) > 10


def test_digital_twin_fusion_critical_vitals_high_risk():
    """Verify fusion model flags high/critical deterioration risk for severely abnormal vitals."""
    fusion = DigitalTwinFusion()

    dummy_visual_embedding = torch.randn(1024)
    critical_vitals = {
        "heart_rate": 138.0,      # Severe tachycardia
        "spo2": 84.0,            # Severe hypoxemia
        "sbp": 78.0,             # Severe hypotension / shock
        "respiratory_rate": 32.0  # Severe tachypnea
    }

    result = fusion.compute_risk(
        visual_embedding=dummy_visual_embedding,
        vitals=critical_vitals,
        cxr_top_finding="Pneumonia",
        cxr_top_probability=0.92
    )

    assert result.deterioration_risk_score >= 50.0
    assert result.risk_tier in ["High", "Critical"]
    assert any("Tachycardia" in factor for factor in result.risk_factors)
    assert any("Hypoxem" in factor for factor in result.risk_factors)
    assert any("Hypotension" in factor for factor in result.risk_factors)
    assert any("Tachypnea" in factor for factor in result.risk_factors)
    assert "EMERGENCY" in result.clinical_recommendation or "URGENT" in result.clinical_recommendation


def test_digital_twin_fusion_input_format_flexibility():
    """Verify fusion handles Python list, numpy array, and torch.Tensor embeddings with grace."""
    fusion = DigitalTwinFusion()
    vitals = {"heart_rate": 80.0, "spo2": 97.0, "sbp": 118.0, "respiratory_rate": 16.0}

    # 1. Python List
    res_list = fusion.compute_risk(visual_embedding=[0.01] * 1024, vitals=vitals)
    assert 0.0 <= res_list.deterioration_risk_score <= 100.0

    # 2. Dimension Mismatch Handling (512-dim input padded to 1024)
    res_pad = fusion.compute_risk(visual_embedding=[0.05] * 512, vitals=vitals)
    assert 0.0 <= res_pad.deterioration_risk_score <= 100.0


# =============================================================================
# 3. FASTAPI DIGITAL TWIN ENDPOINT TESTS
# =============================================================================

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
    """FastAPI TestClient with lifespan context manager."""
    with TestClient(app) as test_client:
        yield test_client


def test_get_digital_twin_endpoint_success(client):
    """Test GET /api/digital-twin/{patient_id}/{study_id} returns valid DigitalTwinResponse."""
    response = client.get("/api/digital-twin/10003502/s50084553")
    assert response.status_code == 200

    data = response.json()
    # Validate against Pydantic schema
    twin_obj = DigitalTwinResponse.model_validate(data)

    assert twin_obj.patient_id == "10003502"
    assert twin_obj.study_id == "s50084553"
    assert 0.0 <= twin_obj.deterioration_risk_score <= 100.0
    assert twin_obj.risk_tier in ["Low", "Moderate", "High", "Critical"]
    assert len(twin_obj.cxr_probabilities) == 8
    assert "Pneumonia" in twin_obj.cxr_probabilities
    assert twin_obj.current_vitals.heart_rate > 0.0
    assert twin_obj.current_vitals.spo2 > 0.0
    assert twin_obj.vitals_trajectory_24h is None  # Defaults to False


def test_get_digital_twin_endpoint_with_trajectory(client):
    """Test GET /api/digital-twin/{patient_id}/{study_id}?include_trajectory=true returns 96 steps."""
    response = client.get("/api/digital-twin/10003502/s50084553?include_trajectory=true")
    assert response.status_code == 200

    data = response.json()
    twin_obj = DigitalTwinResponse.model_validate(data)

    assert twin_obj.vitals_trajectory_24h is not None
    assert len(twin_obj.vitals_trajectory_24h) == 96
    assert twin_obj.vitals_trajectory_24h[0].step_index == 0
    assert twin_obj.vitals_trajectory_24h[-1].step_index == 95


def test_get_digital_twin_endpoint_study_not_found(client):
    """Test 404 Not Found error handling for non-existent study."""
    response = client.get("/api/digital-twin/10003502/s99999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_digital_twin_endpoint_mismatched_patient(client):
    """Test 404 Not Found when patient_id does not match study's subject_id."""
    response = client.get("/api/digital-twin/99999999/s50084553")
    assert response.status_code == 404
    assert "belongs to patient" in response.json()["detail"]


def test_vitals_simulator_trajectory_around_vitals():
    """Verify trajectory generation around custom vitals aligns with target on the final step."""
    simulator = VitalsSimulator()
    target = {"heart_rate": 115.0, "spo2": 88.0, "sbp": 92.0, "respiratory_rate": 26.0}
    traj = simulator.generate_trajectory_around_vitals(target_vitals=target, patient_id="CUSTOM-TEST")
    assert len(traj) == 96
    assert traj[-1].heart_rate == pytest.approx(115.0, abs=0.1)
    assert traj[-1].spo2 == pytest.approx(88.0, abs=0.1)
    assert traj[-1].sbp == pytest.approx(92.0, abs=0.1)
    assert traj[-1].respiratory_rate == pytest.approx(26.0, abs=0.1)


def test_ad_hoc_infer_endpoint_success(client):
    """Test POST /api/digital-twin/ad-hoc-infer with valid radiograph and manual vitals."""
    import io
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    form_data = {
        "patient_id": "CUSTOM-999",
        "heart_rate": "108.5",
        "spo2": "89.0",
        "sbp": "96.0",
        "respiratory_rate": "24.0",
        "age": "62",
        "gender": "F"
    }

    response = client.post(
        "/api/digital-twin/ad-hoc-infer",
        data=form_data,
        files={"file": ("custom_cxr.png", buf, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    twin_obj = DigitalTwinResponse.model_validate(data)

    assert twin_obj.patient_id == "CUSTOM-999"
    assert twin_obj.study_id.startswith("s_adhoc_")
    assert 0.0 <= twin_obj.deterioration_risk_score <= 100.0
    assert twin_obj.current_vitals.heart_rate == pytest.approx(108.5, abs=0.1)
    assert twin_obj.current_vitals.spo2 == pytest.approx(89.0, abs=0.1)
    assert twin_obj.vitals_trajectory_24h is not None
    assert len(twin_obj.vitals_trajectory_24h) == 96
    assert twin_obj.heatmap_base64 is not None
    assert twin_obj.heatmap_base64.startswith("data:image/png;base64,")
    assert twin_obj.image_base64 is not None
    assert twin_obj.image_base64.startswith("data:image/png;base64,")
    assert twin_obj.heatmap_image_base64 is not None
    assert twin_obj.heatmap_image_base64.startswith("data:image/png;base64,")
    assert twin_obj.input_image_base64 is not None
    assert twin_obj.input_image_base64.startswith("data:image/png;base64,")


def test_ad_hoc_infer_out_of_range_vitals(client):
    """Test POST /api/digital-twin/ad-hoc-infer rejects out-of-range vitals with 400."""
    import io
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Heart rate 350 bpm is out of range
    form_data = {
        "patient_id": "CUSTOM-ERR",
        "heart_rate": "350.0",
        "spo2": "95.0",
        "sbp": "120.0",
        "respiratory_rate": "18.0",
    }

    response = client.post(
        "/api/digital-twin/ad-hoc-infer",
        data=form_data,
        files={"file": ("custom_cxr.png", buf, "image/png")}
    )
    assert response.status_code == 400
    assert "Heart rate" in response.json()["detail"]


def test_ad_hoc_infer_invalid_image_file(client):
    """Test POST /api/digital-twin/ad-hoc-infer rejects non-image files with 400."""
    import io

    bad_file = io.BytesIO(b"this is plain text and not a chest radiograph image")

    form_data = {
        "patient_id": "CUSTOM-BAD-IMG",
        "heart_rate": "80.0",
        "spo2": "98.0",
        "sbp": "120.0",
        "respiratory_rate": "16.0",
    }

    response = client.post(
        "/api/digital-twin/ad-hoc-infer",
        data=form_data,
        files={"file": ("corrupt.txt", bad_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "valid image" in response.json()["detail"].lower()
