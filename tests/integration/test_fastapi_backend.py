"""
Integration Tests for FastAPI Backend API Endpoints & ML Services.
Tests patient retrieval, live telemetry, CXR DenseNet-121 inference, Grad-CAM XAI, and What-If simulations.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["primary_disease_domain"] == "Pulmonary / ICU Healthcare"


def test_patients_endpoints():
    # 1. List Patients
    response = client.get("/api/patients")
    assert response.status_code == 200
    patients = response.json()
    assert len(patients) >= 4
    
    # 2. Patient Detail
    p_id = patients[0]["patient_id"]
    detail_res = client.get(f"/api/patients/{p_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["patient_id"] == p_id
    assert "medical_history" in detail
    assert "allergies" in detail

    # 3. Patient Events
    evt_res = client.get(f"/api/patients/{p_id}/events")
    assert evt_res.status_code == 200

    # 4. Patient Medications
    med_res = client.get(f"/api/patients/{p_id}/medications")
    assert med_res.status_code == 200


def test_monitoring_vitals():
    response = client.get("/api/patients/PAT-102/vitals")
    assert response.status_code == 200
    vitals = response.json()
    assert "heart_rate" in vitals
    assert "spo2" in vitals
    assert "systolic_bp" in vitals
    assert "respiratory_rate" in vitals
    assert "body_temperature" in vitals


def test_cxr_vision_pipeline():
    # 1. Available studies
    r_studies = client.get("/api/patients/PAT-102/cxr")
    assert r_studies.status_code == 200
    studies = r_studies.json()
    assert len(studies) > 0
    study_id = studies[0]["study_id"]

    # 2. DenseNet-121 Inference
    r_infer = client.post(
        "/api/cxr/predict",
        json={"patient_id": "PAT-102", "study_id": study_id, "target_pathology": "Pneumonia"}
    )
    assert r_infer.status_code == 200
    infer_data = r_infer.json()
    assert infer_data["is_model_available"] is True
    assert len(infer_data["findings"]) == 8

    # 3. Grad-CAM Heatmap
    r_heatmap = client.get(f"/api/cxr/studies/{study_id}/heatmap?pathology=Pneumonia")
    assert r_heatmap.status_code == 200
    hm_data = r_heatmap.json()
    assert len(hm_data["heatmap_overlay_base64"]) > 1000


def test_risk_and_digital_twin_state():
    # 1. Multi-horizon risk
    r_risk = client.get("/api/patients/PAT-102/risk")
    assert r_risk.status_code == 200
    risk_data = r_risk.json()
    assert "multi_horizon" in risk_data
    assert "news2" in risk_data
    assert "forecast" in risk_data

    # 2. 4-Partition Digital Twin State
    r_twin = client.get("/api/patients/PAT-102/digital-twin")
    assert r_twin.status_code == 200
    twin_data = r_twin.json()
    assert "observed_state" in twin_data
    assert "derived_state" in twin_data
    assert "predicted_state" in twin_data
    assert "simulation_state" in twin_data


def test_simulation_controls_and_whatif():
    # 1. Anomaly injection
    r_anom = client.post(
        "/api/simulation/anomaly",
        json={"patient_id": "PAT-102", "anomaly_type": "septic_spike", "intensity": 1.0}
    )
    assert r_anom.status_code == 200
    assert "septic_spike" in r_anom.json()["active_anomalies"]["PAT-102"]

    # 2. What-If Recalibration
    overrides = {
        "heart_rate": 135.0,
        "spo2": 86.0,
        "systolic_bp": 82.0,
        "respiratory_rate": 30.0,
        "body_temperature": 39.4
    }
    r_whatif = client.post("/api/simulation/what-if/PAT-102", json=overrides)
    assert r_whatif.status_code == 200
    whatif_data = r_whatif.json()
    assert whatif_data["news2"]["risk_tier"] == "High"
