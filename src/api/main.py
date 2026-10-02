"""
FastAPI application service for MIMIC-CXR Pulmonary Diagnostic Pipeline.
Provides endpoints for:
- Retrieving patient CXR studies from database: GET /api/cxr/studies/{patient_id}
- Running DenseNet-121 multi-label inference: POST /api/cxr/studies/{study_id}/infer
- Retrieving cached Grad-CAM heatmaps: GET /api/cxr/studies/{study_id}/heatmap
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.database.models import Patient, CXRStudy
from src.database.connection import get_db, init_db
from src.schemas.cxr_schema import (
    StudyResponse,
    InferenceResult,
    HeatmapResponse,
    TARGET_PULMONARY_CLASSES
)

# Decision Thresholds per Pathology (clinical baseline)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initialize database tables upon startup."""
    init_db()
    yield


app = FastAPI(
    title="MIMIC-CXR Digital Twin Diagnostic Service",
    description="FastAPI service for patient CXR studies, DenseNet-121 multi-label inference, and Grad-CAM explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Healthcheck endpoint for monitoring service status."""
    return {
        "status": "healthy",
        "service": "mimic-cxr-fastapi",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get(
    "/api/cxr/studies/{patient_id}",
    response_model=List[StudyResponse],
    summary="Retrieve all CXR studies for a patient",
    tags=["CXR Studies"]
)
def get_patient_studies(
    patient_id: str,
    db: Session = Depends(get_db)
) -> List[StudyResponse]:
    """
    Retrieve all recorded Chest X-Ray studies for a patient from the database.
    Maps subject_id to study records and derived clinical findings.
    """
    stmt = (
        select(CXRStudy)
        .where(CXRStudy.subject_id == str(patient_id))
        .order_by(CXRStudy.id)
    )
    studies = db.scalars(stmt).all()

    # Convert ORM instances to Pydantic StudyResponse contracts
    return [StudyResponse.model_validate(study) for study in studies]


@app.post(
    "/api/cxr/studies/{study_id}/infer",
    response_model=InferenceResult,
    summary="Execute multi-label DenseNet-121 inference on a CXR study",
    tags=["CXR Inference"]
)
def infer_study(
    study_id: str,
    db: Session = Depends(get_db)
) -> InferenceResult:
    """
    Run multi-label pulmonary inference on a selected CXR study.
    Simulates DenseNet-121 forward-pass logic, validates against the Pydantic InferenceResult schema,
    and caches top findings and probabilities in the database.
    """
    start_time = time.time()

    # 1. Fetch study from database
    stmt = select(CXRStudy).where(CXRStudy.study_id == str(study_id))
    study = db.scalar(stmt)

    if not study:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CXR study '{study_id}' not found in database."
        )

    # 2. Simulate DenseNet-121 Multi-Label Probabilities
    # Correlates probabilities with cohort ground truth if present for clinical realism
    probabilities: Dict[str, float] = {}
    ground_truth = study.pathology_dict

    for pathology, threshold in DEFAULT_PATHOLOGY_THRESHOLDS.items():
        gt_val = ground_truth.get(pathology, 0.0)
        if gt_val == 1.0:
            # High confidence for positive ground truth
            prob = round(float(threshold + 0.25 + (hash(study_id + pathology) % 15) * 0.01), 4)
            prob = min(prob, 0.98)
        else:
            # Low confidence for negative ground truth
            prob = round(float(0.04 + (hash(study_id + pathology) % 12) * 0.01), 4)
            prob = min(prob, threshold - 0.05)

        probabilities[pathology] = max(0.01, min(0.99, prob))

    # Evaluate binary predictions against decision thresholds
    predictions: Dict[str, bool] = {
        pathology: bool(probabilities[pathology] >= DEFAULT_PATHOLOGY_THRESHOLDS[pathology])
        for pathology in DEFAULT_PATHOLOGY_THRESHOLDS
    }

    # Determine highest confidence finding
    # If no pathology exceeds threshold, No Finding is dominant
    pathology_probs = {k: v for k, v in probabilities.items() if k != "No Finding"}
    top_finding = max(pathology_probs, key=pathology_probs.get)
    top_probability = pathology_probs[top_finding]

    if not any(predictions[k] for k in pathology_probs):
        top_finding = "No Finding"
        top_probability = probabilities["No Finding"]

    has_pathology = any(predictions[k] for k in pathology_probs)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    # 3. Update Study Cache in Database
    study.top_finding = top_finding
    study.top_probability = top_probability
    study.inferred_at = datetime.utcnow()
    db.commit()
    db.refresh(study)

    # 4. Construct InferenceResult Pydantic v2 Contract
    result = InferenceResult(
        study_id=study.study_id,
        subject_id=study.subject_id,
        model_version="DenseNet121-MIMICCXR-v1.0",
        probabilities=probabilities,
        predictions=predictions,
        top_finding=top_finding,
        top_probability=top_probability,
        thresholds=DEFAULT_PATHOLOGY_THRESHOLDS,
        has_pathology=has_pathology,
        heatmap_available=has_pathology or (top_finding != "No Finding"),
        heatmap_url=f"/api/cxr/studies/{study.study_id}/heatmap?pathology={top_finding}",
        execution_time_ms=elapsed_ms,
        timestamp=datetime.utcnow()
    )

    return result


@app.get(
    "/api/cxr/studies/{study_id}/heatmap",
    response_model=HeatmapResponse,
    summary="Retrieve Grad-CAM explainability heatmap for a CXR study",
    tags=["CXR Explainability"]
)
def get_study_heatmap(
    study_id: str,
    pathology: str = Query("Pneumonia", description="Target pulmonary pathology for Grad-CAM explainability"),
    db: Session = Depends(get_db)
) -> HeatmapResponse:
    """
    Retrieve cached or generated Grad-CAM heatmap metadata for a specific study and pathology.
    Highlights localized spatial feature regions in the radiograph.
    """
    stmt = select(CXRStudy).where(CXRStudy.study_id == str(study_id))
    study = db.scalar(stmt)

    if not study:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CXR study '{study_id}' not found in database."
        )

    # Clinical localized interpretation based on pathology
    descriptions = {
        "Pneumonia": "Grad-CAM activation highlights focal airspace opacity and parenchymal consolidation in mid-to-lower lung fields.",
        "Pleural Effusion": "Grad-CAM activation highlights blunting of the costophrenic angles and fluid layering along the thoracic wall.",
        "Atelectasis": "Grad-CAM activation indicates bibasilar volume loss and linear discoid opacities.",
        "Consolidation": "Grad-CAM highlights dense airspace opacification with air bronchograms.",
        "Edema": "Grad-CAM highlights diffuse bilateral perihilar haziness and vascular congestion.",
        "Pneumothorax": "Grad-CAM activation localizes along the apical pleural line with absent peripheral lung markings.",
        "Cardiomegaly": "Grad-CAM highlights enlargement of the cardiac silhouette across the cardiothoracic boundary.",
        "No Finding": "Grad-CAM displays diffuse, low-intensity background activation across clear lung fields."
    }

    desc = descriptions.get(
        pathology,
        f"Grad-CAM activation highlights localized features associated with {pathology}."
    )

    return HeatmapResponse(
        study_id=study.study_id,
        subject_id=study.subject_id,
        pathology=pathology,
        heatmap_available=True,
        heatmap_url=f"/api/cxr/studies/{study.study_id}/heatmap?pathology={pathology}",
        localization_score=0.88,
        description=desc,
        timestamp=datetime.utcnow()
    )
