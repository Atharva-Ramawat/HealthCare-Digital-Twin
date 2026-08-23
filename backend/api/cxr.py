"""
Chest X-Ray (CXR) API Router - Pulmonary diagnostics, DenseNet-121 inference, and Grad-CAM heatmaps.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from backend.schemas.cxr import (
    CXRStudy,
    CXRInferenceRequest,
    CXRInferenceResponse,
    CXRHeatmapResponse
)
from backend.services.cxr_service import CXRService

router = APIRouter(tags=["Chest X-Ray (CXR)"])


def get_cxr_service() -> CXRService:
    from backend.main import cxr_service
    return cxr_service


@router.get("/api/patients/{patient_id}/cxr", response_model=List[CXRStudy], summary="List patient CXR studies")
def get_patient_cxr_studies(patient_id: str, service: CXRService = Depends(get_cxr_service)):
    """Retrieve all recorded or historical Chest X-Ray studies for a patient."""
    return service.get_patient_cxr_studies(patient_id)


@router.post("/api/cxr/predict", response_model=CXRInferenceResponse, summary="Execute DenseNet-121 CXR model inference")
def predict_cxr(request: CXRInferenceRequest, service: CXRService = Depends(get_cxr_service)):
    """
    Run PyTorch DenseNet-121 pulmonary multi-label classifier on a selected study or uploaded CXR image.
    Returns probabilities across 8 pulmonary conditions with threshold classifications.
    """
    return service.predict_cxr(
        patient_id=request.patient_id,
        study_id=request.study_id,
        image_base64=request.image_base64,
        target_pathology=request.target_pathology or "Pneumonia"
    )


@router.get("/api/cxr/studies/{study_id}/heatmap", response_model=CXRHeatmapResponse, summary="Get Grad-CAM XAI heatmap")
def get_cxr_heatmap(
    study_id: str,
    pathology: str = Query("Pneumonia", description="Target pulmonary condition for Grad-CAM"),
    service: CXRService = Depends(get_cxr_service)
):
    """
    Generate visual Grad-CAM heatmap overlay highlighting localized thoracic pathology regions.
    """
    return service.generate_cxr_heatmap(study_id=study_id, target_pathology=pathology)
