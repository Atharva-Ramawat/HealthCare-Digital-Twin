"""
Patients API Router - Demographic profiles, historical trends, clinical events, and medications.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Depends, Query

from backend.schemas.patient import (
    PatientSummary,
    PatientDetail,
    PatientEvent,
    PatientMedication
)
from backend.schemas.monitoring import VitalsFrame
from backend.services.digital_twin_service import DigitalTwinService

router = APIRouter(prefix="/api/patients", tags=["Patients"])


# Dependency injection helper
def get_digital_twin_service() -> DigitalTwinService:
    from backend.main import twin_service
    return twin_service


@router.get("", response_model=List[PatientSummary], summary="List all ICU ward patients")
def list_patients(service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Retrieve all virtual ICU patients with real-time risk scores and NEWS 2 indicators."""
    return service.list_patients()


@router.get("/{patient_id}", response_model=PatientDetail, summary="Get patient profile detail")
def get_patient(patient_id: str, service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Retrieve detailed demographics, medical history, allergies, and admission notes for a patient."""
    detail = service.get_patient_detail(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    return detail


@router.get("/{patient_id}/history", response_model=List[VitalsFrame], summary="Get patient vital history")
def get_patient_history(
    patient_id: str,
    limit: int = Query(60, ge=1, le=500, description="Max number of historical time-steps to return"),
    service: DigitalTwinService = Depends(get_digital_twin_service)
):
    """Retrieve chronological recorded vital sign history."""
    if patient_id not in service.replay.patients_meta:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    return service.replay.get_patient_history(patient_id, limit=limit)


@router.get("/{patient_id}/events", response_model=List[PatientEvent], summary="Get clinical timeline events")
def get_patient_events(patient_id: str, service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Retrieve clinical milestones, alert triggers, and care escalation events."""
    if patient_id not in service.replay.patients_meta:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    return service.get_patient_events(patient_id)


@router.get("/{patient_id}/medications", response_model=List[PatientMedication], summary="Get active medications")
def get_patient_medications(patient_id: str, service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Retrieve active and historical medication orders."""
    if patient_id not in service.replay.patients_meta:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    return service.get_patient_medications(patient_id)
