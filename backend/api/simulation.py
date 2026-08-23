"""
Simulation Controls API Router - Start, pause, reset, playback speed, anomaly injection, and what-if counterfactuals.
"""

from typing import Dict
from fastapi import APIRouter, HTTPException, Depends
from backend.schemas.monitoring import (
    SimulationControlRequest,
    SimulationStateResponse,
    AnomalyInjectionRequest
)
from backend.schemas.prediction import RiskPredictionResponse
from backend.services.digital_twin_service import DigitalTwinService

router = APIRouter(prefix="/api/simulation", tags=["Simulation Controls"])


def get_digital_twin_service() -> DigitalTwinService:
    from backend.main import twin_service
    return twin_service


@router.get("/status", response_model=SimulationStateResponse, summary="Get simulation engine status")
def get_simulation_status(service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Retrieve current replay engine state, speed, active anomalies, and operating mode."""
    return service.replay.get_simulation_state()


@router.post("/start", response_model=SimulationStateResponse, summary="Start / Resume simulation streaming")
def start_simulation(service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Resume asynchronous clinical data streaming."""
    service.replay.play()
    return service.replay.get_simulation_state()


@router.post("/pause", response_model=SimulationStateResponse, summary="Pause simulation streaming")
def pause_simulation(service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Pause real-time streaming playback."""
    service.replay.pause()
    return service.replay.get_simulation_state()


@router.post("/reset", response_model=SimulationStateResponse, summary="Reset simulation stream")
def reset_simulation(service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Reset simulation timeline and patient sliding window buffers to initial state."""
    service.replay.reset()
    return service.replay.get_simulation_state()


@router.post("/control", response_model=SimulationStateResponse, summary="Configure playback speed or mode")
def control_simulation(request: SimulationControlRequest, service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Configure speed multiplier (1x, 2x, 5x, 10x), mode ('simulation' | 'mimic_replay' | 'inference'), or action."""
    if request.speed_multiplier is not None:
        service.replay.set_speed(request.speed_multiplier)
    if request.mode is not None:
        service.replay.set_mode(request.mode)
    if request.action == "play":
        service.replay.play()
    elif request.action == "pause":
        service.replay.pause()
    elif request.action == "reset":
        service.replay.reset()
    elif request.action == "step":
        service.replay.step()
        
    return service.replay.get_simulation_state()


@router.post("/anomaly", response_model=SimulationStateResponse, summary="Inject physiological shock or perturbation")
def inject_anomaly(request: AnomalyInjectionRequest, service: DigitalTwinService = Depends(get_digital_twin_service)):
    """
    Artificially alter patient vitals on demand.
    Supported types: 'septic_spike', 'hypoxia_drop', 'cardiac_arrhythmia', 'reset'
    """
    if request.patient_id not in service.replay.patients_meta:
        raise HTTPException(status_code=404, detail=f"Patient '{request.patient_id}' not found")
    
    service.replay.inject_anomaly(request.patient_id, request.anomaly_type)
    return service.replay.get_simulation_state()


@router.post("/what-if/{patient_id}", response_model=RiskPredictionResponse, summary="Execute counterfactual What-If recalibration")
def run_what_if_recalibration(
    patient_id: str,
    vital_overrides: Dict[str, float],
    service: DigitalTwinService = Depends(get_digital_twin_service)
):
    """
    Dynamically recalculate Digital Twin risk scores and 15-minute forecasts
    when evaluators manually tweak slider values for current vitals.
    """
    if patient_id not in service.replay.patients_meta:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    
    service.replay.set_what_if_overrides(patient_id, vital_overrides)
    
    # Compute new sliding window matrix with overridden latest step
    matrix = service.replay.get_sliding_window_matrix(patient_id)
    return service.prediction.predict_risk(patient_id, matrix)
