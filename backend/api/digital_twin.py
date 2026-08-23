"""
Digital Twin State API Router - Access to the complete 4-partition Digital Twin State.
"""

from fastapi import APIRouter, HTTPException, Depends
from backend.schemas.prediction import DigitalTwinStateResponse
from backend.services.digital_twin_service import DigitalTwinService

router = APIRouter(tags=["Digital Twin Core"])


def get_digital_twin_service() -> DigitalTwinService:
    from backend.main import twin_service
    return twin_service


@router.get("/api/patients/{patient_id}/digital-twin", response_model=DigitalTwinStateResponse, summary="Get 4-partition Digital Twin State")
def get_digital_twin(patient_id: str, service: DigitalTwinService = Depends(get_digital_twin_service)):
    """
    Retrieve the full unified Digital Twin representation partitioned into:
    - ObservedState (raw measured vitals & demographics)
    - DerivedState (causal baselines, MAP, Shock Index, NEWS2)
    - PredictedState (multi-horizon risk, forecast horizon, XAI attributions)
    - SimulationState (replay controls, what-if overrides, active anomalies)
    """
    twin_state = service.get_digital_twin_state(patient_id)
    if not twin_state:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    return twin_state
