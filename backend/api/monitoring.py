"""
Monitoring & Telemetry API Router - Real-time vitals, sliding window buffers, and WebSocket telemetry stream.
"""

import asyncio
import json
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from backend.schemas.monitoring import VitalsFrame, SlidingWindowVitals, StreamVitalsMessage
from backend.services.digital_twin_service import DigitalTwinService

router = APIRouter(tags=["Monitoring & Telemetry"])


def get_digital_twin_service() -> DigitalTwinService:
    from backend.main import twin_service
    return twin_service


@router.get("/api/patients/{patient_id}/vitals", response_model=VitalsFrame, summary="Get latest live vitals")
def get_patient_vitals(patient_id: str, service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Retrieve the latest observed/simulated vital signs for a patient."""
    vitals = service.replay.get_latest_vitals(patient_id)
    if not vitals:
        raise HTTPException(status_code=404, detail=f"No vitals found for patient '{patient_id}'")
    return vitals


@router.get("/api/patients/{patient_id}/window", response_model=SlidingWindowVitals, summary="Get sliding window buffer")
def get_sliding_window(patient_id: str, service: DigitalTwinService = Depends(get_digital_twin_service)):
    """Retrieve the 24-step sliding window buffer formatted for neural network input."""
    if patient_id not in service.replay.patients_meta:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    return service.replay.get_sliding_window(patient_id)


@router.websocket("/api/stream/{patient_id}")
async def websocket_vital_stream(websocket: WebSocket, patient_id: str):
    """
    Continuous low-latency WebSocket vital telemetry stream for React UI dashboards.
    Transmits live frames every (1.0 / speed_multiplier) seconds.
    """
    from backend.main import twin_service
    
    await websocket.accept()
    if patient_id not in twin_service.replay.patients_meta:
        await websocket.send_json({"error": f"Patient {patient_id} not recognized"})
        await websocket.close()
        return

    try:
        while True:
            # Advance simulation step if active
            twin_service.replay.step()
            
            latest = twin_service.replay.get_latest_vitals(patient_id)
            window_mat = twin_service.replay.get_sliding_window_matrix(patient_id)
            pred = twin_service.prediction.predict_risk(patient_id, window_mat)
            
            has_anomaly = len(twin_service.replay.active_anomalies.get(patient_id, [])) > 0
            
            msg = StreamVitalsMessage(
                patient_id=patient_id,
                timestamp=latest.timestamp if latest else "",
                step=latest.step if latest else 0,
                vitals=latest.dict() if latest else {},
                news2_score=pred.news2.total_score,
                news2_tier=pred.news2.risk_tier,
                health_risk_score=pred.health_risk_score,
                is_anomaly_active=has_anomaly,
                mode=twin_service.replay.mode
            )
            
            await websocket.send_json(msg.dict())
            
            # Sleep duration determined by playback speed
            delay = 1.0 / max(0.1, twin_service.replay.speed_multiplier)
            await asyncio.sleep(delay)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Stream disconnected for {patient_id}: {e}")
