"""
FastAPI Main Application Entrypoint - AI-Driven Digital Twin for Smart Healthcare.
Mounts REST & WebSocket routers, configures CORS for React + TypeScript frontend, and initializes services.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in Python module search path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.services.replay_service import ClinicalReplayService
from backend.services.prediction_service import PredictionService
from backend.services.cxr_service import CXRService
from backend.services.digital_twin_service import DigitalTwinService

from backend.api.patients import router as patients_router
from backend.api.monitoring import router as monitoring_router
from backend.api.cxr import router as cxr_router
from backend.api.risk import router as risk_router
from backend.api.digital_twin import router as digital_twin_router
from backend.api.simulation import router as simulation_router

# Initialize Core Services (Singletons)
replay_service = ClinicalReplayService(window_size=24)
prediction_service = PredictionService()
cxr_service = CXRService()
twin_service = DigitalTwinService(replay_service=replay_service, prediction_service=prediction_service)

# Initialize FastAPI App
app = FastAPI(
    title="AI-Driven Digital Twin for Smart Healthcare API",
    description=(
        "FastAPI REST and WebSocket backend powering the React + TypeScript Digital Twin interface. "
        "Integrates simulated real-time clinical data streaming, CNN-BiLSTM multi-horizon deterioration risk, "
        "DenseNet-121 MIMIC-CXR pulmonary diagnostics, and Grad-CAM / Integrated Gradients explainability."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(patients_router)
app.include_router(monitoring_router)
app.include_router(cxr_router)
app.include_router(risk_router)
app.include_router(digital_twin_router)
app.include_router(simulation_router)


@app.get("/", tags=["System"])
def root():
    """System health check and operational status."""
    return {
        "system": "AI-Driven Predictive Patient Digital Twin",
        "status": "online",
        "version": "2.0.0",
        "primary_disease_domain": "Pulmonary / ICU Healthcare",
        "mode": replay_service.mode,
        "is_model_available": prediction_service.is_model_available and cxr_service.is_model_available,
        "docs_url": "/docs",
        "active_patients_count": len(replay_service.patients_meta)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
