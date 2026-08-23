"""
Pydantic schemas for real-time vital signs monitoring, telemetry sliding windows, and simulation controls.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class VitalsFrame(BaseModel):
    timestamp: str = Field(..., description="Observation timestamp")
    step: int = Field(0, description="Discrete simulation/replay step index")
    heart_rate: float = Field(..., ge=20.0, le=250.0, description="Heart Rate (bpm)")
    spo2: float = Field(..., ge=50.0, le=100.0, description="Pulse Oximetry Oxygen Saturation (%)")
    systolic_bp: float = Field(..., ge=40.0, le=280.0, description="Systolic Blood Pressure (mmHg)")
    diastolic_bp: Optional[float] = Field(None, ge=20.0, le=180.0, description="Diastolic Blood Pressure (mmHg)")
    respiratory_rate: float = Field(..., ge=4.0, le=60.0, description="Respiratory Rate (breaths/min)")
    body_temperature: float = Field(..., ge=30.0, le=44.0, description="Body Temperature (°C)")
    mean_arterial_pressure: Optional[float] = Field(None, description="MAP = DBP + (SBP - DBP)/3 or 2/3 DBP + 1/3 SBP")
    shock_index: Optional[float] = Field(None, description="Shock Index = HR / SBP")


class SlidingWindowVitals(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    window_size: int = Field(24, description="Number of temporal observations in buffer")
    timestamps: List[str] = Field(default_factory=list, description="Sequence timestamps")
    features: List[str] = Field(default_factory=list, description="Feature column names")
    sequence: List[Dict[str, float]] = Field(default_factory=list, description="Ordered time-series observations")


class AnomalyInjectionRequest(BaseModel):
    patient_id: str = Field(..., description="Target patient ID")
    anomaly_type: str = Field(..., description="'septic_spike', 'hypoxia_drop', 'cardiac_arrhythmia', or 'reset'")
    intensity: float = Field(1.0, ge=0.1, le=5.0, description="Multiplier for anomaly distortion magnitude")


class SimulationControlRequest(BaseModel):
    action: str = Field(..., description="'play', 'pause', 'step', 'reset', or 'set_speed'")
    speed_multiplier: Optional[float] = Field(1.0, ge=0.1, le=20.0, description="Replay speed factor (1x, 2x, 5x, 10x)")
    mode: Optional[str] = Field(None, description="Switch mode: 'simulation', 'mimic_replay', or 'inference'")


class SimulationStateResponse(BaseModel):
    is_playing: bool = Field(..., description="Whether simulation is currently running")
    speed_multiplier: float = Field(..., description="Current speed multiplier")
    mode: str = Field(..., description="Current operating mode: 'simulation', 'mimic_replay', or 'inference'")
    current_step: int = Field(..., description="Current global simulation step")
    active_anomalies: Dict[str, List[str]] = Field(default_factory=dict, description="Active injected anomaly types per patient")
    status_message: str = Field("Simulation running normally", description="Operational status message")


class StreamVitalsMessage(BaseModel):
    patient_id: str = Field(..., description="Patient ID")
    timestamp: str = Field(..., description="Timestamp")
    step: int = Field(..., description="Simulation step")
    vitals: Dict[str, float] = Field(..., description="Latest vital readings")
    news2_score: int = Field(..., description="Calculated NEWS 2 score")
    news2_tier: str = Field(..., description="Risk tier: Low / Medium / High")
    health_risk_score: float = Field(..., description="Deep learning risk probability")
    is_anomaly_active: bool = Field(False, description="Whether an injected anomaly is currently distorting vitals")
    mode: str = Field("simulation", description="'simulation', 'mimic_replay', or 'inference'")
