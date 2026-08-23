"""
Pydantic schemas for multi-horizon deterioration risk, trajectory forecasting, XAI explanations, and the Digital Twin state.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class MultiHorizonRisk(BaseModel):
    horizon_1h: float = Field(..., ge=0.0, le=1.0, description="1-hour deterioration risk probability")
    horizon_3h: float = Field(..., ge=0.0, le=1.0, description="3-hour deterioration risk probability")
    horizon_6h: float = Field(..., ge=0.0, le=1.0, description="6-hour deterioration risk probability")
    primary_endpoint: str = Field("Respiratory Failure / Mechanical Ventilation Initiation", description="Clinical event target")


class NEWS2Breakdown(BaseModel):
    total_score: int = Field(..., ge=0, le=20, description="Composite National Early Warning Score 2")
    risk_tier: str = Field(..., description="'Low', 'Medium', or 'High'")
    respiratory_rate_score: int = Field(0)
    spo2_score: int = Field(0)
    systolic_bp_score: int = Field(0)
    heart_rate_score: int = Field(0)
    temperature_score: int = Field(0)
    clinical_recommendation: str = Field(..., description="NEWS 2 protocol clinical monitoring guidance")


class VitalForecastHorizon(BaseModel):
    forecast_horizon_minutes: int = Field(15, description="Forecast duration in minutes")
    timestamps: List[str] = Field(default_factory=list, description="Future projected minute timestamps")
    predicted_vitals: Dict[str, List[float]] = Field(default_factory=dict, description="Projected trajectory values per vital")


class XAIAttributionItem(BaseModel):
    feature_key: str = Field(..., description="Vital sign parameter key e.g. 'spo2'")
    display_name: str = Field(..., description="Human readable name e.g. 'Oxygen Saturation'")
    impact_percentage: float = Field(..., ge=0.0, le=100.0, description="Relative percentage contribution to risk score")
    direction: str = Field(..., description="'Risk Escalator' or 'Protective/Normal'")
    current_value: float = Field(..., description="Patient's current reading")
    baseline_value: float = Field(..., description="Patient's personal baseline reading")


class RiskPredictionResponse(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: str = Field(..., description="Prediction timestamp")
    health_risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall health deterioration risk score")
    is_model_available: bool = Field(True, description="Whether real PyTorch model weights executed")
    model_version: str = Field("CNN-BiLSTM-MIMICIV-v1.0", description="Model architecture identifier")
    news2: NEWS2Breakdown = Field(..., description="NEWS 2 clinical score breakdown")
    multi_horizon: MultiHorizonRisk = Field(..., description="Multi-horizon risk probabilities (1h, 3h, 6h)")
    forecast: VitalForecastHorizon = Field(..., description="15-minute vital trajectory predictions")
    xai_attributions: List[XAIAttributionItem] = Field(default_factory=list, description="Integrated Gradients vital contributions")


class DigitalTwinStateResponse(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: str = Field(..., description="State timestamp")
    mode: str = Field("simulation", description="'simulation', 'mimic_replay', or 'inference'")
    observed_state: Dict[str, Any] = Field(..., description="Raw vitals, demographics, data source, step")
    derived_state: Dict[str, Any] = Field(..., description="Baselines, deltas, rates, MAP, Shock Index, NEWS2")
    predicted_state: Dict[str, Any] = Field(..., description="Risk score, multi-horizon, forecast, XAI, model validity")
    simulation_state: Dict[str, Any] = Field(..., description="Replay status, speed, active anomalies, what-if overrides")
