"""
Pydantic v2 schemas and API contracts for Digital Twin Multimodal Fusion and ICU Risk Scoring.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class VitalSignSnapshot(BaseModel):
    """Snapshot of latest patient vital signs."""
    heart_rate: float = Field(..., description="Heart Rate in beats per minute (bpm)", ge=20.0, le=250.0)
    spo2: float = Field(..., description="Blood Oxygen Saturation percentage (%)", ge=50.0, le=100.0)
    sbp: float = Field(..., description="Systolic Blood Pressure in mmHg", ge=40.0, le=260.0)
    respiratory_rate: float = Field(..., description="Respiratory Rate in breaths per minute", ge=4.0, le=70.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of vital observation")

    model_config = ConfigDict(from_attributes=True)


class VitalReadingItem(BaseModel):
    """Single discrete vital reading in a temporal trajectory."""
    step_index: int = Field(..., description="15-minute interval step index (0 to 95)")
    timestamp: datetime = Field(..., description="Observation timestamp")
    heart_rate: float = Field(..., description="Heart Rate (bpm)")
    spo2: float = Field(..., description="SpO2 (%)")
    sbp: float = Field(..., description="Systolic BP (mmHg)")
    respiratory_rate: float = Field(..., description="Respiratory Rate (breaths/min)")

    model_config = ConfigDict(from_attributes=True)


class DigitalTwinResponse(BaseModel):
    """Complete multimodal Digital Twin response schema integrating vision inference and vitals."""
    patient_id: str = Field(..., description="Patient subject identifier")
    study_id: str = Field(..., description="CXR study identifier")
    deterioration_risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Unified ICU Deterioration Risk Score in percentage (0% to 100%)"
    )
    risk_tier: str = Field(..., description="Clinical risk category: Low, Moderate, High, or Critical")
    cxr_probabilities: Dict[str, float] = Field(..., description="8-class pulmonary pathology probabilities from DenseNet-121")
    cxr_predictions: Dict[str, bool] = Field(..., description="Binary classifications using clinical decision thresholds")
    cxr_top_finding: str = Field(..., description="Primary detected pulmonary finding")
    cxr_top_probability: float = Field(..., ge=0.0, le=1.0, description="Confidence of primary pulmonary finding")
    current_vitals: VitalSignSnapshot = Field(..., description="Latest snapshot of patient vital signs")
    vitals_trajectory_24h: Optional[List[VitalReadingItem]] = Field(
        default=None,
        description="Optional full 24-hour temporal trajectory with 15-minute resolution (96 steps)"
    )
    visual_risk_contribution: float = Field(..., description="Percentage contribution of visual findings to risk score")
    vitals_risk_contribution: float = Field(..., description="Percentage contribution of vital signs to risk score")
    risk_factors: List[str] = Field(default_factory=list, description="Primary clinical drivers of deterioration risk")
    heatmap_base64: Optional[str] = Field(
        default=None,
        description="Base64 PNG data URL of Grad-CAM feature attribution heatmap overlay"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 PNG data URL of uploaded input chest radiograph"
    )
    heatmap_image_base64: Optional[str] = Field(
        default=None,
        description="Base64 PNG data URL of Grad-CAM feature attribution heatmap overlay"
    )
    input_image_base64: Optional[str] = Field(
        default=None,
        description="Base64 PNG data URL of uploaded input chest radiograph"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of evaluation")

    model_config = ConfigDict(from_attributes=True)
