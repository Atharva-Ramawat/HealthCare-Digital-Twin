"""
Pydantic schemas for patient demographics, summaries, clinical timeline events, and medications.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PatientBase(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier e.g. PAT-101 or MIMIC subject_id")
    name: str = Field(..., description="Full patient name")
    age: int = Field(..., ge=0, le=125, description="Patient age in years")
    gender: str = Field(..., description="Gender: M / F / Other")
    bed: str = Field(..., description="Assigned ICU bed identifier")
    admission_time: str = Field(..., description="ISO 8601 admission timestamp")
    primary_diagnosis: str = Field(..., description="Primary clinical condition / ICU admission reason")


class PatientSummary(PatientBase):
    mode: str = Field("simulation", description="Active mode: 'simulation', 'mimic_replay', or 'inference'")
    trajectory_template: Optional[str] = Field(None, description="Active simulation trajectory template if applicable")
    latest_vitals: Dict[str, Any] = Field(default_factory=dict, description="Current latest vital sign measurements")
    news2_score: int = Field(0, description="Latest calculated NEWS 2 score")
    news2_tier: str = Field("Low", description="Clinical risk tier: Low / Medium / High")
    health_risk_score: float = Field(0.0, ge=0.0, le=1.0, description="Deep learning predicted health risk probability")
    is_model_available: bool = Field(True, description="Whether ML prediction comes from an active model or placeholder")


class PatientDetail(PatientSummary):
    medical_history: List[str] = Field(default_factory=list, description="Chronic pre-existing conditions")
    allergies: List[str] = Field(default_factory=list, description="Known drug and medical allergies")
    attending_physician: str = Field("Dr. A. Sharma (ICU Lead)", description="Assigned clinical physician")
    icu_stay_id: Optional[str] = Field(None, description="MIMIC-IV ICU stay ID if applicable")


class PatientEvent(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    patient_id: str = Field(..., description="Associated patient ID")
    timestamp: str = Field(..., description="Event timestamp")
    event_type: str = Field(..., description="Category: e.g. 'admission', 'vital_anomaly', 'lab_result', 'medication_change', 'cxr_ordered'")
    title: str = Field(..., description="Brief clinical headline")
    description: str = Field(..., description="Detailed clinical notes / context")
    severity: str = Field("normal", description="'normal', 'warning', or 'critical'")


class PatientMedication(BaseModel):
    medication_id: str = Field(..., description="Unique medication order ID")
    patient_id: str = Field(..., description="Associated patient ID")
    name: str = Field(..., description="Pharmaceutical drug name")
    dosage: str = Field(..., description="Prescribed dose e.g. '500 mg', '0.05 mcg/kg/min'")
    route: str = Field(..., description="Administration route: IV / Oral / Inhalation")
    frequency: str = Field(..., description="Frequency: e.g. 'Continuous IV', 'q6h'")
    start_time: str = Field(..., description="Start timestamp")
    is_active: bool = Field(True, description="Whether currently active")
