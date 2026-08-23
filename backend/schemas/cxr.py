"""
Pydantic schemas for Chest X-Ray (CXR) pulmonary vision diagnostics, studies, and Grad-CAM explainability.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CXRFinding(BaseModel):
    name: str = Field(..., description="Pathology name: Pneumonia, Pleural Effusion, Atelectasis, Pneumothorax, etc.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Model predicted probability score [0.0 - 1.0]")
    threshold: float = Field(0.5, description="Decision threshold for clinical binary classification")
    is_positive: bool = Field(False, description="Whether predicted probability exceeds classification threshold")
    severity: str = Field("none", description="Estimated severity: 'none', 'mild', 'moderate', 'severe'")


class CXRStudy(BaseModel):
    study_id: str = Field(..., description="Unique CXR study / DICOM accession ID")
    patient_id: str = Field(..., description="Associated patient identifier")
    timestamp: str = Field(..., description="Acquisition timestamp")
    view_position: str = Field("PA", description="Radiographic view: PA, AP, or Lateral")
    image_url: str = Field(..., description="API endpoint or static URL to retrieve the processed CXR image")
    radiology_indication: Optional[str] = Field(None, description="Clinical reason for CXR exam")
    radiology_report_summary: Optional[str] = Field(None, description="De-identified radiologist impression text")
    ground_truth_findings: Optional[List[str]] = Field(default_factory=list, description="Expert verified findings if from MIMIC-CXR")


class CXRInferenceRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    study_id: Optional[str] = Field(None, description="Study ID from patient's records, or None if raw image provided")
    image_base64: Optional[str] = Field(None, description="Optional base64 encoded CXR image string for upload")
    target_pathology: Optional[str] = Field("Pneumonia", description="Target pulmonary condition for Grad-CAM explainability")


class CXRInferenceResponse(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    study_id: str = Field(..., description="CXR study identifier")
    timestamp: str = Field(..., description="Inference execution timestamp")
    model_version: str = Field("DenseNet121-MIMICCXR-v1.0", description="Deep learning architecture and version")
    is_model_available: bool = Field(True, description="False if model checkpoint is missing/unavailable (honest fallback)")
    model_status_message: str = Field("Inference executed successfully on PyTorch DenseNet-121", description="Status note")
    findings: List[CXRFinding] = Field(..., description="List of evaluated thoracic pathologies and probabilities")
    top_finding: str = Field(..., description="Highest confidence detected pulmonary finding")
    top_probability: float = Field(..., description="Probability of top finding")
    heatmap_available: bool = Field(True, description="Whether Grad-CAM explainability heatmap is generated")


class CXRHeatmapResponse(BaseModel):
    study_id: str = Field(..., description="CXR study identifier")
    patient_id: str = Field(..., description="Patient identifier")
    target_pathology: str = Field(..., description="Pathology explained by heatmap")
    heatmap_overlay_base64: str = Field(..., description="Base64 encoded PNG of CXR image with Grad-CAM heatmap overlay")
    localization_score: float = Field(..., description="Spatial activation intensity score")
    description: str = Field(..., description="Clinical interpretation of activated anatomical region")
