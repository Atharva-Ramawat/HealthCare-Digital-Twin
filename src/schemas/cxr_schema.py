"""
Pydantic v2 schemas and API contracts for MIMIC-CXR studies and deep learning inference.
Defines StudyBase, StudyResponse, and InferenceResult contracts with ORM mode compatibility.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


# Target 8 Pulmonary Pathologies matching cohort metadata
TARGET_PULMONARY_CLASSES = [
    "Pneumonia",
    "Pleural Effusion",
    "Atelectasis",
    "Consolidation",
    "Edema",
    "Pneumothorax",
    "Cardiomegaly",
    "No Finding"
]


class StudyBase(BaseModel):
    """Base schema for a Chest X-Ray study and its clinical metadata."""
    study_id: str = Field(..., description="Unique CXR study identifier (e.g. 's50084553')")
    subject_id: str = Field(..., description="Associated patient subject ID from MIMIC-CXR")
    dicom_id: Optional[str] = Field(None, description="DICOM image identifier")
    view_position: str = Field("PA", description="Radiographic view: PA, AP, Lateral")
    image_path: Optional[str] = Field(None, description="Relative storage path starting with files/...")
    split: str = Field("train", description="Dataset split partition: 'train', 'validate', or 'test'")
    report_text: Optional[str] = Field(None, description="Associated de-identified radiology report text")

    # 8 Pulmonary Pathology Ground-Truth Labels from Cohort Metadata
    pneumonia: float = Field(0.0, ge=0.0, le=1.0, description="Pneumonia label (0.0 or 1.0)")
    pleural_effusion: float = Field(0.0, ge=0.0, le=1.0, description="Pleural Effusion label (0.0 or 1.0)")
    atelectasis: float = Field(0.0, ge=0.0, le=1.0, description="Atelectasis label (0.0 or 1.0)")
    consolidation: float = Field(0.0, ge=0.0, le=1.0, description="Consolidation label (0.0 or 1.0)")
    edema: float = Field(0.0, ge=0.0, le=1.0, description="Edema label (0.0 or 1.0)")
    pneumothorax: float = Field(0.0, ge=0.0, le=1.0, description="Pneumothorax label (0.0 or 1.0)")
    cardiomegaly: float = Field(0.0, ge=0.0, le=1.0, description="Cardiomegaly label (0.0 or 1.0)")
    no_finding: float = Field(0.0, ge=0.0, le=1.0, description="No Finding label (0.0 or 1.0)")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class StudyResponse(StudyBase):
    """Response schema for a persisted CXR study including database ID and resolved image URL."""
    id: int = Field(..., description="Unique database primary key")
    image_url: Optional[str] = Field(None, description="API endpoint to fetch or stream the CXR image")
    resolved_path: Optional[str] = Field(None, description="Physical path on disk if available")
    ground_truth_findings: List[str] = Field(
        default_factory=list,
        description="List of positive findings where label == 1.0"
    )
    top_finding: Optional[str] = Field(None, description="Model-inferred top pathology if evaluated")
    top_probability: Optional[float] = Field(None, description="Inferred confidence of top pathology")
    inferred_at: Optional[datetime] = Field(None, description="Timestamp of inference execution")
    created_at: Optional[datetime] = Field(None, description="Record creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Record last update timestamp")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="after")
    def populate_derived_fields(self) -> "StudyResponse":
        """Automatically derive ground_truth_findings and image_url if not explicitly provided."""
        if not self.ground_truth_findings:
            findings = []
            if self.pneumonia == 1.0: findings.append("Pneumonia")
            if self.pleural_effusion == 1.0: findings.append("Pleural Effusion")
            if self.atelectasis == 1.0: findings.append("Atelectasis")
            if self.consolidation == 1.0: findings.append("Consolidation")
            if self.edema == 1.0: findings.append("Edema")
            if self.pneumothorax == 1.0: findings.append("Pneumothorax")
            if self.cardiomegaly == 1.0: findings.append("Cardiomegaly")
            if self.no_finding == 1.0: findings.append("No Finding")
            self.ground_truth_findings = findings

        if not self.image_url and self.study_id:
            self.image_url = f"/api/cxr/studies/{self.study_id}/image"
        return self


class InferenceResult(BaseModel):
    """Schema for deep learning model predictions and Grad-CAM explainability outputs."""
    study_id: str = Field(..., description="Unique CXR study identifier")
    subject_id: str = Field(..., description="Associated patient subject ID")
    model_version: str = Field("DenseNet121-MIMICCXR-v1.0", description="Model architecture identifier and version")
    probabilities: Dict[str, float] = Field(..., description="Multi-label probability distribution per target pathology")
    predictions: Dict[str, bool] = Field(..., description="Binary classifications using decision thresholds")
    top_finding: str = Field(..., description="Pathology with the highest predicted confidence probability")
    top_probability: float = Field(..., ge=0.0, le=1.0, description="Highest predicted probability score")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Decision thresholds applied per class")
    has_pathology: bool = Field(False, description="True if any pathology is classified positive beyond No Finding")
    heatmap_available: bool = Field(False, description="Whether Grad-CAM spatial activation map is generated")
    heatmap_url: Optional[str] = Field(None, description="Endpoint to retrieve Grad-CAM overlay visualization")
    latent_embedding: Optional[List[float]] = Field(None, description="Optional 1024-dim dense feature embedding")
    execution_time_ms: Optional[float] = Field(None, description="Inference runtime in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of inference execution")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class HeatmapResponse(BaseModel):
    """Schema for Grad-CAM explainability heatmap retrieval."""
    study_id: str = Field(..., description="Unique CXR study identifier")
    subject_id: str = Field(..., description="Associated patient subject ID")
    pathology: str = Field(..., description="Pathology explained by heatmap")
    heatmap_available: bool = Field(True, description="Whether heatmap is available")
    heatmap_url: Optional[str] = Field(None, description="Direct URL to view/stream the heatmap image")
    heatmap_overlay_base64: Optional[str] = Field(None, description="Base64 encoded PNG of heatmap overlay if requested")
    localization_score: float = Field(0.85, description="Spatial activation intensity score")
    description: str = Field(..., description="Clinical interpretation of activated anatomical region")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of retrieval")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

