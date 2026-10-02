"""
SQLAlchemy ORM models for the MIMIC-CXR pipeline and Pulmonary ICU Patient Digital Twin.
Defines Patient and CXRStudy relational entities mapping directly to MIMIC-CXR cohort metadata.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Integer, String, Float, Text, DateTime, ForeignKey, Index, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base class for SQLAlchemy 2.0 models."""
    pass


class Patient(Base):
    """
    Patient demographic and cohort entity.
    Represents an ICU patient in the Digital Twin and links to longitudinal CXR studies.
    """
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    split: Mapped[str] = mapped_column(String(32), default="train", index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # 1-to-many relationship: One patient has multiple CXR studies across their ICU stay
    studies: Mapped[List["CXRStudy"]] = relationship(
        "CXRStudy",
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="CXRStudy.id"
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, subject_id='{self.subject_id}', split='{self.split}')>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert Patient model instance to dictionary representation."""
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "split": self.split,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CXRStudy(Base):
    """
    Chest X-Ray Study entity.
    Maps directly to MIMIC-CXR cohort metadata and holds ground-truth pathology labels,
    image disk references, radiology report impressions, and inference results.
    """
    __tablename__ = "cxr_studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    dicom_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    subject_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("patients.subject_id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )
    view_position: Mapped[str] = mapped_column(String(16), default="PA", nullable=False)
    image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    resolved_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    split: Mapped[str] = mapped_column(String(32), default="train", index=True, nullable=False)
    report_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 8 Pulmonary Pathology Ground-Truth Labels from Cohort Metadata
    pneumonia: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    pleural_effusion: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    atelectasis: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    consolidation: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    edema: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    pneumothorax: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cardiomegaly: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    no_finding: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Deep Learning Inference & Explainability Cache
    predicted_findings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON-encoded predictions
    top_finding: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    top_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    inferred_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship back to Patient
    patient: Mapped["Patient"] = relationship("Patient", back_populates="studies")

    __table_args__ = (
        Index("idx_study_dicom", "study_id", "dicom_id"),
        Index("idx_study_patient", "subject_id", "study_id"),
    )

    def __repr__(self) -> str:
        return f"<CXRStudy(id={self.id}, study_id='{self.study_id}', dicom_id='{self.dicom_id}', subject_id='{self.subject_id}')>"

    @property
    def pathology_dict(self) -> Dict[str, float]:
        """Return dictionary of 8 target pulmonary pathologies with ground-truth values."""
        return {
            "Pneumonia": self.pneumonia,
            "Pleural Effusion": self.pleural_effusion,
            "Atelectasis": self.atelectasis,
            "Consolidation": self.consolidation,
            "Edema": self.edema,
            "Pneumothorax": self.pneumothorax,
            "Cardiomegaly": self.cardiomegaly,
            "No Finding": self.no_finding,
        }

    @property
    def positive_findings(self) -> List[str]:
        """Return list of pathology names where ground-truth label == 1.0."""
        return [name for name, val in self.pathology_dict.items() if val == 1.0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert CXRStudy model instance to dictionary representation."""
        return {
            "id": self.id,
            "study_id": self.study_id,
            "dicom_id": self.dicom_id,
            "subject_id": self.subject_id,
            "view_position": self.view_position,
            "image_path": self.image_path,
            "resolved_path": self.resolved_path,
            "split": self.split,
            "report_text": self.report_text,
            "pneumonia": self.pneumonia,
            "pleural_effusion": self.pleural_effusion,
            "atelectasis": self.atelectasis,
            "consolidation": self.consolidation,
            "edema": self.edema,
            "pneumothorax": self.pneumothorax,
            "cardiomegaly": self.cardiomegaly,
            "no_finding": self.no_finding,
            "positive_findings": self.positive_findings,
            "top_finding": self.top_finding,
            "top_probability": self.top_probability,
            "inferred_at": self.inferred_at.isoformat() if self.inferred_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
