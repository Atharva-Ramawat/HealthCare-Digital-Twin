"""
src/ingestion/__init__.py
Phase 1 — Ingestion module public API.
"""
from src.ingestion.data_availability import (
    DataAvailabilityChecker,
    DataNotAvailableError,
    AvailabilityReport,
)
from src.ingestion.schema import (
    VARIABLE_REGISTRY,
    ALL_VARIABLE_SPECS,
    CORE_VITAL_CANDIDATES,
    LAB_CANDIDATES,
    DEMOGRAPHIC_CANDIDATES,
    CHARTEVENTS_VITAL_ITEM_IDS,
    LABEVENTS_ITEM_IDS,
    VariableCategory,
    DataAvailabilityStatus,
)
from src.ingestion.mimic_iv_loader import MIMICIVLoader
from src.ingestion.mimic_iv_demo_loader import MIMICIVDemoLoader
from src.ingestion.synthetic_generator import SyntheticClinicalGenerator

__all__ = [
    "DataAvailabilityChecker",
    "DataNotAvailableError",
    "AvailabilityReport",
    "VARIABLE_REGISTRY",
    "ALL_VARIABLE_SPECS",
    "CORE_VITAL_CANDIDATES",
    "LAB_CANDIDATES",
    "DEMOGRAPHIC_CANDIDATES",
    "CHARTEVENTS_VITAL_ITEM_IDS",
    "LABEVENTS_ITEM_IDS",
    "VariableCategory",
    "DataAvailabilityStatus",
    "MIMICIVLoader",
    "MIMICIVDemoLoader",
    "SyntheticClinicalGenerator",
]
