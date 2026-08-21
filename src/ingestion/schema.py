"""
schema.py
---------
Phase 1 — Canonical observation schema and variable registry.

Defines the contract between ingestion and all downstream modules.
All loaders must return DataFrames whose columns conform to this schema.

This schema proposal is based on known MIMIC-IV table structure.
Final feature set is a student research team decision.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VariableCategory(str, Enum):
    CORE_VITAL = "core_vital"
    LAB_CANDIDATE = "lab_candidate"
    DEMOGRAPHIC = "demographic"
    IDENTIFIER = "identifier"
    TEMPORAL = "temporal"
    DERIVED = "derived"


class DataAvailabilityStatus(str, Enum):
    CONFIRMED = "confirmed"          # Empirically confirmed in loaded data
    CANDIDATE = "candidate"          # From public documentation; unverified locally
    UNAVAILABLE = "unavailable"      # Confirmed absent
    UNKNOWN = "unknown"              # Not yet checked


@dataclass
class VariableSpec:
    """
    Describes a single canonical schema variable.
    
    availability_status distinguishes documentation-derived mappings
    from empirically confirmed ones.
    """
    canonical_name: str
    category: VariableCategory
    description: str
    unit: str
    source_table: str
    source_column: str
    item_id: Optional[int]          # MIMIC-IV itemid if applicable
    expected_frequency: str         # descriptive (e.g. "hourly", "per-episode")
    typical_missingness: str        # descriptive (e.g. "low", "moderate", "high")
    clinical_range_min: Optional[float] = None
    clinical_range_max: Optional[float] = None
    notes: str = ""
    availability_status: DataAvailabilityStatus = DataAvailabilityStatus.CANDIDATE


# ---------------------------------------------------------------------------
# IDENTIFIERS
# ---------------------------------------------------------------------------

IDENTIFIERS: list[VariableSpec] = [
    VariableSpec(
        canonical_name="subject_id",
        category=VariableCategory.IDENTIFIER,
        description="Unique patient identifier",
        unit="id",
        source_table="patients",
        source_column="subject_id",
        item_id=None,
        expected_frequency="once-per-patient",
        typical_missingness="none",
        notes="Primary patient key across all MIMIC tables",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="hadm_id",
        category=VariableCategory.IDENTIFIER,
        description="Hospital admission identifier",
        unit="id",
        source_table="admissions",
        source_column="hadm_id",
        item_id=None,
        expected_frequency="once-per-admission",
        typical_missingness="none",
        notes="Hospital admission key",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="stay_id",
        category=VariableCategory.IDENTIFIER,
        description="ICU stay identifier",
        unit="id",
        source_table="icustays",
        source_column="stay_id",
        item_id=None,
        expected_frequency="once-per-stay",
        typical_missingness="none",
        notes="Primary ICU stay key",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
]


# ---------------------------------------------------------------------------
# CORE VITAL SIGN CANDIDATES
# Source: MIMIC-IV chartevents table
# item_ids: candidate mappings from public MIMIC-IV documentation
# Source documentation: https://mimic.mit.edu/docs/iv/modules/icu/chartevents/
# IMPORTANT: These are CANDIDATE item IDs. Actual presence in the Demo
# dataset must be confirmed by the Phase 1 EDA script.
# ---------------------------------------------------------------------------

CORE_VITAL_CANDIDATES: list[VariableSpec] = [
    VariableSpec(
        canonical_name="heart_rate",
        category=VariableCategory.CORE_VITAL,
        description="Heart rate (beats per minute)",
        unit="bpm",
        source_table="chartevents",
        source_column="value",
        item_id=220045,
        expected_frequency="hourly (charted)",
        typical_missingness="low",
        clinical_range_min=20.0,
        clinical_range_max=300.0,
        notes="Most frequently charted vital in MIMIC-IV ICU stays",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="sbp_arterial",
        category=VariableCategory.CORE_VITAL,
        description="Systolic blood pressure — invasive arterial line",
        unit="mmHg",
        source_table="chartevents",
        source_column="value",
        item_id=220050,
        expected_frequency="continuous (if arterial line)",
        typical_missingness="moderate (only patients with arterial lines)",
        clinical_range_min=40.0,
        clinical_range_max=300.0,
        notes="Invasive; only present if arterial catheter is in situ",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="sbp_noninvasive",
        category=VariableCategory.CORE_VITAL,
        description="Systolic blood pressure — non-invasive (cuff)",
        unit="mmHg",
        source_table="chartevents",
        source_column="value",
        item_id=220179,
        expected_frequency="hourly (charted)",
        typical_missingness="low-moderate",
        clinical_range_min=40.0,
        clinical_range_max=300.0,
        notes="Non-invasive; broader patient coverage than arterial",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="dbp_arterial",
        category=VariableCategory.CORE_VITAL,
        description="Diastolic blood pressure — invasive arterial line",
        unit="mmHg",
        source_table="chartevents",
        source_column="value",
        item_id=220051,
        expected_frequency="continuous (if arterial line)",
        typical_missingness="moderate",
        clinical_range_min=10.0,
        clinical_range_max=200.0,
        notes="Invasive; paired with sbp_arterial",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="dbp_noninvasive",
        category=VariableCategory.CORE_VITAL,
        description="Diastolic blood pressure — non-invasive (cuff)",
        unit="mmHg",
        source_table="chartevents",
        source_column="value",
        item_id=220180,
        expected_frequency="hourly (charted)",
        typical_missingness="low-moderate",
        clinical_range_min=10.0,
        clinical_range_max=200.0,
        notes="Paired with sbp_noninvasive",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="map_arterial",
        category=VariableCategory.CORE_VITAL,
        description="Mean arterial pressure — invasive",
        unit="mmHg",
        source_table="chartevents",
        source_column="value",
        item_id=220052,
        expected_frequency="continuous (if arterial line)",
        typical_missingness="moderate",
        clinical_range_min=20.0,
        clinical_range_max=200.0,
        notes="Also derivable as (SBP+2*DBP)/3 but direct charted value preferred",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="map_noninvasive",
        category=VariableCategory.CORE_VITAL,
        description="Mean arterial pressure — non-invasive",
        unit="mmHg",
        source_table="chartevents",
        source_column="value",
        item_id=220181,
        expected_frequency="hourly (charted)",
        typical_missingness="low-moderate",
        clinical_range_min=20.0,
        clinical_range_max=200.0,
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="spo2",
        category=VariableCategory.CORE_VITAL,
        description="Peripheral oxygen saturation",
        unit="%",
        source_table="chartevents",
        source_column="value",
        item_id=220277,
        expected_frequency="continuous or hourly",
        typical_missingness="low",
        clinical_range_min=50.0,
        clinical_range_max=100.0,
        notes="Recorded from pulse oximetry",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="respiratory_rate",
        category=VariableCategory.CORE_VITAL,
        description="Respiratory rate (breaths per minute)",
        unit="breaths/min",
        source_table="chartevents",
        source_column="value",
        item_id=220210,
        expected_frequency="hourly (charted)",
        typical_missingness="low",
        clinical_range_min=0.0,
        clinical_range_max=80.0,
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="temperature_c",
        category=VariableCategory.CORE_VITAL,
        description="Body temperature — Celsius",
        unit="°C",
        source_table="chartevents",
        source_column="value",
        item_id=223762,
        expected_frequency="every 4-8 hours",
        typical_missingness="moderate",
        clinical_range_min=25.0,
        clinical_range_max=45.0,
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="temperature_f",
        category=VariableCategory.CORE_VITAL,
        description="Body temperature — Fahrenheit (needs unit conversion)",
        unit="°F",
        source_table="chartevents",
        source_column="value",
        item_id=223761,
        expected_frequency="every 4-8 hours",
        typical_missingness="moderate",
        clinical_range_min=77.0,
        clinical_range_max=113.0,
        notes="Convert to Celsius: (F - 32) * 5/9. Item 223762 (Celsius) preferred.",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
]


# ---------------------------------------------------------------------------
# LAB CANDIDATES
# Source: MIMIC-IV labevents table (itemid from d_labitems)
# Source documentation: https://mimic.mit.edu/docs/iv/modules/hosp/labevents/
# IMPORTANT: Candidate item IDs only. Confirm presence in EDA.
# ---------------------------------------------------------------------------

LAB_CANDIDATES: list[VariableSpec] = [
    VariableSpec(
        canonical_name="glucose_lab",
        category=VariableCategory.LAB_CANDIDATE,
        description="Blood glucose (laboratory)",
        unit="mg/dL",
        source_table="labevents",
        source_column="value",
        item_id=50931,
        expected_frequency="per episode (ordered)",
        typical_missingness="moderate",
        clinical_range_min=10.0,
        clinical_range_max=2000.0,
        notes="Also available in chartevents (item 220621) as bedside glucose",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="glucose_chart",
        category=VariableCategory.LAB_CANDIDATE,
        description="Blood glucose (bedside/chartevents)",
        unit="mg/dL",
        source_table="chartevents",
        source_column="value",
        item_id=220621,
        expected_frequency="per episode",
        typical_missingness="moderate",
        clinical_range_min=10.0,
        clinical_range_max=2000.0,
        notes="Bedside glucose measurement; may overlap with glucose_lab",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="lactate",
        category=VariableCategory.LAB_CANDIDATE,
        description="Serum lactate",
        unit="mmol/L",
        source_table="labevents",
        source_column="value",
        item_id=50813,
        expected_frequency="per episode (ordered on suspicion)",
        typical_missingness="high (ordered selectively)",
        clinical_range_min=0.0,
        clinical_range_max=30.0,
        notes="High clinical significance for sepsis/shock; high missingness is expected",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="creatinine",
        category=VariableCategory.LAB_CANDIDATE,
        description="Serum creatinine",
        unit="mg/dL",
        source_table="labevents",
        source_column="value",
        item_id=50912,
        expected_frequency="daily or per episode",
        typical_missingness="low-moderate",
        clinical_range_min=0.01,
        clinical_range_max=30.0,
        notes="Indicator of renal function; frequently ordered",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="wbc",
        category=VariableCategory.LAB_CANDIDATE,
        description="White blood cell count",
        unit="K/uL",
        source_table="labevents",
        source_column="value",
        item_id=51301,
        expected_frequency="daily or per episode",
        typical_missingness="low-moderate",
        clinical_range_min=0.0,
        clinical_range_max=500.0,
        notes="Part of complete blood count (CBC); frequently ordered",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
]


# ---------------------------------------------------------------------------
# DEMOGRAPHIC / CONTEXT CANDIDATES
# ---------------------------------------------------------------------------

DEMOGRAPHIC_CANDIDATES: list[VariableSpec] = [
    VariableSpec(
        canonical_name="age",
        category=VariableCategory.DEMOGRAPHIC,
        description="Patient age at ICU admission",
        unit="years",
        source_table="patients + admissions",
        source_column="anchor_age (patients) / admittime (admissions)",
        item_id=None,
        expected_frequency="once-per-stay",
        typical_missingness="none",
        notes=(
            "MIMIC-IV uses anchor_age shifted to anchor_year for privacy. "
            "Age > 89 is capped at 91 in MIMIC-IV."
        ),
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
    VariableSpec(
        canonical_name="sex",
        category=VariableCategory.DEMOGRAPHIC,
        description="Patient biological sex",
        unit="M/F",
        source_table="patients",
        source_column="gender",
        item_id=None,
        expected_frequency="once-per-patient",
        typical_missingness="none",
        availability_status=DataAvailabilityStatus.CANDIDATE,
    ),
]


# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------

ALL_VARIABLE_SPECS: list[VariableSpec] = (
    IDENTIFIERS
    + CORE_VITAL_CANDIDATES
    + LAB_CANDIDATES
    + DEMOGRAPHIC_CANDIDATES
)

VARIABLE_REGISTRY: dict[str, VariableSpec] = {
    spec.canonical_name: spec for spec in ALL_VARIABLE_SPECS
}

# CHARTEVENTS item IDs for the core vital candidates (for filtering)
CHARTEVENTS_VITAL_ITEM_IDS: list[int] = [
    spec.item_id
    for spec in CORE_VITAL_CANDIDATES + LAB_CANDIDATES
    if spec.item_id is not None and spec.source_table == "chartevents"
]

# LABEVENTS item IDs for lab candidates (for filtering)
LABEVENTS_ITEM_IDS: list[int] = [
    spec.item_id
    for spec in LAB_CANDIDATES
    if spec.item_id is not None and spec.source_table == "labevents"
]


# ---------------------------------------------------------------------------
# Canonical output column order for processed observation DataFrames
# ---------------------------------------------------------------------------

CANONICAL_OBSERVATION_COLUMNS: list[str] = [
    # Identifiers
    "subject_id",
    "hadm_id",
    "stay_id",
    # Temporal
    "charttime",
    "storetime",
    # Source
    "itemid",
    "canonical_name",
    "value",
    "valuenum",
    "valueuom",
]
