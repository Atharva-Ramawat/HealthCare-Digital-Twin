"""
mimic_iv_item_ids.py
--------------------
Phase 1 — MIMIC-IV candidate item ID reference.

PURPOSE
-------
This file is a DOCUMENTATION REFERENCE, not executable production code.
It lists candidate item IDs for physiological variables of interest, derived
from publicly available MIMIC-IV documentation.

IMPORTANT — DOCUMENTATION vs EMPIRICAL
---------------------------------------
All item IDs below are CANDIDATE MAPPINGS derived from:
  - MIMIC-IV official documentation: https://mimic.mit.edu/docs/iv/
  - MIMIC-IV GitHub: https://github.com/MIT-LCP/mimic-iv
  - Published MIMIC-IV papers and tutorials

They are NOT guaranteed to be present in any particular dataset extract,
including the MIMIC-IV Demo.

Status values:
  CANDIDATE  = From public documentation; must be verified in EDA
  CONFIRMED  = Verified empirically in loaded data (updated by EDA script)

Source documentation version: MIMIC-IV v2.2
Last reviewed: August 2026
"""
from __future__ import annotations

# =============================================================================
# CHARTEVENTS ITEM IDs (icu/d_items)
# Source: MIMIC-IV d_items table, ICU module
# =============================================================================

CHARTEVENTS_ITEMS = {
    # -------------------------------------------------------------------------
    # Heart Rate
    # -------------------------------------------------------------------------
    220045: {
        "label": "Heart Rate",
        "abbreviation": "HR",
        "category": "Routine Vital Signs",
        "unit": "bpm",
        "canonical_name": "heart_rate",
        "clinical_range": (20, 300),
        "status": "CANDIDATE",
        "notes": "Primary heart rate item; most frequently charted vital in ICU",
    },

    # -------------------------------------------------------------------------
    # Blood Pressure — Arterial (invasive)
    # -------------------------------------------------------------------------
    220050: {
        "label": "Arterial Blood Pressure systolic",
        "abbreviation": "ABPs",
        "category": "Routine Vital Signs",
        "unit": "mmHg",
        "canonical_name": "sbp_arterial",
        "clinical_range": (40, 300),
        "status": "CANDIDATE",
        "notes": "Invasive; requires arterial line. High-frequency continuous recording.",
    },
    220051: {
        "label": "Arterial Blood Pressure diastolic",
        "abbreviation": "ABPd",
        "category": "Routine Vital Signs",
        "unit": "mmHg",
        "canonical_name": "dbp_arterial",
        "clinical_range": (10, 200),
        "status": "CANDIDATE",
        "notes": "Invasive; paired with 220050.",
    },
    220052: {
        "label": "Arterial Blood Pressure mean",
        "abbreviation": "ABPm",
        "category": "Routine Vital Signs",
        "unit": "mmHg",
        "canonical_name": "map_arterial",
        "clinical_range": (20, 200),
        "status": "CANDIDATE",
        "notes": "Invasive mean arterial pressure. Also derivable as (SBP+2*DBP)/3.",
    },

    # -------------------------------------------------------------------------
    # Blood Pressure — Non-invasive (cuff)
    # -------------------------------------------------------------------------
    220179: {
        "label": "Non Invasive Blood Pressure systolic",
        "abbreviation": "NBPs",
        "category": "Routine Vital Signs",
        "unit": "mmHg",
        "canonical_name": "sbp_noninvasive",
        "clinical_range": (40, 300),
        "status": "CANDIDATE",
        "notes": "Non-invasive (cuff); broader patient coverage than arterial.",
    },
    220180: {
        "label": "Non Invasive Blood Pressure diastolic",
        "abbreviation": "NBPd",
        "category": "Routine Vital Signs",
        "unit": "mmHg",
        "canonical_name": "dbp_noninvasive",
        "clinical_range": (10, 200),
        "status": "CANDIDATE",
        "notes": "Non-invasive; paired with 220179.",
    },
    220181: {
        "label": "Non Invasive Blood Pressure mean",
        "abbreviation": "NBPm",
        "category": "Routine Vital Signs",
        "unit": "mmHg",
        "canonical_name": "map_noninvasive",
        "clinical_range": (20, 200),
        "status": "CANDIDATE",
        "notes": "Non-invasive mean arterial pressure.",
    },

    # -------------------------------------------------------------------------
    # SpO2
    # -------------------------------------------------------------------------
    220277: {
        "label": "O2 Saturation Pulseoxymetry",
        "abbreviation": "SpO2",
        "category": "Routine Vital Signs",
        "unit": "%",
        "canonical_name": "spo2",
        "clinical_range": (50, 100),
        "status": "CANDIDATE",
        "notes": "Pulse oximetry reading; frequently sampled.",
    },

    # -------------------------------------------------------------------------
    # Respiratory Rate
    # -------------------------------------------------------------------------
    220210: {
        "label": "Respiratory Rate",
        "abbreviation": "RR",
        "category": "Routine Vital Signs",
        "unit": "insp/min",
        "canonical_name": "respiratory_rate",
        "clinical_range": (0, 80),
        "status": "CANDIDATE",
        "notes": "Charted respiratory rate.",
    },

    # -------------------------------------------------------------------------
    # Temperature
    # -------------------------------------------------------------------------
    223762: {
        "label": "Temperature Celsius",
        "abbreviation": "Temp C",
        "category": "Routine Vital Signs",
        "unit": "°C",
        "canonical_name": "temperature_c",
        "clinical_range": (25, 45),
        "status": "CANDIDATE",
        "notes": "Preferred temperature item (Celsius). Use this over Fahrenheit.",
    },
    223761: {
        "label": "Temperature Fahrenheit",
        "abbreviation": "Temp F",
        "category": "Routine Vital Signs",
        "unit": "°F",
        "canonical_name": "temperature_f",
        "clinical_range": (77, 113),
        "status": "CANDIDATE",
        "notes": "Convert to Celsius: (F - 32) * 5/9. Item 223762 preferred.",
    },

    # -------------------------------------------------------------------------
    # Glucose (bedside / chartevents)
    # -------------------------------------------------------------------------
    220621: {
        "label": "Glucose (serum)",
        "abbreviation": "Glucose",
        "category": "Labs",
        "unit": "mg/dL",
        "canonical_name": "glucose_chart",
        "clinical_range": (10, 2000),
        "status": "CANDIDATE",
        "notes": "Bedside glucose in chartevents. Also available in labevents (50931).",
    },
}


# =============================================================================
# LABEVENTS ITEM IDs (hosp/d_labitems)
# Source: MIMIC-IV d_labitems table, hosp module
# =============================================================================

LABEVENTS_ITEMS = {
    # -------------------------------------------------------------------------
    # Glucose (laboratory)
    # -------------------------------------------------------------------------
    50931: {
        "label": "Glucose",
        "fluid": "Blood",
        "category": "Chemistry",
        "unit": "mg/dL",
        "loinc_code": "2345-7",
        "canonical_name": "glucose_lab",
        "clinical_range": (10, 2000),
        "status": "CANDIDATE",
        "notes": "Lab glucose; may overlap with chartevents item 220621.",
    },

    # -------------------------------------------------------------------------
    # Lactate
    # -------------------------------------------------------------------------
    50813: {
        "label": "Lactate",
        "fluid": "Blood",
        "category": "Blood Gas",
        "unit": "mmol/L",
        "loinc_code": "2524-7",
        "canonical_name": "lactate",
        "clinical_range": (0, 30),
        "status": "CANDIDATE",
        "notes": (
            "High clinical significance for sepsis/shock detection. "
            "Ordered selectively — expect high missingness."
        ),
    },

    # -------------------------------------------------------------------------
    # Creatinine
    # -------------------------------------------------------------------------
    50912: {
        "label": "Creatinine",
        "fluid": "Blood",
        "category": "Chemistry",
        "unit": "mg/dL",
        "loinc_code": "2160-0",
        "canonical_name": "creatinine",
        "clinical_range": (0.01, 30),
        "status": "CANDIDATE",
        "notes": "Renal function marker; frequently ordered in ICU.",
    },

    # -------------------------------------------------------------------------
    # White Blood Cells
    # -------------------------------------------------------------------------
    51301: {
        "label": "White Blood Cells",
        "fluid": "Blood",
        "category": "Hematology",
        "unit": "K/uL",
        "loinc_code": "6690-2",
        "canonical_name": "wbc",
        "clinical_range": (0, 500),
        "status": "CANDIDATE",
        "notes": "Part of CBC; frequently ordered. Relevant for infection/sepsis markers.",
    },
}


# =============================================================================
# Convenience exports
# =============================================================================

ALL_TARGET_CHARTEVENTS_IDS: list[int] = list(CHARTEVENTS_ITEMS.keys())
ALL_TARGET_LABEVENTS_IDS: list[int] = list(LABEVENTS_ITEMS.keys())


def get_canonical_name(item_id: int) -> str | None:
    """Return canonical variable name for a given item_id, or None."""
    if item_id in CHARTEVENTS_ITEMS:
        return CHARTEVENTS_ITEMS[item_id]["canonical_name"]
    if item_id in LABEVENTS_ITEMS:
        return LABEVENTS_ITEMS[item_id]["canonical_name"]
    return None


def get_unit(item_id: int) -> str | None:
    """Return unit for a given item_id, or None."""
    if item_id in CHARTEVENTS_ITEMS:
        return CHARTEVENTS_ITEMS[item_id]["unit"]
    if item_id in LABEVENTS_ITEMS:
        return LABEVENTS_ITEMS[item_id]["unit"]
    return None


if __name__ == "__main__":
    print("MIMIC-IV Candidate Item IDs (Phase 1 Reference)")
    print("=" * 60)
    print(f"\nChartevents items: {len(CHARTEVENTS_ITEMS)}")
    for item_id, info in CHARTEVENTS_ITEMS.items():
        print(f"  {item_id}: {info['label']} ({info['unit']}) [{info['status']}]")
    print(f"\nLabevents items: {len(LABEVENTS_ITEMS)}")
    for item_id, info in LABEVENTS_ITEMS.items():
        print(f"  {item_id}: {info['label']} ({info['unit']}) [{info['status']}]")
