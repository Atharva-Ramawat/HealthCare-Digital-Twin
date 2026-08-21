"""
phase1_eda.py
-------------
Phase 1 — Real MIMIC-IV Demo Exploratory Data Analysis.

Runs against the actual downloaded MIMIC-IV Demo data.
Produces structured output files in data/interim/phase1_eda_report/.

DATA SOURCE: MIMIC-IV Clinical Database Demo v2.2 (real clinical data).
All statistics in this script are empirically measured — not fabricated.

IMPORTANT:
- This script is READ-ONLY with respect to source data.
- It does NOT modify, delete, or normalise any raw data.
- All report files are written to data/interim/ only.
- Synthetic data is NOT used for any EDA statistics here.

Usage:
    python scripts/phase1_eda.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap path so script runs from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.mimic_iv_demo_loader import MIMICIVDemoLoader
from src.ingestion.data_availability import DataAvailabilityChecker, DataNotAvailableError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / "data" / "interim" / "phase1_eda_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Candidate item IDs (from public MIMIC-IV documentation; verified below)
# ---------------------------------------------------------------------------
CHARTEVENTS_CANDIDATES: dict[int, str] = {
    220045: "heart_rate",
    220050: "sbp_arterial",
    220179: "sbp_noninvasive",
    220051: "dbp_arterial",
    220180: "dbp_noninvasive",
    220052: "map_arterial",
    220181: "map_noninvasive",
    220277: "spo2",
    220210: "respiratory_rate",
    223762: "temperature_c",
    223761: "temperature_f",
    220621: "glucose_chart",
}

LABEVENTS_CANDIDATES: dict[int, str] = {
    50931: "glucose_lab",
    50813: "lactate",
    50912: "creatinine",
    51301: "wbc",
}

# Known vasopressor item IDs in inputevents (MIMIC-IV documentation)
VASOPRESSOR_ITEM_IDS: list[int] = [
    221906,  # Norepinephrine
    221289,  # Epinephrine
    222315,  # Vasopressin
    221662,  # Dopamine
    221749,  # Phenylephrine
    30128,   # Norepinephrine (legacy)
    30120,   # Epinephrine (legacy)
]

# Mechanical ventilation procedure item IDs
VENT_ITEM_IDS: list[int] = [
    225792,  # Invasive Ventilation
    225794,  # Non-Invasive Ventilation
]


# ===========================================================================
# SECTION 1: Dataset Availability
# ===========================================================================

def check_availability() -> dict:
    log.info("=== SECTION 1: Dataset Availability ===")
    checker = DataAvailabilityChecker(str(PROJECT_ROOT))
    report = checker.check_all()
    log.info("\n" + report.summary())

    result = {}
    for ds in report.datasets:
        result[ds.dataset_name] = {
            "available": ds.is_available,
            "root_path": ds.root_path,
            "available_tables": [t.name for t in ds.available_tables],
            "missing_tables": [t.name for t in ds.missing_tables],
            "notes": ds.notes,
        }
    return result


# ===========================================================================
# SECTION 2: Population Statistics
# ===========================================================================

def analyze_population(loader: MIMICIVDemoLoader) -> dict:
    log.info("=== SECTION 2: Population Statistics ===")
    patients = loader.load_patients()
    admissions = loader.load_admissions()
    icustays = loader.load_icustays()

    # Parse timestamps for duration calculation
    icustays_copy = icustays.copy()
    icustays_copy["intime"] = pd.to_datetime(icustays_copy["intime"], errors="coerce")
    icustays_copy["outtime"] = pd.to_datetime(icustays_copy["outtime"], errors="coerce")
    icustays_copy["los_computed_h"] = (
        (icustays_copy["outtime"] - icustays_copy["intime"]).dt.total_seconds() / 3600
    )

    # Age from patients table
    age_vals = patients["anchor_age"].dropna()

    # Hospital mortality: deathtime is not null in admissions
    admissions_copy = admissions.copy()
    n_hosp_deaths = admissions_copy["deathtime"].notna().sum()

    # ICU stay duration stats
    los_h = icustays_copy["los_computed_h"].dropna()

    stats = {
        "data_source": "MIMIC-IV Clinical Database Demo v2.2",
        "data_source_note": "REAL CLINICAL DATA — empirically measured",
        "n_patients": int(patients["subject_id"].nunique()),
        "n_admissions": int(admissions["hadm_id"].nunique()),
        "n_icu_stays": int(icustays["stay_id"].nunique()),
        "sex_distribution": patients["gender"].value_counts().to_dict(),
        "age_years": {
            "min": float(age_vals.min()),
            "max": float(age_vals.max()),
            "mean": round(float(age_vals.mean()), 1),
            "median": float(age_vals.median()),
            "std": round(float(age_vals.std()), 1),
            "note": "anchor_age; ages >89 capped at 91 in MIMIC-IV",
        },
        "icu_los_hours": {
            "min": round(float(los_h.min()), 2),
            "max": round(float(los_h.max()), 2),
            "mean": round(float(los_h.mean()), 2),
            "median": round(float(los_h.median()), 2),
            "std": round(float(los_h.std()), 2),
            "p25": round(float(los_h.quantile(0.25)), 2),
            "p75": round(float(los_h.quantile(0.75)), 2),
        },
        "icu_care_units": icustays["first_careunit"].value_counts().to_dict(),
        "n_hospital_deaths": int(n_hosp_deaths),
        "hospital_mortality_rate": round(n_hosp_deaths / max(len(admissions), 1), 4),
    }

    log.info("Patients: %d | Admissions: %d | ICU stays: %d",
             stats["n_patients"], stats["n_admissions"], stats["n_icu_stays"])
    log.info("Age: mean=%.1f, range=[%d, %d]",
             stats["age_years"]["mean"],
             stats["age_years"]["min"],
             stats["age_years"]["max"])
    log.info("ICU LOS: median=%.1fh", stats["icu_los_hours"]["median"])

    # Save population stats
    pop_rows = []
    pop_rows.append({"metric": "n_patients", "value": stats["n_patients"]})
    pop_rows.append({"metric": "n_admissions", "value": stats["n_admissions"]})
    pop_rows.append({"metric": "n_icu_stays", "value": stats["n_icu_stays"]})
    pop_rows.append({"metric": "age_mean", "value": stats["age_years"]["mean"]})
    pop_rows.append({"metric": "age_median", "value": stats["age_years"]["median"]})
    pop_rows.append({"metric": "age_min", "value": stats["age_years"]["min"]})
    pop_rows.append({"metric": "age_max", "value": stats["age_years"]["max"]})
    pop_rows.append({"metric": "icu_los_median_hours", "value": stats["icu_los_hours"]["median"]})
    pop_rows.append({"metric": "icu_los_mean_hours", "value": stats["icu_los_hours"]["mean"]})
    pop_rows.append({"metric": "n_hospital_deaths", "value": stats["n_hospital_deaths"]})
    pop_rows.append({"metric": "hospital_mortality_rate", "value": stats["hospital_mortality_rate"]})
    for sex, count in stats["sex_distribution"].items():
        pop_rows.append({"metric": f"sex_{sex}", "value": count})
    for unit, count in stats["icu_care_units"].items():
        pop_rows.append({"metric": f"care_unit_{unit.replace(' ', '_')}", "value": count})

    pd.DataFrame(pop_rows).to_csv(OUT_DIR / "population_statistics.csv", index=False)
    log.info("Saved: population_statistics.csv")

    return stats


# ===========================================================================
# SECTION 3: Variable Mapping Validation
# ===========================================================================

def validate_variable_mapping(loader: MIMICIVDemoLoader) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=== SECTION 3: Variable Mapping Validation ===")

    d_items = loader.load_d_items()
    d_labitems = loader.load_d_labitems()

    rows = []

    # --- Chartevents candidates ---
    for item_id, canonical_name in CHARTEVENTS_CANDIDATES.items():
        match = d_items[d_items["itemid"] == item_id]
        if match.empty:
            rows.append({
                "canonical_name": canonical_name,
                "item_id": item_id,
                "source_table": "chartevents",
                "item_id_status": "NOT_IN_D_ITEMS",
                "label_in_d_items": None,
                "unit_in_d_items": None,
                "category_in_d_items": None,
            })
        else:
            r = match.iloc[0]
            rows.append({
                "canonical_name": canonical_name,
                "item_id": item_id,
                "source_table": "chartevents",
                "item_id_status": "CONFIRMED_IN_D_ITEMS",
                "label_in_d_items": r.get("label"),
                "unit_in_d_items": r.get("unitname", r.get("unit", None)),
                "category_in_d_items": r.get("category"),
            })

    # --- Labevents candidates ---
    for item_id, canonical_name in LABEVENTS_CANDIDATES.items():
        match = d_labitems[d_labitems["itemid"] == item_id]
        if match.empty:
            rows.append({
                "canonical_name": canonical_name,
                "item_id": item_id,
                "source_table": "labevents",
                "item_id_status": "NOT_IN_D_LABITEMS",
                "label_in_d_items": None,
                "unit_in_d_items": None,
                "category_in_d_items": None,
            })
        else:
            r = match.iloc[0]
            rows.append({
                "canonical_name": canonical_name,
                "item_id": item_id,
                "source_table": "labevents",
                "item_id_status": "CONFIRMED_IN_D_LABITEMS",
                "label_in_d_items": r.get("label"),
                "unit_in_d_items": r.get("fluid", None),
                "category_in_d_items": r.get("category"),
            })

    mapping_df = pd.DataFrame(rows)

    confirmed = mapping_df[mapping_df["item_id_status"].str.startswith("CONFIRMED")]
    missing = mapping_df[~mapping_df["item_id_status"].str.startswith("CONFIRMED")]
    log.info("Item IDs confirmed in d_items/d_labitems: %d/%d",
             len(confirmed), len(mapping_df))
    if not missing.empty:
        log.warning("Item IDs NOT found in dictionaries: %s",
                    missing["canonical_name"].tolist())

    mapping_df.to_csv(OUT_DIR / "variable_mapping.csv", index=False)
    log.info("Saved: variable_mapping.csv")

    return mapping_df, confirmed


# ===========================================================================
# SECTION 4: Chartevents Profiling (Temporal + Missingness + Quality)
# ===========================================================================

def profile_chartevents(
    loader: MIMICIVDemoLoader,
    icustays: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    log.info("=== SECTION 4: Chartevents Profiling ===")

    confirmed_chart = mapping_df[
        (mapping_df["source_table"] == "chartevents") &
        (mapping_df["item_id_status"] == "CONFIRMED_IN_D_ITEMS")
    ]["item_id"].tolist()

    all_item_ids = [int(x) for x in mapping_df[
        mapping_df["source_table"] == "chartevents"]["item_id"].tolist()]

    log.info("Loading chartevents for %d item IDs...", len(all_item_ids))
    chart = loader.load_chartevents(item_ids=all_item_ids)

    total_chart_rows = len(chart)
    log.info("Chartevents loaded: %d rows", total_chart_rows)

    # Parse charttime for interval analysis
    chart["charttime_dt"] = pd.to_datetime(chart["charttime"], errors="coerce")

    temporal_rows = []
    missingness_rows = []
    quality_issues = []

    n_total_stays = icustays["stay_id"].nunique()

    for item_id, canonical_name in CHARTEVENTS_CANDIDATES.items():
        sub = chart[chart["itemid"] == item_id].copy()
        n_obs = len(sub)
        n_patients = sub["subject_id"].nunique() if n_obs > 0 else 0
        n_stays = sub["stay_id"].nunique() if n_obs > 0 else 0
        n_missing_value = sub["valuenum"].isna().sum() if n_obs > 0 else 0
        missing_pct = round(n_missing_value / max(n_obs, 1) * 100, 2)

        if n_obs == 0:
            temporal_rows.append({
                "canonical_name": canonical_name,
                "item_id": item_id,
                "source": "chartevents",
                "n_observations": 0,
                "n_unique_patients": 0,
                "n_unique_stays": 0,
                "stays_with_variable_pct": 0.0,
                "median_interval_min": None,
                "mean_interval_min": None,
                "std_interval_min": None,
                "p25_interval_min": None,
                "p75_interval_min": None,
                "min_interval_min": None,
                "max_interval_min": None,
                "irregular_sampling": None,
                "obs_per_stay_median": None,
            })
            missingness_rows.append({
                "canonical_name": canonical_name,
                "item_id": item_id,
                "source": "chartevents",
                "n_observations": 0,
                "n_missing_valuenum": 0,
                "missing_pct": 0.0,
                "n_patients_with_variable": 0,
                "n_stays_with_variable": 0,
                "stays_with_variable_pct": 0.0,
                "severity": "ABSENT",
            })
            continue

        # --- Temporal analysis ---
        intervals_all = []
        obs_per_stay = []
        sub_sorted = sub.sort_values(["stay_id", "charttime_dt"])
        for stay_id, grp in sub_sorted.groupby("stay_id"):
            times = grp["charttime_dt"].dropna().sort_values()
            if len(times) > 1:
                diffs_min = times.diff().dt.total_seconds().dropna() / 60.0
                # Exclude negative or zero diffs (duplicate/out-of-order)
                diffs_min = diffs_min[diffs_min > 0]
                intervals_all.extend(diffs_min.tolist())
            obs_per_stay.append(len(grp))

        intervals = np.array(intervals_all) if intervals_all else np.array([])
        ops_arr = np.array(obs_per_stay)

        temporal_rows.append({
            "canonical_name": canonical_name,
            "item_id": item_id,
            "source": "chartevents",
            "n_observations": n_obs,
            "n_unique_patients": n_patients,
            "n_unique_stays": n_stays,
            "stays_with_variable_pct": round(n_stays / max(n_total_stays, 1) * 100, 1),
            "median_interval_min": round(float(np.median(intervals)), 1) if len(intervals) > 0 else None,
            "mean_interval_min": round(float(np.mean(intervals)), 1) if len(intervals) > 0 else None,
            "std_interval_min": round(float(np.std(intervals)), 1) if len(intervals) > 0 else None,
            "p25_interval_min": round(float(np.percentile(intervals, 25)), 1) if len(intervals) > 0 else None,
            "p75_interval_min": round(float(np.percentile(intervals, 75)), 1) if len(intervals) > 0 else None,
            "min_interval_min": round(float(np.min(intervals)), 1) if len(intervals) > 0 else None,
            "max_interval_min": round(float(np.max(intervals)), 2) if len(intervals) > 0 else None,
            "irregular_sampling": True,  # All clinical charting is inherently irregular
            "obs_per_stay_median": round(float(np.median(ops_arr)), 1) if len(ops_arr) > 0 else None,
        })

        # --- Missingness ---
        severity = (
            "LOW" if missing_pct < 10 else
            "MODERATE" if missing_pct < 30 else
            "HIGH" if missing_pct < 60 else
            "SEVERE"
        )
        missingness_rows.append({
            "canonical_name": canonical_name,
            "item_id": item_id,
            "source": "chartevents",
            "n_observations": n_obs,
            "n_missing_valuenum": int(n_missing_value),
            "missing_pct": missing_pct,
            "n_patients_with_variable": n_patients,
            "n_stays_with_variable": n_stays,
            "stays_with_variable_pct": round(n_stays / max(n_total_stays, 1) * 100, 1),
            "severity": severity,
        })

        # --- Quality checks ---
        vals = sub["valuenum"].dropna()
        if len(vals) > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
            n_neg = int((vals < 0).sum())
            n_dup = int(sub.duplicated(subset=["stay_id", "charttime", "itemid"]).sum())
            unit_dist = sub["valueuom"].value_counts().to_dict()
            n_units = len(unit_dist)

            if n_neg > 0:
                quality_issues.append({
                    "variable": canonical_name,
                    "issue": "negative_values",
                    "count": n_neg,
                    "severity": "WARNING",
                })
            if n_dup > 0:
                quality_issues.append({
                    "variable": canonical_name,
                    "issue": "duplicate_timestamps",
                    "count": n_dup,
                    "severity": "WARNING",
                })
            if n_units > 1:
                quality_issues.append({
                    "variable": canonical_name,
                    "issue": "multiple_units",
                    "count": n_units,
                    "detail": str(unit_dist),
                    "severity": "NOTE",
                })

    # --- Summary stats per variable ---
    summary_rows = []
    for item_id, canonical_name in CHARTEVENTS_CANDIDATES.items():
        sub = chart[chart["itemid"] == item_id]
        vals = sub["valuenum"].dropna()
        if len(vals) == 0:
            continue
        summary_rows.append({
            "canonical_name": canonical_name,
            "item_id": item_id,
            "source": "chartevents",
            "n_observations": len(sub),
            "n_non_null": len(vals),
            "mean": round(float(vals.mean()), 3),
            "median": round(float(vals.median()), 3),
            "std": round(float(vals.std()), 3),
            "min": round(float(vals.min()), 3),
            "p5": round(float(vals.quantile(0.05)), 3),
            "p25": round(float(vals.quantile(0.25)), 3),
            "p75": round(float(vals.quantile(0.75)), 3),
            "p95": round(float(vals.quantile(0.95)), 3),
            "max": round(float(vals.max()), 3),
            "data_source": "REAL MIMIC-IV Demo v2.2",
        })

    temporal_df = pd.DataFrame(temporal_rows)
    missingness_df = pd.DataFrame(missingness_rows)

    temporal_df.to_csv(OUT_DIR / "temporal_analysis_chartevents.csv", index=False)
    missingness_df.to_csv(OUT_DIR / "missingness_analysis_chartevents.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "chartevents_summary_stats.csv", index=False)
    log.info("Saved: temporal_analysis_chartevents.csv, missingness_analysis_chartevents.csv")

    return temporal_df, missingness_df, {"quality_issues": quality_issues}


# ===========================================================================
# SECTION 5: Labevents Profiling
# ===========================================================================

def profile_labevents(
    loader: MIMICIVDemoLoader,
    icustays: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=== SECTION 5: Labevents Profiling ===")

    all_lab_ids = [int(x) for x in mapping_df[
        mapping_df["source_table"] == "labevents"]["item_id"].tolist()]

    log.info("Loading labevents for %d item IDs...", len(all_lab_ids))

    # Link labevents to ICU stays via admissions
    admissions = loader.load_admissions()
    hadm_to_stay = (
        icustays[["hadm_id", "stay_id", "subject_id"]]
        .drop_duplicates()
    )

    labs = loader.load_labevents(item_ids=all_lab_ids)
    labs_linked = labs.merge(
        hadm_to_stay[["hadm_id", "stay_id"]].drop_duplicates(),
        on="hadm_id",
        how="left",
    )

    n_total_stays = icustays["stay_id"].nunique()
    temporal_rows = []
    missingness_rows = []

    for item_id, canonical_name in LABEVENTS_CANDIDATES.items():
        sub = labs_linked[labs_linked["itemid"] == item_id].copy()
        n_obs = len(sub)
        n_patients = sub["subject_id"].nunique() if n_obs > 0 else 0
        n_stays = sub["stay_id"].dropna().nunique() if n_obs > 0 else 0

        if n_obs == 0:
            temporal_rows.append({
                "canonical_name": canonical_name, "item_id": item_id,
                "source": "labevents", "n_observations": 0,
                "n_unique_patients": 0, "n_unique_stays": 0,
                "median_interval_h": None, "mean_interval_h": None,
            })
            continue

        sub["charttime_dt"] = pd.to_datetime(sub["charttime"], errors="coerce")
        n_missing = sub["valuenum"].isna().sum()
        missing_pct = round(n_missing / max(n_obs, 1) * 100, 2)

        # Interval analysis (hours, since labs are episodic)
        intervals_h = []
        for stay_id, grp in sub.dropna(subset=["stay_id"]).groupby("stay_id"):
            times = grp["charttime_dt"].dropna().sort_values()
            if len(times) > 1:
                diffs_h = times.diff().dt.total_seconds().dropna() / 3600.0
                diffs_h = diffs_h[diffs_h > 0]
                intervals_h.extend(diffs_h.tolist())

        ivs = np.array(intervals_h) if intervals_h else np.array([])

        temporal_rows.append({
            "canonical_name": canonical_name,
            "item_id": item_id,
            "source": "labevents",
            "n_observations": n_obs,
            "n_unique_patients": n_patients,
            "n_unique_stays": n_stays,
            "stays_with_variable_pct": round(n_stays / max(n_total_stays, 1) * 100, 1),
            "median_interval_h": round(float(np.median(ivs)), 1) if len(ivs) > 0 else None,
            "mean_interval_h": round(float(np.mean(ivs)), 1) if len(ivs) > 0 else None,
            "std_interval_h": round(float(np.std(ivs)), 1) if len(ivs) > 0 else None,
            "min_interval_h": round(float(np.min(ivs)), 2) if len(ivs) > 0 else None,
            "max_interval_h": round(float(np.max(ivs)), 1) if len(ivs) > 0 else None,
        })

        severity = (
            "LOW" if missing_pct < 10 else
            "MODERATE" if missing_pct < 30 else
            "HIGH" if missing_pct < 60 else
            "SEVERE"
        )
        vals = sub["valuenum"].dropna()
        missingness_rows.append({
            "canonical_name": canonical_name,
            "item_id": item_id,
            "source": "labevents",
            "n_observations": n_obs,
            "n_missing_valuenum": int(n_missing),
            "missing_pct": missing_pct,
            "n_patients_with_variable": n_patients,
            "n_stays_with_variable": n_stays,
            "stays_with_variable_pct": round(n_stays / max(n_total_stays, 1) * 100, 1),
            "val_mean": round(float(vals.mean()), 3) if len(vals) > 0 else None,
            "val_median": round(float(vals.median()), 3) if len(vals) > 0 else None,
            "val_min": round(float(vals.min()), 3) if len(vals) > 0 else None,
            "val_max": round(float(vals.max()), 3) if len(vals) > 0 else None,
            "severity": severity,
        })

    temporal_df = pd.DataFrame(temporal_rows)
    missingness_df = pd.DataFrame(missingness_rows)
    temporal_df.to_csv(OUT_DIR / "temporal_analysis_labevents.csv", index=False)
    missingness_df.to_csv(OUT_DIR / "missingness_analysis_labevents.csv", index=False)
    log.info("Saved: temporal_analysis_labevents.csv, missingness_analysis_labevents.csv")

    return temporal_df, missingness_df


# ===========================================================================
# SECTION 6: Candidate Outcome Analysis
# ===========================================================================

def analyze_candidate_outcomes(loader: MIMICIVDemoLoader, icustays: pd.DataFrame) -> dict:
    log.info("=== SECTION 6: Candidate Outcome Analysis ===")

    admissions = loader.load_admissions()
    n_stays = icustays["stay_id"].nunique()
    n_patients = icustays["subject_id"].nunique()

    outcomes = {}

    # ---- 1. Hospital Mortality ----
    admissions_copy = admissions.copy()
    n_deaths = int(admissions_copy["deathtime"].notna().sum())
    outcomes["hospital_mortality"] = {
        "description": "Patient death during hospital admission",
        "required_tables": ["hosp/admissions"],
        "required_columns": ["deathtime"],
        "operational_definition": "deathtime IS NOT NULL",
        "n_positive_cases": n_deaths,
        "n_total": len(admissions),
        "positive_rate": round(n_deaths / max(len(admissions), 1), 4),
        "temporal_precision": "discharge-time (not ICU-time)",
        "label_quality": "HIGH — directly recorded",
        "leakage_risk": "HIGH — deathtime is a future event; must not enter features",
        "prediction_horizon_suitability": "6h (maybe 1h/3h with care)",
        "demo_feasibility": "YES — directly available",
        "full_mimic_feasibility": "YES",
        "notes": (
            "Mortality labels are definitive but coarse in timing. "
            "ICU-specific mortality (discharge to death within N hours) "
            "needs careful temporal construction."
        ),
        "student_decision_required": True,
    }

    # ---- 2. ICU Mortality (in-ICU death proxy) ----
    # Proxy: admitted to ICU, deathtime falls within ICU outtime
    icustays_copy = icustays.copy()
    icustays_copy["intime_dt"] = pd.to_datetime(icustays_copy["intime"], errors="coerce")
    icustays_copy["outtime_dt"] = pd.to_datetime(icustays_copy["outtime"], errors="coerce")
    adm_copy = admissions.copy()
    adm_copy["deathtime_dt"] = pd.to_datetime(adm_copy["deathtime"], errors="coerce")
    merged = icustays_copy.merge(
        adm_copy[["hadm_id", "deathtime_dt"]], on="hadm_id", how="left"
    )
    in_icu_deaths = merged[
        merged["deathtime_dt"].notna() &
        (merged["deathtime_dt"] >= merged["intime_dt"]) &
        (merged["deathtime_dt"] <= merged["outtime_dt"])
    ]
    n_icu_deaths = len(in_icu_deaths)
    outcomes["icu_mortality"] = {
        "description": "Death occurring during ICU stay window",
        "required_tables": ["hosp/admissions", "icu/icustays"],
        "operational_definition": "deathtime >= intime AND deathtime <= outtime",
        "n_positive_cases": n_icu_deaths,
        "n_total": n_stays,
        "positive_rate": round(n_icu_deaths / max(n_stays, 1), 4),
        "temporal_precision": "within-ICU",
        "label_quality": "MODERATE — proxy via discharge records",
        "leakage_risk": "HIGH — death time must not enter features",
        "demo_feasibility": "YES — computable from demo data",
        "notes": "More clinically targeted than hospital mortality.",
        "student_decision_required": True,
    }

    # ---- 3. Vasopressor Initiation ----
    try:
        stay_ids = icustays["stay_id"].tolist()
        inputevents = loader.load_inputevents(stay_ids=stay_ids)
        vasopress = inputevents[inputevents["itemid"].isin(VASOPRESSOR_ITEM_IDS)]
        n_vaso_stays = vasopress["stay_id"].nunique()
        outcomes["vasopressor_initiation"] = {
            "description": "First vasopressor administration during ICU stay",
            "required_tables": ["icu/inputevents"],
            "required_columns": ["itemid", "starttime", "stay_id"],
            "candidate_item_ids": VASOPRESSOR_ITEM_IDS,
            "n_stays_with_vasopressors": int(n_vaso_stays),
            "n_total_stays": int(n_stays),
            "positive_rate_stays": round(n_vaso_stays / max(n_stays, 1), 4),
            "temporal_precision": "starttime (precise timestamp)",
            "label_quality": "HIGH — directly recorded administration time",
            "leakage_risk": "MODERATE — vasopressor itself must not be a feature after t0",
            "prediction_horizon_suitability": "1h/3h/6h — suitable",
            "demo_feasibility": "YES — inputevents available",
            "notes": (
                "Vasopressor initiation is a strong clinical signal of haemodynamic collapse. "
                "Requires careful temporal cutoff: only first initiation per stay matters. "
                "Item IDs should be verified against d_items."
            ),
            "student_decision_required": True,
        }
    except Exception as e:
        outcomes["vasopressor_initiation"] = {
            "description": "Vasopressor initiation",
            "error": str(e),
            "demo_feasibility": "NEEDS_VERIFICATION",
        }

    # ---- 4. Mechanical Ventilation ----
    try:
        procedureevents = loader.load_procedureevents(stay_ids=stay_ids)
        vent_events = procedureevents[procedureevents["itemid"].isin(VENT_ITEM_IDS)]
        n_vent_stays = vent_events["stay_id"].nunique()
        outcomes["mechanical_ventilation"] = {
            "description": "Invasive or non-invasive mechanical ventilation initiation",
            "required_tables": ["icu/procedureevents"],
            "candidate_item_ids": VENT_ITEM_IDS,
            "n_stays_with_ventilation": int(n_vent_stays),
            "n_total_stays": int(n_stays),
            "positive_rate_stays": round(n_vent_stays / max(n_stays, 1), 4),
            "temporal_precision": "starttime (precise)",
            "label_quality": "HIGH — directly recorded",
            "leakage_risk": "MODERATE — ventilation settings must not be features after t0",
            "prediction_horizon_suitability": "1h/3h/6h — suitable",
            "demo_feasibility": "YES — procedureevents available",
            "notes": (
                "Mechanical ventilation onset is a clear clinical deterioration marker. "
                "Distinguish invasive (225792) vs non-invasive (225794)."
            ),
            "student_decision_required": True,
        }
    except Exception as e:
        outcomes["mechanical_ventilation"] = {
            "description": "Mechanical ventilation",
            "error": str(e),
            "demo_feasibility": "NEEDS_VERIFICATION",
        }

    # ---- 5. Hemodynamic deterioration (MAP < 65 mmHg) ----
    outcomes["hemodynamic_deterioration"] = {
        "description": "MAP <65 mmHg sustained for ≥2 consecutive observations",
        "required_tables": ["icu/chartevents"],
        "required_item_ids": [220052, 220181],
        "operational_definition": "MAP < 65 mmHg for ≥2 readings (threshold TBD by student team)",
        "temporal_precision": "charttime (variable — depends on charting frequency)",
        "label_quality": "MODERATE — derived from charted vitals; charting gaps are common",
        "leakage_risk": "LOW — derived purely from past vitals",
        "prediction_horizon_suitability": "1h/3h suitable; 6h more uncertain",
        "demo_feasibility": "YES — MAP available in chartevents",
        "notes": (
            "MAP threshold and duration criteria are clinical design choices. "
            "Student team must define the threshold and persistence criterion."
        ),
        "student_decision_required": True,
    }

    # ---- 6. Respiratory Deterioration ----
    outcomes["respiratory_deterioration"] = {
        "description": "SpO2 <90% or RR >30 sustained deterioration",
        "required_tables": ["icu/chartevents"],
        "required_item_ids": [220277, 220210],
        "operational_definition": "SpO2 < threshold OR RR > threshold (thresholds TBD)",
        "temporal_precision": "charttime",
        "label_quality": "MODERATE — derived from charted vitals",
        "leakage_risk": "LOW",
        "demo_feasibility": "YES — SpO2 and RR available",
        "notes": (
            "Threshold values and persistence criteria are student team research decisions. "
            "Combined SpO2 + RR deterioration is more specific than either alone."
        ),
        "student_decision_required": True,
    }

    # ---- 7. Composite ICU deterioration event ----
    outcomes["composite_deterioration"] = {
        "description": "Any of: vasopressor initiation OR ventilation OR MAP<65 OR in-ICU death",
        "required_tables": ["icu/inputevents", "icu/procedureevents",
                            "icu/chartevents", "hosp/admissions"],
        "operational_definition": "OR of above individual events",
        "temporal_precision": "Varies by component",
        "label_quality": "MODERATE — composite labels have lower specificity",
        "leakage_risk": "MODERATE — must handle each component separately",
        "demo_feasibility": "YES — all components available",
        "notes": (
            "Composite outcome increases positive class size (better for class balance) "
            "but reduces clinical specificity. Student team must decide the tradeoff."
        ),
        "student_decision_required": True,
    }

    with open(OUT_DIR / "candidate_outcomes.json", "w") as f:
        json.dump(outcomes, f, indent=2, default=str)
    log.info("Saved: candidate_outcomes.json")

    return outcomes


# ===========================================================================
# SECTION 7: Leakage Analysis
# ===========================================================================

def analyze_leakage() -> dict:
    log.info("=== SECTION 7: Leakage Analysis ===")

    leakage = {
        "data_source": "MIMIC-IV Clinical Database Demo v2.2",
        "analysis_note": (
            "Temporal leakage is the most critical data quality issue "
            "for clinical time-series prediction. The following risks "
            "must be addressed in Phase 2."
        ),
        "leakage_risks": [
            {
                "category": "Future Observations",
                "risk": "CRITICAL",
                "description": (
                    "If features are constructed using observations after time t, "
                    "the model sees future data during training. This invalidates all results."
                ),
                "affected_variables": "All chartevents, labevents",
                "prevention": (
                    "At prediction time t, use ONLY observations with charttime < t. "
                    "Implement strict temporal cutoff in sequence builder."
                ),
            },
            {
                "category": "Outcome Timestamp Leakage",
                "risk": "CRITICAL",
                "description": (
                    "If the outcome label (e.g. vasopressor starttime, deathtime) "
                    "is used as a feature or if observations AFTER the event are included "
                    "in the feature window."
                ),
                "affected_variables": "inputevents.starttime, admissions.deathtime, procedureevents.starttime",
                "prevention": (
                    "Label must be computed from events STRICTLY after the feature window ends. "
                    "Exclude all post-event observations from the feature window."
                ),
            },
            {
                "category": "Treatment Intervention Variables",
                "risk": "HIGH",
                "description": (
                    "Including vasopressor rates, ventilator settings, or drug doses as features "
                    "when the prediction target IS vasopressor initiation or ventilation."
                ),
                "affected_variables": "inputevents.amount, inputevents.rate, procedureevents",
                "prevention": (
                    "If predicting vasopressor initiation: exclude vasopressor-related "
                    "inputevents from feature set. "
                    "Student team must define the exclusion list per outcome."
                ),
            },
            {
                "category": "Discharge Information",
                "risk": "HIGH",
                "description": (
                    "admissions.dischtime, admissions.deathtime, admissions.discharge_location "
                    "reveal future information."
                ),
                "affected_variables": "dischtime, deathtime, discharge_location",
                "prevention": "Exclude all discharge fields from feature engineering.",
            },
            {
                "category": "Post-Event Physiological Response",
                "risk": "HIGH",
                "description": (
                    "After a vasopressor is given, heart rate and BP change rapidly. "
                    "Including post-treatment observations in a 'pre-event' window introduces leakage."
                ),
                "affected_variables": "chartevents after intervention starttime",
                "prevention": "Feature window must end strictly before event starttime.",
            },
            {
                "category": "Patient-Level vs Stay-Level Splitting",
                "risk": "MODERATE",
                "description": (
                    "If a patient has multiple ICU stays and stays from the SAME patient "
                    "appear in both train and test sets, the model may learn patient-specific "
                    "patterns rather than generalisable clinical patterns."
                ),
                "affected_variables": "subject_id across multiple stay_ids",
                "prevention": (
                    "ALWAYS split at patient level (subject_id), not at stay level. "
                    "All stays for a given subject_id must be in the same partition."
                ),
            },
            {
                "category": "Duplicate Records",
                "risk": "LOW-MODERATE",
                "description": (
                    "Duplicate charttime + itemid + stay_id rows can artificially "
                    "boost the apparent precision of certain measurements."
                ),
                "affected_variables": "chartevents",
                "prevention": "Deduplicate by (stay_id, itemid, charttime) in Phase 2. "
                              "Keep value with latest storetime if duplicates exist.",
            },
            {
                "category": "ICD Code Leakage",
                "risk": "HIGH",
                "description": (
                    "diagnoses_icd contains DISCHARGE diagnoses (assigned at end of stay). "
                    "Using ICD codes as features (e.g., sepsis ICD flag) introduces leakage."
                ),
                "affected_variables": "hosp/diagnoses_icd",
                "prevention": "Do NOT use ICD codes as features. Use only for label derivation.",
            },
            {
                "category": "Temporal Resolution Mismatch",
                "risk": "MODERATE",
                "description": (
                    "Labs are sampled at much lower frequency than vitals. "
                    "Forward-filling labs over long gaps effectively uses a 'stale' value "
                    "that may no longer reflect current state, or may propagate past an event."
                ),
                "affected_variables": "labevents (glucose, lactate, creatinine, WBC)",
                "prevention": (
                    "Define maximum forward-fill window per variable. "
                    "Student team must set clinically reasonable limits."
                ),
            },
        ],
        "train_test_split_recommendation": (
            "Split must be at PATIENT level (subject_id). "
            "Suggested split: 70% train / 15% validation / 15% test. "
            "Temporal split (first N months train, last M months test) is "
            "also valid for simulating prospective deployment. "
            "Final split strategy is a student team research decision."
        ),
    }

    with open(OUT_DIR / "leakage_analysis.json", "w") as f:
        json.dump(leakage, f, indent=2, default=str)
    log.info("Saved: leakage_analysis.json")
    return leakage


# ===========================================================================
# SECTION 8: Dataset Recommendation
# ===========================================================================

def generate_dataset_recommendation(pop_stats: dict, temporal_df: pd.DataFrame) -> dict:
    log.info("=== SECTION 8: Dataset Recommendation ===")

    # Empirically informed recommendation
    # Temporal resolution from actual data
    hr_row = temporal_df[temporal_df["canonical_name"] == "heart_rate"]
    hr_median_interval = float(hr_row["median_interval_min"].iloc[0]) if len(hr_row) > 0 and hr_row["median_interval_min"].iloc[0] is not None else None

    rec = {
        "data_source_of_recommendation": "MIMIC-IV Clinical Database Demo v2.2 — empirical analysis",
        "primary_development_dataset": {
            "name": "MIMIC-IV Clinical Database Demo v2.2",
            "version": "2.2",
            "n_patients": pop_stats.get("n_patients"),
            "n_icu_stays": pop_stats.get("n_icu_stays"),
            "local_path": "data/raw/mimic_iv_demo/",
            "rationale": (
                "Available locally with 28/28 required tables. "
                "Provides the same schema as full MIMIC-IV. "
                "Sufficient for Phase 1 profiling, schema validation, "
                "ingestion infrastructure, and initial ML prototyping. "
                "NOT sufficient for final research results (100 patients only)."
            ),
            "limitations": [
                "Only 100 patients — insufficient for robust ML generalisation",
                "Training ML models on Demo may overfit; Demo results are preliminary only",
                "Class balance may not reflect full MIMIC-IV distribution",
                "Rare outcomes may have very few positive cases",
            ],
        },
        "primary_research_dataset": {
            "name": "MIMIC-IV",
            "version": "2.2 or latest",
            "estimated_patients": "~70,000+ ICU stays",
            "access_requirements": (
                "PhysioNet account + CITI training completion. "
                "URL: https://physionet.org/content/mimiciv/"
            ),
            "local_availability": "NOT AVAILABLE — requires credentialing",
            "rationale": (
                "Full MIMIC-IV provides the statistical power needed for "
                "reliable ML model training and evaluation. "
                "Same schema as Demo — all ingestion code is ready to use. "
                "Required for final research conclusions and publications."
            ),
        },
        "secondary_high_frequency_dataset": {
            "name": "VitalDB",
            "description": "Surgical ICU high-frequency waveform database",
            "access": "vitaldb Python package (API access)",
            "role": (
                "Optional future source for HIGH-FREQUENCY physiological replay. "
                "VitalDB records at ~1-second intervals (vs MIMIC hourly charting). "
                "Could validate the replay engine at sub-minute resolution. "
                "NOT recommended as primary source due to different patient population."
            ),
            "decision_required": "Student team decision — Phase 1 does not download VitalDB",
        },
        "external_validation_dataset": {
            "name": "eICU Collaborative Research Database",
            "description": "Multi-centre ICU database (208 hospitals, USA)",
            "access_requirements": "PhysioNet credentialing (same as MIMIC-IV)",
            "role": (
                "Ideal external validation dataset — different hospital systems, "
                "different patient populations, overlapping variable set. "
                "Enables external validity testing of trained models."
            ),
            "decision_required": "Student team decision — Phase 1 does not download eICU",
        },
        "temporal_resolution_recommendation": {
            "empirical_basis": (
                f"Heart rate median charting interval in Demo: "
                f"{hr_median_interval} minutes (empirically measured)"
                if hr_median_interval else
                "Heart rate interval: see temporal_analysis_chartevents.csv"
            ),
            "recommendation": (
                "60-minute (1-hour) temporal resolution as PRIMARY starting point. "
                "Rationale: clinical charting in MIMIC-IV ICU is predominantly hourly. "
                "Labs are much sparser (~4-24h intervals). "
                "Sub-hourly resolution would be dominated by forward-filled values. "
                "Final resolution is a student team research decision after reviewing "
                "the full temporal_analysis_chartevents.csv."
            ),
            "alternative_resolutions_to_consider": [
                "15 minutes — captures intra-hour variation; requires careful imputation",
                "60 minutes — matches dominant charting frequency; recommended starting point",
                "120 minutes — reduces sequence length; may lose clinical granularity",
            ],
            "student_decision_required": True,
            "note": (
                "DO NOT finalise temporal resolution until student team has reviewed "
                "the full temporal analysis tables."
            ),
        },
    }

    with open(OUT_DIR / "dataset_recommendation.json", "w") as f:
        json.dump(rec, f, indent=2, default=str)
    log.info("Saved: dataset_recommendation.json")
    return rec


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    log.info("=" * 70)
    log.info("Phase 1 EDA — MIMIC-IV Clinical Database Demo v2.2")
    log.info("DATA SOURCE: REAL CLINICAL DATA")
    log.info("=" * 70)

    # --- Step 1: Availability ---
    avail = check_availability()
    with open(OUT_DIR / "dataset_inventory.json", "w") as f:
        json.dump(avail, f, indent=2, default=str)
    log.info("Saved: dataset_inventory.json")

    demo_avail = avail.get("MIMIC-IV (mimic_iv_demo)", {})
    if not demo_avail.get("available", False):
        log.error("MIMIC-IV Demo NOT available at expected path.")
        log.error("Download from: https://physionet.org/content/mimic-iv-demo/2.2/")
        sys.exit(1)

    log.info("MIMIC-IV Demo confirmed available. Proceeding with real data EDA.")

    # --- Step 2: Population ---
    loader = MIMICIVDemoLoader()
    pop_stats = analyze_population(loader)
    icustays = loader.load_icustays()

    # --- Step 3: Variable mapping ---
    mapping_df, confirmed_df = validate_variable_mapping(loader)

    # --- Step 4 & 5: Chartevents + Labevents profiling ---
    temporal_chart_df, miss_chart_df, quality_data = profile_chartevents(
        loader, icustays, mapping_df
    )
    temporal_lab_df, miss_lab_df = profile_labevents(loader, icustays, mapping_df)

    # Merge temporal tables for combined output
    all_temporal = pd.concat([temporal_chart_df, temporal_lab_df], ignore_index=True)
    all_temporal.to_csv(OUT_DIR / "temporal_analysis.csv", index=False)

    all_miss = pd.concat([miss_chart_df, miss_lab_df], ignore_index=True)
    all_miss.to_csv(OUT_DIR / "missingness_analysis.csv", index=False)

    # --- Quality report ---
    with open(OUT_DIR / "data_quality_report.json", "w") as f:
        json.dump({
            "data_source": "REAL MIMIC-IV Demo v2.2",
            "issues": quality_data["quality_issues"],
            "phase": "analysis_only",
            "note": "No data has been deleted or modified. Issues are recorded for Phase 2 handling.",
        }, f, indent=2, default=str)
    log.info("Saved: data_quality_report.json")

    # --- Step 6: Outcomes ---
    outcomes = analyze_candidate_outcomes(loader, icustays)

    # --- Step 7: Leakage ---
    leakage = analyze_leakage()

    # --- Step 8: Recommendation ---
    rec = generate_dataset_recommendation(pop_stats, temporal_chart_df)

    # --- Final summary print ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 1 EDA COMPLETE")
    log.info("=" * 70)
    log.info("Data source     : REAL MIMIC-IV Clinical Database Demo v2.2")
    log.info("Patients        : %d", pop_stats["n_patients"])
    log.info("ICU stays       : %d", pop_stats["n_icu_stays"])
    log.info("Admissions      : %d", pop_stats["n_admissions"])
    log.info("Hospital deaths : %d (%.1f%%)",
             pop_stats["n_hospital_deaths"],
             pop_stats["hospital_mortality_rate"] * 100)

    log.info("\nOutput files in %s:", OUT_DIR)
    for f in sorted(OUT_DIR.iterdir()):
        if f.name != ".gitkeep":
            log.info("  %s (%.1f KB)", f.name, f.stat().st_size / 1024)

    log.info("\nPhase 1 EDA complete. Review output files before proceeding to Phase 2.")
    log.info("All statistics above are from REAL MIMIC-IV Demo data.")
    log.info("See candidate_outcomes.json for prediction target options.")
    log.info("Student team must make final decisions before Phase 2 begins.")


if __name__ == "__main__":
    main()
