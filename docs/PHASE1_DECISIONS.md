# Phase 1 Research Decisions
## AI-Driven Predictive Patient Digital Twin for ICU Healthcare

**Status:** APPROVED AND LOCKED  
**Approved by:** Student research team  
**Locked in commit:** `2304693` — phase-1: dataset discovery and real MIMIC-IV demo profiling  
**Decision document committed:** phase-1: lock research decisions and phase review  
**Date:** August 2026

> [!IMPORTANT]
> These decisions are binding for Phase 2 and all subsequent phases.
> They may only be revised with explicit student team approval and a new documented decision.
> Do NOT proceed into Phase 2 without referencing this document.

---

## Decision 1 — Primary Prediction Target

**APPROVED:** Mechanical ventilation initiation.

- Operational definition: first onset of mechanical ventilation during an ICU stay,
  recorded as a procedureevents entry for the patient.
- Source table: `icu/procedureevents`
- Empirical class balance in MIMIC-IV Demo: **66 / 140 stays (47.1%)** — well-balanced.
- Suitable for 1h / 3h / 6h prediction horizons.
- Label derivation methodology is a student team responsibility (Phase 5).

---

## Decision 2 — Secondary Prediction Target

**APPROVED:** Vasopressor initiation.

- Operational definition: first administration of a vasopressor agent during an ICU stay,
  recorded as an inputevents entry.
- Source table: `icu/inputevents`
- Empirical class balance in MIMIC-IV Demo: **52 / 140 stays (37.1%)** — well-balanced.
- Suitable for 1h / 3h / 6h prediction horizons.
- Vasopressor agent list and dose thresholds are a student team decision (Phase 5).

---

## Decision 3 — Mortality

**APPROVED (exclusion rationale):** Hospital and ICU mortality are NOT the primary prediction targets.

Rationale:
- Hospital deaths in MIMIC-IV Demo: **15 / 275 admissions (5.5%)** — severe class imbalance.
- ICU deaths in MIMIC-IV Demo: **12 / 140 stays (8.6%)** — insufficient positive cases.
- The Demo dataset is too small for a robust mortality model.
- Mortality may remain as an **exploratory** outcome if full MIMIC-IV access is obtained.
- This decision is revisable if full MIMIC-IV data (>>70,000 stays) becomes available.

---

## Decision 4 — Prediction Horizons

**APPROVED:** Three prediction horizons:

| Horizon | Value |
|---|---|
| Short | **1 hour** |
| Medium | **3 hours** |
| Long | **6 hours** |

All three horizons will be modelled. Horizon selection for the final research report
is a student team decision.

---

## Decision 5 — Canonical Temporal Resolution

**APPROVED:** 60 minutes.

> [!IMPORTANT]
> **This is a modeling resampling decision, NOT a claim that MIMIC-IV data has a
> fixed 60-minute sampling rate.**
>
> MIMIC-IV ICU observations are inherently **irregular**. The empirical median charting
> interval for heart rate, SpO2, and respiratory rate in the MIMIC-IV Demo is 60 minutes,
> but individual observations may be separated by seconds or several hours.
>
> The preprocessing pipeline will resample the irregular time series to a regular
> 60-minute grid using forward-filling (within approved limits). This choice is
> empirically motivated by Phase 1 EDA results but is a modeling decision, not a
> statement about clinical data frequency.

Empirical basis: Phase 1 EDA (`temporal_analysis_chartevents.csv`):
- Heart Rate (220045): median interval = 60 min
- SpO2 (220277): median interval = 60 min
- Respiratory Rate (220210): median interval = 60 min
- Non-invasive BP (220179/220180/220181): median interval = 60 min

---

## Decision 6 — Temperature Representation

**APPROVED:**
- **Retain Fahrenheit observations** (`itemid 223761`) during ingestion and intermediate
  representations. Coverage in MIMIC-IV Demo: **98.6% of ICU stays**.
- **Celsius observations** (`itemid 223762`) have only **10.7% coverage** in the Demo
  and must not be used as the primary temperature source.
- **During preprocessing (Phase 2):** convert Fahrenheit to Celsius using
  `°C = (°F - 32) × 5/9` so the canonical representation is always Celsius.
- The ingestion layer continues to preserve both item IDs without modification.

---

## Decision 7 — Blood Pressure Representation

**APPROVED:**
- **Preserve invasive (arterial) and non-invasive (cuff) measurements separately**
  in the raw and intermediate representations.
- **Do NOT silently merge or average** arterial and non-invasive BP during ingestion.
- Arterial BP (`item_ids 220050, 220051, 220052`) is present in only **46.4% of stays**
  (patients with arterial lines).
- Non-invasive BP (`item_ids 220179, 220180, 220181`) is present in **99.3% of stays**.
- A **canonical BP signal** (with documented precedence: arterial preferred when available,
  non-invasive otherwise) may be derived during preprocessing (Phase 2) after student
  team review of the merging rules.

---

## Decision 8 — Full MIMIC-IV Access

**APPROVED:**
- **MIMIC-IV Demo** is the development and pipeline validation dataset for Phase 2.
- **Full MIMIC-IV** should be pursued for research-scale model training if and when
  PhysioNet CITI credentialing is obtained.
- Phase 2 preprocessing infrastructure must be compatible with full MIMIC-IV
  (same schema, larger volume — chunked loading already implemented).
- No research conclusions may be drawn from Demo-only model training.

---

## Decision 9 — VitalDB and eICU

**APPROVED (exclusion from Phase 2):**
- **VitalDB** and **eICU** will NOT be expanded or integrated during Phase 2.
- Both remain documented as future optional external-validation datasets.
- Their ingestion stubs are preserved but will not be developed until explicitly approved.

---

## Decision 10 — Synthetic Data

**CONFIRMED:**
- Synthetic data (from `SyntheticClinicalGenerator`) remains available for:
  - Unit and integration testing
  - Infrastructure pipeline validation
  - UI scaffolding demonstrations
- Synthetic data **must NEVER be presented as evidence of clinical model performance**.
- All synthetic DataFrames carry `data_source = "SYNTHETIC — DEVELOPMENT / TESTING ONLY"`.
- This policy carries forward from Phase 0 PROJECT_CONSTITUTION.md Section 1.3.

---

*End of Phase 1 Research Decisions document.*
