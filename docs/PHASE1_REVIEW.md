# Phase 1 Review
## AI-Driven Predictive Patient Digital Twin for ICU Healthcare

**Phase:** 1 — Dataset Discovery, Ingestion, and Data Profiling  
**Status:** COMPLETE AND ACCEPTED  
**Commits:**
- `2304693` — phase-1: dataset discovery and real MIMIC-IV demo profiling
- _(this document)_ — phase-1: lock research decisions and phase review

**Date:** August 2026

---

## 1. Phase 1 Objectives

Phase 1 was defined as the complete **Data Discovery + Ingestion + Real MIMIC-IV Demo Profiling** foundation. The objectives were:

1. Implement a production-quality ingestion layer for MIMIC-IV data
2. Download and validate MIMIC-IV Clinical Database Demo v2.2 (28 tables)
3. Empirically profile all candidate physiological variables using real data
4. Validate candidate item ID mappings against the actual `d_items`/`d_labitems` tables
5. Analyse temporal sampling behaviour to recommend a modeling resolution
6. Identify and document data quality issues requiring Phase 2 attention
7. Analyse candidate prediction outcomes and their class balance
8. Perform an explicit leakage analysis and document rules for Phase 2
9. Produce machine-readable EDA output files (JSON, CSV)
10. Produce a 88-test test suite covering all ingestion infrastructure
11. Obtain student team approval on 10 key research decisions

---

## 2. What Was Actually Implemented

### 2.1 New Modules

| Module | Description |
|---|---|
| [`src/ingestion/mimic_iv_loader.py`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/src/ingestion/mimic_iv_loader.py) | Full MIMIC-IV table loader. 10 load methods. Raises `DataNotAvailableError` with actionable PhysioNet instructions when data is absent. Timestamps preserved as strings. |
| [`src/ingestion/mimic_iv_demo_loader.py`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/src/ingestion/mimic_iv_demo_loader.py) | Thin subclass of `MIMICIVLoader` pointing to local Demo download. |
| [`src/ingestion/synthetic_generator.py`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/src/ingestion/synthetic_generator.py) | `SyntheticClinicalGenerator` with 4 trajectories. MIMIC-IV-schema-compatible. Carries synthetic disclaimer on every row. |
| [`src/ingestion/schema.py`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/src/ingestion/schema.py) | Canonical variable registry. 21 `VariableSpec` entries. `VariableCategory` and `DataAvailabilityStatus` enums. |
| [`src/ingestion/data_availability.py`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/src/ingestion/data_availability.py) | `DataAvailabilityChecker` scans `data/raw/`. Detects `.csv.gz` and uncompressed `.csv`. Returns structured `AvailabilityReport`. |

### 2.2 Scripts

| Script | Description |
|---|---|
| [`scripts/phase1_eda.py`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/scripts/phase1_eda.py) | 8-section EDA pipeline. Runs against real MIMIC-IV Demo. Produces 14 output files in `data/interim/phase1_eda_report/`. |
| [`scripts/mimic_iv_item_ids.py`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/scripts/mimic_iv_item_ids.py) | Documentation-derived item ID reference. Not a production module. |

### 2.3 Tests — 88 / 88 Passing

| Test File | Tests | What Is Tested |
|---|---|---|
| `test_data_availability.py` | 15 | Checker, table detection, compressed/uncompressed, summary string |
| `test_schema.py` | 15 | Registry completeness, no duplicates, item IDs, enum membership |
| `test_mimic_iv_loader.py` | 14 | Fake `.csv.gz` fixtures, error handling, filter logic, timestamp dtype |
| `test_mimic_iv_demo_loader.py` | 17 | Unit + real-data integration, 100-patient confirmation |
| `test_synthetic_generator.py` | 25 | All 4 trajectories, schema columns, disclaimer, reproducibility, ward dataset |
| `test_feature_engineer.py` | 1 | Phase 0 placeholder — preserved |
| `test_patient_state.py` | 1 | Phase 0 placeholder — preserved |

### 2.4 Documentation

| Document | Description |
|---|---|
| [`docs/PHASE1_DATA_ANALYSIS.md`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/docs/PHASE1_DATA_ANALYSIS.md) | Main Phase 1 report with all empirical findings |
| [`docs/DATASET_INVENTORY.md`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/docs/DATASET_INVENTORY.md) | Dataset status table and access instructions |
| [`docs/CANONICAL_SCHEMA.md`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/docs/CANONICAL_SCHEMA.md) | Schema contract for ingestion ↔ downstream |
| [`docs/PHASE1_DECISIONS.md`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/docs/PHASE1_DECISIONS.md) | 10 approved and locked research decisions |

---

## 3. Real MIMIC-IV Demo Findings

> [!IMPORTANT]
> All statistics below are from **real MIMIC-IV Clinical Database Demo v2.2**.
> No numbers are fabricated. Source files: `data/interim/phase1_eda_report/`.

### 3.1 Population

| Metric | Value |
|---|---|
| Patients | 100 |
| Hospital admissions | 275 |
| ICU stays | 140 |
| Age — mean / median | 61.8 / 63 years |
| Age — range | 21 – 91 years |
| Sex | 57% M / 43% F |
| Hospital mortality | 15/275 (5.5%) |
| ICU mortality | 12/140 (8.6%) |
| ICU LOS — median | 51.7 hours |
| ICU LOS — mean | 88.3 hours |

### 3.2 Temporal Sampling (empirical medians)

| Variable | Median Interval | Stay Coverage |
|---|---|---|
| Heart Rate (220045) | **60 min** | 100% |
| SpO2 (220277) | **60 min** | 100% |
| Respiratory Rate (220210) | **60 min** | 100% |
| Non-invasive BP (220179/180/181) | **60 min** | 99% |
| Arterial BP (220050/051/052) | **60 min** | 46% |
| Temperature Fahrenheit (223761) | **240 min** | 99% |
| Temperature Celsius (223762) | 60 min | **11%** |
| Glucose chart (220621) | 611 min | 98% |
| Glucose lab (50931) | 16.5 h | 100% |
| Creatinine (50912) | 16.1 h | 100% |
| WBC (51301) | 22.4 h | 100% |
| Lactate (50813) | 6.0 h | 74% |

### 3.3 Data Quality Issues Found

| Variable | Issue | Phase 2 Action |
|---|---|---|
| MAP arterial | Min = −23 mmHg, Max = 801 mmHg | Clip to [0, 300] |
| Temperature Celsius | Max = 99°C (impossible) | Exclude >45°C |
| Heart Rate | Min = 0 bpm | Exclude zeros |
| SpO2 | Min = 29% | Exclude <50% |
| Respiratory Rate | Min = 0 bpm | Exclude zeros |

*Records are preserved as-is in Phase 1. All cleaning is Phase 2.*

### 3.4 Candidate Outcome Class Balance

| Outcome | Positive | Total | Rate |
|---|---|---|---|
| Hospital mortality | 15 | 275 admissions | 5.5% |
| ICU mortality | 12 | 140 stays | 8.6% |
| **Vasopressor initiation** | **52** | **140 stays** | **37.1%** |
| **Mechanical ventilation** | **66** | **140 stays** | **47.1%** |

### 3.5 Item ID Validation

All 16 candidate item IDs (12 chartevents + 4 labevents) confirmed present in the Demo `d_items` / `d_labitems` tables.

---

## 4. Dataset Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Only 100 patients | Insufficient for final ML training | Full MIMIC-IV for research scale |
| Hospital mortality n=15 | Cannot train a robust mortality model | Excluded as primary target |
| No sub-hourly vital signs | Digital Twin replay limited to ≥60 min resolution | VitalDB optional if needed |
| Single-centre data | Potential generalisation bias | eICU external validation (future) |
| Arterial BP in only 46% of stays | Feature sparsity for some patients | Preserve separately; merge in Phase 2 |
| Temperature Celsius sparse (11%) | Cannot use as primary temperature source | Use Fahrenheit; convert in Phase 2 |
| Timestamps shifted (MIMIC de-identification) | Absolute calendar dates are artificial | Use relative timestamps (hours from ICU admission) |

---

## 5. Approved Research Decisions

All 10 decisions are documented in full in [`docs/PHASE1_DECISIONS.md`](file:///C:/Users/athar/.gemini/antigravity/scratch/healthcare-digital-twin/docs/PHASE1_DECISIONS.md).

| # | Decision | Approved Value |
|---|---|---|
| 1 | Primary prediction target | Mechanical ventilation initiation |
| 2 | Secondary prediction target | Vasopressor initiation |
| 3 | Mortality | Excluded (too sparse in Demo); exploratory only |
| 4 | Prediction horizons | 1h, 3h, 6h |
| 5 | Temporal resolution | **60 min** (modeling resampling decision — data is irregular) |
| 6 | Temperature | Retain °F during ingestion; convert to °C in preprocessing |
| 7 | Blood pressure | Preserve invasive and non-invasive separately |
| 8 | Full MIMIC-IV | Pursue credentialing for research-scale training |
| 9 | VitalDB / eICU | Do NOT expand in Phase 2 |
| 10 | Synthetic data | Development/testing only; never evidence of model performance |

---

## 6. What Phase 2 Will Implement

Phase 2 scope (clinical preprocessing and temporal pipeline):

1. **Timestamp parsing** — parse `charttime` strings to `datetime64`; compute hours since ICU admission (`charttime_relative_h`)
2. **Temperature conversion** — Fahrenheit → Celsius (`°C = (°F − 32) × 5/9`)
3. **Data quality filtering** — remove/clip physiological impossibilities documented in Phase 1 (Section 3.3)
4. **Regular grid resampling** — resample irregular observations to 60-minute grid using forward-fill (with approved maximum fill limits per variable type)
5. **Lab forward-fill windows** — apply maximum fill limits: glucose ~24h, creatinine/WBC ~48h, lactate ~12h (student team to confirm thresholds)
6. **BP merging rules** — canonical BP signal: arterial preferred when available, non-invasive otherwise (student team to confirm)
7. **Patient-level train/val/test split** — splits MUST be at `subject_id` level; no stay-level splitting
8. **Outcome label construction** — mechanical ventilation and vasopressor onset labels (student team defines exact logic in Phase 5; Phase 2 prepares the raw data and alignment windows)
9. **Sequence tensor construction** — build fixed-length input windows for CNN-BiLSTM consumption
10. **Per-variable normalisation** — z-score or min-max per training set statistics (method to be approved)

---

## 7. What Phase 2 Explicitly Will NOT Implement

| Out of Scope | Reason |
|---|---|
| CNN-BiLSTM model definition | Phase 5 — student team ML research |
| Model training or hyperparameter tuning | Phase 5 — student team |
| SHAP / XAI integration | Phase 8 |
| What-if simulation | Phase 7 |
| Dashboard / Streamlit | Phase 9 |
| Digital Twin State and Engine | Phase 3 |
| Simulated replay engine | Phase 4 |
| Full MIMIC-IV data download | Separate prerequisite — not Phase 2 scope |
| VitalDB / eICU ingestion | Excluded by Decision 9 |
| Mortality label construction | Excluded by Decision 3 |

---

## 8. Known Risks Entering Phase 2

| Risk | Severity | Mitigation |
|---|---|---|
| Outcome label leakage | **CRITICAL** | Strict Phase 1 leakage rules documented in `leakage_analysis.json`; patient-level splits mandatory |
| Excessive forward-fill masking real missingness | HIGH | Maximum fill windows per variable; missingness flags preserved |
| Temperature unit confusion | MODERATE | Decision 6 explicitly documents conversion; tests must cover it |
| BP merging creating phantom data | MODERATE | Decision 7 requires transparent precedence rules |
| Demo too small for sequence validation | MODERATE | Full MIMIC-IV required for meaningful ML results |
| Irregular sampling confusing CNN-BiLSTM | MODERATE | Resolution decision confirmed; forward-fill limits enforced |
| Train/test leakage across patients | **CRITICAL** | Split at `subject_id` — enforced by Phase 2 code |

---

## 9. Git Commit References

| Commit | Contents |
|---|---|
| `91938b2` | Phase 0: project constitution, modular architecture scaffold, all stubs |
| `8b9c0b8` | Phase 0: architecture corrections, ML ownership, ARCHITECTURE.md |
| `2304693` | Phase 1: complete ingestion layer, EDA on real MIMIC-IV Demo, 88 tests |
| _(this commit)_ | Phase 1: locked research decisions and phase review |

---

*End of Phase 1 Review — Phase 2 may begin after student team confirmation.*
