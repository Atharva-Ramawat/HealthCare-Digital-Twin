# Canonical Observation Schema
## AI-Driven Predictive Patient Digital Twin for ICU Healthcare

**Phase:** 1 — Data Discovery and Ingestion  
**Status:** Candidate schema — empirically validated against MIMIC-IV Demo v2.2

> [!NOTE]
> This schema defines the contract between the ingestion layer and all downstream modules.
> All loaders return DataFrames that conform to this column set.
> Final feature selection is a **student team research decision**.

---

## Raw Observation Table Schema (MIMIC-IV chartevents compatible)

| Column | Type | Description | Notes |
|---|---|---|---|
| `subject_id` | int | Unique patient identifier | Primary key across all tables |
| `hadm_id` | int | Hospital admission ID | Links to admissions table |
| `stay_id` | int | ICU stay ID | Links to icustays table |
| `itemid` | int | MIMIC-IV item identifier | Links to d_items or d_labitems |
| `charttime` | str | Observation timestamp | **Preserved as string — NOT parsed in Phase 1** |
| `storetime` | str | Time charted by nurse | Preserved as string |
| `value` | str | Raw string value | Original as recorded |
| `valuenum` | float | Numeric value | NaN if missing/non-numeric |
| `valueuom` | str | Unit of measure | As recorded in MIMIC-IV |

> [!IMPORTANT]
> `charttime` and `storetime` are deliberately preserved as **strings** in Phase 1.
> Timestamp parsing, timezone handling, and temporal alignment are **Phase 2** concerns.

---

## Synthetic Output Schema (additional columns)

When using `SyntheticClinicalGenerator`, the following additional columns are appended to mark data as synthetic:

| Column | Value | Purpose |
|---|---|---|
| `canonical_name` | e.g. `heart_rate` | Human-readable variable name |
| `data_source` | `SYNTHETIC — DEVELOPMENT / TESTING ONLY` | Explicit label |
| `trajectory` | e.g. `normal`, `gradual_deterioration` | Trajectory type |

> [!CAUTION]
> Synthetic data must **never** be mixed with real MIMIC-IV analysis results.
> The `data_source` column provides a machine-readable way to filter.

---

## Canonical Variable Registry (Phase 1 — CANDIDATE Status)

All item IDs below are **CANDIDATE** mappings from public MIMIC-IV documentation.
Status confirmed against d_items/d_labitems during Phase 1 EDA.

### Core Vitals (chartevents)

| Canonical Name | Item ID | Unit | Coverage (Demo) | Status |
|---|---|---|---|---|
| `heart_rate` | 220045 | bpm | 100% stays | CANDIDATE (confirmed in d_items) |
| `spo2` | 220277 | % | 100% stays | CANDIDATE (confirmed in d_items) |
| `respiratory_rate` | 220210 | breaths/min | 100% stays | CANDIDATE (confirmed in d_items) |
| `sbp_noninvasive` | 220179 | mmHg | 99.3% stays | CANDIDATE (confirmed in d_items) |
| `dbp_noninvasive` | 220180 | mmHg | 99.3% stays | CANDIDATE (confirmed in d_items) |
| `map_noninvasive` | 220181 | mmHg | 98.6% stays | CANDIDATE (confirmed in d_items) |
| `temperature_f` | 223761 | °F | 98.6% stays | CANDIDATE (confirmed in d_items) |
| `glucose_chart` | 220621 | mg/dL | 97.9% stays | CANDIDATE (confirmed in d_items) |
| `sbp_arterial` | 220050 | mmHg | 46.4% stays | CANDIDATE (invasive only) |
| `dbp_arterial` | 220051 | mmHg | 46.4% stays | CANDIDATE (invasive only) |
| `map_arterial` | 220052 | mmHg | 46.4% stays | CANDIDATE (invasive only) |
| `temperature_c` | 223762 | °C | 10.7% stays | CANDIDATE (sparse in Demo) |

### Lab Candidates (labevents)

| Canonical Name | Item ID | Unit | Coverage (Demo) | Median Interval |
|---|---|---|---|---|
| `glucose_lab` | 50931 | mg/dL | 100% stays | ~16.5 hours |
| `creatinine` | 50912 | mg/dL | 100% stays | ~16.1 hours |
| `wbc` | 51301 | K/uL | 100% stays | ~22.4 hours |
| `lactate` | 50813 | mmol/L | 73.6% stays | ~6.0 hours |

### Identifiers

| Canonical Name | Source Table | Description |
|---|---|---|
| `subject_id` | patients | Unique patient key |
| `hadm_id` | admissions | Hospital admission key |
| `stay_id` | icustays | ICU stay key |

---

## Phase 2 Schema Extensions (Not implemented in Phase 1)

The following columns will be added during preprocessing (Phase 2):

| Column | Description |
|---|---|
| `charttime_dt` | Parsed datetime (from charttime string) |
| `charttime_relative_h` | Hours since ICU admission (intime) |
| `canonical_name` | Human-readable variable name from item_id mapping |
| `value_clean` | Value after quality filtering |
| `is_outlier` | Boolean flag for implausible values |
| `split` | `train` / `val` / `test` (patient-level) |
