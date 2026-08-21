# Phase 1 Data Analysis Report
## AI-Driven Predictive Patient Digital Twin for ICU Healthcare

**Data Source:** MIMIC-IV Clinical Database Demo v2.2  
**Source Type:** REAL CLINICAL DATA — empirically measured  
**Analysis Date:** August 2026  
**Status:** Phase 1 Complete

> [!IMPORTANT]
> All statistics in this document are derived from the **real MIMIC-IV Demo data** (100 patients, v2.2).  
> No statistics are fabricated. Where empirical measurement was not possible, this is explicitly stated.

---

## 1. Dataset Population — Empirical Results

| Metric | Value (Real MIMIC-IV Demo) |
|---|---|
| Unique patients | **100** |
| Hospital admissions | **275** |
| ICU stays | **140** |
| ICU LOS — median | **51.7 hours** |
| ICU LOS — mean | **88.3 hours** |
| Age — mean | **61.8 years** |
| Age — median | **63 years** |
| Age — range | **21 – 91 years** (ages >89 capped at 91) |
| Sex distribution | 57% Male / 43% Female |
| Hospital deaths | **15 / 275 admissions (5.5%)** |

### ICU Care Unit Distribution

| Unit | Stays |
|---|---|
| Medical ICU (MICU) | 29 |
| Surgical ICU (SICU) | 29 |
| Cardiac Vascular ICU (CVICU) | 25 |
| Medical/Surgical ICU (MICU/SICU) | 23 |
| Trauma SICU (TSICU) | 16 |
| Coronary Care Unit (CCU) | 13 |
| Neuro Surgical ICU | 3 |
| Neuro Stepdown / Neuro Intermediate | 2 |

---

## 2. Variable Availability — Real Data Validation

All 16 candidate item IDs (12 chartevents + 4 labevents) were **confirmed present** in the MIMIC-IV Demo d_items / d_labitems tables.

### Chartevents Variables (icu/chartevents)

| Variable | Item ID | Observations | ICU Stays | Stay Coverage | Missingness |
|---|---|---|---|---|---|
| heart_rate | 220045 | 13,913 | 140/140 | **100%** | 0% |
| spo2 | 220277 | 13,540 | 140/140 | **100%** | 0% |
| respiratory_rate | 220210 | 13,913 | 140/140 | **100%** | 0% |
| sbp_noninvasive | 220179 | 8,347 | 139/140 | 99.3% | 0% |
| dbp_noninvasive | 220180 | 8,349 | 139/140 | 99.3% | 0% |
| map_noninvasive | 220181 | 8,342 | 138/140 | 98.6% | 0% |
| temperature_f | 223761 | 3,379 | 138/140 | 98.6% | 0% |
| glucose_chart | 220621 | 931 | 137/140 | 97.9% | 0% |
| sbp_arterial | 220050 | 5,525 | 65/140 | 46.4% | 0% |
| dbp_arterial | 220051 | 5,524 | 65/140 | 46.4% | 0% |
| map_arterial | 220052 | 5,560 | 65/140 | 46.4% | 0% |
| temperature_c | 223762 | 391 | 15/140 | **10.7%** | 0% |

**Key finding:** Arterial (invasive) variables cover only ~46% of stays — only patients with arterial lines. Temperature in Celsius is sparsely recorded (10.7%); Temperature in Fahrenheit is the dominant recording (98.6%).

### Labevents Variables (hosp/labevents)

| Variable | Item ID | Observations | ICU Stays | Median Interval |
|---|---|---|---|---|
| glucose_lab | 50931 | 3,024 | 140/140 | ~16.5 hours |
| creatinine | 50912 | 3,321 | 140/140 | ~16.1 hours |
| wbc | 51301 | 3,032 | 140/140 | ~22.4 hours |
| lactate | 50813 | 879 | 103/140 | ~6.0 hours |

---

## 3. Temporal Analysis — Empirical Sampling Intervals

> [!IMPORTANT]
> These are MEASURED sampling intervals from the real MIMIC-IV Demo, not assumed values.

### Chartevents — Core Vitals

All high-frequency vitals (HR, SpO2, RR, BP) have a **median charting interval of 60 minutes** — hourly charting is the dominant pattern.

| Variable | Median Interval | Obs/Stay (median) |
|---|---|---|
| heart_rate | **60 min** | 57 |
| respiratory_rate | **60 min** | 57 |
| spo2 | **60 min** | 56 |
| sbp_noninvasive | **60 min** | 33 |
| dbp_noninvasive | **60 min** | 33 |
| map_noninvasive | **60 min** | 34 |
| sbp_arterial | **60 min** | 43 |
| temperature_f | 240 min (4h) | 13 |
| glucose_chart | 611 min (~10h) | 3 |

### Labevents — Episodic Sampling

| Variable | Median Interval |
|---|---|
| glucose_lab | 16.5 hours |
| creatinine | 16.1 hours |
| wbc | 22.4 hours |
| lactate | 6.0 hours (ordered more urgently) |

### Temporal Resolution Recommendation

> [!IMPORTANT]
> **Recommended starting resolution: 60 minutes (1 hour).**
> 
> Rationale: Empirical median charting interval for HR, SpO2, RR, BP is exactly 60 minutes.  
> Sub-hourly resolution would be dominated by forward-filled values and is **not empirically justified**.  
> Lab values (16–22h intervals) must use longer forward-fill windows.  
> **Final resolution is a student team research decision.** Review `temporal_analysis.csv` before finalising.

---

## 4. Data Quality Findings

| Variable | Min | Max | Mean | Median | Notable Issues |
|---|---|---|---|---|---|
| heart_rate | 0.0 | 200.0 | 91.1 | 90.0 | Min=0 is likely artefact; max=200 plausible |
| sbp_arterial | 25.0 | 207.0 | 113.8 | 112.0 | Plausible range |
| sbp_noninvasive | 46.0 | 215.0 | 115.3 | 113.0 | Plausible |
| map_arterial | **-23.0** | **801.0** | 77.7 | 76.0 | **⚠ Negative MAP and MAP=801 are data artefacts** |
| spo2 | 29.0 | 100.0 | 96.8 | 97.0 | Min=29 likely artefact (sensor off) |
| respiratory_rate | 0.0 | 58.0 | 20.0 | 20.0 | Min=0 likely artefact |
| temperature_c | 31.1 | **99.0** | 37.3 | 36.9 | **⚠ Max=99°C is impossible (artefact or °F entered in °C field)** |
| temperature_f | 94.0 | 103.4 | 98.6 | 98.4 | Plausible range |
| glucose_chart | 34.0 | 942.0 | 151.3 | 130.0 | Wide range but plausible in ICU |

> [!WARNING]
> **Action required in Phase 2 (not Phase 1):**  
> - MAP values < 0 and > 300: clip or exclude  
> - Temperature C = 99°C: exclude as impossible (may be Fahrenheit entered in Celsius field)  
> - HR = 0, SpO2 = 29, RR = 0: exclude as monitoring artefacts  
> These records are **preserved as-is** in Phase 1; cleaning happens in Phase 2.

---

## 5. Candidate Prediction Targets

> [!IMPORTANT]
> The student research team must select the final prediction target.  
> The following analysis presents empirical feasibility only.  
> Antigravity does NOT select the clinical target.

| Outcome | Positive Cases | Total | Rate | Data Source | Leakage Risk | Student Decision |
|---|---|---|---|---|---|---|
| Hospital Mortality | 15 | 275 admissions | 5.5% | admissions.deathtime | HIGH — exclude from features | Required |
| ICU Mortality (in-ICU death) | 12 | 140 stays | 8.6% | admissions.deathtime + icustays times | HIGH | Required |
| Vasopressor Initiation | **52** | 140 stays | **37.1%** | inputevents | MODERATE | Required |
| Mechanical Ventilation | **66** | 140 stays | **47.1%** | procedureevents | MODERATE | Required |
| Hemodynamic Deterioration (MAP<65) | *derivable* | 140 stays | *TBD by student team* | chartevents | LOW | Required |
| Respiratory Deterioration (SpO2<90) | *derivable* | 140 stays | *TBD by student team* | chartevents | LOW | Required |

**Key observations:**
- Hospital and ICU mortality have **small positive class sizes** (15 and 12 events) in the Demo — insufficient for reliable ML training on Demo alone
- Vasopressor initiation (37.1%) and mechanical ventilation (47.1%) have **more balanced class distributions** and are directly measurable from the data
- Derived vital-sign deterioration labels are empirically possible but require student team threshold decisions

---

## 6. Leakage Analysis Summary

> [!CAUTION]
> Temporal leakage is the most critical failure mode in clinical ML. The following rules MUST be enforced in Phase 2.

| Risk | Category | Severity | Phase 2 Rule |
|---|---|---|---|
| Future observations in feature window | Temporal | **CRITICAL** | Only charttime < prediction_time |
| Outcome timestamp in features | Temporal | **CRITICAL** | Exclude deathtime, vasopressor starttime |
| Vasopressor rate as feature when predicting vasopressors | Clinical | **HIGH** | Exclude by outcome definition |
| Discharge information | Clinical | **HIGH** | Exclude dischtime, discharge_location |
| Post-event physiology in feature window | Temporal | **HIGH** | Feature window must end before event |
| Patient stays split across train/test | Splitting | **MODERATE** | Always split at subject_id level |
| ICD codes as features | Clinical | **HIGH** | ICD codes are discharge diagnoses — labels only |
| Long lab forward-fill past event | Temporal | **MODERATE** | Define max forward-fill window per lab |

**Mandatory rule:** Train/test split MUST be at **patient level** (`subject_id`). Never split at stay level.

---

## 7. Dataset Role Recommendations

### Primary Development Dataset
**MIMIC-IV Demo v2.2** (locally available, 100 patients, 140 ICU stays)  
- Suitable for: ingestion testing, EDA, initial ML prototyping, unit tests  
- NOT suitable for: final ML training, statistical generalisation, published results

### Primary Research Dataset
**Full MIMIC-IV** (requires PhysioNet CITI credentialing)  
- ~70,000+ ICU stays; same schema as Demo  
- Required for: training CNN-BiLSTM, final evaluation, research conclusions

### External Validation
**eICU Collaborative Research Database** — multi-centre US hospitals  
- Different patient population — ideal for external validity testing  
- Requires PhysioNet credentialing  
- Student team decision whether to pursue

### High-Frequency Waveform Source
**VitalDB** — surgical ICU, ~1-second resolution  
- Not a substitute for MIMIC-IV  
- Useful only if sub-minute replay engine validation is required  
- Phase 1 does not download VitalDB data

---

## 8. Decisions Required from Student Team Before Phase 2

1. **Prediction target selection** — which outcome to predict (see Section 5)
2. **Temporal resolution** — 60-minute recommended starting point (see Section 3)
3. **Vital sign combinations** — invasive vs. non-invasive BP strategy
4. **Temperature handling** — use Fahrenheit (98.6% coverage) or Celsius (10.7% coverage)
5. **Lab forward-fill windows** — maximum allowed fill time per variable
6. **Train/test split strategy** — temporal split vs. random patient split
7. **Data quality thresholds** — which artefacts to clip vs. exclude
8. **Full MIMIC-IV access** — initiate PhysioNet credentialing for research-scale training

---

## 9. Phase 2 Scope (NOT started in Phase 1)

Phase 2 will build on these Phase 1 findings to implement:

- Timestamp parsing and temporal alignment
- Irregular time-series resampling to chosen resolution
- Clinical range filtering and artefact removal
- Lab variable forward-filling (with student-approved limits)
- Patient-level train/validation/test split
- Outcome label construction (student team decision required first)
- Sequence tensor construction for CNN-BiLSTM input

Phase 2 does NOT start until the student team has reviewed this report and provided decisions on the items above.
