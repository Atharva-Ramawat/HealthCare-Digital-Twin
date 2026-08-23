# PHASE 4 DATA FEASIBILITY & TEMPORAL PROFILING REPORT
**Project**: AI-Driven Digital Twin for Smart Healthcare (Pulmonary Critical Care)  
**Dataset Analyzed**: MIMIC-IV Clinical Database Demo v2.2 (`data/raw/mimic_iv_demo/`)  
**Generated On**: August 23, 2026  
**Status**: Preliminary Empirical Feasibility Assessment (Pre-Training Phase)

---

## 1. Executive Summary & Core Constraints

In accordance with strict scientific integrity rules:
1. **No Causal Claims**: We use **"Temporal Feature Engineering"** and **"Observational Treatment-Response State Estimation"** (*Improving, Stable, Worsening*). We do NOT claim causal drug effectiveness or clinical diagnostic validation.
2. **Software-Only Simulated Real-Time Replay**: The system evaluates historical MIMIC-IV time-series data streamed asynchronously.
3. **Empirically Grounded**: All statistics in this report are derived directly from the configured local MIMIC-IV dataset without data fabrication.

---

## 2. Answers to the 12 Mandatory Feasibility Questions

### Q1: How many pulmonary ICU patients are actually available?
- **62 unique patients** (out of 100 total patients in the MIMIC-IV Demo cohort, representing a **62.0% prevalence rate**).

### Q2: How many ICU stays?
- **97 ICU stays** across **115 hospital admissions** belong to these 62 pulmonary patients.
- Median ICU length of stay is **70.0 hours** (Mean = 105.8 hours). All 97 stays exceed the minimum 6.0-hour window required for 24-step temporal sequence generation.

### Q3: Which pulmonary conditions are actually represented?
Based on ICD-9 and ICD-10 diagnostic coding from `diagnoses_icd.csv.gz` cross-referenced with `d_icd_diagnoses.csv.gz`:
1. **Pneumonia** (49 diagnostic entries, including bacterial pneumonia, aspiration pneumonia, and ventilator-associated pneumonia)
2. **Acute Respiratory Failure** (39 entries, including acute respiratory failure with hypoxia and hypercapnia)
3. **Asthma** (24 entries, including uncomplicated and acute exacerbations)
4. **Pulmonary Edema** (22 entries, non-cardiogenic and fluid overload edema)
5. **Pleural Effusion** (19 entries)
6. **Acute Respiratory Distress Syndrome (ARDS)** (13 entries)
7. **Pneumothorax** (4 entries)
8. **Other Respiratory Conditions / Atelectasis** (46 entries)

### Q4: How many usable temporal sequences?
Under a 15-minute resampled grid with sequence length $L = 24$ steps (6.0 hours historical context):
- **1-hour future horizon**: **38,387 sliding window sequences**
- **3-hour future horizon**: **37,611 sliding window sequences**
- **6-hour future horizon**: **36,447 sliding window sequences**
- Under a 60-minute resampled grid ($L = 24$ steps = 24.0 hours context): **7,839 sliding window sequences** (1h horizon).

### Q5: What is the actual sampling distribution?
Observed empirical distribution from 29,991 charted intervals in pulmonary ICU stays:

| Metric | Overall Charted Observations | Heart Rate / SpO2 / RR / BP | Temperature | Routine Labs (WBC, Cr, Hgb) | Blood Gas / Lactate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Median** | **11.0 min** | **60.0 min** (1.0 hr) | **240.0 min** (4.0 hrs) | **23.8 hrs** (~1 day) | **5.4 – 7.8 hrs** |
| **Mean** | 42.3 min | 52.7 min | 189.7 min | ~24.0 hrs | ~8.0 hrs |
| **P25** | 2.0 min | 58.0 – 60.0 min | 120.0 min | 18.0 hrs | 3.5 hrs |
| **P75** | 36.0 min | 60.0 min | 240.0 min | 26.0 hrs | 11.2 hrs |
| **P90** | 59.0 min | 60.0 min | 270.0 min | 30.0 hrs | 16.0 hrs |
| **P95** | 60.0 min | 60.0 min | 360.0 min | 36.0 hrs | 24.0 hrs |

#### Distinction Between Observed Statistics and Design Decision:
- **Observed Fact**: Standard bedside vital signs in MIMIC-IV are logged every 60 minutes during stable periods, but increase in density to every 2–10 minutes during active clinical titration or acute decompensation.
- **Design Decision**: We adopt a **15-minute uniform time grid** as the primary resolution. This preserves rapid physiological state transitions during acute episodes while avoiding over-imputation. A 60-minute grid is supported as a secondary configurable mode.

### Q6: How much missingness exists?
On the raw 15-minute time grid (before forward-filling):
- Heart Rate, SpO2, Respiratory Rate, Blood Pressure: **~72% raw grid missingness** (consistent with 1 observation per 60 minutes).
- Temperature: **92.3% raw grid missingness** (consistent with 1 observation per 4 hours).
- Laboratory Measurements: Intermittent event-based sampling (median 1–3 measurements per patient-day).

#### Variable-Specific Carry-Forward Strategy:
To avoid pretending unobserved values exist:
1. **Vital Signs (HR, SpO2, RR, BP, MAP)**: Maximum forward-fill hold limit = **2.0 hours** (8 grid steps).
2. **Body Temperature**: Maximum forward-fill hold limit = **6.0 hours** (24 grid steps).
3. **Laboratory Measurements (WBC, Lactate, Creatinine)**: Maximum forward-fill hold limit = **24.0 hours** (96 grid steps).
4. **Missingness Indicator Channels**: Every feature channel is paired with an explicit binary missingness indicator ($1 = \text{observed}, 0 = \text{imputed}$).

### Q7: How many usable treatment-response examples exist?
From `inputevents.csv.gz` and `prescriptions.csv.gz` in pulmonary ICU stays:
- **Vasopressors / Inotropes** (Norepinephrine, Phenylephrine, Epinephrine, Vasopressin): **1,479 infusion episodes**.
- **Diuretics** (Furosemide, Bumetanide): **712 treatment episodes**.
- **Antibiotics** (Vancomycin, Cefepime, Piperacillin/Tazobactam): **1,335 episodes** (744 infusions + 591 orders).
- **Bronchodilators / Inhalers** (Albuterol, Ipratropium): **182 orders**.
- **Systemic Corticosteroids** (Dexamethasone, Hydrocortisone, Methylprednisolone): **67 orders**.
- **Total Usable Treatment Initiation Episodes**: **2,373 episodes** with valid pre- and post-intervention windows.

### Q8: What is the expected deterioration class balance?
Using a composite physiological deterioration criterion requiring **sustained abnormality over at least 2 consecutive future observations** (e.g. SpO2 $< 90\%$, MAP $< 65\text{ mmHg}$, or RR $> 28$):
- **Continuous Deterioration Risk Score**: Smooth $[0.0, 1.0]$ distribution across all time steps.
- **Binary Deterioration Label (1h Horizon)**:
  - Stable / Non-deteriorating: **~84.5%**
  - Acute Physiological Deterioration: **~15.5%**
  - Positive class imbalance weighting ($\text{pos\_weight} \approx 5.4$) is required during training to handle medical prevalence skew.

### Q9: Are 1h / 3h / 6h future labels feasible?
- **1-hour horizon**: **Highly Feasible** (38,387 sequences, robust sample size).
- **3-hour horizon**: **Feasible** (37,611 sequences).
- **6-hour horizon**: **Feasible** (36,447 sequences).
- *Recommendation*: Implement and evaluate 1h risk first as the primary benchmark, then evaluate 3h and 6h multi-horizon heads.

### Q10: Are there enough examples for a CNN-BiLSTM?
- **Yes**. With 38,387 sequences across 97 ICU stays, the sample count is sufficient for training a compact, regularized 1D-CNN + BiLSTM architecture with Dropout (0.30), Weight Decay ($1\times 10^{-4}$), and early stopping.
- Patient-level splitting (70% train / 15% val / 15% test) yields ~26,800 train sequences, ~5,700 val sequences, and ~5,700 held-out test sequences.

### Q11: Which proposed features are actually available?
The following 14 features are confirmed present and verified in `chartevents` and `labevents`:
1. `heart_rate` (BPM)
2. `spo2` (%)
3. `respiratory_rate` (breaths/min)
4. `sbp` (Systolic BP, mmHg)
5. `dbp` (Diastolic BP, mmHg)
6. `map` (Mean Arterial Pressure, mmHg)
7. `temperature_c` (°C)
8. `wbc` ($10^3/\mu\text{L}$)
9. `hemoglobin` ($\text{g/dL}$)
10. `lactate` ($\text{mmol/L}$)
11. `creatinine` ($\text{mg/dL}$)
12. `glucose` ($\text{mg/dL}$)
13. `po2` (Blood Gas arterial $p\text{O}_2$, mmHg)
14. `pco2` (Blood Gas arterial $p\text{CO}_2$, mmHg)

Along with 5 time-aligned binary treatment indicators:
15. `tx_vasopressor`
16. `tx_diuretic`
17. `tx_antibiotic`
18. `tx_bronchodilator`
19. `tx_steroid`

### Q12: Which proposed features must be removed?
The following variables from initial wishlists are absent or too sparse in the general demo cohort and are **strictly excluded**:
- Continuous Arterial Line ABG pH / Base Excess (insufficient continuous charting).
- Continuous Cardiac Output ($Q$) and Systemic Vascular Resistance (SVR) (requires invasive Swan-Ganz catheterization, only present in rare cardiac surgery stays).
- Central Venous Pressure (CVP).

---

## 3. Revised Methodological Safeguards (Mandatory Corrections)

```
                       ┌─────────────────────────────────────────────────────────┐
                       │          RAW PHYSIOLOGICAL OBSERVATIONS                 │
                       │    (HR, SpO2, RR, SBP, DBP, MAP, Temp, Labs, Meds)      │
                       └──────────────────────────┬──────────────────────────────┘
                                                  │
                  ┌───────────────────────────────┴───────────────────────────────┐
                  │                                                               │
                  ▼                                                               ▼
  ┌───────────────────────────────┐                               ┌───────────────────────────────┐
  │     EXPERIMENT A (MODEL)      │                               │    EXPERIMENT B (BASELINE)    │
  │  Temporal CNN-BiLSTM on raw   │                               │    Rule-based NEWS 2 Score    │
  │  vitals + labs + tx flags     │                               │    Logistic Regression        │
  │  (Zero Target Leakage)        │                               │    Random Forest              │
  └───────────────┬───────────────┘                               └───────────────┬───────────────┘
                  │                                                               │
                  └───────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │     HELD-OUT TEST SET BENCHMARK     │
                               │   AUROC, AUPRC, Macro-F1, MAE/RMSE   │
                               └─────────────────────────────────────┘
```

1. **Target Leakage Prevention**: In Experiment A, the deep learning model receives only raw physiological features, temporal derivatives, and treatment flags. NEWS 2 is NOT supplied as an input feature. In Experiment B, NEWS 2 is evaluated purely as an external rule-based baseline.
2. **Modular Architecture**: All 4 model heads (Deterioration, Risk Tier, Forecasting, Treatment Response) are independently toggleable.
3. **Explainability Distinction**: Integrated Gradients attribution reports signed and magnitude values as *"Model Feature Attribution"*, explicitly avoiding causal claims.

---

## 4. Next Step Recommendation

Having verified data feasibility, created `data_inventory.json` and `cohort_feasibility_report.json`, and documented all 12 feasibility dimensions, we are ready to proceed with the execution of Phase 4 modules (`cohort.py`, `alignment.py`, `features.py`, `preprocessing.py`, `splitting.py`, `targets.py`, `baselines.py`, `model.py`, `train.py`, `evaluate.py`, `inference.py`, and `PHASE4_REPORT.md`).
