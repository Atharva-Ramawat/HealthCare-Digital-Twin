# PROJECT CONSTITUTION
## AI-Driven Predictive Patient Digital Twin for Healthcare
### B.Tech Major Project — Phase 0 Architecture Document

> **Status:** Phase 0 Complete — Architecture Defined, No Models Implemented  
> **Date:** August 2026  
> **Revision:** 1.0

---

## 1. Project Statement

This project builds a research-grade **AI-Driven Predictive Patient Digital Twin** for ICU patients.

The Digital Twin maintains a continuously-updating computational representation of an ICU patient reconstructed from historical clinical data replayed as a simulated real-time stream. It performs multi-horizon deterioration risk prediction, future physiological state forecasting, explainable AI attribution, uncertainty quantification, and controlled what-if scenario simulation — all visualised through an interactive clinical dashboard.

### 1.1 Clinical Focus
ICU patient physiological deterioration / early deterioration prediction.

Sepsis may be used as an evaluation case in Phase 3/4 if clinically and technically justified.

### 1.2 Critical Boundaries
- NO physical IoT devices. Input is a **simulated real-time clinical data replay** layer.
- NO treatment recommendations. This is a research simulation tool.
- NO clinical claims without defined label generation methodology.
- Data is de-identified, publicly available clinical research data (MIMIC-IV, eICU).

---

## 2. Architecture Overview

```
                    HISTORICAL CLINICAL DATA
                    (MIMIC-IV / VitalDB / eICU / Synthetic)
                              │
                              ▼
                    ┌─────────────────────┐
                    │   DATA INGESTION    │
                    │  (src/ingestion/)   │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  DATA CLEANING &    │
                    │  PREPROCESSING      │
                    │ (src/preprocessing/)│
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ TEMPORAL ALIGNMENT  │
                    │ (uniform time grid) │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ FEATURE ENGINEERING │
                    │   (src/features/)   │
                    │ Δ, dX/dt, baseline  │
                    │ deviation, NEWS2    │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  PATIENT TIMELINE   │
                    │   (per patient,     │
                    │  temporally indexed)│
                    └─────────┬───────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
                ▼                            ▼
    ┌───────────────────┐        ┌──────────────────────┐
    │  DIGITAL TWIN     │        │  STREAMING/REPLAY    │
    │  STATE STORE      │◄───────│  SIMULATOR           │
    │  (src/twin/)      │ update │  (src/simulation/)   │
    └─────────┬─────────┘        └──────────────────────┘
              │
              ▼ (sliding window matrix)
    ┌─────────────────────────────────────────┐
    │              MODEL LAYER                │
    │            (src/models/)               │
    │                                         │
    │  ┌──────────────┐  ┌──────────────────┐ │
    │  │  RISK        │  │  FUTURE STATE    │ │
    │  │  PREDICTION  │  │  PREDICTION      │ │
    │  │  (1h/3h/6h)  │  │  (forecast_      │ │
    │  │  probabilities│  │   horizon steps) │ │
    │  └──────┬───────┘  └───────┬──────────┘ │
    └─────────┼──────────────────┼────────────┘
              └─────────┬────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   WHAT-IF SIMULATION  │
            │  (src/simulation/     │
            │   scenario_simulator) │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   EXPLAINABILITY      │
            │   (src/explainability)│
            │   SHAP / Integrated   │
            │   Gradients           │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  UNCERTAINTY ANALYSIS │
            │  MC Dropout / Ensemble│
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  DIGITAL TWIN UI      │
            │   (dashboard/)        │
            │   Streamlit + Plotly  │
            └───────────────────────┘
```

---

## 3. Module Responsibility Table

| Module | Package Path | Primary Responsibility | Phase |
|---|---|---|---|
| **MIMIC-IV Loader** | `src/ingestion/mimic_iv_loader.py` | Load MIMIC-IV ICU tables (chartevents, labevents, icustays, patients, admissions) | 1 |
| **VitalDB Loader** | `src/ingestion/vitaldb_loader.py` | Load VitalDB high-frequency physiological recordings via API | 2 |
| **Synthetic Generator** | `src/ingestion/synthetic_generator.py` | Generate realistic synthetic vital-sign time series (fallback for dev/demo) | 0/1 |
| **Clinical Data Cleaner** | `src/preprocessing/cleaner.py` | Missing value imputation, outlier detection, physiological clipping | 1 |
| **Temporal Aligner** | `src/preprocessing/temporal_aligner.py` | Resample to uniform time grid (default: 1-minute resolution) | 1 |
| **Vitals Normalizer** | `src/preprocessing/normalizer.py` | Min-Max / Z-score / patient-adaptive normalisation | 1 |
| **Feature Engineer** | `src/features/feature_engineer.py` | Compute Δ, dX/dt, rolling stats, baseline deviation, NEWS2, derived vitals | 1 |
| **Baseline Calculator** | `src/features/baseline_calculator.py` | Patient-specific physiological baseline (EWMA) | 1 |
| **Sequence Builder** | `src/features/sequence_builder.py` | Sliding-window sequence matrices for model input | 1 |
| **Replay Engine** | `src/simulation/replay_engine.py` | Historical clinical data replay as simulated real-time stream | 1 |
| **Scenario Simulator** | `src/simulation/scenario_simulator.py` | What-if / counterfactual trajectory projection | 3 |
| **Digital Twin State** | `src/twin/patient_state.py` | Core data structure representing patient's twin at all times | 1 |
| **Digital Twin Engine** | `src/twin/twin_engine.py` | Orchestrate state updates: observe → feature → predict → explain | 1 |
| **Base Model** | `src/models/base_model.py` | Abstract interface for all models | 1 |
| **Statistical Baseline** | `src/models/baselines.py` | Last-value carry-forward + threshold rules | 2 |
| **Logistic Regression** | `src/models/baselines.py` | Logistic regression on temporal features | 2 |
| **Random Forest** | `src/models/baselines.py` | Random forest on engineered features | 2 |
| **Simple LSTM** | `src/models/baselines.py` | Unidirectional LSTM single-task | 2 |
| **CNN-BiLSTM** | `src/models/cnn_bilstm.py` | Primary deep learning model (multi-task, multi-horizon) | 3 |
| **Model Trainer** | `src/models/trainer.py` | Training loop, early stopping, checkpoint saving | 2 |
| **SHAP Explainer** | `src/explainability/shap_explainer.py` | SHAP feature attributions | 3 |
| **IG Explainer** | `src/explainability/ig_explainer.py` | Integrated Gradients (alternative) | 3 |
| **Clinical Evaluator** | `src/evaluation/metrics.py` | AUROC, AUPRC, calibration, lead-time metrics | 2 |
| **Model Comparator** | `src/evaluation/model_comparator.py` | Side-by-side comparison table of all models | 2 |
| **Streamlit App** | `dashboard/app.py` | Dashboard entry point | 1 |
| **Vital Charts** | `dashboard/components/vital_charts.py` | Time-series vital charts with forecast overlay | 1 |
| **Risk Gauge** | `dashboard/components/risk_gauge.py` | Risk score gauge and multi-horizon bar | 1 |
| **XAI Panel** | `dashboard/components/xai_panel.py` | SHAP attribution visualisation | 3 |
| **What-If Panel** | `dashboard/components/whatif_panel.py` | Counterfactual slider panel | 3 |
| **Ward Grid** | `dashboard/components/ward_grid.py` | ICU ward overview grid | 1 |

---

## 4. Data Flow

### 4.1 Training Data Flow

```
raw/mimic_iv/ ──► MIMICIVLoader ──► ClinicalDataCleaner ──► TemporalAligner
    ──► VitalsNormalizer ──► ClinicalFeatureEngineer ──► SequenceBuilder
    ──► ModelTrainer ──► models/checkpoints/
```

### 4.2 Inference (Simulated Real-Time) Data Flow

```
Synthetic / Replay Data ──► ClinicalReplayEngine
    ──► (per timestep) new_vitals ──► DigitalTwinEngine.update()
        ├── append to vital_history
        ├── update sliding_window buffer
        ├── ClinicalFeatureEngineer (deltas, baseline deviation)
        ├── PatientBaselineCalculator.update_baseline()
        ├── Model.predict(sliding_window)
        │     ├── risk_scores (1h/3h/6h)
        │     ├── forecast_vitals
        │     └── news2_tier
        ├── SHAPExplainer.explain()
        └── update DigitalTwinState ──► Dashboard reads state
```

### 4.3 What-If Simulation Data Flow

```
User sets vital overrides ──► ScenarioDefinition
    ──► ScenarioSimulator.simulate(current_state_snapshot, [scenarios])
        ├── Project each scenario forward N steps through model
        └── ScenarioResult list ──► Dashboard renders trajectory comparison
```

---

## 5. Digital Twin State Schema

The `DigitalTwinState` (defined in `src/twin/patient_state.py`) is the central data structure.

```python
@dataclass
class DigitalTwinState:

    # IDENTITY
    patient_id:           str
    demographics:         dict          # age, gender, weight, comorbidities

    # CURRENT PHYSIOLOGICAL STATE
    current_vitals:       dict          # {vital_key: latest_value}
    vital_history:        DataFrame     # temporally indexed, all observed vitals
    sliding_window:       np.ndarray    # (window_size, num_features) — model input

    # PERSONALISED BASELINE
    patient_baseline:     dict          # {vital_key: ewma_baseline}
    baseline_valid:       bool          # True when >= min_observations
    baseline_deviations:  dict          # Deviation_t = X_t - PatientBaseline

    # TEMPORAL TRENDS
    vital_deltas:         dict          # ΔX_t = X_t - X_{t-1}
    vital_rates:          dict          # dX/dt approximation
    rolling_stats:        dict          # {window: {vital: {mean, std, min, max}}}

    # RISK ASSESSMENT
    news2_score:          int           # Computed NEWS2 integer score
    news2_tier:           str           # "Low" | "Medium" | "High"
    risk_scores:          dict          # {60: p_1h, 180: p_3h, 360: p_6h}
    risk_tier:            str

    # PREDICTIVE STATE
    forecast_vitals:      np.ndarray    # (forecast_horizon, num_features)
    forecast_uncertainty: np.ndarray    # (forecast_horizon,) — optional
    multi_horizon_risk:   dict          # {1h: p, 3h: p, 6h: p}

    # EXPLAINABILITY
    last_shap_values:     dict          # {vital_key: shap_contribution}

    # SIMULATION
    active_scenarios:     list[ScenarioResult]

    # METADATA
    last_updated_at:      datetime
    replay_step_index:    int
    data_source:          str
```

### 5.1 Key Derived Quantities

| Quantity | Formula | Description |
|---|---|---|
| Delta | `ΔX_t = X_t - X_{t-1}` | First-order change |
| Rate of change | `dX/dt ≈ (X_t - X_{t-w}) / w` | Change over window w |
| Baseline deviation | `Dev_t = X_t - Baseline_t` | Personalised deviation |
| Baseline update | `B_t = α · X_t + (1-α) · B_{t-1}` | EWMA personalised baseline |
| Shock Index | `SI = HR / SBP` | Haemodynamic instability indicator |
| MAP | `MAP = (SBP + 2·DBP) / 3` | Mean arterial pressure |

---

## 6. Model Architecture (Planned)

### 6.1 Primary Architecture: CNN-BiLSTM (Phase 3)

```
Input: (Batch, window_size=24, num_features=6+)
    ↓
Conv1D (filters=64, kernel=3, padding=same) + BatchNorm + ReLU
    ↓
BiLSTM (hidden=128, layers=2, dropout=0.2)
    ↓
Last hidden state: (Batch, 256)
    ↓
┌──────────────────┬──────────────────────┬───────────────────────┐
│ Head 1: Risk     │ Head 2: NEWS2 Tier   │ Head 3: Forecast      │
│ Linear(256,32)   │ Linear(256,32)       │ Linear(256,64)        │
│ ReLU             │ ReLU                 │ ReLU                  │
│ Linear(32,3)     │ Linear(32,3)         │ Linear(64,horizon×F)  │
│ Sigmoid (×3)     │ Softmax (3 class)    │ Reshape (horizon,F)   │
│ (1h,3h,6h prob)  │                      │                       │
└──────────────────┴──────────────────────┴───────────────────────┘
```

**Note:** Head 1 is extended from legacy prototype's single-horizon to multi-horizon (1h/3h/6h) probabilities. This is the primary architectural enhancement.

### 6.2 Comparison Ladder (Phase 2 → Phase 3)

| Model | Type | Horizon | Features |
|---|---|---|---|
| Statistical Baseline | Heuristic | Single | Raw vitals |
| Logistic Regression | Classical ML | Single | Engineered features |
| Random Forest | Classical ML | Single | Engineered features |
| Simple LSTM | DL | Single | Raw vitals |
| **CNN-BiLSTM** | **DL (primary)** | **Multi (1h/3h/6h)** | **Full features** |

### 6.3 Label Generation

**Risk Labels:**
- Will be derived from MIMIC-IV outcomes: ICU mortality, sepsis onset (Sepsis-3 criteria), organ failure flags
- Labels must NOT be fabricated — derivation methodology must be explicitly documented in Phase 2

**Forecast Labels:**
- Future vital-sign observations from the same patient timeline (supervised regression)

---

## 7. Storage and Data Structure

```
data/
├── raw/           # Raw source data — never modified — not committed
├── interim/       # Partially processed (per-patient DataFrames) — not committed
├── processed/     # Full analysis-ready dataset — not committed
├── features/      # Feature matrices and sequence arrays — not committed
└── synthetic/     # Synthetic ward data (safe to commit for testing)

models/
├── checkpoints/   # Saved model weights (.pt files) — not committed to git
└── registry/      # JSON model metadata (hyperparams, eval metrics, git hash) — committed
```

**Storage Format:**
| Stage | Format | Rationale |
|---|---|---|
| Raw MIMIC-IV | CSV / Parquet | Native MIMIC-IV format |
| Interim / Processed | Parquet | Efficient columnar I/O |
| Feature matrices | NumPy .npz | Fast array serialisation |
| Model weights | PyTorch .pt | Native format |
| Configuration | YAML | Human-readable |

---

## 8. Development Roadmap

### Phase 0 — Architecture Scaffold ✅ COMPLETE
**Deliverables:**
- [x] Full modular directory structure
- [x] Project Constitution document
- [x] Digital Twin State schema
- [x] Module responsibility table
- [x] Data flow definition
- [x] Configuration file (configs/settings.yaml)
- [x] Placeholder stubs for all modules
- [x] Legacy prototype archived in docs/legacy_prototype/
- [x] README, .gitignore, pyproject.toml

---

### Phase 1 — Data Pipeline & Digital Twin Foundation 🔜 NEXT
**Estimated Duration:** 3-4 weeks  
**Key Deliverables:**
- [ ] Port and refactor synthetic generator with config integration
- [ ] MIMIC-IV ingestion (start with MIMIC-IV Demo subset)
- [ ] Data cleaning pipeline (cleaner.py)
- [ ] Temporal alignment to 1-minute grid
- [ ] Feature engineering (deltas, rolling stats, baseline deviation, NEWS2)
- [ ] DigitalTwinState fully implemented and tested
- [ ] DigitalTwinEngine (orchestrator) implemented
- [ ] ClinicalReplayEngine operational with synthetic data
- [ ] Minimal but functional Streamlit dashboard showing live state updates
- [ ] Unit tests for all Phase 1 components
- [ ] EDA notebook: MIMIC-IV data exploration

**Success Criteria:**
- A synthetic patient's DigitalTwinState updates correctly at each timestep
- Dashboard displays current vitals, deltas, baseline deviations, and NEWS2 score
- All Phase 1 unit tests pass

---

### Phase 2 — Baseline Models & Training Pipeline
**Estimated Duration:** 3-4 weeks  
**Key Deliverables:**
- [ ] Sequence builder (sliding window dataset generation from feature DataFrames)
- [ ] Patient-level train/val/test split
- [ ] Label generation from MIMIC-IV outcomes (documented methodology)
- [ ] Statistical baseline
- [ ] Logistic Regression + Random Forest baselines
- [ ] Simple LSTM baseline
- [ ] Training pipeline with early stopping
- [ ] Evaluation framework: AUROC, AUPRC, calibration, lead time
- [ ] Model comparison table (notebook)

**Success Criteria:**
- All baseline models produce AUROC > 0.65 on validation set (minimum viability)
- Training/evaluation pipeline is fully reproducible

---

### Phase 3 — CNN-BiLSTM, XAI, What-If
**Estimated Duration:** 4-5 weeks  
**Key Deliverables:**
- [ ] CNN-BiLSTM multi-task multi-horizon model
- [ ] Multi-horizon risk prediction heads (1h / 3h / 6h)
- [ ] Vital forecast head
- [ ] MC Dropout uncertainty quantification
- [ ] SHAP explainer for CNN-BiLSTM
- [ ] ScenarioSimulator for what-if trajectory projection
- [ ] Full dashboard integration (XAI panel, what-if panel, uncertainty display)
- [ ] Comparison: CNN-BiLSTM vs all baselines

**Success Criteria:**
- CNN-BiLSTM outperforms all baselines on at least one primary metric
- SHAP explanations are stable and clinically plausible
- What-if scenarios produce meaningfully different trajectories

---

### Phase 4 — Evaluation, Validation & Documentation
**Estimated Duration:** 2-3 weeks  
**Key Deliverables:**
- [ ] eICU external validation (if access obtained)
- [ ] MIMIC-III Waveform data integration (optional)
- [ ] VitalDB integration (optional)
- [ ] Full evaluation report notebook
- [ ] Project report / thesis writeup data
- [ ] API documentation
- [ ] Final dashboard polish

---

## 9. What Is NOT Implemented Yet

As of Phase 0, **nothing beyond scaffolding is implemented**. Specifically:

| Component | Implementation Status |
|---|---|
| MIMIC-IV data loading | ❌ Not implemented |
| VitalDB loading | ❌ Not implemented |
| Data cleaning | ❌ Not implemented |
| Temporal alignment | ❌ Not implemented |
| Feature engineering | ❌ Not implemented |
| Patient baseline calculator | ❌ Not implemented |
| Sequence builder | ❌ Not implemented |
| Clinical replay engine | ❌ Not implemented |
| Digital Twin State (full) | 🟡 Schema defined, not wired |
| Digital Twin Engine | ❌ Not implemented |
| Statistical baseline model | ❌ Not implemented |
| Logistic Regression model | ❌ Not implemented |
| Random Forest model | ❌ Not implemented |
| Simple LSTM | ❌ Not implemented |
| CNN-BiLSTM | ❌ Not implemented |
| SHAP explainer | ❌ Not implemented |
| Integrated Gradients explainer | ❌ Not implemented |
| Scenario simulator | ❌ Not implemented |
| Uncertainty quantification | ❌ Not implemented |
| Training pipeline | ❌ Not implemented |
| Evaluation framework | ❌ Not implemented |
| Full dashboard | 🟡 Placeholder only |
| Any model training results | ❌ None — no fabricated metrics |
| Label generation methodology | ❌ Pending Phase 2 |

**The legacy prototype (docs/legacy_prototype/) is a fully synthetic, monolithic demo. It is NOT used in the new modular architecture and is preserved for reference only.**

---

## 10. Known Risks and Unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| MIMIC-IV access and download time | Medium | Start with MIMIC-IV Demo subset; obtain full access in parallel |
| Irregular sampling in MIMIC-IV chartevents | High | Temporal aligner handles resampling; document assumptions |
| Class imbalance in deterioration labels | High | AUPRC as primary metric; class weighting in loss function |
| Label quality: deterioration labels may be noisy | High | Use established definitions (Sepsis-3); document label derivation fully |
| MIMIC-IV has per-patient temporal heterogeneity | Medium | Patient-level train/val/test split; not random |
| CNN-BiLSTM may not outperform simpler baselines | Medium | Comparison ladder is built in; report honestly |
| SHAP for temporal CNNs requires careful implementation | Medium | Use DeepSHAP or TimeShap; validate attribution stability |
| MC Dropout uncertainty may be poorly calibrated | Medium | Calibration curve + Brier score in evaluation |
| What-if simulation is physically unconstrained | Low | Use for trajectory comparison only; add disclaimer in UI |
| eICU access (Phase 4) may not be obtainable in time | Low | Mark as optional; skip if unavailable |

---

## 11. Next Phase Recommendation

**Proceed to Phase 1: Data Pipeline & Digital Twin Foundation.**

### Recommended Phase 1 Start Sequence

1. **Apply for MIMIC-IV access** on PhysioNet (if not already held). Begin with MIMIC-IV Demo.
2. **Port and refactor the synthetic generator** from the legacy prototype into `src/ingestion/synthetic_generator.py` with config integration. This gives an immediately usable data source.
3. **Implement DigitalTwinState fully** — wire the dataclass, add `to_dict()` and `summary()`, write unit tests.
4. **Implement ClinicalFeatureEngineer** — deltas, rolling stats, baseline deviation, NEWS2. Use synthetic data to validate outputs.
5. **Implement ClinicalReplayEngine** (synthetic mode) — get the end-to-end `replay → state update` loop working.
6. **Implement DigitalTwinEngine** — wire ingestion → features → state updates.
7. **Build minimal Streamlit dashboard** — display live synthetic patient state (vitals, NEWS2, deltas).
8. **Start MIMIC-IV ingestion** once data access is confirmed.
9. **Write EDA notebook** on MIMIC-IV ICU stays.

**Do NOT start model training (Phase 2) until the complete data → feature → Digital Twin State pipeline is validated on real MIMIC-IV data.**

---

*End of Project Constitution — Phase 0*
