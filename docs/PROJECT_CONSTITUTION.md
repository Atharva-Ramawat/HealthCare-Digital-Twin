# PROJECT CONSTITUTION
## AI-Driven Predictive Patient Digital Twin for ICU Healthcare
### B.Tech Major Project — Living Architecture Document

> **Status:** Phase 1 Complete — Research Decisions Locked
> **Date:** August 2026
> **Revision:** 1.2 (Phase 1 decisions incorporated)

---

## 0. Ownership and Responsibility Matrix

### 0.1 Student Research Team Owns

The student research team owns all ML research decisions:

- Clinical prediction target definition and finalisation
- Clinical label derivation methodology
- Feature selection decisions
- Final CNN-BiLSTM architecture choices
- Model training and hyperparameter tuning
- Model evaluation design and execution
- Experimental design and ablation studies
- Interpretation of ML results
- Final research conclusions

### 0.2 Antigravity Is Responsible For

- Software architecture and design
- Data ingestion implementations
- Preprocessing pipeline implementation
- Temporal alignment pipeline
- Digital Twin infrastructure (state, engine, interfaces)
- Simulated real-time clinical data replay layer
- Model-agnostic ML integration interfaces
- Dashboard engineering
- Testing infrastructure
- Engineering documentation

### 0.3 Antigravity Must NEVER

- Fabricate ML results, metrics, or predictions
- Fabricate clinical labels or outcome definitions
- Claim synthetic data represents real patient behaviour
- Make unsupported clinical claims
- Decide the final ML architecture without student team approval
- Present placeholder inference outputs as validated research results

### 0.4 Placeholder Models

Placeholder models are permitted ONLY for software integration testing, demonstrating
data flow, and UI scaffolding with dummy outputs.

Every placeholder model output MUST be clearly labelled in code, logs, and UI.
No placeholder output may be presented as a research result.

---

## 1. Project Statement

This project builds a research-grade **AI-Driven Predictive Patient Digital Twin** for ICU patients.

The Digital Twin maintains a continuously-updating computational representation of an ICU patient,
reconstructed from historical clinical data replayed as a simulated real-time stream.

### 1.1 Clinical Focus

**ICU patient physiological deterioration / early deterioration prediction.**

> [!IMPORTANT]
> The prediction targets have been finalised by the student research team after Phase 1 EDA.
> See `docs/PHASE1_DECISIONS.md` for the full decision record.

**Approved primary target:** Mechanical ventilation initiation  
**Approved secondary target:** Vasopressor initiation  
**Approved prediction horizons:** 1 hour, 3 hours, 6 hours  
**Mortality:** Excluded as primary target (class imbalance — 15/275 admissions in Demo).
May be explored as a secondary outcome if full MIMIC-IV access is obtained.

**Sepsis-3 is NOT used.** The approved targets are operationally defined from
`icu/procedureevents` (ventilation) and `icu/inputevents` (vasopressors).
Final label derivation logic is a student team responsibility (Phase 5).


### 1.2 Critical Boundaries

- NO physical IoT devices. Input is a **simulated real-time clinical data replay** layer.
- NO treatment recommendations. This is a research simulation tool only.
- NO clinical claims without an explicitly documented label generation methodology.
- Data is de-identified, publicly available clinical research data.
- **Synthetic data is NOT research data.** See Section 1.3.

### 1.3 Synthetic Data Policy

Synthetic data is permitted ONLY for:

- Unit testing of software components
- Software development and pipeline integration testing
- Replay-engine and edge-case testing
- UI demonstration when real clinical data is unavailable

Synthetic data MUST NOT be used to:

- Claim clinical model performance
- Report model accuracy or evaluation metrics
- Represent real-world patient behaviour
- Support any clinical or research conclusion

All ML performance claims must be derived from legitimate clinical research datasets
(MIMIC-IV, MIMIC-IV Demo, VitalDB, eICU) under their respective data use agreements.

---

## 2. Architecture Overview

### 2.1 Layered Architecture

The system comprises five distinct layers.
**The Digital Twin is NOT the dashboard.**
The Digital Twin Engine and its state exist independently of any visualisation layer.

```
LAYER 1: DATA ACQUISITION
  Historical Clinical Data
  (MIMIC-IV / MIMIC-IV Demo / VitalDB / MIMIC Waveform / eICU)
       |                         |
  [Replay Simulator]       [Training Pipeline]
  (simulated real-time)    (offline, batch)
       |
LAYER 2: PREPROCESSING PIPELINE
  Ingestion -> Cleaning -> Temporal Alignment -> Feature Engineering
  (src/ingestion/ -> src/preprocessing/ -> src/features/)
       |
LAYER 3: DIGITAL TWIN CORE  <- INDEPENDENT OF DASHBOARD
  DigitalTwinState (Observed | Derived | Predicted | Simulation)
  DigitalTwinEngine (orchestrator)
       |
LAYER 4: INTELLIGENCE LAYER  (student-owned ML research)
  PredictionInterface (model-agnostic)
  Risk Prediction / Future State / Uncertainty / XAI / What-If
       |
LAYER 5: VISUALISATION LAYER  (reads from Digital Twin only)
  Dashboard (Streamlit + Plotly) -- dashboard/ --
```

### 2.2 Digital Twin Independence Principle

The Digital Twin MUST be operable without Streamlit running.

```
  Digital Twin Engine
        | updates
  DigitalTwinState
        | read by
  PredictionInterface -> results written back to state
        | read by
  Dashboard (visualisation only)
```

The dashboard NEVER writes to DigitalTwinState directly.
The dashboard NEVER triggers ML inference directly.

---

## 3. Module Responsibility Table

| Module | Path | Responsibility | Phase | Owner |
|---|---|---|---|---|
| MIMIC-IV Loader | `src/ingestion/mimic_iv_loader.py` | Load MIMIC-IV ICU tables | 1 | Antigravity |
| VitalDB Loader | `src/ingestion/vitaldb_loader.py` | Load VitalDB recordings | 1 | Antigravity |
| Synthetic Generator | `src/ingestion/synthetic_generator.py` | Synthetic vitals (dev/test ONLY) | 0/1 | Antigravity |
| Data Cleaner | `src/preprocessing/cleaner.py` | Missing values, outliers, clipping | 2 | Antigravity |
| Temporal Aligner | `src/preprocessing/temporal_aligner.py` | Resample to configurable resolution | 2 | Antigravity |
| Vitals Normalizer | `src/preprocessing/normalizer.py` | Normalisation (strategy Phase 1) | 2 | Antigravity |
| Feature Engineer | `src/features/feature_engineer.py` | Delta, rates, rolling stats, NEWS2 | 2 | Antigravity |
| Baseline Calculator | `src/features/baseline_calculator.py` | Patient-specific baseline (causal) | 2 | Antigravity |
| Sequence Builder | `src/features/sequence_builder.py` | Sliding-window input matrices | 2 | Antigravity |
| Replay Engine | `src/simulation/replay_engine.py` | Simulated real-time data stream | 4 | Antigravity |
| Prediction Interface | `src/models/prediction_interface.py` | Model-agnostic I/O contract | 3 | Antigravity |
| Base Model | `src/models/base_model.py` | Abstract model interface | 3 | Antigravity |
| Baseline Models | `src/models/baselines.py` | Statistical, LR, RF, Simple LSTM | 5 | Student team |
| CNN-BiLSTM | `src/models/cnn_bilstm.py` | Primary DL model (team decides arch) | 5 | Student team |
| Model Trainer | `src/models/trainer.py` | Training loop infrastructure | 5 | Antigravity/Student |
| Scenario Simulator | `src/simulation/scenario_simulator.py` | Hypothetical trajectory projection | 7 | Antigravity/Student |
| Digital Twin State | `src/twin/patient_state.py` | Core patient state data structure | 3 | Antigravity |
| Digital Twin Engine | `src/twin/twin_engine.py` | Per-timestep state update orchestrator | 3 | Antigravity |
| XAI Explainer | `src/explainability/explainer.py` | XAI attribution (method TBD) | 8 | Antigravity/Student |
| Clinical Evaluator | `src/evaluation/metrics.py` | AUROC, AUPRC, calibration, lead-time | 5 | Antigravity/Student |
| Model Comparator | `src/evaluation/model_comparator.py` | Baseline vs main model comparison | 5 | Student team |
| Streamlit App | `dashboard/app.py` | Dashboard entry point (vis only) | 4 | Antigravity |
| Dashboard Components | `dashboard/components/*.py` | Vital charts, risk, XAI, what-if UI | 4-9 | Antigravity |

---

## 4. Data Flow

### 4.1 Offline Training Data Flow

```
raw/mimic_iv/
    -> MIMICIVLoader
    -> ClinicalDataCleaner
    -> TemporalAligner (configurable resolution, confirmed Phase 1)
    -> VitalsNormalizer
    -> ClinicalFeatureEngineer
    -> SequenceBuilder
    -> [Student team: label generation + model training]
    -> models/checkpoints/
```

### 4.2 Simulated Real-Time Inference Data Flow

```
Historical Dataset (MIMIC-IV / Synthetic fallback)
    -> ClinicalReplayEngine
    -> (per simulated timestep) new_vitals observation
    -> DigitalTwinEngine.update(patient_id, new_vitals)
          |
          +- append to ObservedState.vital_history
          +- update ObservedState.sliding_window
          +- ClinicalFeatureEngineer -> update DerivedState
          |     (Delta, dX/dt, rolling stats, baseline deviation, NEWS2)
          +- PatientBaselineCalculator.update(new_vitals)  [causal: past only]
          +- PredictionInterface.predict(window)
          |     -> PredictedState.risk_scores
          |     -> PredictedState.future_state (optional)
          |     -> PredictedState.uncertainty (if available)
          +- updated DigitalTwinState -> Dashboard reads (visualisation only)
```

### 4.3 What-If Simulation Data Flow

```
User defines ScenarioDefinition(s)
    -> ScenarioSimulator.simulate(observed_state_snapshot, [scenarios])
          +- Project via PredictionInterface (hypothetical only)
          +- Return list[ScenarioResult]
    -> SimulationState.scenario_results updated
    -> Dashboard renders trajectory comparison

NOTE: All what-if output is labelled:
"Hypothetical model projection - NOT a clinical recommendation"
```

---

## 5. Digital Twin State Schema

The DigitalTwinState is partitioned into four sub-states.

### 5.1 Observed State

Contains only directly measured/reported information.

```python
@dataclass
class ObservedState:
    patient_id:        str
    demographics:      dict
    current_vitals:    dict          # {vital_key: latest_raw_value}
    vital_history:     pd.DataFrame  # full temporally-indexed record
    latest_timestamp:  datetime
    data_source:       str           # "synthetic" | "mimic_iv" | "vitaldb"
    replay_step_index: int
```

### 5.2 Derived State

Computed causally from observed history only.
**All derivations use ONLY observations available at or before time t.**

```python
@dataclass
class DerivedState:
    sliding_window:      np.ndarray  # (window_size, num_features)
    vital_deltas:        dict        # DeltaX_t = X_t - X_{t-1}
    vital_rates:         dict        # dX/dt over short window
    rolling_stats:       dict        # {window: {vital: {mean, std, min, max}}}
    patient_baseline:    dict        # patient-specific baseline (method TBD)
    baseline_valid:      bool        # True when >= min_observations
    baseline_deviations: dict        # X_t - PatientBaseline_t
    news2_score:         int
    news2_tier:          str         # "Low" | "Medium" | "High"
    derived_vitals:      dict        # MAP, Shock Index, Pulse Pressure
```

### 5.3 Predicted State

Populated by PredictionInterface after each model call.
All fields None/False until a real model is integrated (Phase 6).

```python
@dataclass
class PredictedState:
    risk_scores:         dict                  # {horizon_steps: probability}
    risk_tier:           str
    future_state:        Optional[np.ndarray]  # optional, if model supports it
    uncertainty:         Optional[dict]        # PredictionUncertainty (method TBD)
    xai_attributions:    Optional[dict]        # {vital_key: attribution} (method TBD)
    prediction_valid:    bool                  # False until real model integrated
    last_predicted_at:   Optional[datetime]
```

### 5.4 Simulation State

Populated by ScenarioSimulator. Empty until Phase 7.

```python
@dataclass
class SimulationState:
    active_scenarios:     list
    scenario_results:     list    # hypothetical projections only
    simulation_valid:     bool
    simulation_timestamp: Optional[datetime]
```

### 5.5 Top-Level Wrapper

```python
@dataclass
class DigitalTwinState:
    observed:        ObservedState
    derived:         DerivedState
    predicted:       PredictedState
    simulation:      SimulationState
    last_updated_at: datetime
```

---

## 6. Prediction Interface (Model-Agnostic)

The Digital Twin Engine depends on PredictionInterface, NOT CNN-BiLSTM directly.

```python
@dataclass
class ModelInput:
    sliding_window:   np.ndarray  # (window_size, num_features)
    patient_id:       str
    derived_features: dict        # supplementary features

@dataclass
class PredictionOutput:
    risk_scores:     dict                  # {horizon_steps: float} -- required
    risk_tier:       str
    future_state:    Optional[np.ndarray]  # optional
    uncertainty:     Optional[dict]        # optional, method TBD
    is_placeholder:  bool = True           # MUST be False for real trained models

class PredictionInterface(ABC):
    @abstractmethod
    def predict(self, model_input: ModelInput) -> PredictionOutput: ...
    @abstractmethod
    def is_ready(self) -> bool: ...
```

Rules:
- `is_placeholder = True` for all stub/demo models.
- Dashboard MUST visually distinguish placeholder outputs from real model outputs.
- `future_state` is OPTIONAL. Digital Twin functions without it.
- `uncertainty` is OPTIONAL. Method selected in Phase 8.

---

## 7. Storage and Data Structure

Only `data/synthetic/` may be committed.
No real clinical data is ever committed to the repository.

| Stage | Format | Committed? |
|---|---|---|
| Raw clinical data | CSV / Parquet | NO |
| Interim / Processed | Parquet | NO |
| Feature matrices | NumPy .npz | NO |
| Model weights | PyTorch .pt | NO |
| Synthetic (dev/test) | CSV | YES |
| Model registry metadata | JSON | YES |
| Configuration | YAML | YES |

---

## 8. Temporal Resolution Policy

**Temporal resolution is CONFIGURABLE and NOT fixed at Phase 0.**

Clinical datasets have different sampling frequencies:
- MIMIC-IV chartevents: typically hourly for many vitals
- MIMIC-III waveform: up to 125 Hz
- VitalDB: 1-500 Hz

The final resolution(s) will be confirmed after Phase 1 dataset profiling.

In configs/settings.yaml:

```yaml
preprocessing:
  temporal_resolution_minutes: null   # TBD after Phase 1 EDA
```

No component may hardcode `resolution = 1 minute` as a universal assumption.

---

## 9. Patient Baseline Methodology

### 9.1 Engineering Initial Approach

For Phase 2/3 engineering, an EWMA is used as a starting point:

  B_t = alpha * X_t + (1 - alpha) * B_{t-1}

This is an **initial engineering approach**, NOT the final research methodology.

### 9.2 Research Methodology (Phase 5)

The final baseline methodology is a student team research decision and may differ.

### 9.3 Causal Constraint (Non-Negotiable)

**The patient baseline at time t must only use observations available up to time t.**

Future observations MUST NEVER influence the baseline used for an earlier prediction.
This constraint applies to ALL baseline methods chosen.

---

## 10. What-If Simulation — Research Boundary

### What It Is

A controlled **hypothetical trajectory simulation** tool.
Compares model-projected patient trajectories under different assumed vital-sign scenarios.

### What It Is NOT

- NOT a treatment recommendation engine
- NOT a clinical prescription tool
- NOT a causal model
- NOT a validated clinical tool

All what-if outputs must be labelled:
"Hypothetical model projection — not a clinical recommendation"

The scenario methodology will be designed by the student team in Phase 7.

---

## 11. Uncertainty Estimation

Uncertainty is architecturally supported via `PredictedState.uncertainty`.

**The uncertainty estimation method is NOT fixed at Phase 0.**

It is a research decision for the student team in Phase 8.

Candidate methods (illustrative):
- Monte Carlo (MC) Dropout
- Deep Ensembles
- Conformal Prediction
- Bayesian approximations

---

## 12. Explainability (XAI)

XAI is supported via `PredictedState.xai_attributions`.

**The XAI method is NOT coupled to one library at Phase 0.**

A generic `ExplainerInterface` will be defined in Phase 3.
The specific method will be selected by the student team in Phase 8.

Candidate methods (illustrative):
- SHAP (currently favoured)
- Integrated Gradients (captum)
- TimeShap
- Attention-weight visualisation

---

## 13. Model Architecture

### 13.1 Current Direction

CNN-BiLSTM is the current research direction, subject to student team finalisation.

The values in `configs/settings.yaml` under `models.cnn_bilstm` are illustrative starting points only.

### 13.2 Candidate Prediction Targets (Not Finalised)

The following are candidate deterioration labels under consideration.
The final selection is a Phase 5 research decision:

- ICU mortality within 6/12/24 hours
- Sepsis onset (Sepsis-3 or another justified definition)
- Acute respiratory deterioration (SpO2-based)
- Composite deterioration score
- Other outcome that Phase 1 dataset analysis reveals as tractable

### 13.3 Model Comparison Ladder

| Model | Type | Phase | Owner |
|---|---|---|---|
| Statistical Baseline | Heuristic | 5 | Student team |
| Logistic Regression | Classical ML | 5 | Student team |
| Random Forest | Classical ML | 5 | Student team |
| Simple LSTM | DL temporal | 5 | Student team |
| **CNN-BiLSTM** | **DL primary (planned)** | **5** | **Student team** |

---

## 14. Phase-Gated Development Roadmap

### PHASE GATING RULE (Mandatory)

Every phase ends with a phase gate review.
The next phase MUST NOT begin until the student team explicitly approves.

Each gate summary must include:
1. Implementation summary
2. Files created or modified
3. Tests performed and results
4. Known limitations
5. Research decisions requiring student approval
6. Next-phase recommendation

**Antigravity does not advance phases autonomously.**

---

PHASE 0 -- Architecture and Project Constitution -- COMPLETE (v1.1)
PHASE 1 -- Dataset Discovery, Ingestion and Data Profiling -- NEXT
PHASE 2 -- Clinical Preprocessing and Temporal Pipeline
PHASE 3 -- Digital Twin State and Engine
PHASE 4 -- Simulated Real-Time Replay + Minimal Visualisation  [*** 30% MILESTONE ***]
PHASE 5 -- Student ML Research and Model Development  (student-owned)
PHASE 6 -- ML Integration with Digital Twin
PHASE 7 -- What-If Trajectory Simulation
PHASE 8 -- XAI and Uncertainty Integration
PHASE 9 -- Complete Dashboard
PHASE 10 -- External Validation
PHASE 11 -- Final Integration, Testing and Research Evaluation

---

## 15. 30% Milestone Definition

The 30% milestone is reached when:

  Real clinical/demo dataset (MIMIC-IV Demo)
          |
  Data ingestion (MIMICIVLoader)
          |
  Preprocessing + Feature Engineering
          |
  DigitalTwinState populated correctly
          |
  DigitalTwinEngine driving state updates
          |
  Sequential simulated clinical observations (replay)
          |
  Twin state updates per timestep
          |
  Basic visualisation (real derived features shown in dashboard)

This milestone does NOT require:
- Trained ML models
- CNN-BiLSTM
- SHAP / XAI
- What-if simulation
- Final dashboard

---

## 16. What Is NOT Implemented

As of Phase 0 v1.1, nothing beyond architectural scaffold and stubs is implemented.

| Component | Status |
|---|---|
| MIMIC-IV data loading | NOT IMPLEMENTED |
| VitalDB loading | NOT IMPLEMENTED |
| Data cleaning | NOT IMPLEMENTED |
| Temporal alignment | NOT IMPLEMENTED |
| Feature engineering | NOT IMPLEMENTED |
| Patient baseline calculator | NOT IMPLEMENTED |
| Sequence builder | NOT IMPLEMENTED |
| Clinical replay engine | NOT IMPLEMENTED |
| DigitalTwinState (four-partition, wired) | SCHEMA DEFINED ONLY |
| Digital Twin Engine | NOT IMPLEMENTED |
| PredictionInterface | INTERFACE DEFINED ONLY |
| PlaceholderPredictor | NOT IMPLEMENTED |
| Any prediction model | NOT IMPLEMENTED |
| XAI / SHAP | NOT IMPLEMENTED |
| Uncertainty quantification | NOT IMPLEMENTED |
| What-if simulation | NOT IMPLEMENTED |
| Model training pipeline | NOT IMPLEMENTED |
| Evaluation framework | NOT IMPLEMENTED |
| Prediction target / labels | PENDING PHASE 5 RESEARCH DECISION |
| Full dashboard | PLACEHOLDER SCAFFOLD ONLY |
| Any training results or metrics | NONE - NO FABRICATED NUMBERS |

---

## 17. Known Risks

| Risk | Severity | Mitigation |
|---|---|---|
| MIMIC-IV credentialing delay | Medium | Start with MIMIC-IV Demo |
| Sparse sampling in MIMIC-IV chartevents | High | Resolution determined after Phase 1 EDA |
| Class imbalance in deterioration labels | High | AUPRC primary metric; assessed Phase 5 |
| Label definition uncertainty | High | Phase 5 research decision |
| Per-patient temporal heterogeneity | Medium | Patient-level splits enforced |
| CNN-BiLSTM may not outperform baselines | Medium | Results reported honestly |
| XAI method / model compatibility | Medium | Generic interface allows substitution |
| Causal leakage in baseline computation | High | Strict causal constraint in code |
| eICU access not guaranteed | Low | External validation optional |

---

## 18. Architecture Validation Checklist

- [x] ML ownership belongs to student team (Section 0)
- [x] Sepsis-3 is NOT hardcoded as the final prediction target (Sections 1.1, 13.2)
- [x] Synthetic data is clearly separated from research data (Sections 1.3, 7)
- [x] Digital Twin is independent of dashboard (Sections 2.1, 2.2)
- [x] Prediction interface is model-agnostic (Section 6)
- [x] Observed/Derived/Predicted/Simulation state separated (Section 5)
- [x] Temporal resolution is configurable (Section 8)
- [x] EWMA is not treated as final baseline methodology (Section 9)
- [x] What-if simulation is clearly hypothetical (Section 10)
- [x] Uncertainty method is not prematurely fixed (Section 11)
- [x] XAI is not prematurely coupled to one library (Section 12)
- [x] Phase gates are enforced (Section 14)
- [x] 30% milestone is clearly defined (Section 15)
- [x] No ML results have been fabricated (Section 16)
- [x] No research conclusions have been fabricated

---

*End of Project Constitution -- Phase 0 v1.1*
