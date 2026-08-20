# ARCHITECTURE
## AI-Driven Predictive Patient Digital Twin for ICU Healthcare

> Cross-reference: docs/PROJECT_CONSTITUTION.md (authoritative design document)
> This file provides a compact visual and structural reference.

---

## System Layers

```
LAYER 1: DATA ACQUISITION
  MIMIC-IV / MIMIC-IV Demo / VitalDB / MIMIC Waveform / eICU / Synthetic (dev only)
       |                         |
  [Replay Simulator]       [Training Pipeline]
  simulated real-time      offline / batch
  (src/simulation/)        (src/models/trainer.py)

LAYER 2: PREPROCESSING PIPELINE  (src/preprocessing/, src/features/)
  Ingestion -> Cleaning -> Temporal Alignment -> Normalisation -> Feature Engineering

LAYER 3: DIGITAL TWIN CORE  (src/twin/)  << INDEPENDENT OF DASHBOARD >>
  DigitalTwinEngine (orchestrator)
  DigitalTwinState:
    ObservedState  |  DerivedState  |  PredictedState  |  SimulationState

LAYER 4: INTELLIGENCE LAYER  (src/models/, src/simulation/, src/explainability/)
  PredictionInterface (model-agnostic -- does not depend on CNN-BiLSTM directly)
    -> Risk Prediction (multi-horizon)
    -> Future State Prediction (optional)
    -> PredictionUncertainty (method TBD Phase 8)
    -> XAI Attribution (method TBD Phase 8)
  ScenarioSimulator
    -> Hypothetical trajectory projection (NOT treatment recommendations)

LAYER 5: VISUALISATION ONLY  (dashboard/)
  Streamlit + Plotly
  Reads from DigitalTwinState
  Does NOT write to state or trigger inference directly
```

---

## Digital Twin State Partition

```
DigitalTwinState
  |
  +-- ObservedState          <- raw measured/reported data only
  |     patient_id, demographics, current_vitals,
  |     vital_history, latest_timestamp, data_source
  |
  +-- DerivedState           <- computed causally from ObservedState
  |     sliding_window, vital_deltas, vital_rates, rolling_stats,
  |     patient_baseline, baseline_valid, baseline_deviations,
  |     news2_score, news2_tier, derived_vitals (MAP, SI, PP)
  |
  +-- PredictedState         <- populated by PredictionInterface (empty until Phase 6)
  |     risk_scores, risk_tier,
  |     future_state (optional), uncertainty (method TBD), xai_attributions (method TBD),
  |     prediction_valid (False until real model), last_predicted_at
  |
  +-- SimulationState        <- populated by ScenarioSimulator (empty until Phase 7)
        active_scenarios, scenario_results (hypothetical only),
        simulation_valid, simulation_timestamp
```

**Causal constraint:** DerivedState at time t uses ONLY observations from time <= t.

---

## Prediction Interface Contract

The Digital Twin Engine depends on PredictionInterface, NOT CNN-BiLSTM internals.

```
ModelInput
  sliding_window (window_size, num_features)
  patient_id
  derived_features (optional)
        |
        v
PredictionInterface.predict()
        |
        v
PredictionOutput
  risk_scores  {horizon_steps: probability}   <- REQUIRED
  risk_tier    str
  future_state np.ndarray                     <- OPTIONAL
  uncertainty  dict                           <- OPTIONAL, method TBD
  is_placeholder bool                         <- MUST be True for stubs
```

---

## Key Engineering Rules

1. The dashboard NEVER writes to DigitalTwinState.
2. The dashboard NEVER triggers ML inference.
3. PredictionOutput.is_placeholder MUST be True for all stub/demo models.
4. Temporal resolution is configurable (not hardcoded to 1 minute).
5. Patient baseline at time t uses only observations up to time t (causal).
6. Synthetic data is for development and testing ONLY -- not for research claims.
7. What-if outputs are labelled "HYPOTHETICAL PROJECTION -- NOT A CLINICAL RECOMMENDATION".
8. Phases advance only after explicit student team approval.

---

## Key Files

| File | Purpose |
|---|---|
| docs/PROJECT_CONSTITUTION.md | Authoritative design document |
| configs/settings.yaml | All configuration (temporal resolution set after Phase 1 EDA) |
| src/twin/patient_state.py | DigitalTwinState four-partition schema |
| src/twin/twin_engine.py | Per-timestep state update orchestrator |
| src/models/prediction_interface.py | Model-agnostic prediction I/O contract |
| src/models/base_model.py | Abstract model interface |
| src/simulation/replay_engine.py | Simulated real-time data stream |
| src/simulation/scenario_simulator.py | Hypothetical trajectory projection |
| src/explainability/explainer.py | Generic XAI interface (method TBD Phase 8) |
| dashboard/app.py | Streamlit entry point -- visualisation only |

---

## What-If Simulation Boundary

```
"Hypothetical trajectory simulation"
        |
What it IS:                     What it is NOT:
- Comparative trajectory view   - Treatment recommendation
- Sensitivity analysis          - Clinical prescription
- Research exploration          - Causal intervention model
- Decision-support research     - Validated clinical tool
```

---

## Ownership Summary

| Component | Owner |
|---|---|
| Architecture, infrastructure, dashboard | Antigravity |
| Data ingestion, preprocessing | Antigravity |
| Digital Twin State and Engine | Antigravity |
| PredictionInterface (contract only) | Antigravity |
| Prediction target definition | Student team |
| Label derivation methodology | Student team |
| Model architecture decisions | Student team |
| Model training and evaluation | Student team |
| XAI method selection | Student team |
| Uncertainty method selection | Student team |
| Scenario design (what-if) | Student team |
| Research conclusions | Student team |
