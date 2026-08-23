# ARCHITECTURE SPECIFICATION
## AI-Driven Digital Twin for Smart Healthcare (Pulmonary & ICU Focus)

This document specifies the end-to-end system architecture, component boundaries, data flow pipelines, state partitions, and API contracts for the React + FastAPI + PyTorch Digital Twin platform.

---

## 1. System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                            REACT + TYPESCRIPT FRONTEND                            |
|  - ICU Ward Grid          - Patient Twin View        - CXR Viewer & Grad-CAM     |
|  - What-If Simulator      - Vital Trajectory Graphs  - Early Warning Alerts      |
+-----------------------------------------------------------------------------------+
                                         │  ▲
                     HTTP REST / JSON   │  │   WebSockets (Live Telemetry)
                                         ▼  │
+-----------------------------------------------------------------------------------+
|                                FASTAPI BACKEND                                    |
|  /api/patients   | /api/monitoring | /api/cxr | /api/risk | /api/digital-twin    |
+-----------------------------------------------------------------------------------+
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
+──────────────────+            +──────────────────+            +──────────────────+
|  REPLAY SERVICE  |            |   CXR SERVICE    |            | PREDICTION & XAI |
|  - Replay Engine |            | - DenseNet121    |            | - CNN-BiLSTM     |
|  - Synthetic Gen |            | - MIMIC-CXR Pre  |            | - NEWS 2 Scorer  |
|  - Shock Inject  |            | - Grad-CAM XAI   |            | - Int. Gradients |
+──────────────────+            +──────────────────+            +──────────────────+
        │                                │                                │
        └────────────────────────────────┬────────────────────────────────┘
                                         ▼
+-----------------------------------------------------------------------------------+
|                              DIGITAL TWIN CORE                                    |
|  - DigitalTwinEngine & DigitalTwinState (Observed | Derived | Predicted | Sim)    |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                                 DATA LAYER                                        |
|  - MIMIC-IV Clinical (Vitals/Labs)  - MIMIC-CXR Images  - Synthetic Generator     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Directory Layout & Module Responsibilities

```
healthcare-digital-twin/
├── PROJECT_CONTEXT.md              # Global project context & principles
├── ARCHITECTURE.md                 # System architecture (this document)
├── DEVELOPMENT_ROADMAP.md          # Multi-phase roadmap and milestones
├── requirements.txt                # Python backend dependencies
├── configs/
│   └── settings.yaml               # Global configuration parameters
├── backend/
│   ├── main.py                     # FastAPI entrypoint, middleware & routing
│   ├── api/                        # REST & WebSocket API Routers
│   │   ├── __init__.py
│   │   ├── patients.py             # Patient CRUD, demographics, timeline, history
│   │   ├── monitoring.py           # Live vitals streaming, sliding window, WS
│   │   ├── cxr.py                  # Chest X-ray metadata, prediction, Grad-CAM
│   │   ├── risk.py                 # Multi-horizon risk scores & trajectory forecasts
│   │   ├── digital_twin.py         # Full unified DigitalTwinState access
│   │   └── simulation.py           # Replay controls & anomaly shock injections
│   ├── services/                   # Business Logic & Model Inference Services
│   │   ├── __init__.py
│   │   ├── cxr_service.py          # Vision pipeline & DenseNet121 inference
│   │   ├── replay_service.py       # Asynchronous stream orchestrator & buffer
│   │   ├── prediction_service.py   # CNN-BiLSTM, NEWS 2 & Integrated Gradients
│   │   └── digital_twin_service.py # State aggregation & management
│   └── schemas/                    # Pydantic v2 Contract Schemas
│       ├── __init__.py
│       ├── patient.py              # Patient metadata & history models
│       ├── monitoring.py           # Vitals, telemetry & simulation schemas
│       ├── cxr.py                  # CXR findings, study & inference schemas
│       └── prediction.py           # Risk scores, forecasts & XAI schemas
├── src/                            # Core ML, Feature & Twin Libraries
│   ├── ingestion/                  # MIMIC-IV / CXR loaders & synthetic generator
│   ├── preprocessing/              # Cleaning, normalizers, temporal alignment
│   ├── features/                   # Feature engineering & baseline calculators
│   ├── twin/                       # DigitalTwinState & DigitalTwinEngine
│   ├── models/                     # PyTorch architectures (CNN-BiLSTM, DenseNet)
│   └── explainability/             # Integrated Gradients & Grad-CAM algorithms
└── data/                           # Data storage (raw, processed, synthetic)
```

---

## 3. Digital Twin State Architecture & Causality

The Digital Twin State maintains a strictly partitioned representation for each patient:

1. **ObservedState**: Raw measurements received up to the current timestamp (vitals, labs, medications, CXR image references).
2. **DerivedState**: Causally computed metrics (Exponential Moving Average baselines, vital delta rates, Shock Index, Mean Arterial Pressure, NEWS 2 score). **Rule**: Value at time $t$ depends strictly on observations at time $\le t$.
3. **PredictedState**: Multi-horizon deterioration probabilities ($t+1\text{h}, t+3\text{h}, t+6\text{h}$), 15-minute vital trajectory predictions, and feature attributions.
4. **SimulationState**: Hypothetical counterfactual projections resulting from user-adjusted what-if interventions.

---

## 4. API Endpoints Contract

### Patient Management
- `GET /api/patients`: List all virtual ICU patients with summary metrics.
- `GET /api/patients/{patient_id}`: Retrieve detailed patient demographic profile.
- `GET /api/patients/{patient_id}/history`: Fetch historical vital trends and timeline.
- `GET /api/patients/{patient_id}/events`: List clinical milestone events.
- `GET /api/patients/{patient_id}/medications`: List active medications and dosages.

### Real-Time Monitoring & Streaming
- `GET /api/patients/{patient_id}/vitals`: Current latest vital signs and sliding window buffer.
- `WS /api/stream/{patient_id}`: Continuous low-latency WebSocket vital telemetry.

### Chest X-Ray & Pulmonary Vision
- `GET /api/patients/{patient_id}/cxr`: Available CXR studies for patient.
- `POST /api/cxr/predict`: Run DenseNet-121 inference on uploaded/selected CXR image.
- `GET /api/cxr/{study_id}/heatmap`: Retrieve Grad-CAM visual explainability heatmap.

### AI Risk & Digital Twin
- `GET /api/patients/{patient_id}/risk`: Multi-horizon deterioration risk scores and NEWS 2 tier.
- `GET /api/patients/{patient_id}/digital-twin`: Full unified 4-partition Digital Twin state.

### Simulation & What-If Controls
- `POST /api/simulation/start`: Resume streaming replay.
- `POST /api/simulation/pause`: Pause streaming replay.
- `POST /api/simulation/reset`: Reset sliding window buffers to initial state.
- `POST /api/simulation/anomaly`: Inject physiological shock (e.g. Septic Shock, ARDS).
- `POST /api/simulation/what-if`: Compute real-time counterfactual twin recalibration.
