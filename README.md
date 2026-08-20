# AI-Driven Predictive Patient Digital Twin for ICU Healthcare

> **B.Tech Major Project** — Research-grade ICU patient Digital Twin with simulated real-time
> clinical data replay, multi-horizon physiological deterioration prediction, explainability,
> and hypothetical what-if trajectory simulation.

---

## Important Notices

- **No live clinical data is connected.** The input layer is a *simulated real-time clinical
  data replay* built from historical datasets. This does NOT represent live hospital monitoring.
- **No medical advice or treatment recommendations are provided.** This is a research tool only.
- **Synthetic data is for development and testing only.** All ML performance claims must come
  from real clinical research datasets (MIMIC-IV, VitalDB, eICU).
- **Current Status: Phase 0 v1.1 — Architecture Scaffold and Constitution.**
  No models are trained or deployed.
- **ML research is owned by the student team.** See docs/PROJECT_CONSTITUTION.md Section 0.

---

## Project Goal

Build a research-grade Digital Twin of an ICU patient that:

1. Reconstructs the patient's physiological state from historical clinical data
2. Continuously updates a computational representation of the patient
3. Replays historical clinical observations as a simulated real-time stream
4. Tracks patient-specific physiological baselines
5. Computes temporal trends and derived clinical features
6. Predicts deterioration risk over configurable future horizons (horizons TBD in Phase 5)
7. Optionally predicts future physiological states
8. Provides prediction explanations (XAI method TBD in Phase 8)
9. Estimates prediction uncertainty (method TBD in Phase 8)
10. Supports hypothetical what-if trajectory simulation (NOT treatment recommendations)
11. Visualises the complete Digital Twin through an interactive Streamlit dashboard

---

## Architecture Principle

The **Digital Twin is NOT the dashboard.**

```
  Digital Twin Engine
        | updates
  DigitalTwinState (Observed | Derived | Predicted | Simulation)
        | read by
  PredictionInterface (model-agnostic)
        | results written to state
  Dashboard (visualisation only — does NOT drive the Twin)
```

The Digital Twin operates independently of Streamlit.
See docs/PROJECT_CONSTITUTION.md for full architecture.

---

## Repository Structure

```
healthcare-digital-twin/
|-- data/
|   |-- raw/
|   |   |-- mimic_iv/         (NOT committed)
|   |   |-- mimic_waveform/   (NOT committed)
|   |   |-- vitaldb/          (NOT committed)
|   |   +-- eicu/             (NOT committed)
|   |-- interim/              (NOT committed)
|   |-- processed/            (NOT committed)
|   |-- features/             (NOT committed)
|   +-- synthetic/            (safe to commit - dev/test only)
|
|-- notebooks/
|
|-- src/
|   |-- ingestion/            (MIMIC-IV, VitalDB loaders; synthetic generator)
|   |-- preprocessing/        (cleaner, temporal aligner, normalizer)
|   |-- features/             (feature engineering, baseline calculator, sequence builder)
|   |-- simulation/           (replay engine, scenario simulator)
|   |-- twin/                 (DigitalTwinState, DigitalTwinEngine)
|   |-- models/               (PredictionInterface, baselines, CNN-BiLSTM)
|   |-- explainability/       (XAI interface -- method TBD)
|   +-- evaluation/           (metrics, model comparator)
|
|-- dashboard/
|   |-- app.py                (Streamlit entry point -- visualisation only)
|   +-- components/
|
|-- tests/
|   |-- unit/
|   +-- integration/
|
|-- configs/
|   +-- settings.yaml         (central config -- all modules read from here)
|
|-- models/
|   |-- checkpoints/          (NOT committed)
|   +-- registry/             (model metadata -- committed)
|
|-- docs/
|   |-- PROJECT_CONSTITUTION.md
|   |-- ARCHITECTURE.md
|   +-- legacy_prototype/     (original flat-file demo -- reference only)
|
|-- scripts/
|-- .env.example
|-- .gitignore
|-- pyproject.toml
|-- requirements.txt
+-- README.md
```

---

## Quick Start (Phase 0 -- Scaffold Only)

```bash
# 1. Clone and enter project
git clone <repo-url>
cd healthcare-digital-twin

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up project directories
python scripts/setup_project.py

# 5. Copy and fill environment variables
copy .env.example .env

# 6. Launch placeholder dashboard
streamlit run dashboard/app.py
```

---

## Data Sources

| Source | Type | Use | Status |
|---|---|---|---|
| MIMIC-IV / Demo | Structured clinical (ICU) | Primary training data | Phase 1 |
| MIMIC Waveform | High-frequency physiological | Waveform features (optional) | Phase 1+ |
| VitalDB | High-frequency OR/ICU | Additional physiological data | Phase 1 |
| eICU | Multi-centre ICU | External validation | Phase 10 |
| Synthetic Generator | Algorithmic | Development / testing ONLY | Phase 0/1 |

---

## Development Roadmap (Phase-Gated)

Every phase requires explicit student team approval before the next begins.

| Phase | Focus | Status |
|---|---|---|
| 0 | Architecture, scaffold, constitution | Done (v1.1) |
| 1 | Dataset discovery, ingestion, data profiling | Next |
| 2 | Clinical preprocessing and temporal pipeline | Pending |
| 3 | Digital Twin State and Engine | Pending |
| **4** | **Simulated real-time replay + minimal visualisation** | **Pending (30% Milestone)** |
| 5 | Student ML research and model development | Pending (student-owned) |
| 6 | ML integration with Digital Twin | Pending |
| 7 | What-if trajectory simulation | Pending |
| 8 | XAI and uncertainty integration | Pending |
| 9 | Complete dashboard | Pending |
| 10 | External validation | Pending (optional) |
| 11 | Final integration, testing, evaluation | Pending |

---

## Ownership

| Area | Owner |
|---|---|
| Software architecture and infrastructure | Antigravity |
| Data ingestion and preprocessing | Antigravity |
| Digital Twin State and Engine | Antigravity |
| Dashboard | Antigravity |
| Prediction target definition | Student team |
| Label derivation methodology | Student team |
| Model training and evaluation | Student team |
| Research conclusions | Student team |

---

## Ethical Statement

This system is built for educational and research purposes only.
It processes de-identified publicly available clinical data under their respective
data use agreements. It does not provide clinical diagnosis, prognosis, or treatment
recommendations. All risk scores and predictions are research outputs only.

---

*Last updated: Phase 0 v1.1 -- August 2026*
