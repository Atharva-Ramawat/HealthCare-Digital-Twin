# AI-Driven Predictive Patient Digital Twin for Healthcare

> **B.Tech Major Project** — Research-grade ICU patient Digital Twin with multi-horizon physiological deterioration prediction, real-time data simulation, explainability, and what-if counterfactual analysis.

---

## ⚠️ Important Notices

- **NO live clinical data is connected.** The input layer is a *simulated real-time clinical data replay* built from historical/synthetic datasets. This system does NOT represent live hospital monitoring.
- **This system provides NO medical advice and NO treatment recommendations.** It is a research/educational tool.
- **Current Status: Phase 0 — Architecture Scaffold.** No production models are trained or deployed.

---

## Project Goal

Build a research-oriented Digital Twin of an ICU patient that:

1. Reconstructs the patient's physiological state from historical clinical data
2. Continuously updates a computational representation of the patient
3. Replays historical clinical observations as a simulated real-time stream
4. Learns temporal physiological trajectories and patient-specific baselines
5. Predicts deterioration risk over 1h / 3h / 6h horizons
6. Predicts future physiological states
7. Provides prediction explanations (SHAP)
8. Quantifies prediction uncertainty (MC Dropout)
9. Supports controlled what-if / counterfactual scenario simulation
10. Visualises the complete Digital Twin through an interactive Streamlit dashboard

---

## Repository Structure

```
healthcare-digital-twin/
├── data/
│   ├── raw/
│   │   ├── mimic_iv/         # MIMIC-IV tables (not committed)
│   │   ├── mimic_waveform/   # MIMIC-III waveforms (not committed)
│   │   ├── vitaldb/          # VitalDB cases (not committed)
│   │   └── eicu/             # eICU tables (not committed)
│   ├── interim/              # Partially processed (not committed)
│   ├── processed/            # Analysis-ready (not committed)
│   ├── features/             # Feature matrices (not committed)
│   └── synthetic/            # Synthetic data (safe to commit)
│
├── notebooks/                # Exploratory notebooks (not production code)
│
├── src/
│   ├── ingestion/            # Data loading (MIMIC-IV, VitalDB, synthetic)
│   ├── preprocessing/        # Cleaning, alignment, normalisation
│   ├── features/             # Feature engineering, baseline calculation
│   ├── simulation/           # Clinical data replay & what-if simulation
│   ├── twin/                 # Digital Twin State & Engine
│   ├── models/               # ML/DL models (baselines → CNN-BiLSTM)
│   ├── explainability/       # SHAP / Integrated Gradients
│   └── evaluation/           # Metrics and model comparison
│
├── dashboard/
│   ├── app.py                # Streamlit entry point
│   ├── components/           # Reusable UI components
│   └── pages/                # Multi-page navigation
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── configs/
│   └── settings.yaml         # Central configuration (all modules read from here)
│
├── models/
│   ├── checkpoints/          # Trained model weights (not committed)
│   └── registry/             # Model metadata and versioning
│
├── docs/
│   ├── PROJECT_CONSTITUTION.md
│   ├── architecture/         # Architecture diagrams
│   ├── data_schema/          # Data schema definitions
│   ├── api/                  # Module API documentation
│   └── legacy_prototype/     # Original flat-file prototype (reference only)
│
├── scripts/
│   ├── setup_project.py
│   ├── generate_synthetic_data.py
│   └── train_model.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick Start (Phase 0 — Scaffold Only)

```bash
# 1. Clone and enter project
git clone <repo-url>
cd healthcare-digital-twin

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up project directories
python scripts/setup_project.py

# 5. Copy and fill environment variables
cp .env.example .env

# 6. Launch placeholder dashboard
streamlit run dashboard/app.py
```

---

## Data Sources

| Source | Type | Use | Status |
|---|---|---|---|
| MIMIC-IV | Structured clinical (ICU) | Primary training data | Phase 1 |
| MIMIC-III Waveform | High-frequency physiological | Waveform features | Phase 2 |
| VitalDB | High-frequency OR/ICU | Additional physiological data | Phase 2 |
| eICU | Multi-centre ICU | External validation | Phase 4 |
| Synthetic Generator | Algorithmic | Development / demo fallback | Phase 0/1 |

---

## Development Roadmap

| Phase | Focus | Status |
|---|---|---|
| **Phase 0** | Architecture, scaffold, constitution | ✅ Done |
| **Phase 1** | MIMIC-IV ingestion, preprocessing, feature engineering, Digital Twin State, Replay Engine, Synthetic rewrite, minimal dashboard | 🔜 Next |
| **Phase 2** | Baseline models, training pipeline, evaluation framework | 🔴 Pending |
| **Phase 3** | CNN-BiLSTM main model, SHAP explainability, What-if simulation, uncertainty quantification | 🔴 Pending |
| **Phase 4** | Multi-horizon prediction, eICU validation, full dashboard integration, documentation | 🔴 Pending |

---

## Architecture Overview

See [docs/PROJECT_CONSTITUTION.md](docs/PROJECT_CONSTITUTION.md) for the full architecture, data flow, module responsibilities, and Digital Twin state schema.

---

## Ethical Statement

This system is built for educational and research purposes only.
It processes de-identified publicly available clinical data (MIMIC-IV, eICU) under their respective data use agreements.
It does not provide clinical diagnosis, prognosis, or treatment recommendations.
All risk scores and predictions are research outputs only.

---

*Last updated: Phase 0 — August 2026*
