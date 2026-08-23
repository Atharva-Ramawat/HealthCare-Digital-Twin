# DEVELOPMENT ROADMAP
## AI-Driven Digital Twin for Smart Healthcare

This roadmap details the progression from initial data replay foundations to full multimodal deep learning and React frontend integration.

---

## Phase Overview

| Phase | Milestone Description | Target Timeline | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Mathematical Synthetic Generator, Asynchronous Replay Engine, Base Schemas | Sprint 1 | ✅ Completed |
| **Phase 2** | FastAPI Backend API Contracts, REST & WebSockets, React UI Integration Layer | Sprint 2 | ✅ Completed |
| **Phase 3** | MIMIC-CXR Pulmonary Vision Model, GPU Training Pipeline, Grad-CAM XAI, Evaluation | Sprint 3 | ✅ Completed |
| **Phase 4** | MIMIC-IV Clinical Feature Engineering & Temporal CNN-BiLSTM Multi-Horizon Training | Sprint 4 | ⏳ Planned (Next) |
| **Phase 5** | Multimodal Fusion (Vitals + CXR + Labs) & Explainability Engine (Integrated Gradients) | Sprint 5 | ⏳ Planned |
| **Phase 6** | React + TypeScript Frontend Integration with Live Telemetry & CXR Diagnostics | Sprint 6 | ⏳ Planned |
| **Phase 7** | What-If Simulation Center, Counterfactual Trajectory Evaluation & Final Benchmarking | Sprint 7 | ⏳ Planned |

---

## Detailed Breakdown: Current & Upcoming Phases

### Phase 1: Foundations & Real-Time Replay Engine (Completed ✅)
- Pure software fallback synthetic data generator (circadian rhythm, sine waves, noise).
- Asynchronous sliding window buffer replay engine.
- Physiological anomaly injection (Septic Shock, ARDS, Cardiac Arrhythmia).
- Initial Digital Twin state representation (`Observed`, `Derived`, `Predicted`, `Simulation`).

### Phase 2: MIMIC-CXR Pipeline & FastAPI API Contracts (Current 🚀)
- **CXR Data Pipeline**: Ingestion and preprocessing transforms for MIMIC-CXR / CheXpert images.
- **Pulmonary Vision Model**: PyTorch DenseNet-121 / ResNet multi-label classifier for 8 common pulmonary pathologies (Pneumonia, Pneumothorax, Pleural Effusion, Atelectasis, Consolidation, Edema, Cardiomegaly, Normal).
- **CXR Explainability (XAI)**: Grad-CAM spatial activation mapping highlighting thoracic lesions.
- **Backend Architecture**: FastAPI application structure with Pydantic v2 schemas and modular service layers (`cxr_service`, `replay_service`, `prediction_service`, `digital_twin_service`).
- **REST & WebSocket Contracts**: Clean endpoints for React UI consumption with strict separation of simulation, historical replay, and model inference modes.

### Phase 3: MIMIC-IV Clinical Feature Engineering & Preprocessing (Next ⏳)
- Cohort extraction from MIMIC-IV (ICU admissions with pulmonary and sepsis conditions).
- Time-series cleaning, outlier detection, and physiological clipping.
- Dynamic causal feature extraction: Exponential Moving Average (EMA) baselines, vital delta rates, Shock Index (HR / SBP), Mean Arterial Pressure (MAP).
- Sequence batch generation for sliding window neural models.

### Phase 4: Temporal Deep Learning Training (Planned ⏳)
- 1D-CNN + Bidirectional LSTM implementation in PyTorch.
- Multi-horizon risk scoring (1-hour, 3-hour, 6-hour deterioration probabilities).
- Short-term 15-minute vital trajectory forecasting head.
- Training and evaluation on campus NVIDIA GPUs using BCE and MSE multi-task loss.

### Phase 5: Multimodal Fusion & Explainability (Planned ⏳)
- Late fusion architecture combining CXR image embeddings with temporal vital embeddings.
- Integrated Gradients for time-series attribution (vital parameter risk escalation breakdown).
- Uncertainty estimation using Monte Carlo Dropout / ensemble variance.

### Phase 6: React + TypeScript UI Integration (Planned ⏳)
- Connect React dashboard to FastAPI endpoints and WebSocket telemetry.
- Integrate interactive DICOM / CXR image viewer with Grad-CAM heatmap overlays.
- Real-time Plotly / WebGL charts for multi-horizon deterioration risk and forecasted vital paths.

### Phase 7: Verification & Final Major Project Deliverables (Planned ⏳)
- End-to-end latency testing and benchmark evaluation.
- What-If counterfactual scenario verification.
- Project report, architecture documentation, and academic presentation assets.
