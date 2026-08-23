# AI-Driven Digital Twin for Smart Healthcare — Project Context

## 1. Project Overview & Goal
This project is an academic research prototype building a 100% software-based **"AI-Driven Digital Twin for Smart Healthcare using Deep Learning and Real-Time Clinical Data Simulation"** for a B.Tech Major Project.

The core objective is to create a computational Digital Twin of an ICU/hospitalized patient that continuously aggregates historical clinical observations, simulates real-time patient progression via an asynchronous replay engine, and uses deep learning (temporal CNN-BiLSTM, computer vision for Chest X-Rays, and multi-modal fusion) to predict physiological deterioration, forecast vital trajectories, and provide explainable clinical risk scores (NEWS 2 & multi-horizon risk).

---

## 2. Key Operational Constraints & Principles
- **Strictly Software-Only**: ZERO IoT or physical patient monitoring hardware dependencies.
- **Data Streaming Replay**: Real-time patient monitoring is emulated using an asynchronous replay engine over historical datasets (MIMIC-IV, MIMIC-CXR, PhysioNet, VitalDB) supported by a mathematical synthetic data fallback generator.
- **Primary Disease Domain**: Pulmonary / Lung Disease (Pneumonia, ARDS, Pleural Effusion, Pneumothorax, Atelectasis, Respiratory Failure).
- **Compute Environment**: Local development + NVIDIA campus GPUs for offline model training and inference acceleration.
- **Academic Research Prototype**: The system is NOT a certified medical device and does NOT provide live medical diagnosis or treatment prescriptions.
- **No Fabricated Medical Predictions**: Clearly distinguish:
  1. *Real dataset historical playback*
  2. *Synthetic simulation fallback*
  3. *Trained deep learning model inference*
  4. *Model-unavailable / placeholder state* (Explicitly returned with boolean flags when weights are not loaded).

---

## 3. Technology Stack & Component Boundaries

### Frontend
- **Framework**: React + TypeScript (Vite / Next.js).
- **Design Baseline**: Existing teammate UI skeleton featuring:
  - ICU Ward View (Patient Grid & Real-Time Status Cards)
  - Patient Digital Twin Dashboard (Vitals, Multi-Horizon Risk, Trajectory Graphs)
  - Chest X-Ray (CXR) Diagnostic & Heatmap Viewer (Grad-CAM)
  - "What-If" Counterfactual Simulation Center
  - Clinical Early Warning Alerts Feed & Patient History Timeline
- **Rule**: Do NOT replace the React frontend with Streamlit. Maintain clean API boundaries.

### Backend
- **Framework**: Python 3.10+ with FastAPI.
- **Communication**: REST APIs for CRUD/predictions + WebSockets (`/api/stream/{patient_id}`) for low-latency live telemetry streams.
- **Data Validation**: Strict Pydantic v2 schemas for all request/response contracts.
- **Data Serialization**: Clean JSON responses only. Never expose raw PyTorch tensors or internal Python exceptions to the frontend.

### Machine Learning & Digital Twin Services
- **Framework**: PyTorch.
- **Deep Learning Architectures**:
  - **Temporal Sequence Model**: 1D-CNN + Bidirectional LSTM (CNN-BiLSTM) for 24-step sliding window vitals, predicting multi-horizon deterioration risk (1h, 3h, 6h) and 15-minute vital trajectory forecasting.
  - **Pulmonary CXR Vision Model**: DenseNet-121 / ResNet backbone for multi-label thoracic pathology detection (8+ findings) from MIMIC-CXR.
  - **Explainable AI (XAI)**: Integrated Gradients for vital time-series attribution and Grad-CAM for CXR spatial heatmap localization.
- **Digital Twin State Architecture**: Partitioned into `ObservedState`, `DerivedState`, `PredictedState`, and `SimulationState` with strict temporal causality (no future information leakage).

---

## 4. Operating Modes
The system supports three transparent operating modes selectable via API and displayed in the React UI:
1. **Demo / Synthetic Simulation Mode**: Mathematical sine wave, circadian rhythm, baseline drift, and injected anomaly scenarios (Septic Shock, ARDS, Cardiac Arrhythmia).
2. **Historical MIMIC Replay Mode**: Step-by-step playback of real de-identified ICU admissions from MIMIC-IV / PhysioNet.
3. **ML Inference Mode**: Active PyTorch deep learning models generating live predictions and Grad-CAM / Integrated Gradients attributions.

---

## 5. Development Phases Summary
- **Phase 1**: Core Architecture, Synthetic Generator, Asynchronous Replay Engine, and State Schemas. ✅
- **Phase 2 (Current)**: MIMIC-CXR Pulmonary Image Pipeline, Vision Model Architecture, Grad-CAM XAI, and FastAPI Backend API Contracts for React Integration. 🚀
- **Phase 3**: MIMIC-IV Clinical Data Preprocessing, Lab & Medication Feature Engineering, and Multi-Horizon Temporal Dataset Alignment.
- **Phase 4**: Temporal CNN-BiLSTM Training, Multi-Modal Fusion (Vitals + CXR + Labs), and Validation.
- **Phase 5**: Full Integration of React UI with FastAPI WebSockets, "What-If" Counterfactual Simulation, and System Verification.
