# Multimodal Patient Digital Twin for Pulmonary ICU Care

An AI-driven clinical decision support system integrating deep radiographic representations and high-frequency physiological trajectories for deterioration risk prediction in intensive care units.

---

## Abstract and Project Aim

Patients admitted to Intensive Care Units (ICUs) with acute respiratory and pulmonary conditions—such as pneumonia, pulmonary edema, atelectasis, and pneumothorax—experience rapid and volatile physiological changes. Effective clinical management requires the continuous interpretation of heterogeneous data streams spanning multiple temporal resolutions:

1. High-dimensional radiographic imaging (Chest X-Rays / CXR) capturing structural changes in pulmonary parenchyma and pleural spaces.
2. High-frequency temporal vital sign trajectories (Heart Rate, Blood Oxygen Saturation, Systolic Blood Pressure, and Respiratory Rate) capturing real-time hemodynamic and respiratory dynamics.

This project implements an end-to-end Multimodal Patient Digital Twin platform designed to continuously reconstruct, monitor, and predict patient physiological states. By fusing deep visual representations extracted from chest radiographs with normalized 24-hour vital sign trajectories, the system produces an objective ICU Deterioration Risk Score (0 to 100%) and stratified risk tiers.

The system is engineered according to the Digital Twin Independence Principle: the core computational state representation, machine learning models, and data ingestion pipelines function independently of any user interface or presentation layer.

---

## System Architecture

The platform follows a layered, decoupled architecture designed for high-throughput, low-latency execution in clinical environments.

```
+-------------------------------------------------------------------------+
|                      LAYER 5: CLINICAL DASHBOARD                        |
|   React 19, TypeScript, Tailwind CSS, Recharts                          |
|   - Multi-Bed ICU Command Center                                        |
|   - Synchronized 24-Hour Telemetry Grid (HR, SpO2, SBP, RR)             |
|   - CXR & Grad-CAM Explainability Image Viewer                          |
|   - Longitudinal Intervention Tracker & Ad-Hoc Sandbox                  |
+-------------------------------------------------------------------------+
                                    | HTTP REST (JSON / Base64 / Binary)
+-------------------------------------------------------------------------+
|                      LAYER 4: ASYNCHRONOUS API GATEWAY                  |
|   FastAPI, Uvicorn, Pydantic v2                                         |
|   - POST /api/digital-twin/ad-hoc-infer (Multimodal Upload)             |
|   - GET  /api/digital-twin/{patient_id}/{study_id} (Twin State)         |
|   - GET  /api/cxr/studies/{study_id}/heatmap (Grad-CAM PNG Stream)      |
|   - POST /api/cxr/studies/{study_id}/infer (Vision Inference)           |
|   - CUDA Cache Management & Non-blocking Thread Offloading              |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                      LAYER 3: MULTIMODAL FUSION ENGINE                  |
|   DigitalTwinFusion Engine                                              |
|   - 1024-dimensional DenseNet-121 Visual Embedding Extraction           |
|   - 4-Parameter Vital Sign Snapshot Normalization                       |
|   - Multimodal Fusion MLP & Heuristic ICU Deterioration Risk Engine     |
|   - Four-Tier Risk Stratification (Low, Moderate, High, Critical)       |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                      LAYER 2: VISION & EXPLAINABILITY                   |
|   DenseNet-121 Pulmonary Classifier (Torchvision)                       |
|   - Multi-Label Classification Head: Linear(1024, 8)                    |
|   - Target-Specific Grad-CAM on denseblock4.denselayer16.conv2          |
|   - Structural Out-of-Distribution (OOD) CXR Modality Gatekeeper        |
|   - PyTorch Automatic Mixed Precision (AMP CUDA / CPU Fallback)         |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                      LAYER 1: CLINICAL DATA INGESTION                   |
|   PhysioNet MIMIC-IV Clinical Database (v2.2) & MIMIC-CXR               |
|   - Relational Tables: patients, admissions, icustays, chartevents      |
|   - SQLite Database & Canonical Clinical Observation Schema             |
|   - 24-Hour Vital Sign Trajectory Simulator (96 Steps @ 15-min Intervals|
+-------------------------------------------------------------------------+
```

### Key Architectural Characteristics

1. **Digital Twin State Representation**: Patient state is modeled as an immutable snapshot containing current vitals, demographic parameters, active radiographic findings, extracted embeddings, and projected deterioration risk.
2. **Explainable AI (XAI)**: Visual explanations are computed using Gradient-weighted Class Activation Mapping (Grad-CAM) directly from the final feature maps of DenseNet-121, providing anatomical localization for each diagnosed pulmonary pathology.
3. **Out-of-Distribution Gatekeeping**: To prevent model hallucinations from non-radiological images, uploaded inputs pass through a structural gatekeeper analyzing color channel variance, intensity histograms, and aspect ratios prior to model inference.
4. **GPU Memory and Concurrency Management**: The backend is hardened against CUDA Out of Memory (OOM) failures through explicit gradient zeroing, elimination of persistent computational graphs (`retain_graph=False`), explicit tensor deletion, and scheduled cache reclamation via `torch.cuda.empty_cache()`.

---

## Target Pulmonary Pathologies

The computer vision subsystem classifies chest radiographs across 8 clinically significant pulmonary conditions:

| Index | Pathology Target | Clinical Relevance in ICU |
|---|---|---|
| 0 | Atelectasis | Partial or complete collapse of lung segments, common in mechanically ventilated patients. |
| 1 | Cardiomegaly | Enlargement of cardiac silhouette, indicating heart failure or fluid overload. |
| 2 | Consolidation | Alveolar air spaces replaced by fluid or exudate, typical of severe pneumonia. |
| 3 | Edema | Pulmonary fluid accumulation impairing gas exchange; common in acute decompensated heart failure. |
| 4 | Pleural Effusion | Pathological fluid accumulation in the pleural cavity compressing lung parenchyma. |
| 5 | Pneumonia | Infectious inflammation of lung parenchyma requiring targeted antimicrobial intervention. |
| 6 | Pneumothorax | Presence of air in pleural cavity requiring urgent decompression. |
| 7 | No Finding | Absence of acute radiological abnormalities across monitored categories. |

---

## Technology Stack

- **Machine Learning and Vision**:
  - Python 3.11
  - PyTorch 2.x
  - Torchvision (DenseNet-121)
  - NumPy, SciPy, Pillow, Matplotlib
- **Backend Infrastructure**:
  - FastAPI (Asynchronous ASGI Web Framework)
  - Uvicorn (High-performance ASGI Server)
  - Pydantic v2 (Strict Data Validation & Canonical Contracts)
  - SQLAlchemy & SQLite (Relational Storage)
- **Frontend Dashboard**:
  - React 19 (Component-Driven Architecture)
  - TypeScript 5 (Static Type Safety)
  - Vite (Build Tool & HMR Server)
  - Tailwind CSS (Utility-First Styling)
  - Recharts (Time-Series Physiological Telemetry Visualizations)
  - Lucide React (Clinical Iconography)
  - Axios (HTTP Client with Binary Blob Handling)
- **Testing & Quality Assurance**:
  - Pytest (155 automated unit and integration tests)
  - AnyIO (Asynchronous Test Harness)

---

## Repository Structure

```
healthcare-digital-twin/
|-- backend/                       # Legacy backend services and auxiliary routers
|   |-- api/                       # API endpoints (patients, monitoring, cxr, risk)
|   |-- services/                  # Service implementations (replay, prediction, twin)
|   +-- main.py                    # Multi-service FastAPI application
|
|-- src/                           # Core Digital Twin system
|   |-- api/                       # Primary FastAPI application
|   |   |-- main.py                # Lifespan manager, CORS, and endpoint definitions
|   |   +-- routers/               # Specialized routers (twin.py, etc.)
|   |-- database/                  # Database connections and SQLAlchemy models
|   |   |-- connection.py          # Session factory and database initialization
|   |   +-- models.py              # Patient and CXRStudy relational tables
|   |-- digital_twin/              # Digital Twin core engine
|   |   |-- fusion.py              # Multimodal fusion model & risk scoring logic
|   |   +-- simulator.py           # 24-hour vital sign trajectory generator
|   |-- ml/                        # Machine learning implementations
|   |   +-- cxr/                   # Vision models, Grad-CAM, and validators
|   |       |-- model.py           # DenseNet-121 pulmonary classifier
|   |       +-- validator.py       # Structural OOD gatekeeper for CXR validation
|   +-- schemas/                   # Pydantic v2 data transfer schemas
|       |-- cxr_schema.py          # Vision inference schemas
|       +-- twin_schema.py         # Digital Twin state and response contracts
|
|-- frontend/                      # React 19 + TypeScript clinical dashboard
|   |-- src/
|   |   |-- components/            # UI components (CommandCenter, CXRFusionViewer, etc.)
|   |   |-- services/              # API clients and HTTP abstraction layers
|   |   |-- types/                 # TypeScript type declarations
|   |   |-- App.tsx                # View routing and top-level state
|   |   +-- main.tsx               # Application entrypoint
|   |-- package.json               # Node dependencies and build scripts
|   +-- vite.config.ts             # Vite configuration and backend reverse proxy
|
|-- docs/                          # Technical documentation
|   |-- SYSTEM_DOCUMENTATION.md    # Complete system technical specifications
|   +-- legacy_prototype/          # Reference baseline implementation
|
|-- tests/                         # Automated test suite
|   |-- integration/               # API endpoint and end-to-end integration tests
|   +-- unit/                      # Model, fusion, and schema unit tests
|
|-- pyproject.toml                 # Pytest configuration and project metadata
|-- requirements.txt               # Python package dependencies
+-- README.md                      # Project documentation and setup guide
```

---

## Quick Start Guide

### Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher with npm
- Git

### 1. Environment Setup

Clone the repository and initialize the Python virtual environment:

```bash
git clone https://github.com/example/healthcare-digital-twin.git
cd healthcare-digital-twin

python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

### 2. Backend Execution

Launch the FastAPI application server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API service initializes database connections and preloads the DenseNet-121 vision weights during startup. Verify the backend status:

- Swagger UI Documentation: `http://localhost:8000/docs`
- Service Health Endpoint: `http://localhost:8000/api/health`

### 3. Frontend Execution

In a separate terminal, enter the frontend directory and install the Node.js packages:

```bash
cd frontend
npm install
```

Start the Vite development server:

```bash
npm run dev
```

The frontend dashboard will be available at `http://localhost:3000`. The Vite server automatically proxies requests targeting `/api` to the backend running at `http://localhost:8000`.

### 4. Running Automated Tests

Run the complete backend test suite:

```bash
pytest
```

Run specific test modules:

```bash
# Vision validator and out-of-distribution unit tests
pytest tests/unit/test_cxr_validator.py

# Multimodal fusion and trajectory simulation unit tests
pytest tests/unit/test_digital_twin_fusion.py

# API integration tests
pytest tests/integration/test_fastapi_backend.py
```

Verify frontend compilation and type safety:

```bash
cd frontend
npm run build
```

---

## Technical Specifications and Documentation

For detailed architectural diagrams, mathematical formulations of the fusion models, Pydantic data contracts, and DGX cluster deployment instructions, refer to:

- [System Documentation](docs/SYSTEM_DOCUMENTATION.md)

---

## Research and Ethical Disclaimers

1. **Research and Educational Prototype**: This system is developed strictly for academic research and educational evaluation as part of a final-year B.Tech engineering project. It is not an FDA-approved or CE-marked medical device.
2. **No Clinical Advice**: The predictions, risk percentages, and saliency maps produced by this platform are algorithmic estimates intended for retrospective analysis and computational research. They must not be used for diagnosis, clinical triage, or patient treatment decisions.
3. **Data Compliance**: Research using MIMIC-IV and MIMIC-CXR datasets complies with PhysioNet Credentialed Health Data Use Agreements. All patient identifiers in the source datasets have been de-identified in accordance with HIPAA safe harbor regulations.
