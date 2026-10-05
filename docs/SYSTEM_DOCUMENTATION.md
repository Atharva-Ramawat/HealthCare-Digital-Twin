# System Documentation

## AI-Driven Multimodal Patient Digital Twin for Pulmonary ICU Care

### 1. System Overview

This project implements an AI-driven multimodal Digital Twin designed for intensive care unit (ICU) clinical decision support, with a specific focus on acute pulmonary critical care and physiological deterioration prediction. 

The system continuously reconstructs, monitors, and predicts patient physiological states by integrating two complementary data modalities:
1. High-dimensional radiographic imaging (Chest X-Rays / CXR) capturing structural pulmonary pathology.
2. High-frequency temporal vital sign trajectories capturing real-time hemodynamic and respiratory dynamics.

The central architecture adheres to the Digital Twin Independence Principle: the computational representation of the patient, the data ingestion pipeline, and the predictive machine learning models operate independently of any user interface or visualization layer.

---

### 2. System Architecture and Component Layers

The platform is structured into five distinct operational layers:

```
+-------------------------------------------------------------------------+
|                      LAYER 5: CLINICAL DASHBOARD                        |
|   React 19 + TypeScript + Tailwind CSS + Recharts                       |
|   [Command Center]   [Live Ward]   [Treatment Tracker]   [Ad-Hoc Sandbx]|
+-------------------------------------------------------------------------+
                                    | HTTP / REST
+-------------------------------------------------------------------------+
|                      LAYER 4: API & SERVING ENGINE                      |
|   FastAPI Asynchronous Gateway                                          |
|   - POST /api/digital-twin/ad-hoc-infer                                 |
|   - GET  /api/digital-twin/{patient_id}/{study_id}                      |
|   - GET  /api/cxr/studies/{study_id}/heatmap                            |
|   - POST /api/cxr/studies/{study_id}/infer                              |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                      LAYER 3: MULTIMODAL FUSION                         |
|   DigitalTwinFusion Engine                                              |
|   - 1024-dim DenseNet-121 Visual Embedding                              |
|   - Normalized Vital Sign Snapshot (HR, SpO2, SBP, RR)                  |
|   - Fusion MLP & Heuristic ICU Deterioration Risk Score (0-100%)        |
|   - Risk Tier Classification (Low, Moderate, High, Critical)            |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                      LAYER 2: ML & COMPUTER VISION                      |
|   DenseNet-121 Pulmonary Classifier (ImageNet Pretrained)               |
|   - Head: Linear(1024, 8) for Target Pulmonary Pathologies             |
|   - Grad-CAM Attribution on features.denseblock4.denselayer16.conv2    |
|   - Out-of-Distribution (OOD) Modality Gatekeeper Validator             |
|   - PyTorch Mixed Precision (AMP CUDA)                                  |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                      LAYER 1: DATA ACQUISITION                          |
|   PhysioNet MIMIC-IV Clinical Database (v2.2) & MIMIC-CXR               |
|   - 28 Relational ICU & Hospital Tables (chartevents, labevents, etc.)  |
|   - SQLite Database & Canonical Clinical Observation Schema             |
|   - 24-Hour Vitals Simulator (96 Timesteps @ 15-min Intervals)          |
+-------------------------------------------------------------------------+
```

#### 2.1 Layer 1: Data Acquisition and Canonical Schema
- Ingestion supports full MIMIC-IV and MIMIC-IV Demo datasets across core relational modules (`hosp/` and `icu/`).
- Standardized observation schema preserves raw identifiers (`subject_id`, `hadm_id`, `stay_id`, `itemid`, `charttime`, `valuenum`, `valueuom`).
- Temporal sampling analysis identified an empirical median charting interval of 60 minutes for core vitals in MIMIC-IV chartevents, which serves as the grid resampling baseline.
- For bedside simulation, a synthetic 24-hour physiological trajectory generator models 96 sequential 15-minute observations under configurable profiles (stable, gradual deterioration, acute collapse, septic shock).

#### 2.2 Layer 2: Vision Modeling and OOD Validation
- Backbone: DenseNet-121 initialized with ImageNet weights.
- Classifier Head: Modified from 1000 ImageNet logits to `Linear(1024, 8)` to predict 8 target pulmonary pathologies:
  - Pneumonia
  - Pleural Effusion
  - Consolidation
  - Edema
  - Atelectasis
  - Cardiomegaly
  - Pneumothorax
  - No Finding
- Feature Extraction: Global Average Pooling outputs a 1024-dimensional dense representation capturing morphological findings.
- Explainability: Grad-CAM forward and backward hooks attached to the final convolutional layer (`features.denseblock4.denselayer16.conv2`). Alpha weights are derived via global average pooling of gradients, followed by ReLU filtering, min-max normalization, and bilinear interpolation to 224x224.
- Out-of-Distribution (OOD) Gatekeeper: Prior to running inference, uploaded images are checked by a lightweight structural validator (`src/ml/cxr/validator.py`). Images violating monochromatic color saturation ($\text{chroma} > 15.0$), contrast bounds ($\text{std} < 15.0$), white background dominance ($> 70\%$), or abnormal edge density are rejected with an HTTP 400 error to prevent hallucinated predictions on non-medical inputs.

#### 2.3 Layer 3: Multimodal Fusion Engine
- Combines the 1024-dimensional visual embedding from DenseNet-121 with normalized bedside vitals ($\text{Heart Rate}, \text{SpO}_2, \text{Systolic BP}, \text{Respiratory Rate}$).
- Computes a unified ICU Deterioration Risk Score scaled from 0% to 100%.
- Calculates relative contribution percentages: visual findings contribution vs. vital signs contribution.
- Assigns deterministic risk tiers: Low ($< 30\%$), Moderate ($30\% - 55\%$), High ($55\% - 75\%$), and Critical ($> 75\%$).
- Generates clinical recommendations and identifies driving risk factors (e.g., severe hypoxemia, dense lobar consolidations, hemodynamic collapse).

#### 2.4 Layer 4: FastAPI Backend Gateway
- Implemented in `src/api/main.py` and modular routers in `src/api/routers/twin.py`.
- Threading: Runs PyTorch forward passes and Grad-CAM generation asynchronously in background thread pools (`asyncio.to_thread`) to prevent blocking the event loop.
- Memory Hardening: Grad-CAM backward pass explicitly avoids graph retention (`retain_graph=False`), clears gradients with `zero_grad()`, and triggers `torch.cuda.empty_cache()` in endpoint `finally:` blocks to prevent Out-Of-Memory (OOM) failures on 6GB VRAM GPUs.
- Ad-Hoc Inference: Supports `POST /api/digital-twin/ad-hoc-infer` accepting multipart form data (image file + bedside vitals), returning real-time predictions, Grad-CAM overlays encoded as base64 PNG data URLs, and synthesized 24-hour trajectories.

#### 2.5 Layer 5: High-Density Clinical Frontend
- Built with React 19, TypeScript, Vite, Tailwind CSS, and Recharts.
- Contains four isolated views:
  - Dashboard (ICU Command Center): Multi-bed cohort overview with 6 active beds, micro-sparklines of vital trends, and a real-time Critical Alerts Feed.
  - Live Ward: Dedicated view for verified MIMIC database patients with 96-timestep synchronized telemetry charts, CXR viewer, Grad-CAM overlay toggle, and longitudinal history logs.
  - Treatment Analysis: Longitudinal side-by-side comparison tool (Scan A: Pre-Intervention vs. Scan B: Post-Intervention) with automated Delta Metrics ($\Delta\text{Risk Score}$, $\Delta\text{Pathologies}$, $\Delta\text{Vitals}$).
  - Ad-Hoc Sandbox: Isolated simulation environment with prominent yellow warning banners to test manual uploads without corrupting database records.

---

### 3. Canonical Data Schema and Database Contracts

#### 3.1 Observation Table Contract (MIMIC-IV chartevents)

| Column | Type | Description |
|---|---|---|
| `subject_id` | INTEGER | Unique patient identifier |
| `hadm_id` | INTEGER | Hospital admission identifier |
| `stay_id` | INTEGER | ICU stay identifier |
| `itemid` | INTEGER | MIMIC-IV item code (linked to d_items) |
| `charttime` | TEXT | Timestamp of observation (ISO 8601 string) |
| `valuenum` | REAL | Numerical measurement value |
| `valueuom` | TEXT | Unit of measurement |

#### 3.2 Canonical Item Mappings

| Variable Name | Item ID | Canonical Unit | Typical Normal Range |
|---|---|---|---|
| Heart Rate | 220045 | bpm | 60 - 100 |
| SpO2 | 220277 | % | 95 - 100 |
| Respiratory Rate | 220210 | breaths/min | 12 - 20 |
| Systolic BP (Non-invasive) | 220179 | mmHg | 90 - 120 |
| Diastolic BP (Non-invasive) | 220180 | mmHg | 60 - 80 |
| Mean Arterial Pressure | 220181 | mmHg | 70 - 100 |
| Temperature | 223761 | °F (converted to °C) | 36.5 - 37.5 °C |
| Glucose | 220621 | mg/dL | 70 - 140 |

#### 3.3 Pydantic v2 API Response Schemas

```python
class VitalSignSnapshot(BaseModel):
    heart_rate: float
    spo2: float
    sbp: float
    respiratory_rate: float
    timestamp: datetime

class VitalReadingItem(BaseModel):
    step_index: int
    timestamp: datetime
    heart_rate: float
    spo2: float
    sbp: float
    respiratory_rate: float

class DigitalTwinResponse(BaseModel):
    patient_id: str
    study_id: str
    deterioration_risk_score: float
    risk_tier: str
    cxr_probabilities: Dict[str, float]
    cxr_predictions: Dict[str, bool]
    cxr_top_finding: str
    cxr_top_probability: float
    current_vitals: VitalSignSnapshot
    vitals_trajectory_24h: Optional[List[VitalReadingItem]]
    visual_risk_contribution: float
    vitals_risk_contribution: float
    risk_factors: List[str]
    clinical_recommendation: str
    heatmap_base64: Optional[str] = None
    image_base64: Optional[str] = None
    timestamp: datetime
```

---

### 4. Machine Learning & Model Specifications

#### 4.1 Vision Model: DenseNet-121
- Architecture: Densely Connected Convolutional Network with 4 dense blocks and transition layers.
- Feature Extractor: Features before the final classification head produce a 1024-dimensional feature vector:
  $$\mathbf{v} = \text{GAP}(\text{features}(\mathbf{x})) \in \mathbb{R}^{1024}$$
- Classifier Layer:
  $$\mathbf{z} = \mathbf{W}\mathbf{v} + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{8 \times 1024}$$
  $$\mathbf{p} = \sigma(\mathbf{z})$$
- Loss Function: Multi-label binary cross-entropy with logits (`BCEWithLogitsLoss`).

#### 4.2 Explainability: Grad-CAM
- Gradient computation:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y_c}{\partial A_{i,j}^k}$$
- Heatmap generation:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- Normalization: Heatmap is rescaled to $[0, 1]$, colorized via the Jet colormap, and blended with the radiograph using an alpha weighting of 0.45.

#### 4.3 Multimodal Deterioration Risk Scoring
The unified risk score $R \in [0, 100]$ combines visual probability features and physiological vital sign deviations:
1. Visual Risk Component ($R_{\text{vis}}$): Weighted sum of detected pulmonary pathology probabilities, prioritized by clinical severity (Pneumothorax, Pneumonia, Consolidation, Edema weighted higher).
2. Vitals Risk Component ($R_{\text{vit}}$): Non-linear penalty formulation based on deviations from normal adult physiological ranges (tachycardia, tachypnea, hypoxemia, hypotension):
   $$R_{\text{vit}} = f(\text{HR}, \text{SpO}_2, \text{SBP}, \text{RR})$$
3. Unified Score:
   $$R = \alpha R_{\text{vis}} + (1 - \alpha) R_{\text{vit}}$$
   where $\alpha \approx 0.45$, adjusted by the clinical severity weighting of active findings.

---

### 5. Architectural Decisions and Governance

1. Primary Prediction Targets:
   - Primary: Mechanical ventilation initiation within 1h, 3h, and 6h windows.
   - Secondary: Vasopressor therapy initiation.
   - Exclusion: In-hospital mortality was excluded as a primary modeling target due to severe class imbalance in standard cohorts ($< 6\%$).
2. Causal Integrity & Leakage Prevention:
   - Time-series features at time $t$ use only observations recorded at $\tau \le t$.
   - Preprocessing steps (imputation, scaling) must fit solely on training splits.
   - Patient-level splitting: Patients in validation and test folds are strictly segregated to avoid identity leakage.
3. Separation of Real and Synthetic Data:
   - Real clinical evaluations are anchored to verified MIMIC-CXR and MIMIC-IV database IDs.
   - Synthetic trajectory data is restricted to testing and the Ad-Hoc Sandbox, with explicit UI disclaimers.
4. Out-of-Distribution Modality Rejection:
   - Uploaded non-radiographic images are rejected at the API boundary (HTTP 400) to prevent deep learning model hallucinations.

---

### 6. High-Performance Cluster (DGX / HPC) Execution

For distributed GPU training on cluster nodes (A100 / H100 / RTX GPUs), scripts adhere to environment variable configuration:
- Dataset Root: `MIMIC_IV_ROOT` or `DATA_ROOT`
- Output Directory: `OUTPUT_DIR`
- Checkpoint Directory: `CHECKPOINT_DIR`

Execution command:
```bash
python -m ml.temporal.train --config configs/experiments/d5_multitask_full.yaml --epochs 30 --batch-size 128
```
All training runs log metadata (`run_metadata.json`), per-epoch metrics (`history.json`), and PyTorch model checkpoints (`checkpoints/best.pt`).
