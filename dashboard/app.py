"""
dashboard/app.py
----------------
PHASE 4 PLACEHOLDER
Main Streamlit dashboard entry point -- Visualisation Layer ONLY.

ARCHITECTURAL RULE:
  The dashboard is NOT the Digital Twin.
  The dashboard reads from DigitalTwinState and renders it.
  The dashboard does NOT write to DigitalTwinState.
  The dashboard does NOT trigger ML inference.

  Digital Twin Engine -> DigitalTwinState -> Dashboard (read only)

The Digital Twin Engine operates independently of this Streamlit application.
"""
import streamlit as st

st.set_page_config(
    page_title="AI Clinical Digital Twin",
    page_icon="[ICU]",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Phase 0 -- Architecture Scaffold")

st.warning(
    "PLACEHOLDER DASHBOARD -- Phase 0 Architecture Only.  "
    "No real models are loaded. No clinical data is connected. "
    "See docs/PROJECT_CONSTITUTION.md for the development roadmap."
)

st.info(
    "The Digital Twin operates independently of this dashboard.  "
    "This view reads from DigitalTwinState only -- it does not drive inference."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Module Implementation Status")
    st.markdown("""
| Module | Status |
|---|---|
| MIMIC-IV Loader | NOT IMPLEMENTED |
| VitalDB Loader | NOT IMPLEMENTED |
| Synthetic Generator | PORTING (Phase 1) |
| Data Cleaner | NOT IMPLEMENTED |
| Temporal Aligner | NOT IMPLEMENTED (resolution TBD Phase 1) |
| Feature Engineer | NOT IMPLEMENTED |
| Baseline Calculator | NOT IMPLEMENTED |
| Replay Engine | NOT IMPLEMENTED |
| DigitalTwinState | SCHEMA DEFINED ONLY |
| DigitalTwinEngine | NOT IMPLEMENTED |
| PredictionInterface | INTERFACE DEFINED ONLY |
| Prediction Model | NOT IMPLEMENTED |
| SHAP / XAI | NOT IMPLEMENTED (method TBD Phase 8) |
| What-If Simulation | NOT IMPLEMENTED |
| Dashboard (full) | NOT IMPLEMENTED |
""")

with col2:
    st.subheader("Architecture Principles")
    st.markdown("""
**The Digital Twin is NOT the dashboard.**

```
Digital Twin Engine
      | updates
DigitalTwinState
  Observed | Derived | Predicted | Simulation
      | read by
Dashboard (this view)
```

**PredictionInterface is model-agnostic.**
CNN-BiLSTM is the planned model but is not yet implemented.
The final architecture is a student team research decision (Phase 5).

**Synthetic data is for development only.**
No ML claims may be made from synthetic data.

**Phase gating is enforced.**
Each phase requires explicit student team approval before proceeding.
""")

st.subheader("Phase Roadmap")
st.markdown("""
| Phase | Focus | Status |
|---|---|---|
| 0 | Architecture + Constitution | **Done (v1.1)** |
| 1 | Dataset discovery + ingestion | **Next** |
| 2 | Preprocessing + temporal pipeline | Pending |
| 3 | Digital Twin State + Engine | Pending |
| **4** | **Simulated replay + minimal viz** | **Pending (30% Milestone)** |
| 5 | Student ML research | Pending (student-owned) |
| 6 | ML integration | Pending |
| 7 | What-if simulation | Pending |
| 8 | XAI + uncertainty | Pending |
| 9 | Complete dashboard | Pending |
| 10 | External validation | Pending |
| 11 | Final integration + evaluation | Pending |
""")
