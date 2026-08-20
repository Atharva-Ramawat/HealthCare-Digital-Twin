"""
app.py
------
PHASE 1 PLACEHOLDER (enhanced from legacy prototype)
Main Streamlit dashboard entry point.

Views:
  1. Ward Overview    — grid of all active patient twins
  2. Patient Twin     — detailed single-patient Digital Twin view
  3. What-If Panel    — counterfactual scenario simulation interface
  4. Alert Feed       — NEWS2 threshold breach log

NOT IMPLEMENTED (modular rewrite; legacy prototype is flat app.py).
"""
import streamlit as st

st.set_page_config(
    page_title="AI Clinical Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚧 Phase 0 — Architecture Scaffold")
st.info(
    "This dashboard is a placeholder. "
    "The full Digital Twin UI will be built in Phase 1 (Simulation Layer) "
    "and incrementally enhanced through Phase 4. "
    "See docs/PROJECT_CONSTITUTION.md for the development roadmap."
)
st.markdown("""
**Module Status:**
| Module | Status |
|---|---|
| Ingestion | 🔴 Not Implemented |
| Preprocessing | 🔴 Not Implemented |
| Feature Engineering | 🔴 Not Implemented |
| Synthetic Generator (Fallback) | 🟡 Porting from Prototype |
| Replay Engine | 🔴 Not Implemented |
| Digital Twin State | 🟡 Schema Defined |
| Risk Prediction (Baselines) | 🔴 Not Implemented |
| CNN-BiLSTM Model | 🔴 Not Implemented |
| SHAP Explainability | 🔴 Not Implemented |
| What-If Simulation | 🔴 Not Implemented |
| Dashboard (Full) | 🔴 Not Implemented |
""")
