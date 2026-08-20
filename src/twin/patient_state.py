"""
patient_state.py
-----------------
PHASE 3 PLACEHOLDER
Defines the DigitalTwinState dataclass -- the core in-memory representation
of a single patient's Digital Twin.

The state is partitioned into four logically separate sub-states.
See docs/PROJECT_CONSTITUTION.md Section 5 for the authoritative schema.

CAUSAL CONSTRAINT: DerivedState at time t must ONLY use observations up to
and including time t. No future information may influence past-timestep state.

NOT IMPLEMENTED. Schema defined here as architectural reference.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd


# =============================================================================
# A. OBSERVED STATE
# Contains only directly measured or reported clinical information.
# No inference, no computation beyond raw recording.
# =============================================================================
@dataclass
class ObservedState:
    """
    Raw observed clinical data for this patient.
    Populated by DigitalTwinEngine from each incoming timestep observation.
    """
    patient_id:        str = ""
    demographics:      dict = field(default_factory=dict)  # age, gender, weight, etc.
    current_vitals:    dict = field(default_factory=dict)  # {vital_key: latest_raw_value}
    vital_history:     Optional[pd.DataFrame] = None       # full temporally-indexed record
    latest_timestamp:  Optional[datetime] = None           # timestamp of latest observation
    data_source:       str = "synthetic"  # "synthetic" | "mimic_iv" | "vitaldb" | etc.
    replay_step_index: int = 0


# =============================================================================
# B. DERIVED STATE
# Quantities computed causally from ObservedState.
# CRITICAL: All values at time t use ONLY observations up to time t.
# Populated by ClinicalFeatureEngineer and PatientBaselineCalculator.
# =============================================================================
@dataclass
class DerivedState:
    """
    Causally-computed features derived from the observed patient history.

    CAUSAL CONSTRAINT: Every field here must be computed using only
    vital_history up to and including the current timestep. This is a
    hard correctness requirement -- no future data may be used.

    patient_baseline: Initial engineering approach uses EWMA.
    The final baseline methodology is a student team research decision (Phase 5).
    """
    sliding_window:      Optional[np.ndarray] = None  # (window_size, num_features)
    vital_deltas:        dict = field(default_factory=dict)  # DeltaX_t = X_t - X_{t-1}
    vital_rates:         dict = field(default_factory=dict)  # dX/dt over short window
    rolling_stats:       dict = field(default_factory=dict)  # {window: {vital: stats}}
    patient_baseline:    dict = field(default_factory=dict)  # patient-specific baseline
    baseline_valid:      bool = False   # True when >= min_observations accumulated
    baseline_deviations: dict = field(default_factory=dict)  # X_t - PatientBaseline_t
    news2_score:         int = 0
    news2_tier:          str = "Unknown"   # "Low" | "Medium" | "High"
    derived_vitals:      dict = field(default_factory=dict)  # MAP, ShockIndex, PP


# =============================================================================
# C. PREDICTED STATE
# Populated by PredictionInterface after each model inference call.
# All fields remain None / False until a REAL trained model is integrated (Phase 6).
# is_placeholder MUST remain True until a validated model is loaded.
# =============================================================================
@dataclass
class PredictedState:
    """
    Model inference outputs for this patient.

    prediction_valid: False until a real trained model is loaded.
    is_placeholder: True for all stub or demo predictor outputs.

    future_state: OPTIONAL. Only populated if the student team decides to
    implement future vital-sign prediction (Phase 5 decision).

    uncertainty: OPTIONAL. Method TBD by student team (Phase 8).
    xai_attributions: OPTIONAL. Method TBD by student team (Phase 8).
    """
    risk_scores:       dict = field(default_factory=dict)  # {horizon_steps: probability}
    risk_tier:         str = "Unknown"
    future_state:      Optional[np.ndarray] = None  # (horizon, num_features) -- optional
    uncertainty:       Optional[dict] = None         # PredictionUncertainty -- method TBD
    xai_attributions:  Optional[dict] = None         # {vital_key: attribution} -- method TBD
    prediction_valid:  bool = False   # False until a REAL model is integrated
    is_placeholder:    bool = True    # MUST be True for stub/demo outputs
    last_predicted_at: Optional[datetime] = None


# =============================================================================
# D. SIMULATION STATE
# Populated by ScenarioSimulator. Empty until Phase 7.
# All results are HYPOTHETICAL PROJECTIONS -- not clinical recommendations.
# =============================================================================
@dataclass
class SimulationState:
    """
    What-if scenario simulation results.

    IMPORTANT: All scenario_results are HYPOTHETICAL MODEL PROJECTIONS.
    They do NOT represent causal intervention predictions or treatment recommendations.
    They must be labelled as such in the UI and in any research outputs.
    """
    active_scenarios:     list = field(default_factory=list)   # list[ScenarioDefinition]
    scenario_results:     list = field(default_factory=list)   # list[ScenarioResult]
    simulation_valid:     bool = False
    simulation_timestamp: Optional[datetime] = None


# =============================================================================
# TOP-LEVEL DIGITAL TWIN STATE
# =============================================================================
@dataclass
class DigitalTwinState:
    """
    Core computational representation of a single ICU patient Digital Twin.

    This is the central data structure of the entire system.
    All modules either read from or write to this structure via well-defined interfaces.

    The Digital Twin operates independently of the Streamlit dashboard.
    The dashboard reads from this state; it does not write to it or trigger inference.

    Phase 3: Full implementation.
    Phase 6: PredictedState populated by real model.
    Phase 7: SimulationState populated by ScenarioSimulator.
    """
    observed:        ObservedState = field(default_factory=ObservedState)
    derived:         DerivedState = field(default_factory=DerivedState)
    predicted:       PredictedState = field(default_factory=PredictedState)
    simulation:      SimulationState = field(default_factory=SimulationState)
    last_updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Serialise state to a JSON-compatible dictionary."""
        raise NotImplementedError("Phase 3 TODO")

    def summary(self) -> str:
        """Return human-readable summary of current twin state."""
        raise NotImplementedError("Phase 3 TODO")

    def is_prediction_real(self) -> bool:
        """Return True only when a real validated model is producing predictions."""
        return self.predicted.prediction_valid and not self.predicted.is_placeholder
