"""
twin_engine.py
--------------
PHASE 3 PLACEHOLDER
The Digital Twin Engine -- orchestrator of the entire per-timestep update cycle.

The engine depends on PredictionInterface (model-agnostic), NOT CNN-BiLSTM directly.
This allows any model implementation to be plugged in by the student team.

Responsibilities (per timestep):
  1. Receive new vital observation from ClinicalReplayEngine
  2. Append to ObservedState.vital_history
  3. Update ObservedState.current_vitals and latest_timestamp
  4. Update DerivedState via ClinicalFeatureEngineer
     (Deltas, rates, rolling stats, baseline deviation, NEWS2)
  5. Update DerivedState.patient_baseline via PatientBaselineCalculator (causal)
  6. Call PredictionInterface.predict(sliding_window)
  7. Write output to PredictedState (risk_scores, future_state, uncertainty)
  8. Call ExplainerInterface.explain() if available
  9. Write attributions to PredictedState.xai_attributions
  10. Store updated DigitalTwinState

The Dashboard reads DigitalTwinState -- it does NOT call the engine directly.

NOT IMPLEMENTED.
"""
from __future__ import annotations
from typing import Dict, Optional
from .patient_state import DigitalTwinState, ObservedState, DerivedState, PredictedState, SimulationState


class DigitalTwinEngine:
    """
    Orchestrates the full Digital Twin state update cycle per patient per timestep.

    Dependencies (all injected -- not hardcoded):
      - predictor: PredictionInterface implementation (real or placeholder)
      - feature_engineer: ClinicalFeatureEngineer
      - baseline_calc: PatientBaselineCalculator
      - explainer: ExplainerInterface (optional)
      - config: loaded settings.yaml
    """

    def __init__(self, config: dict, predictor, feature_engineer, baseline_calc, explainer=None):
        """
        Args:
            predictor: PredictionInterface -- model-agnostic. PlaceholderPredictor acceptable.
            explainer: ExplainerInterface -- optional. None if not yet configured.
        """
        raise NotImplementedError("DigitalTwinEngine: Phase 3 TODO")

    def register_patient(self, patient_id: str, demographics: dict) -> None:
        """Register a new patient and initialise their DigitalTwinState."""
        raise NotImplementedError

    def update(self, patient_id: str, new_vitals: dict) -> DigitalTwinState:
        """
        Process one new observation timestep for a patient.

        1. Update ObservedState
        2. Update DerivedState (causal -- no future data)
        3. Call PredictionInterface.predict()  -> update PredictedState
        4. Call ExplainerInterface.explain()   -> update PredictedState.xai_attributions
        5. Return updated DigitalTwinState

        NOTE: If predictor.is_ready() is False or is_placeholder is True,
        PredictedState.prediction_valid remains False and outputs are labelled placeholder.
        """
        raise NotImplementedError

    def get_state(self, patient_id: str) -> Optional[DigitalTwinState]:
        """Retrieve current state for a patient."""
        raise NotImplementedError

    def get_all_states(self) -> Dict[str, DigitalTwinState]:
        """Retrieve states for all registered patients."""
        raise NotImplementedError
