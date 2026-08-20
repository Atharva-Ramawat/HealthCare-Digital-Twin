"""
prediction_interface.py
------------------------
PHASE 3 PLACEHOLDER
Model-agnostic prediction interface for the Digital Twin.

The DigitalTwinEngine depends on PredictionInterface -- NOT on CNN-BiLSTM directly.
This decouples the Digital Twin infrastructure from the specific ML model architecture.

The student team may plug any model into the Digital Twin by implementing this interface.

PLACEHOLDER vs REAL models:
  - PredictionOutput.is_placeholder MUST be True for all stub/demo predictors.
  - PredictionOutput.is_placeholder MUST be False for real trained models.
  - The dashboard MUST visually distinguish placeholder from real outputs.

NOT IMPLEMENTED.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ModelInput:
    """Standardised input to any prediction model."""
    sliding_window:   np.ndarray   # (window_size, num_features) -- normalised
    patient_id:       str
    derived_features: dict = field(default_factory=dict)   # optional supplementary


@dataclass
class PredictionOutput:
    """
    Standardised output from any prediction model.

    risk_scores is REQUIRED. All other fields are OPTIONAL.

    future_state: only present if the student team implements future vital prediction.
    uncertainty:  only present if an uncertainty method is selected (Phase 8).
    is_placeholder: MUST be True for stub/demo models. MUST be False for real models.
    """
    risk_scores:     dict                  # {horizon_steps: float in [0,1]} -- REQUIRED
    risk_tier:       str                   # "Low" | "Medium" | "High" -- REQUIRED
    future_state:    Optional[np.ndarray] = None   # (horizon, num_features) -- OPTIONAL
    uncertainty:     Optional[dict] = None          # uncertainty estimate -- OPTIONAL (method TBD)
    is_placeholder:  bool = True           # MUST be False for real trained models


class PredictionInterface(ABC):
    """
    Abstract interface that all prediction models must implement.

    The Digital Twin Engine calls predict() at each timestep.
    The interface does not assume CNN-BiLSTM or any specific architecture.
    """

    @abstractmethod
    def predict(self, model_input: ModelInput) -> PredictionOutput:
        """Run inference and return standardised output."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Return True only when a real trained model is loaded.
        Stub/placeholder implementations must return False.
        """
        ...

    def validate_output(self, output: PredictionOutput) -> None:
        """Check that output satisfies the interface contract."""
        if not output.risk_scores:
            raise ValueError("PredictionOutput.risk_scores must not be empty")
        if output.is_placeholder and output.prediction_valid if hasattr(output, "prediction_valid") else False:
            raise ValueError("Placeholder output must not be marked as prediction_valid")


class PlaceholderPredictor(PredictionInterface):
    """
    Stub predictor for software integration testing ONLY.

    Returns constant dummy values. MUST be clearly labelled as placeholder
    in all UI, logs, and research outputs.

    PHASE 3 TODO: Implement once DigitalTwinEngine is wired.
    """

    def predict(self, model_input: ModelInput) -> PredictionOutput:
        """Return dummy placeholder output. NOT a research result."""
        raise NotImplementedError(
            "PlaceholderPredictor: Phase 3 TODO. "
            "This is a stub -- outputs must be labelled as PLACEHOLDER."
        )

    def is_ready(self) -> bool:
        return False   # Placeholder is never 'ready' for research use
