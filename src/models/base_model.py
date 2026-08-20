"""
base_model.py
-------------
PHASE 3 PLACEHOLDER
Abstract base class for all prediction models in this project.

Implements PredictionInterface (see prediction_interface.py).
All models must satisfy the predict() contract and return PredictionOutput.

PredictionOutput.future_state is OPTIONAL -- not all models need to forecast vitals.
PredictionOutput.uncertainty is OPTIONAL -- method selected in Phase 8.

NOT IMPLEMENTED.
"""
from __future__ import annotations
from .prediction_interface import PredictionInterface, ModelInput, PredictionOutput
import numpy as np


class BaseDigitalTwinModel(PredictionInterface):
    """
    Abstract base for all Digital Twin prediction models.
    Extends PredictionInterface with save/load capability.

    The student team implements concrete subclasses.
    Antigravity provides the engineering interface and infrastructure.
    """

    def predict(self, model_input: ModelInput) -> PredictionOutput:
        """
        Run inference. Must return PredictionOutput with at minimum:
          risk_scores: dict   {horizon_steps: probability in [0, 1]}
          risk_tier:   str    "Low" | "Medium" | "High"
          is_placeholder: bool

        future_state and uncertainty are OPTIONAL.
        """
        raise NotImplementedError

    def is_ready(self) -> bool:
        raise NotImplementedError

    def train(self, X_train: np.ndarray, y_train: dict, **kwargs) -> None:
        """
        Train the model on prepared feature sequences.
        y_train is a dict keyed by output head (e.g., "risk_labels", "forecast_vitals").
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save model weights and metadata to path."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load model weights from path."""
        raise NotImplementedError
