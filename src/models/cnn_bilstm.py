"""
cnn_bilstm.py
-------------
PHASE 5 PLACEHOLDER (student team research -- architecture TBD)
Primary deep-learning model: CNN-BiLSTM Multi-Task Network.

STUDENT TEAM RESPONSIBILITY:
  The final architecture, hyperparameters, output heads, and training
  configuration are research decisions owned by the student team (Phase 5).
  The values in configs/settings.yaml are illustrative starting points only.

INTERFACE:
  Implements BaseDigitalTwinModel -> PredictionInterface.
  The DigitalTwinEngine accesses this model via PredictionInterface only.
  This ensures the Digital Twin infrastructure does not depend on CNN-BiLSTM internals.

CURRENT DIRECTION (per project synopsis, subject to student team finalisation):
  Input: (Batch, window_size, num_features)
  Conv1D block -> BiLSTM -> shared representation
  Output heads: Risk prediction (multi-horizon), optionally future vital forecast

NOT IMPLEMENTED.
"""
from __future__ import annotations
from .base_model import BaseDigitalTwinModel
from .prediction_interface import ModelInput, PredictionOutput
import numpy as np


class CNNBiLSTMTwin(BaseDigitalTwinModel):
    """
    CNN-BiLSTM Multi-Task Digital Twin prediction model.
    Architecture to be finalised by the student team in Phase 5.
    """

    def __init__(self, config: dict):
        raise NotImplementedError("CNNBiLSTMTwin: Phase 5 TODO -- student team research")

    def predict(self, model_input: ModelInput) -> PredictionOutput:
        raise NotImplementedError

    def is_ready(self) -> bool:
        return False

    def train(self, X_train, y_train, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
