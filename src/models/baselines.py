"""
baselines.py
------------
PHASE 5 PLACEHOLDER (student team research)
Classical baseline models for comparison against the primary CNN-BiLSTM.

All models implement BaseDigitalTwinModel -> PredictionInterface.
The Digital Twin Engine accesses them via PredictionInterface only.

The student team selects which baselines to implement and evaluates them (Phase 5).

Models:
  1. StatisticalBaseline: Last-value carry-forward + NEWS2 threshold rules
  2. LogisticRegressionModel: On engineered temporal features
  3. RandomForestModel: On engineered temporal features
  4. SimpleLSTMModel: Unidirectional, single-task LSTM

NOT IMPLEMENTED.
"""
from __future__ import annotations
from .base_model import BaseDigitalTwinModel
from .prediction_interface import ModelInput, PredictionOutput


class StatisticalBaseline(BaseDigitalTwinModel):
    """Heuristic last-value + NEWS2 threshold baseline."""
    def predict(self, model_input: ModelInput) -> PredictionOutput: raise NotImplementedError
    def is_ready(self) -> bool: return False
    def train(self, X_train, y_train, **kwargs): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError

class LogisticRegressionModel(BaseDigitalTwinModel):
    """Logistic Regression on flattened temporal feature vectors."""
    def predict(self, model_input: ModelInput) -> PredictionOutput: raise NotImplementedError
    def is_ready(self) -> bool: return False
    def train(self, X_train, y_train, **kwargs): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError

class RandomForestModel(BaseDigitalTwinModel):
    """Random Forest on engineered clinical features."""
    def predict(self, model_input: ModelInput) -> PredictionOutput: raise NotImplementedError
    def is_ready(self) -> bool: return False
    def train(self, X_train, y_train, **kwargs): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError

class SimpleLSTMModel(BaseDigitalTwinModel):
    """Unidirectional LSTM single-task baseline."""
    def predict(self, model_input: ModelInput) -> PredictionOutput: raise NotImplementedError
    def is_ready(self) -> bool: return False
    def train(self, X_train, y_train, **kwargs): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError
