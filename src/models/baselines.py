"""
baselines.py
------------
PHASE 2 PLACEHOLDER
Classical baseline models for comparison against the CNN-BiLSTM.

Models:
  1. Statistical Baseline: Last-value carry-forward + threshold rules
  2. Logistic Regression: On engineered temporal features
  3. Random Forest: On engineered temporal features  
  4. Simple LSTM: Single-direction, single-task LSTM

All wrapped with BaseDigitalTwinModel interface.

NOT IMPLEMENTED.
"""
from __future__ import annotations
from .base_model import BaseDigitalTwinModel
import numpy as np


class StatisticalBaseline(BaseDigitalTwinModel):
    """Heuristic last-value + NEWS2 threshold baseline."""
    def train(self, X_train, y_train, **kwargs): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError

class LogisticRegressionModel(BaseDigitalTwinModel):
    """Logistic Regression on flattened temporal feature vectors."""
    def train(self, X_train, y_train, **kwargs): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError

class RandomForestModel(BaseDigitalTwinModel):
    """Random Forest on engineered clinical features."""
    def train(self, X_train, y_train, **kwargs): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError

class SimpleLSTMModel(BaseDigitalTwinModel):
    """Unidirectional LSTM single-task baseline."""
    def train(self, X_train, y_train, **kwargs): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError
