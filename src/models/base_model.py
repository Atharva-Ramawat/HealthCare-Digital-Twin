"""
base_model.py
-------------
PHASE 1 PLACEHOLDER
Abstract base class that all models in this project must implement.
Defines a consistent interface for training, inference, saving, and loading.

NOT IMPLEMENTED.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseDigitalTwinModel(ABC):
    """Abstract base for all Digital Twin prediction models."""

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """Train the model on prepared feature sequences."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> dict:
        """
        Run inference. Must return a dict with at minimum:
          {
            "risk_score": float,          # 0.0 - 1.0
            "risk_tier": str,             # "Low" | "Medium" | "High"
            "forecast_vitals": ndarray,   # (forecast_horizon, num_features)
          }
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...
