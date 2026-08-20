"""
explainer.py
------------
PHASE 8 PLACEHOLDER
Generic XAI (Explainability) interface for the Digital Twin.

The XAI method is NOT fixed or coupled to one library at Phase 0.
The specific method will be selected by the student team in Phase 8,
based on the final model architecture and clinical interpretability requirements.

The Digital Twin depends on ExplainerInterface -- not on SHAP or captum directly.

Candidate methods (illustrative, not exhaustive):
  - SHAP (currently favoured for interpretability)
  - Integrated Gradients (captum)
  - TimeShap (temporal-aware SHAP variant)
  - Attention weight visualisation (if attention-based architecture)

NOT IMPLEMENTED.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class ExplainerInterface(ABC):
    """
    Abstract interface for XAI attribution methods.

    The DigitalTwinEngine calls explain() after each prediction.
    The interface does not assume SHAP or any other specific library.
    """

    @abstractmethod
    def explain(self, window: np.ndarray, baseline: Optional[np.ndarray] = None) -> dict:
        """
        Compute feature attributions for a single inference window.

        Args:
            window: (window_size, num_features) input to the model.
            baseline: Optional reference baseline (method-dependent).

        Returns:
            dict: {vital_key: {attribution: float, impact_pct: float, direction: str}}
            All attributions computed with respect to the model's risk prediction output.
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True only when a real trained model and XAI method are configured."""
        ...


class PlaceholderExplainer(ExplainerInterface):
    """
    Stub explainer for software integration testing ONLY.
    Returns dummy attribution values. Must be labelled as placeholder.

    PHASE 8 TODO: Replace with real ExplainerInterface implementation.
    """

    def explain(self, window: np.ndarray, baseline: Optional[np.ndarray] = None) -> dict:
        raise NotImplementedError(
            "PlaceholderExplainer: Phase 8 TODO. "
            "Outputs must be labelled as PLACEHOLDER."
        )

    def is_ready(self) -> bool:
        return False
