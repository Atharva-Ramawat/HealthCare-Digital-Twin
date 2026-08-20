"""
shap_explainer.py
-----------------
PHASE 3 PLACEHOLDER
SHAP-based feature attribution for the trained CNN-BiLSTM model.

Responsibilities:
  - Compute SHAP values per vital-sign feature per timestep
  - Aggregate temporal attributions to feature-level importance
  - Return explanation dict compatible with DigitalTwinState.last_shap_values

NOTE: Legacy prototype used Integrated Gradients (captum).
SHAP is the primary target per project constitution.
Both may be supported (configurable).

NOT IMPLEMENTED.
"""
from __future__ import annotations
import numpy as np


class SHAPExplainer:
    """SHAP-based prediction explainer for temporal clinical models."""

    def __init__(self, model, config: dict):
        raise NotImplementedError("SHAPExplainer: Phase 3 TODO")

    def explain(self, window: np.ndarray) -> dict:
        """
        Compute feature attributions for a single inference window.
        Returns {vital_key: {shap_value, impact_pct, direction}}.
        """
        raise NotImplementedError
