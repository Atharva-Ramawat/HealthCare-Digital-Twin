"""
ig_explainer.py
---------------
PHASE 3 PLACEHOLDER
Integrated Gradients explainer (alternative / comparison to SHAP).

Based on: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (2017).
Migrated and refactored from legacy prototype (docs/legacy_prototype/explainability.py).

NOT IMPLEMENTED.
"""
from __future__ import annotations
import numpy as np


class IntegratedGradientsExplainer:
    """Integrated Gradients gradient-based attribution for CNN-BiLSTM."""

    def __init__(self, model, device=None, steps: int = 50):
        raise NotImplementedError("IntegratedGradientsExplainer: Phase 3 TODO")

    def explain(self, window: np.ndarray, baseline: np.ndarray = None) -> dict:
        raise NotImplementedError
