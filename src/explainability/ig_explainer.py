"""
ig_explainer.py
---------------
PHASE 8 PLACEHOLDER
Integrated Gradients concrete implementation of ExplainerInterface.

Alternative to SHAP. May be considered if gradient-based attribution
is better suited to the final model architecture.

See src/explainability/explainer.py for the generic ExplainerInterface.
Method selection is a Phase 8 student team decision.

NOT IMPLEMENTED.
"""
from __future__ import annotations
from .explainer import ExplainerInterface
import numpy as np
from typing import Optional


class IntegratedGradientsExplainer(ExplainerInterface):
    """
    Integrated Gradients gradient-based attribution.
    Based on: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (2017).
    """

    def __init__(self, model, device=None, steps: int = 50):
        raise NotImplementedError("IntegratedGradientsExplainer: Phase 8 TODO")

    def explain(self, window: np.ndarray, baseline: Optional[np.ndarray] = None) -> dict:
        raise NotImplementedError

    def is_ready(self) -> bool:
        return False
