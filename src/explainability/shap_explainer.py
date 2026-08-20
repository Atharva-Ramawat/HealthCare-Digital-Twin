"""
shap_explainer.py
-----------------
PHASE 8 PLACEHOLDER
SHAP-based concrete implementation of ExplainerInterface.

SHAP is currently the favoured XAI approach, but is NOT mandatory.
The final XAI method will be selected by the student team in Phase 8.
This file will only be implemented if SHAP is confirmed as the method of choice.

See src/explainability/explainer.py for the generic ExplainerInterface.

NOT IMPLEMENTED.
"""
from __future__ import annotations
from .explainer import ExplainerInterface
import numpy as np
from typing import Optional


class SHAPExplainer(ExplainerInterface):
    """
    SHAP-based prediction explainer.
    Concrete implementation of ExplainerInterface using SHAP library.

    Requires a trained model compatible with SHAP (e.g., DeepSHAP or KernelSHAP).
    Method selection and validation is a Phase 8 student team decision.
    """

    def __init__(self, model, config: dict):
        raise NotImplementedError("SHAPExplainer: Phase 8 TODO")

    def explain(self, window: np.ndarray, baseline: Optional[np.ndarray] = None) -> dict:
        raise NotImplementedError

    def is_ready(self) -> bool:
        return False
