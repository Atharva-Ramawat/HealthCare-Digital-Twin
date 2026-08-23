"""
DenseNet-121 Multi-Label Chest X-Ray (CXR) Pulmonary Vision Model & Grad-CAM Explainability.
Re-exports canonical ML components from ml.cxr for backend service integration.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ml.cxr.model import (
    DenseNet121Pulmonary,
    GradCAMExplainer
)
from ml.cxr.labels import (
    TARGET_PULMONARY_CLASSES as PULMONARY_PATHOLOGIES,
    DEFAULT_PATHOLOGY_THRESHOLDS as PATHOLOGY_THRESHOLDS
)

__all__ = [
    "DenseNet121Pulmonary",
    "GradCAMExplainer",
    "PULMONARY_PATHOLOGIES",
    "PATHOLOGY_THRESHOLDS"
]
