"""
model_comparator.py
--------------------
PHASE 2 PLACEHOLDER
Compares multiple models on the same test dataset.

Produces a comparison table:
  Model | AUROC-1h | AUROC-3h | AUROC-6h | AUPRC | Lead-Time | MSE-Forecast

NOT IMPLEMENTED.
"""
from __future__ import annotations


class ModelComparator:
    def __init__(self, evaluator, models: list): raise NotImplementedError("ModelComparator: Phase 2 TODO")
    def compare(self, X_test, y_test) -> dict: raise NotImplementedError
