"""
metrics.py
----------
PHASE 2 PLACEHOLDER
Clinical and ML evaluation metrics.

Metrics to compute:
  - AUROC (per horizon: 1h, 3h, 6h)
  - AUPRC (clinically important for imbalanced deterioration labels)
  - Sensitivity / Specificity at clinical thresholds
  - F1 Score
  - Early Warning Lead Time (timesteps before event that model fires alert)
  - Calibration (Brier Score, reliability diagram)
  - Forecast MSE / MAE per vital sign

NOT IMPLEMENTED.
"""
from __future__ import annotations
import numpy as np


class ClinicalEvaluator:
    def __init__(self): raise NotImplementedError("ClinicalEvaluator: Phase 2 TODO")
    def compute_auroc(self, y_true, y_pred_proba): raise NotImplementedError
    def compute_auprc(self, y_true, y_pred_proba): raise NotImplementedError
    def compute_early_warning_lead_time(self, y_true, y_pred_proba, threshold): raise NotImplementedError
    def compute_calibration(self, y_true, y_pred_proba): raise NotImplementedError
    def compute_forecast_error(self, y_true_vitals, y_pred_vitals): raise NotImplementedError
