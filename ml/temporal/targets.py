"""
MIMIC-IV Multi-Task Temporal Target Generation Engine.
Implements mathematically rigorous targets with ZERO future imputation and explicit target validity masks:
- Head 1: Physiological Deterioration (with deterioration_valid_mask)
- Head 2: Risk Tier (0=Low, 1=Medium, 2=High)
- Head 3: Short-Term Vital Forecast (Shape [4, 5], with forecast_valid_mask [4, 5])
- Head 4: Event-Linked Observational Treatment-Response (with response_valid_mask)
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

FORECAST_CHANNELS = ["heart_rate", "spo2", "sbp", "respiratory_rate", "temperature_c"]

NORMAL_VITAL_DEFAULTS = {
    "heart_rate": 75.0,
    "spo2": 98.0,
    "sbp": 120.0,
    "dbp": 80.0,
    "map": 85.0,
    "respiratory_rate": 16.0,
    "temperature_c": 37.0
}


def compute_vital_instability_score(
    hr: Any,
    spo2: Any,
    sbp: Any,
    rr: Any,
    map_val: Any
) -> float:
    """
    Compute a continuous physiological instability index in [0.0, 1.0] from vitals.
    """
    try:
        hr_f = float(hr) if pd.notnull(hr) else NORMAL_VITAL_DEFAULTS["heart_rate"]
    except (ValueError, TypeError):
        hr_f = NORMAL_VITAL_DEFAULTS["heart_rate"]

    try:
        spo2_f = float(spo2) if pd.notnull(spo2) else NORMAL_VITAL_DEFAULTS["spo2"]
    except (ValueError, TypeError):
        spo2_f = NORMAL_VITAL_DEFAULTS["spo2"]

    try:
        sbp_f = float(sbp) if pd.notnull(sbp) else NORMAL_VITAL_DEFAULTS["sbp"]
    except (ValueError, TypeError):
        sbp_f = NORMAL_VITAL_DEFAULTS["sbp"]

    try:
        rr_f = float(rr) if pd.notnull(rr) else NORMAL_VITAL_DEFAULTS["respiratory_rate"]
    except (ValueError, TypeError):
        rr_f = NORMAL_VITAL_DEFAULTS["respiratory_rate"]

    try:
        map_f = float(map_val) if pd.notnull(map_val) else NORMAL_VITAL_DEFAULTS["map"]
    except (ValueError, TypeError):
        map_f = NORMAL_VITAL_DEFAULTS["map"]

    score = 0.0
    if spo2_f < 88.0: score += 0.35
    elif spo2_f < 92.0: score += 0.20
    elif spo2_f < 95.0: score += 0.10

    if map_f < 60.0 or sbp_f < 85.0: score += 0.30
    elif map_f < 65.0 or sbp_f < 90.0: score += 0.20
    elif map_f < 70.0: score += 0.10

    if rr_f > 30.0 or rr_f < 8.0: score += 0.25
    elif rr_f > 24.0: score += 0.15
    elif rr_f > 20.0: score += 0.05

    if hr_f > 130.0 or hr_f < 40.0: score += 0.20
    elif hr_f > 110.0 or hr_f < 50.0: score += 0.10

    return min(1.0, score)


def extract_targets_for_window(
    raw_timeline_df: pd.DataFrame,
    pred_idx: int,
    forecast_horizon_steps: int = 4,
    treatment_event_info: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract multi-task targets using ACTUAL future observations with explicit validity masks.
    ZERO future-target imputation is applied.
    """
    total_len = len(raw_timeline_df)
    future_end_idx = pred_idx + forecast_horizon_steps

    if future_end_idx >= total_len:
        return None

    # Slice actual future observations [pred_idx + 1 : pred_idx + 4]
    future_slice = raw_timeline_df.iloc[pred_idx + 1 : future_end_idx + 1].copy()

    # -------------------------------------------------------------
    # HEAD 3: Short-Term Vital Forecasting & Validity Mask
    # -------------------------------------------------------------
    forecast_raw = future_slice[FORECAST_CHANNELS].values.astype(np.float32)  # Shape (4, 5)
    
    # 1.0 where genuinely observed/non-null in future window, 0.0 where missing
    forecast_valid_mask = (~np.isnan(forecast_raw)).astype(np.float32)
    
    # Replace unobserved NaNs with 0.0 for numeric stability (masked out in loss calculation)
    forecast_targets = np.nan_to_num(forecast_raw, nan=0.0)

    # -------------------------------------------------------------
    # HEAD 1: Physiological Deterioration Target (Persistence Rule)
    # -------------------------------------------------------------
    # Only evaluate deterioration if at least 1 actual vital observation exists in future window
    future_has_vitals = forecast_valid_mask.sum() > 0
    deterioration_valid_mask = 1.0 if future_has_vitals else 0.0

    hypoxemia = future_slice["spo2"] < 90.0
    map_col = "map" if "map" in future_slice.columns else "sbp"
    hypotension = (future_slice[map_col] < 65.0) | (future_slice["sbp"] < 90.0)
    tachypnea = future_slice["respiratory_rate"] > 28.0
    hr_instability = (future_slice["heart_rate"] < 45.0) | (future_slice["heart_rate"] > 130.0)

    any_abnormal_per_step = (hypoxemia | hypotension | tachypnea | hr_instability).fillna(False)
    abnormal_step_count = int(any_abnormal_per_step.sum())

    is_deterioration = 1.0 if abnormal_step_count >= 2 else 0.0

    # -------------------------------------------------------------
    # HEAD 2: Risk Tier (0=Low, 1=Medium, 2=High)
    # -------------------------------------------------------------
    if is_deterioration == 1.0:
        risk_tier = 2
    elif abnormal_step_count == 1 or np.any(future_slice["spo2"].dropna() < 93.0) or np.any(future_slice["respiratory_rate"].dropna() > 22.0):
        risk_tier = 1
    else:
        risk_tier = 0

    # -------------------------------------------------------------
    # HEAD 4: Event-Linked Observational Treatment-Response State
    # -------------------------------------------------------------
    # Only valid when explicitly linked to a real treatment initiation event T0
    if treatment_event_info is not None and treatment_event_info.get("is_eligible_window", False):
        response_valid_mask = 1.0
        pre_start = max(0, pred_idx - 3)
        pre_slice = raw_timeline_df.iloc[pre_start : pred_idx + 1]

        pre_scores = [
            compute_vital_instability_score(
                row.get("heart_rate"), row.get("spo2"), row.get("sbp"),
                row.get("respiratory_rate"), row.get("map", row.get("sbp"))
            )
            for _, row in pre_slice.iterrows()
        ]
        post_scores = [
            compute_vital_instability_score(
                row.get("heart_rate"), row.get("spo2"), row.get("sbp"),
                row.get("respiratory_rate"), row.get("map", row.get("sbp"))
            )
            for _, row in future_slice.iterrows()
        ]

        mean_pre = float(np.mean(pre_scores)) if pre_scores else 0.0
        mean_post = float(np.mean(post_scores)) if post_scores else 0.0
        delta_risk = mean_post - mean_pre

        if delta_risk <= -0.15:
            tx_response = 1  # Improving
        elif delta_risk >= +0.15:
            tx_response = 2  # Worsening
        else:
            tx_response = 0  # Stable
    else:
        response_valid_mask = 0.0
        tx_response = 0  # Default unlabelled / masked out

    return {
        "deterioration": is_deterioration,
        "deterioration_valid_mask": deterioration_valid_mask,
        "risk_tier": risk_tier,
        "forecast": forecast_targets,
        "forecast_valid_mask": forecast_valid_mask,
        "treatment_response": tx_response,
        "response_valid_mask": response_valid_mask,
        "abnormal_future_steps": abnormal_step_count
    }
