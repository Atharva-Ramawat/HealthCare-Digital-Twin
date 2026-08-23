"""
MIMIC-IV Multi-Task Temporal Target Generation Engine.
Implements mathematically rigorous targets for:
- Head 1: Physiological Deterioration (Binary 0/1 with persistence requirement)
- Head 2: Risk Tier (Low=0, Medium=1, High=2)
- Head 3: Short-Term 4-Step Vital Forecast (Shape [4, 5]: HR, SpO2, SBP, RR, Temp)
- Head 4: Observational Treatment-Response State (Stable=0, Improving=1, Worsening=2)
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

FORECAST_CHANNELS = ["heart_rate", "spo2", "sbp", "respiratory_rate", "temperature_c"]

# Baseline Normal Defaults used strictly for computing fallback risk scores when unobserved
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
    Safely handles missing or non-numeric values.
    """
    # Safe float conversion
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
    # SpO2 penalty
    if spo2_f < 88.0: score += 0.35
    elif spo2_f < 92.0: score += 0.20
    elif spo2_f < 95.0: score += 0.10

    # Hemodynamic penalty
    if map_f < 60.0 or sbp_f < 85.0: score += 0.30
    elif map_f < 65.0 or sbp_f < 90.0: score += 0.20
    elif map_f < 70.0: score += 0.10

    # Respiratory Rate penalty
    if rr_f > 30.0 or rr_f < 8.0: score += 0.25
    elif rr_f > 24.0: score += 0.15
    elif rr_f > 20.0: score += 0.05

    # Heart Rate penalty
    if hr_f > 130.0 or hr_f < 40.0: score += 0.20
    elif hr_f > 110.0 or hr_f < 50.0: score += 0.10

    return min(1.0, score)


def extract_targets_for_window(
    raw_timeline_df: pd.DataFrame,
    pred_idx: int,
    forecast_horizon_steps: int = 4
) -> Optional[Dict[str, Any]]:
    """
    Extract multi-task ground truth targets for an input window ending at pred_idx (time t).
    Future window spans [pred_idx + 1, pred_idx + forecast_horizon_steps].
    """
    total_len = len(raw_timeline_df)
    future_end_idx = pred_idx + forecast_horizon_steps

    # Check if complete future window exists
    if future_end_idx >= total_len:
        return None

    # Slice future window [pred_idx + 1 : pred_idx + 4]
    future_slice = raw_timeline_df.iloc[pred_idx + 1 : future_end_idx + 1].copy()

    # Forward fill any unobserved future forecast values within the slice from the pre-window
    future_filled = future_slice.copy()
    for ch in FORECAST_CHANNELS:
        if ch in future_filled.columns:
            # If still null, backfill from the prediction point
            if future_filled[ch].isnull().any():
                last_val = raw_timeline_df[ch].iloc[:pred_idx + 1].dropna()
                fill_val = last_val.iloc[-1] if len(last_val) > 0 else NORMAL_VITAL_DEFAULTS.get(ch, 0.0)
                future_filled[ch] = future_filled[ch].fillna(fill_val)
        else:
            future_filled[ch] = NORMAL_VITAL_DEFAULTS.get(ch, 0.0)

    # 1. Head 1: Physiological Deterioration Target (Persistence Rule)
    # Check abnormal conditions across the future 4 steps
    hypoxemia = future_filled["spo2"] < 90.0
    map_col = "map" if "map" in future_filled.columns else "sbp"
    hypotension = (future_filled[map_col] < 65.0) | (future_filled["sbp"] < 90.0)
    tachypnea = future_filled["respiratory_rate"] > 28.0
    hr_instability = (future_filled["heart_rate"] < 45.0) | (future_filled["heart_rate"] > 130.0)

    any_abnormal_per_step = hypoxemia | hypotension | tachypnea | hr_instability
    abnormal_step_count = int(any_abnormal_per_step.sum())

    # Persistence requirement: at least 2 abnormal observations in future 4-step window
    is_deterioration = 1 if abnormal_step_count >= 2 else 0

    # 2. Head 2: Risk Tier (0=Low, 1=Medium, 2=High)
    if is_deterioration == 1:
        risk_tier = 2  # High Risk
    elif abnormal_step_count == 1 or np.any(future_filled["spo2"] < 93.0) or np.any(future_filled["respiratory_rate"] > 22.0):
        risk_tier = 1  # Medium Risk
    else:
        risk_tier = 0  # Low Risk

    # 3. Head 3: Short-Term Vital Forecasting (Shape [4, 5])
    forecast_matrix = future_filled[FORECAST_CHANNELS].values.astype(np.float32)  # Shape (4, 5)

    # 4. Head 4: Observational Treatment-Response State (0=Stable, 1=Improving, 2=Worsening)
    # Compare pre-intervention risk [pred_idx - 3, pred_idx] vs future risk
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
        for _, row in future_filled.iterrows()
    ]

    mean_pre_risk = float(np.mean(pre_scores)) if pre_scores else 0.0
    mean_post_risk = float(np.mean(post_scores)) if post_scores else 0.0
    delta_risk = mean_post_risk - mean_pre_risk

    if delta_risk <= -0.15:
        tx_response_state = 1  # Improving
    elif delta_risk >= +0.15:
        tx_response_state = 2  # Worsening
    else:
        tx_response_state = 0  # Stable

    return {
        "deterioration": is_deterioration,
        "risk_tier": risk_tier,
        "forecast": forecast_matrix,
        "treatment_response": tx_response_state,
        "abnormal_future_steps": abnormal_step_count,
        "pre_risk": round(mean_pre_risk, 3),
        "post_risk": round(mean_post_risk, 3),
        "delta_risk": round(delta_risk, 3)
    }
