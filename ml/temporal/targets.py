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


def compute_vital_instability_score(
    hr: float,
    spo2: float,
    sbp: float,
    rr: float,
    map_val: float
) -> float:
    """
    Compute a continuous physiological instability index in [0.0, 1.0] from raw vitals.
    """
    score = 0.0
    # SpO2 penalty
    if spo2 < 88.0: score += 0.35
    elif spo2 < 92.0: score += 0.20
    elif spo2 < 95.0: score += 0.10

    # Hemodynamic penalty
    if map_val < 60.0 or sbp < 85.0: score += 0.30
    elif map_val < 65.0 or sbp < 90.0: score += 0.20
    elif map_val < 70.0: score += 0.10

    # Respiratory Rate penalty
    if rr > 30.0 or rr < 8.0: score += 0.25
    elif rr > 24.0: score += 0.15
    elif rr > 20.0: score += 0.05

    # Heart Rate penalty
    if hr > 130.0 or hr < 40.0: score += 0.20
    elif hr > 110.0 or hr < 50.0: score += 0.10

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

    future_slice = raw_timeline_df.iloc[pred_idx + 1 : future_end_idx + 1]

    # 1. Head 1: Physiological Deterioration Target (Persistence Rule)
    # Check abnormal conditions across the future 4 steps
    hypoxemia = future_slice["spo2"] < 90.0
    hypotension = (future_slice["map"] < 65.0) | (future_slice["sbp"] < 90.0)
    tachypnea = future_slice["respiratory_rate"] > 28.0
    hr_instability = (future_slice["heart_rate"] < 45.0) | (future_slice["heart_rate"] > 130.0)

    any_abnormal_per_step = hypoxemia | hypotension | tachypnea | hr_instability
    abnormal_step_count = int(any_abnormal_per_step.sum())

    # Persistence requirement: at least 2 abnormal observations in future 4-step window
    is_deterioration = 1 if abnormal_step_count >= 2 else 0

    # 2. Head 2: Risk Tier (0=Low, 1=Medium, 2=High)
    if is_deterioration == 1:
        risk_tier = 2  # High Risk
    elif abnormal_step_count == 1 or np.any(future_slice["spo2"] < 93.0) or np.any(future_slice["respiratory_rate"] > 22.0):
        risk_tier = 1  # Medium Risk
    else:
        risk_tier = 0  # Low Risk

    # 3. Head 3: Short-Term Vital Forecasting (Shape [4, 5])
    # Extract future values for HR, SpO2, SBP, RR, Temp
    forecast_matrix = future_slice[FORECAST_CHANNELS].values.astype(np.float32)  # Shape (4, 5)

    # 4. Head 4: Observational Treatment-Response State (0=Stable, 1=Improving, 2=Worsening)
    # Compare pre-intervention risk [pred_idx - 3, pred_idx] vs future risk
    pre_start = max(0, pred_idx - 3)
    pre_slice = raw_timeline_df.iloc[pre_start : pred_idx + 1]

    pre_scores = [
        compute_vital_instability_score(row["heart_rate"], row["spo2"], row["sbp"], row["respiratory_rate"], row["map"])
        for _, row in pre_slice.iterrows()
    ]
    post_scores = [
        compute_vital_instability_score(row["heart_rate"], row["spo2"], row["sbp"], row["respiratory_rate"], row["map"])
        for _, row in future_slice.iterrows()
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
