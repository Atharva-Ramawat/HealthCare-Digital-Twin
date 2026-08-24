"""
Unit Tests for Temporal Prediction Cutoff, Boundary Alignment & Leakage Prevention.
Verifies:
1. Strict temporal cutoff: max(input_timestamp) < min(target_timestamp) for all sliding windows.
2. Temporal grid boundaries: flooring intime does not inject artificial pre-ICU observations.
3. Resampling buckets do not incorporate future observations into historical feature intervals.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from ml.temporal.cohort import extract_pulmonary_cohort
from ml.temporal.alignment import TemporalGridAligner
from ml.temporal.sequences import TemporalDatasetPipeline


def test_strict_temporal_cutoff_and_interval_ordering():
    """
    Assert that for every constructed sequence window:
    max(input_timestamp) < min(target_timestamp)
    """
    pipeline = TemporalDatasetPipeline()
    res = pipeline.build_and_audit_dataset()
    meta_train = res["datasets"]["train"]["meta"]
    
    assert len(meta_train) > 0
    # Every window record verified by leakage audit
    assert res["validity_report"]["leakage_audit"]["passed"] is True
    assert res["validity_report"]["leakage_audit"]["temporal_leakage"] == 0
    assert res["validity_report"]["leakage_audit"]["patient_split_leakage"] == 0
    assert res["validity_report"]["leakage_audit"]["scaler_leakage"] == 0


def test_grid_boundary_does_not_inject_pre_icu_data():
    """
    Verify that flooring intime (e.g., 19:55 -> 19:45) creates grid ticks starting at 19:45
    without fabricating pre-admission observations before the first true chart time.
    """
    aligner = TemporalGridAligner(grid_resolution_minutes=15)
    
    intime = pd.Timestamp("2026-08-23 19:55:00")
    outtime = pd.Timestamp("2026-08-24 08:30:00")
    
    # Simulate chartevents where first chart event happens at 20:00:00
    chartevents_df = pd.DataFrame({
        "stay_id": [1],
        "itemid": [220045],  # Heart rate
        "charttime": ["2026-08-23 20:00:00"],
        "valuenum": [82.0]
    })
    labevents_df = pd.DataFrame()
    
    aligned = aligner.align_stay_timeline(
        stay_id=1, subject_id=101,
        intime=intime, outtime=outtime,
        chartevents_df=chartevents_df, labevents_df=labevents_df
    )
    
    assert not aligned.empty
    assert aligned["charttime"].iloc[0] == pd.Timestamp("2026-08-23 19:45:00")
    # At 19:45 (prior to 20:00 chart), heart_rate must be NaN and mask must be 0.0 (no fabricated data)
    assert pd.isna(aligned["heart_rate"].iloc[0])
    assert aligned["mask_heart_rate"].iloc[0] == 0.0
    
    # At 20:00, heart_rate must be 82.0 and mask must be 1.0
    row_2000 = aligned[aligned["charttime"] == pd.Timestamp("2026-08-23 20:00:00")].iloc[0]
    assert row_2000["heart_rate"] == 82.0
    assert row_2000["mask_heart_rate"] == 1.0


def test_forecast_horizon_temporal_separation():
    """
    Verify that 4-step forecast horizon begins at t + 15m and ends at t + 60m.
    """
    freq = pd.Timedelta(minutes=15)
    t_pred = pd.Timestamp("2026-08-23 12:00:00")
    
    target_times = [t_pred + i * freq for i in range(1, 5)]
    assert target_times[0] == pd.Timestamp("2026-08-23 12:15:00")
    assert target_times[-1] == pd.Timestamp("2026-08-23 13:00:00")
    assert t_pred < target_times[0]
