"""
Unit Tests for MIMIC-IV Temporal Dataset Construction & Leakage Audits.
Tests cohort extraction, patient-level split segregation, temporal feature engineering,
train-only preprocessor fitting, target extraction, and tensor dimensions.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from ml.temporal.cohort import extract_pulmonary_cohort
from ml.temporal.splitting import create_patient_level_split
from ml.temporal.features import TemporalFeatureEngineer, FULL_FEATURE_LIST
from ml.temporal.preprocessing import ClinicalPreprocessor
from ml.temporal.alignment import TemporalGridAligner
from ml.temporal.targets import extract_targets_for_window, compute_vital_instability_score, FORECAST_CHANNELS
from ml.temporal.sequences import MIMICIVTemporalDataset


def test_cohort_extraction_and_eligibility():
    cohort_df, meta = extract_pulmonary_cohort(min_stay_duration_hours=6.0)
    assert len(cohort_df) > 0
    assert meta["pulmonary_patients_count"] == 62
    assert meta["pulmonary_icustays_count"] == 97
    assert "subject_id" in cohort_df.columns
    assert "stay_id" in cohort_df.columns
    assert "duration_hours" in cohort_df.columns
    assert (cohort_df["duration_hours"] >= 6.0).all()


def test_patient_level_split_leakage():
    cohort_df, _ = extract_pulmonary_cohort(min_stay_duration_hours=6.0)
    split_dict, summary = create_patient_level_split(cohort_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    
    train_p = split_dict["train"]
    val_p = split_dict["validate"]
    test_p = split_dict["test"]

    # Hard mathematical disjointness assertions
    assert len(train_p.intersection(val_p)) == 0
    assert len(train_p.intersection(test_p)) == 0
    assert len(val_p.intersection(test_p)) == 0
    assert summary["leakage_audit"]["passed"] is True
    assert summary["splits"]["train"]["patients_count"] == 43
    assert summary["splits"]["validate"]["patients_count"] == 9
    assert summary["splits"]["test"]["patients_count"] == 10


def test_temporal_feature_engineering_causality():
    engineer = TemporalFeatureEngineer(ema_span=4, rolling_window=4)
    features = engineer.get_feature_names()
    assert len(features) == 45
    assert "shock_index" in features
    assert "delta_spo2" in features
    assert "ema_heart_rate" in features
    # Ensure NEWS2 is strictly NOT in input features
    assert "news2" not in features
    assert "news_score" not in features


def test_train_only_preprocessor():
    preprocessor = ClinicalPreprocessor()
    df_train = pd.DataFrame({
        "subject_id": [1, 1, 1],
        "heart_rate": [80.0, 90.0, 100.0],
        "spo2": [98.0, 96.0, 95.0]
    })
    feature_cols = ["heart_rate", "spo2"]
    preprocessor.fit_on_training_data(df_train, feature_cols, train_patient_ids=[1])

    assert preprocessor.is_fitted is True
    assert preprocessor.channel_means["heart_rate"] == 90.0
    assert preprocessor.fitted_patient_ids == [1]

    # Transform test patient using training parameters
    df_test = pd.DataFrame({
        "subject_id": [2, 2],
        "heart_rate": [90.0, 110.0],
        "spo2": [95.0, np.nan]
    })
    df_norm = preprocessor.transform(df_test, feature_cols)
    assert not df_norm["heart_rate"].isnull().any()
    assert not df_norm["spo2"].isnull().any()
    # At heart_rate=90 (mean), normalized value should be approx 0
    assert df_norm["heart_rate"].iloc[0] == pytest.approx(0.0, abs=1e-3)


def test_target_extraction_shapes():
    timeline = pd.DataFrame({
        "heart_rate": [80.0] * 30,
        "spo2": [98.0] * 30,
        "sbp": [120.0] * 30,
        "map": [85.0] * 30,
        "respiratory_rate": [16.0] * 30,
        "temperature_c": [37.0] * 30
    })
    target = extract_targets_for_window(timeline, pred_idx=23, forecast_horizon_steps=4)
    assert target is not None
    assert target["deterioration"] in [0, 1]
    assert target["risk_tier"] in [0, 1, 2]
    assert target["forecast"].shape == (4, 5)
    assert target["treatment_response"] in [0, 1, 2]


def test_dataset_tensor_integrity():
    X = np.random.randn(10, 24, 45).astype(np.float32)
    y_det = np.zeros(10, dtype=np.float32)
    y_tier = np.zeros(10, dtype=np.int64)
    y_fore = np.random.randn(10, 4, 5).astype(np.float32)
    y_resp = np.zeros(10, dtype=np.int64)
    meta = [{"stay_id": i} for i in range(10)]

    dataset = MIMICIVTemporalDataset(X, y_det, y_tier, y_fore, y_resp, meta)
    assert len(dataset) == 10
    item = dataset[0]
    assert item[0].shape == torch.Size([24, 45])
    assert item[1].shape == torch.Size([])
    assert item[2].shape == torch.Size([])
    assert item[3].shape == torch.Size([4, 5])
    assert item[4].shape == torch.Size([])
