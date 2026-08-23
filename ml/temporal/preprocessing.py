"""
MIMIC-IV Physiological Preprocessing & Train-Only Normalization Engine.
Enforces physiological bounds validation, handles missingness via training population medians,
and fits normalization scalers STRICTLY on the training patient split to guarantee zero data leakage.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Physiological Plausibility Limits for Outlier Rejection & Clipping
PHYSIOLOGICAL_BOUNDS = {
    "heart_rate": (20.0, 250.0),
    "spo2": (50.0, 100.0),
    "respiratory_rate": (4.0, 60.0),
    "sbp": (40.0, 300.0),
    "dbp": (20.0, 200.0),
    "map": (30.0, 220.0),
    "temperature_c": (30.0, 44.0),
    "wbc": (0.1, 150.0),
    "hemoglobin": (2.0, 25.0),
    "lactate": (0.1, 30.0),
    "creatinine": (0.1, 25.0),
    "glucose": (10.0, 1000.0),
    "po2": (20.0, 500.0),
    "pco2": (10.0, 150.0),
    "shock_index": (0.1, 5.0),
}


class ClinicalPreprocessor:
    """
    Fits normalization statistics strictly on training patients and transforms patient timelines.
    """

    def __init__(self, scaler_path: Optional[str] = None):
        self.scaler_path = scaler_path or os.path.join(PROJECT_ROOT, "models", "checkpoints", "temporal_scaler.json")
        self.is_fitted = False
        self.channel_means: Dict[str, float] = {}
        self.channel_stds: Dict[str, float] = {}
        self.channel_medians: Dict[str, float] = {}
        self.fitted_patient_ids: List[int] = []

    def fit_on_training_data(
        self,
        train_features_df: pd.DataFrame,
        feature_columns: List[str],
        train_patient_ids: List[int]
    ):
        """
        Fit mean, standard deviation, and median strictly on the training set.
        """
        self.fitted_patient_ids = list(train_patient_ids)
        self.channel_means = {}
        self.channel_stds = {}
        self.channel_medians = {}

        # Filter strictly training patients
        train_df = train_features_df[train_features_df["subject_id"].isin(train_patient_ids)].copy()

        # Apply physiological bounds clipping
        for col, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
            if col in train_df.columns:
                train_df[col] = train_df[col].clip(lower=low, upper=high)

        for col in feature_columns:
            if col in train_df.columns:
                valid_vals = train_df[col].dropna()
                if len(valid_vals) > 0:
                    m = float(valid_vals.mean())
                    s = float(valid_vals.std()) if valid_vals.std() > 1e-6 else 1.0
                    med = float(valid_vals.median())
                else:
                    m, s, med = 0.0, 1.0, 0.0
            else:
                m, s, med = 0.0, 1.0, 0.0

            self.channel_means[col] = round(m, 6)
            self.channel_stds[col] = round(s, 6)
            self.channel_medians[col] = round(med, 6)

        self.is_fitted = True
        self.save_scaler()

    def transform(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Clip bounds, impute unobserved initial values with training median, and standardize (z-score).
        Missingness masks and binary treatment flags are preserved in {0, 1}.
        """
        if not self.is_fitted:
            raise RuntimeError("ClinicalPreprocessor must be fitted on training patients before transform.")

        df = features_df.copy()

        # 1. Physiological Bounds Clipping
        for col, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=low, upper=high)

        # 2. Impute unobserved initial values using training set population median
        for col in feature_columns:
            if col in df.columns:
                train_med = self.channel_medians.get(col, 0.0)
                df[col] = df[col].fillna(train_med)
            else:
                df[col] = self.channel_medians.get(col, 0.0)

        # 3. Standardize continuous channels (preserve masks & treatment flags as {0, 1})
        continuous_cols = [c for c in feature_columns if not c.startswith("mask_") and not c.startswith("tx_")]
        for col in continuous_cols:
            m = self.channel_means.get(col, 0.0)
            s = self.channel_stds.get(col, 1.0)
            df[col] = (df[col] - m) / (s + 1e-6)

        return df

    def save_scaler(self):
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        data = {
            "is_fitted": self.is_fitted,
            "fitted_train_patient_count": len(self.fitted_patient_ids),
            "fitted_patient_ids": self.fitted_patient_ids,
            "channel_means": self.channel_means,
            "channel_stds": self.channel_stds,
            "channel_medians": self.channel_medians
        }
        with open(self.scaler_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_scaler(self):
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, "r") as f:
                data = json.load(f)
            self.is_fitted = data.get("is_fitted", False)
            self.channel_means = data.get("channel_means", {})
            self.channel_stds = data.get("channel_stds", {})
            self.channel_medians = data.get("channel_medians", {})
            self.fitted_patient_ids = data.get("fitted_patient_ids", [])
