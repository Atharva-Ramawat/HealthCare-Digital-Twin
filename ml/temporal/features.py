"""
MIMIC-IV Temporal Feature Engineering & Feature Registry.
Extracts raw physiological channels, missingness masks, treatment indicators,
temporal deltas, Exponential Moving Average (EMA) baselines, and rolling volatility.

MANDATORY CONSTRAINT: NEWS 2 is strictly excluded from model input features to prevent target leakage.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Canonical Feature Registry for CNN-BiLSTM Input Tensors
RAW_VITAL_FEATURES = [
    "heart_rate", "spo2", "respiratory_rate", "sbp", "dbp", "map", "temperature_c"
]

RAW_LAB_FEATURES = [
    "wbc", "hemoglobin", "lactate", "creatinine", "glucose", "po2", "pco2"
]

TREATMENT_FLAG_FEATURES = [
    "tx_vasopressor", "tx_diuretic", "tx_antibiotic", "tx_bronchodilator", "tx_steroid"
]

MISSINGNESS_MASK_FEATURES = [
    f"mask_{feat}" for feat in (RAW_VITAL_FEATURES + RAW_LAB_FEATURES)
]

DERIVED_TEMPORAL_FEATURES = [
    # First-order rate-of-change (Delta)
    "delta_heart_rate", "delta_spo2", "delta_respiratory_rate", "delta_map",
    # Exponential Moving Average (EMA) Baseline
    "ema_heart_rate", "ema_spo2", "ema_respiratory_rate", "ema_map",
    # Rolling Volatility (Standard Deviation)
    "roll_std_heart_rate", "roll_std_spo2", "roll_std_map",
    # Clinical Ratio
    "shock_index"
]

# Full Canonical Feature List (in fixed index order)
FULL_FEATURE_LIST = (
    RAW_VITAL_FEATURES +
    RAW_LAB_FEATURES +
    TREATMENT_FLAG_FEATURES +
    MISSINGNESS_MASK_FEATURES +
    DERIVED_TEMPORAL_FEATURES
)


class TemporalFeatureEngineer:
    """
    Computes temporal features from aligned ICU timelines without future information leakage.
    """

    def __init__(self, ema_span: int = 4, rolling_window: int = 4):
        self.ema_span = ema_span
        self.rolling_window = rolling_window

    def engineer_features(self, aligned_df: pd.DataFrame, treatment_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute temporal features strictly using causal (backward-looking) operations.
        """
        df = aligned_df.copy()

        # 1. Merge Treatment Flags
        if treatment_df is not None and not treatment_df.empty:
            df = pd.merge(df, treatment_df, on="charttime", how="left")
        for tx_col in TREATMENT_FLAG_FEATURES:
            if tx_col not in df.columns:
                df[tx_col] = 0.0
            else:
                df[tx_col] = df[tx_col].fillna(0.0).astype(float)

        # 2. Compute Shock Index (HR / SBP)
        if "heart_rate" in df.columns and "sbp" in df.columns:
            sbp_safe = df["sbp"].replace(0, np.nan)
            df["shock_index"] = df["heart_rate"] / sbp_safe
        else:
            df["shock_index"] = np.nan

        # 3. Compute Rate of Change (Deltas) — backward difference only
        for col in ["heart_rate", "spo2", "respiratory_rate", "map"]:
            if col in df.columns:
                df[f"delta_{col}"] = df[col].diff().fillna(0.0)
            else:
                df[f"delta_{col}"] = 0.0

        # 4. Compute Exponential Moving Average (EMA) — causal only
        for col in ["heart_rate", "spo2", "respiratory_rate", "map"]:
            if col in df.columns:
                df[f"ema_{col}"] = df[col].ewm(span=self.ema_span, adjust=False).mean()
            else:
                df[f"ema_{col}"] = df[col] if col in df.columns else np.nan

        # 5. Compute Rolling Volatility (Standard Deviation) — backward window only
        for col in ["heart_rate", "spo2", "map"]:
            if col in df.columns:
                df[f"roll_std_{col}"] = df[col].rolling(window=self.rolling_window, min_periods=1).std().fillna(0.0)
            else:
                df[f"roll_std_{col}"] = 0.0

        return df

    @staticmethod
    def get_feature_names() -> List[str]:
        return list(FULL_FEATURE_LIST)


if __name__ == "__main__":
    eng = TemporalFeatureEngineer()
    features = eng.get_feature_names()
    print(f"Temporal Feature Registry: {len(features)} total features.")
    print("Features:", features)
