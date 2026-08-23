"""
MIMIC-IV Temporal Resampling & Grid Alignment Engine.
Aligns irregular vital signs and laboratory events to a uniform time grid (default 15 minutes),
enforcing variable-specific carry-forward limits and generating explicit missingness masks.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Variable-Specific Maximum Carry-Forward Hold Limits (in hours and grid steps at 15-min resolution)
VARIABLE_FORWARD_FILL_LIMITS_HOURS = {
    "heart_rate": 2.0,         # 8 steps @ 15m
    "spo2": 2.0,               # 8 steps @ 15m
    "respiratory_rate": 2.0,   # 8 steps @ 15m
    "sbp": 2.0,                # 8 steps @ 15m
    "dbp": 2.0,                # 8 steps @ 15m
    "map": 2.0,                # 8 steps @ 15m
    "temperature_c": 6.0,      # 24 steps @ 15m
    "wbc": 24.0,               # 96 steps @ 15m
    "hemoglobin": 24.0,        # 96 steps @ 15m
    "lactate": 12.0,           # 48 steps @ 15m
    "creatinine": 24.0,        # 96 steps @ 15m
    "glucose": 12.0,           # 48 steps @ 15m
    "po2": 12.0,               # 48 steps @ 15m
    "pco2": 12.0,              # 48 steps @ 15m
}

VITAL_ITEMID_MAP = {
    220045: "heart_rate",
    220277: "spo2",
    220210: "respiratory_rate",
    220179: "sbp_ni",
    220050: "sbp_art",
    220180: "dbp_ni",
    220051: "dbp_art",
    220181: "map_ni",
    220052: "map_art",
    223762: "temp_c",
    223761: "temp_f"
}

LAB_ITEMID_MAP = {
    51301: "wbc",
    51222: "hemoglobin",
    50813: "lactate",
    50912: "creatinine",
    50931: "glucose",
    50821: "po2",
    50818: "pco2"
}


class TemporalGridAligner:
    """
    Aligns irregularly sampled clinical time-series onto a uniform time grid.
    """

    def __init__(
        self,
        grid_resolution_minutes: int = 15,
        forward_fill_limits_hours: Optional[Dict[str, float]] = None
    ):
        self.grid_resolution_minutes = grid_resolution_minutes
        self.freq_str = f"{grid_resolution_minutes}min"
        self.ffill_limits = forward_fill_limits_hours or VARIABLE_FORWARD_FILL_LIMITS_HOURS

    def align_stay_timeline(
        self,
        stay_id: int,
        subject_id: int,
        intime: pd.Timestamp,
        outtime: pd.Timestamp,
        chartevents_df: pd.DataFrame,
        labevents_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Produce a uniform resampled timeline for a single ICU stay.
        
        Returns:
            aligned_df: DataFrame with raw + imputed physiological variables,
                        missingness indicator channels ('mask_{var}'), and timestamps.
        """
        # 1. Establish Uniform Time Grid
        grid_index = pd.date_range(start=intime, end=outtime, freq=self.freq_str)
        if len(grid_index) < 2:
            return pd.DataFrame()

        grid_df = pd.DataFrame({"charttime": grid_index})
        grid_df["stay_id"] = stay_id
        grid_df["subject_id"] = subject_id

        # 2. Extract & Pivot Vitals
        stay_charts = chartevents_df[chartevents_df["stay_id"] == stay_id].copy() if not chartevents_df.empty else pd.DataFrame()
        vitals_pivoted = pd.DataFrame(index=grid_index)

        if not stay_charts.empty:
            stay_charts = stay_charts[stay_charts["itemid"].isin(VITAL_ITEMID_MAP.keys())].copy()
            stay_charts["vital_name"] = stay_charts["itemid"].map(VITAL_ITEMID_MAP)
            
            # Unit conversions
            is_temp_f = stay_charts["vital_name"] == "temp_f"
            stay_charts.loc[is_temp_f, "valuenum"] = (stay_charts.loc[is_temp_f, "valuenum"] - 32.0) * 5.0 / 9.0
            stay_charts.loc[is_temp_f, "vital_name"] = "temp_c"

            # Merge SBP, DBP, MAP from arterial and non-invasive
            stay_charts["vital_clean"] = stay_charts["vital_name"].replace({
                "sbp_ni": "sbp", "sbp_art": "sbp",
                "dbp_ni": "dbp", "dbp_art": "dbp",
                "map_ni": "map", "map_art": "map",
                "temp_c": "temperature_c"
            })

            stay_charts["charttime"] = pd.to_datetime(stay_charts["charttime"]).dt.floor(self.freq_str)
            piv = stay_charts.pivot_table(index="charttime", columns="vital_clean", values="valuenum", aggfunc="mean")
            vitals_pivoted = piv

        # 3. Extract & Pivot Labs
        stay_labs = labevents_df[labevents_df["subject_id"] == subject_id].copy() if not labevents_df.empty else pd.DataFrame()
        labs_pivoted = pd.DataFrame(index=grid_index)

        if not stay_labs.empty:
            stay_labs = stay_labs[stay_labs["itemid"].isin(LAB_ITEMID_MAP.keys())].copy()
            stay_labs["lab_clean"] = stay_labs["itemid"].map(LAB_ITEMID_MAP)
            stay_labs["charttime"] = pd.to_datetime(stay_labs["charttime"]).dt.floor(self.freq_str)
            piv_lab = stay_labs.pivot_table(index="charttime", columns="lab_clean", values="valuenum", aggfunc="mean")
            labs_pivoted = piv_lab

        # 4. Merge onto Grid
        merged = grid_df.set_index("charttime")
        merged = merged.join(vitals_pivoted, how="left").join(labs_pivoted, how="left")

        # 5. Apply Variable-Specific Forward Filling with Missingness Masks
        all_channels = [
            "heart_rate", "spo2", "respiratory_rate", "sbp", "dbp", "map", "temperature_c",
            "wbc", "hemoglobin", "lactate", "creatinine", "glucose", "po2", "pco2"
        ]

        for col in all_channels:
            if col not in merged.columns:
                merged[col] = np.nan

            # Generate explicit binary missingness indicator: 1 = directly observed, 0 = missing/imputed
            mask_col = f"mask_{col}"
            merged[mask_col] = merged[col].notnull().astype(float)

            # Apply bounded forward-fill
            limit_hours = self.ffill_limits.get(col, 2.0)
            limit_steps = int(np.ceil((limit_hours * 60.0) / self.grid_resolution_minutes))
            merged[col] = merged[col].ffill(limit=limit_steps)

        merged = merged.reset_index()
        return merged
