"""
synthetic_generator.py
-----------------------
Phase 1 — Synthetic clinical data generator.

WARNING
-------
Synthetic trajectories are NOT clinical simulations and must NOT be
interpreted as medically valid disease progression.

Purpose: infrastructure testing and schema validation ONLY.

Trajectories:
  normal               - stable baseline with realistic noise
  gradual_deterioration - slow drift toward abnormal values over time
  acute_deterioration  - rapid multi-variable deterioration (stress test)
  missingness_stress   - high missingness to test imputation pipelines

All outputs carry:
  data_source = 'SYNTHETIC — DEVELOPMENT / TESTING ONLY'

NEVER mix synthetic outputs with real MIMIC-IV analysis results.
Synthetic data must NEVER be used for clinical performance claims.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Mirrors MIMIC-IV candidate item IDs so synthetic DataFrames
# pass through the same downstream infrastructure as real chartevents rows.
SYNTHETIC_ITEM_IDS: dict[str, int] = {
    "heart_rate":       220045,
    "sbp_noninvasive":  220179,
    "dbp_noninvasive":  220180,
    "map_noninvasive":  220181,
    "spo2":             220277,
    "respiratory_rate": 220210,
    "temperature_c":    223762,
    "glucose_chart":    220621,
}

# Engineering bounds for test-data generation.
# These do NOT constitute clinical guidance.
_VITAL_SPECS: dict[str, dict] = {
    "heart_rate":       {"mu": 75.0,  "sigma": 8.0,  "unit": "bpm",         "lo": 40.0,  "hi": 180.0},
    "sbp_noninvasive":  {"mu": 120.0, "sigma": 12.0, "unit": "mmHg",        "lo": 70.0,  "hi": 220.0},
    "dbp_noninvasive":  {"mu": 75.0,  "sigma": 8.0,  "unit": "mmHg",        "lo": 40.0,  "hi": 130.0},
    "map_noninvasive":  {"mu": 90.0,  "sigma": 9.0,  "unit": "mmHg",        "lo": 50.0,  "hi": 150.0},
    "spo2":             {"mu": 97.0,  "sigma": 1.5,  "unit": "%",           "lo": 80.0,  "hi": 100.0},
    "respiratory_rate": {"mu": 16.0,  "sigma": 3.0,  "unit": "breaths/min", "lo": 6.0,   "hi": 50.0},
    "temperature_c":    {"mu": 37.0,  "sigma": 0.4,  "unit": "degC",        "lo": 34.0,  "hi": 40.5},
    "glucose_chart":    {"mu": 110.0, "sigma": 20.0, "unit": "mg/dL",       "lo": 40.0,  "hi": 500.0},
}

VALID_TRAJECTORIES: frozenset[str] = frozenset({
    "normal",
    "gradual_deterioration",
    "acute_deterioration",
    "missingness_stress",
})

_DISCLAIMER = "SYNTHETIC \u2014 DEVELOPMENT / TESTING ONLY"


class SyntheticClinicalGenerator:
    """
    Generates synthetic ICU vital-sign time series in MIMIC-IV chartevents schema.

    Output DataFrames share the same column structure as
    MIMICIVLoader.load_chartevents() so they flow through the same
    downstream infrastructure without modification.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_patient_series(
        self,
        patient_id: int = 10_000_001,
        hadm_id: int = 20_000_001,
        stay_id: int = 30_000_001,
        trajectory: str = "normal",
        num_steps: int = 200,
        step_minutes: int = 60,
        start_time: Optional[pd.Timestamp] = None,
        missing_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate a single ICU stay as a chartevents-compatible DataFrame.

        Start time defaults to 2100-01-01 to make synthetic records
        visually distinct from any real MIMIC-IV timestamps.

        Returns
        -------
        pd.DataFrame with columns matching MIMIC-IV chartevents schema:
          subject_id, hadm_id, stay_id, itemid, charttime, storetime,
          value, valuenum, valueuom, canonical_name,
          data_source (= SYNTHETIC disclaimer), trajectory
        """
        if trajectory not in VALID_TRAJECTORIES:
            raise ValueError(
                f"Unknown trajectory '{trajectory}'. "
                f"Valid: {sorted(VALID_TRAJECTORIES)}"
            )
        if start_time is None:
            start_time = pd.Timestamp("2100-01-01")
        if trajectory == "missingness_stress":
            missing_rate = max(missing_rate, 0.60)

        records = []
        for vital, spec in _VITAL_SPECS.items():
            ts_list, vals = self._generate_trajectory(
                spec=spec,
                trajectory=trajectory,
                num_steps=num_steps,
                step_minutes=step_minutes,
                start_time=start_time,
            )
            item_id = SYNTHETIC_ITEM_IDS[vital]
            for ts, v in zip(ts_list, vals):
                is_missing = self._rng.random() < missing_rate
                records.append({
                    "subject_id":     patient_id,
                    "hadm_id":        hadm_id,
                    "stay_id":        stay_id,
                    "itemid":         item_id,
                    "charttime":      ts,
                    "storetime":      ts,
                    "value":          None if is_missing else str(round(float(v), 2)),
                    "valuenum":       np.nan if is_missing else float(v),
                    "valueuom":       spec["unit"],
                    "canonical_name": vital,
                    "data_source":    _DISCLAIMER,
                    "trajectory":     trajectory,
                })

        df = pd.DataFrame.from_records(records)
        df = df.sort_values(["charttime", "itemid"]).reset_index(drop=True)
        return df

    def _generate_trajectory(
        self,
        spec: dict,
        trajectory: str,
        num_steps: int,
        step_minutes: int,
        start_time: pd.Timestamp,
    ) -> tuple[list[pd.Timestamp], np.ndarray]:
        mu, sigma, lo, hi = spec["mu"], spec["sigma"], spec["lo"], spec["hi"]

        # Realistic irregular timestamps (jitter ~10% of step_minutes)
        jitter = self._rng.normal(0, step_minutes * 0.1, size=num_steps)
        deltas = np.maximum(1, step_minutes + jitter).cumsum().astype(int)
        timestamps = [start_time + pd.Timedelta(minutes=int(d)) for d in deltas]

        if trajectory == "normal":
            vals = self._rng.normal(mu, sigma, num_steps)

        elif trajectory == "gradual_deterioration":
            drift = np.linspace(0, (hi - mu) * 0.45, num_steps)
            noise = self._rng.normal(0, sigma, num_steps)
            # HR and RR drift upward; BP and SpO2 drift downward
            if vital_is_increasing(spec["mu"]):
                vals = mu + drift + noise
            else:
                vals = mu - drift * 0.5 + noise

        elif trajectory == "acute_deterioration":
            pivot = int(num_steps * 0.70)
            stable = self._rng.normal(mu, sigma, pivot)
            if vital_is_increasing(spec["mu"]):
                deteri = self._rng.normal(hi * 0.85, sigma * 2, num_steps - pivot)
            elif spec["mu"] == 97.0:  # SpO2
                deteri = self._rng.normal(lo + 5, sigma * 2, num_steps - pivot)
            else:
                deteri = self._rng.normal(
                    lo + (mu - lo) * 0.25, sigma * 2, num_steps - pivot
                )
            vals = np.concatenate([stable, deteri])

        else:  # missingness_stress — normal values; missing applied externally
            vals = self._rng.normal(mu, sigma, num_steps)

        vals = np.clip(vals, lo, hi)
        return timestamps, vals

    def generate_ward_dataset(
        self,
        n_patients: int = 5,
        trajectory_mix: Optional[dict[str, float]] = None,
        num_steps: int = 200,
        step_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        Generate a multi-patient synthetic ward dataset.

        Issues UserWarning so callers are aware of the synthetic nature.
        All rows carry data_source = SYNTHETIC disclaimer.
        """
        warnings.warn(
            f"{_DISCLAIMER} \u2014 Do not use for research claims.",
            UserWarning,
            stacklevel=2,
        )
        if trajectory_mix is None:
            trajs = sorted(VALID_TRAJECTORIES)
            trajectory_mix = {t: 1.0 / len(trajs) for t in trajs}

        keys = list(trajectory_mix)
        total = sum(trajectory_mix.values())
        weights = [trajectory_mix[k] / total for k in keys]
        chosen = self._rng.choice(keys, size=n_patients, p=weights)

        frames = []
        for i, traj in enumerate(chosen):
            frames.append(self.generate_patient_series(
                patient_id=10_000_000 + i,
                hadm_id=20_000_000 + i,
                stay_id=30_000_000 + i,
                trajectory=str(traj),
                num_steps=num_steps,
                step_minutes=step_minutes,
            ))
        return pd.concat(frames, ignore_index=True)


def vital_is_increasing(mu: float) -> bool:
    """Helper: HR (75) and RR (16) drift up during deterioration."""
    return mu in (75.0, 16.0)
