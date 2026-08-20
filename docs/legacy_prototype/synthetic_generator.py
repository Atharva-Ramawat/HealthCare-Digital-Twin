"""
Fallback Data Engine - Synthetic Clinical Data Generator
Generates realistic physiological vital time-series (HR, SpO2, SBP, RR, Temp)
using mathematical sine waves, circadian variations, baseline drift, and Gaussian noise.
Includes trajectory templates for Normal Stable, Septic Shock, and ARDS sequences.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
import sys

# Ensure local imports work regardless of execution location
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


class SyntheticClinicalDataGenerator:
    """
    Synthetic Clinical Data Engine for emulating real-time patient physiological vitals
    without physical hardware or IoT devices.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_patient_series(
        self,
        patient_id: str = "PAT-101",
        trajectory_type: str = "normal",
        num_steps: int = 200,
        start_time_offset: int = 0
    ) -> pd.DataFrame:
        """
        Generate time-series vitals for a specific patient trajectory sequence.
        
        Trajectories:
        - 'normal': Stable physiological parameters with natural variance.
        - 'septic_shock': Progressive Septic Shock (HR ↑, SBP ↓, Temp ↑, RR ↑).
        - 'ards': Acute Respiratory Distress Sequence (SpO2 ↓, RR ↑, HR ↑).
        """
        t = np.arange(num_steps) + start_time_offset

        # 1. Base Sine Wave for Circadian Rhythm (period ~ 120 steps)
        circadian = np.sin(2 * np.pi * t / 120.0)

        # 2. Gaussian Noise & Random Walk Drift
        def random_walk(std: float, scale: float = 0.3) -> np.ndarray:
            rw = np.cumsum(np.random.normal(0, scale, num_steps))
            return rw - np.mean(rw)

        # Initialize Vitals Baselines from Config
        hr_base = config.VITALS_META['heart_rate']['normal_baseline']
        spo2_base = config.VITALS_META['spo2']['normal_baseline']
        sbp_base = config.VITALS_META['systolic_bp']['normal_baseline']
        rr_base = config.VITALS_META['respiratory_rate']['normal_baseline']
        temp_base = config.VITALS_META['body_temperature']['normal_baseline']

        # Normal fluctuations
        hr = hr_base + 3.0 * circadian + random_walk(1.0) + np.random.normal(0, 1.5, num_steps)
        spo2 = spo2_base - 0.3 * np.abs(circadian) + np.random.normal(0, 0.3, num_steps)
        sbp = sbp_base + 4.0 * circadian + random_walk(1.5) + np.random.normal(0, 2.0, num_steps)
        rr = rr_base + 0.8 * circadian + np.random.normal(0, 0.8, num_steps)
        temp = temp_base + 0.15 * circadian + np.random.normal(0, 0.08, num_steps)

        # Apply Pathological Trajectory Distortions
        onset = int(num_steps * 0.35)  # Shock/Deterioration starts after 35% time-steps
        if trajectory_type.lower() == "septic_shock":
            ramp = np.zeros(num_steps)
            ramp[onset:] = np.linspace(0, 1, num_steps - onset) ** 1.2

            hr += ramp * 55.0               # HR spikes (e.g., 75 -> 130+ bpm)
            sbp -= ramp * 42.0              # SBP drops sharply (e.g., 120 -> 78 mmHg)
            temp += ramp * 2.6              # Temp spikes (e.g., 37.0 -> 39.6 °C)
            rr += ramp * 14.0               # RR increases (e.g., 16 -> 30 breaths/min)
            spo2 -= ramp * 5.0              # Mild secondary SpO2 drop

        elif trajectory_type.lower() == "ards":
            ramp = np.zeros(num_steps)
            ramp[onset:] = np.linspace(0, 1, num_steps - onset) ** 0.9

            spo2 -= ramp * 16.0             # Severe SpO2 drop (e.g., 98% -> 82%)
            rr += ramp * 20.0               # Severe Tachypnea (e.g., 16 -> 36 breaths/min)
            hr += ramp * 35.0               # Compensatory Tachycardia (e.g., 75 -> 110 bpm)
            sbp += ramp * 15.0 - (ramp**2) * 25.0  # Initial transient rise then drop
            temp += ramp * 0.8              # Mild inflammatory temp rise

        # Clip values to physiological realistic boundaries
        hr = np.clip(hr, config.VITALS_META['heart_rate']['min_val'], config.VITALS_META['heart_rate']['max_val'])
        spo2 = np.clip(spo2, config.VITALS_META['spo2']['min_val'], config.VITALS_META['spo2']['max_val'])
        sbp = np.clip(sbp, config.VITALS_META['systolic_bp']['min_val'], config.VITALS_META['systolic_bp']['max_val'])
        rr = np.clip(rr, config.VITALS_META['respiratory_rate']['min_val'], config.VITALS_META['respiratory_rate']['max_val'])
        temp = np.clip(temp, config.VITALS_META['body_temperature']['min_val'], config.VITALS_META['body_temperature']['max_val'])

        timestamps = pd.date_range(start="2026-08-10 08:00:00", periods=num_steps, freq="1min")

        df = pd.DataFrame({
            'timestamp': timestamps,
            'step': t,
            'patient_id': patient_id,
            'heart_rate': np.round(hr, 2),
            'spo2': np.round(spo2, 2),
            'systolic_bp': np.round(sbp, 2),
            'respiratory_rate': np.round(rr, 2),
            'body_temperature': np.round(temp, 2),
            'trajectory_type': trajectory_type
        })

        return df

    def generate_ward_dataset(self, num_steps: int = 300) -> pd.DataFrame:
        """
        Generate datasets for all active ICU Ward patients defined in config.
        """
        all_dfs = []
        for i, profile in enumerate(config.PATIENT_PROFILES):
            df = self.generate_patient_series(
                patient_id=profile['patient_id'],
                trajectory_type=profile['trajectory'],
                num_steps=num_steps,
                start_time_offset=i * 10
            )
            all_dfs.append(df)
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df

    def save_to_csv(self, output_path: str, num_steps: int = 300) -> str:
        """
        Save synthesized patient ward data to CSV file.
        """
        df = self.generate_ward_dataset(num_steps=num_steps)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SyntheticGenerator] Dataset saved successfully to {output_path} ({len(df)} records)")
        return output_path


if __name__ == "__main__":
    generator = SyntheticClinicalDataGenerator(seed=42)
    output_file = os.path.join(os.path.dirname(__file__), "data", "synthetic_clinical_data.csv")
    generator.save_to_csv(output_file, num_steps=150)
    
    # Test print sample
    sample_df = generator.generate_patient_series(patient_id="PAT-102", trajectory_type="septic_shock", num_steps=10)
    print("\nSample Generated Septic Shock Sequence:")
    print(sample_df[['step', 'patient_id', 'heart_rate', 'spo2', 'systolic_bp', 'respiratory_rate', 'body_temperature']])
