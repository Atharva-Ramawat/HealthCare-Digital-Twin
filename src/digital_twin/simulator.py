"""
Temporal Vitals Simulator for ICU Patients.
Generates 15-minute resolution trajectory data over a 24-hour monitoring window (96 steps).
Simulates key ICU vital signs: Heart Rate (HR), SpO2, Systolic Blood Pressure (SBP), and Respiratory Rate (RR).
"""

import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class VitalReading:
    """Single 15-minute interval vital sign measurement."""
    step_index: int
    timestamp: datetime
    heart_rate: float
    spo2: float
    sbp: float
    respiratory_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "timestamp": self.timestamp.isoformat(),
            "heart_rate": round(self.heart_rate, 1),
            "spo2": round(self.spo2, 1),
            "sbp": round(self.sbp, 1),
            "respiratory_rate": round(self.respiratory_rate, 1)
        }


class VitalsSimulator:
    """
    Simulates high-fidelity temporal vital sign dynamics over a 24-hour timeline at 15-minute intervals.
    Generates realistic physiological trajectories including stable, deteriorating, and recovering dynamics.
    """

    TOTAL_STEPS: int = 96  # 24 hours * 4 steps per hour (15-min intervals)
    INTERVAL_MINUTES: int = 15

    # Standard Adult Physiological Baselines and Bounds
    BASELINES = {
        "heart_rate": 72.0,        # bpm
        "spo2": 98.0,              # %
        "sbp": 120.0,              # mmHg
        "respiratory_rate": 16.0    # breaths/min
    }

    BOUNDS = {
        "heart_rate": (35.0, 220.0),
        "spo2": (60.0, 100.0),
        "sbp": (50.0, 240.0),
        "respiratory_rate": (6.0, 60.0)
    }

    def __init__(self, default_seed: Optional[int] = None):
        self.default_seed = default_seed

    def _derive_seed_and_type(self, patient_id: str, trajectory_type: str = "auto") -> tuple[int, str]:
        """Derive reproducible pseudo-random seed and clinical trajectory type from patient_id."""
        hash_digest = hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()
        seed = int(hash_digest[:8], 16)

        if trajectory_type == "auto":
            # Deterministically partition patients into clinical profiles
            profile_idx = seed % 3
            if profile_idx == 0:
                resolved_type = "deteriorating"
            elif profile_idx == 1:
                resolved_type = "recovering"
            else:
                resolved_type = "stable"
        else:
            resolved_type = trajectory_type.lower()

        return seed, resolved_type

    def generate_24h_trajectory(
        self,
        patient_id: str,
        trajectory_type: str = "auto",
        end_time: Optional[datetime] = None,
        seed: Optional[int] = None
    ) -> List[VitalReading]:
        """
        Generate 96 discrete 15-minute vital sign observations spanning the 24 hours preceding end_time.
        
        Args:
            patient_id: Identifier for patient to ensure reproducible deterministic trajectories.
            trajectory_type: 'stable', 'deteriorating', 'recovering', or 'auto'.
            end_time: Final observation timestamp (defaults to current UTC time).
            seed: Optional integer seed overriding the patient-derived seed.
            
        Returns:
            List of 96 VitalReading objects.
        """
        derived_seed, resolved_type = self._derive_seed_and_type(patient_id, trajectory_type)
        rng_seed = seed if seed is not None else derived_seed
        rng = np.random.RandomState(rng_seed)

        if end_time is None:
            end_time = datetime.utcnow()

        start_time = end_time - timedelta(minutes=self.INTERVAL_MINUTES * (self.TOTAL_STEPS - 1))
        steps = np.arange(self.TOTAL_STEPS)
        progress = steps / float(self.TOTAL_STEPS - 1)  # Linear progression 0.0 -> 1.0

        # Base noise vectors
        hr_noise = rng.normal(0.0, 1.8, self.TOTAL_STEPS)
        spo2_noise = rng.normal(0.0, 0.4, self.TOTAL_STEPS)
        sbp_noise = rng.normal(0.0, 2.5, self.TOTAL_STEPS)
        rr_noise = rng.normal(0.0, 0.7, self.TOTAL_STEPS)

        # Baseline starting values with patient-specific baseline shifts
        patient_hr_offset = rng.uniform(-5.0, 5.0)
        patient_sbp_offset = rng.uniform(-8.0, 8.0)
        patient_rr_offset = rng.uniform(-1.5, 1.5)

        base_hr = self.BASELINES["heart_rate"] + patient_hr_offset
        base_spo2 = self.BASELINES["spo2"]
        base_sbp = self.BASELINES["sbp"] + patient_sbp_offset
        base_rr = self.BASELINES["respiratory_rate"] + patient_rr_offset

        if resolved_type == "deteriorating":
            # Progressive deterioration: acute tachycardia, hypoxia, hypotension, tachypnea
            drift = np.power(progress, 1.8)
            hr_series = base_hr + drift * 48.0 + hr_noise
            spo2_series = base_spo2 - drift * 12.0 + spo2_noise
            sbp_series = base_sbp - drift * 36.0 + sbp_noise
            rr_series = base_rr + drift * 15.0 + rr_noise

        elif resolved_type == "recovering":
            # Initially unstable, progressively normalizing to stable ICU baseline
            drift = 1.0 - np.power(progress, 1.2)
            hr_series = base_hr + drift * 35.0 + hr_noise
            spo2_series = (base_spo2 - 7.0) + (1.0 - drift) * 7.0 + spo2_noise
            sbp_series = (base_sbp - 22.0) + (1.0 - drift) * 22.0 + sbp_noise
            rr_series = base_rr + drift * 10.0 + rr_noise

        else:  # "stable"
            # Circadian sinusoidal rhythm + slight stationary random walk
            circadian = np.sin(progress * 2.0 * np.pi)
            hr_series = base_hr + circadian * 4.0 + hr_noise
            spo2_series = base_spo2 - np.abs(circadian * 0.8) + spo2_noise
            sbp_series = base_sbp + circadian * 6.0 + sbp_noise
            rr_series = base_rr + circadian * 1.5 + rr_noise

        # Clamp values to physiological limits
        hr_clamped = np.clip(hr_series, self.BOUNDS["heart_rate"][0], self.BOUNDS["heart_rate"][1])
        spo2_clamped = np.clip(spo2_series, self.BOUNDS["spo2"][0], self.BOUNDS["spo2"][1])
        sbp_clamped = np.clip(sbp_series, self.BOUNDS["sbp"][0], self.BOUNDS["sbp"][1])
        rr_clamped = np.clip(rr_series, self.BOUNDS["respiratory_rate"][0], self.BOUNDS["respiratory_rate"][1])

        readings: List[VitalReading] = []
        for i in range(self.TOTAL_STEPS):
            timestamp = start_time + timedelta(minutes=self.INTERVAL_MINUTES * i)
            readings.append(
                VitalReading(
                    step_index=i,
                    timestamp=timestamp,
                    heart_rate=round(float(hr_clamped[i]), 1),
                    spo2=round(float(spo2_clamped[i]), 1),
                    sbp=round(float(sbp_clamped[i]), 1),
                    respiratory_rate=round(float(rr_clamped[i]), 1)
                )
            )

        return readings

    def get_latest_vitals(
        self,
        patient_id: str,
        trajectory_type: str = "auto",
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to retrieve the single most recent vital signs snapshot for a patient.
        
        Returns:
            Dictionary with keys 'heart_rate', 'spo2', 'sbp', 'respiratory_rate', 'timestamp'.
        """
        trajectory = self.generate_24h_trajectory(patient_id, trajectory_type, end_time)
        latest = trajectory[-1]
        return {
            "heart_rate": latest.heart_rate,
            "spo2": latest.spo2,
            "sbp": latest.sbp,
            "respiratory_rate": latest.respiratory_rate,
            "timestamp": latest.timestamp
        }

    def generate_trajectory_around_vitals(
        self,
        target_vitals: Dict[str, float],
        patient_id: str = "custom",
        end_time: Optional[datetime] = None,
        seed: Optional[int] = None
    ) -> List[VitalReading]:
        """
        Generate a 96-step (24h) baseline trajectory whose final timestep strictly
        aligns with the provided target vital signs, simulating smooth physiological variations
        converging on the target observation.
        """
        if end_time is None:
            end_time = datetime.utcnow()

        start_time = end_time - timedelta(minutes=self.INTERVAL_MINUTES * (self.TOTAL_STEPS - 1))
        steps = np.arange(self.TOTAL_STEPS)
        progress = steps / float(self.TOTAL_STEPS - 1)  # 0.0 to 1.0

        derived_seed, _ = self._derive_seed_and_type(patient_id, "auto")
        rng = np.random.RandomState(seed if seed is not None else derived_seed)

        target_hr = float(target_vitals["heart_rate"])
        target_spo2 = float(target_vitals["spo2"])
        target_sbp = float(target_vitals["sbp"])
        target_rr = float(target_vitals["respiratory_rate"])

        # Subtle noise converging to 0 at the final timestep
        noise_decay = 1.0 - progress
        hr_noise = rng.normal(0.0, 1.2, self.TOTAL_STEPS) * noise_decay
        spo2_noise = rng.normal(0.0, 0.3, self.TOTAL_STEPS) * noise_decay
        sbp_noise = rng.normal(0.0, 2.0, self.TOTAL_STEPS) * noise_decay
        rr_noise = rng.normal(0.0, 0.5, self.TOTAL_STEPS) * noise_decay

        # Circadian baseline shift that settles into target_vitals
        circadian = np.sin(progress * 2.0 * np.pi) * noise_decay
        hr_series = target_hr + circadian * 4.0 + hr_noise
        spo2_series = target_spo2 + circadian * 0.8 + spo2_noise
        sbp_series = target_sbp + circadian * 6.0 + sbp_noise
        rr_series = target_rr + circadian * 1.5 + rr_noise

        # Clamp to bounds
        hr_clamped = np.clip(hr_series, self.BOUNDS["heart_rate"][0], self.BOUNDS["heart_rate"][1])
        spo2_clamped = np.clip(spo2_series, self.BOUNDS["spo2"][0], self.BOUNDS["spo2"][1])
        sbp_clamped = np.clip(sbp_series, self.BOUNDS["sbp"][0], self.BOUNDS["sbp"][1])
        rr_clamped = np.clip(rr_series, self.BOUNDS["respiratory_rate"][0], self.BOUNDS["respiratory_rate"][1])

        # Enforce exact target values on the final timestep
        hr_clamped[-1] = target_hr
        spo2_clamped[-1] = target_spo2
        sbp_clamped[-1] = target_sbp
        rr_clamped[-1] = target_rr

        readings: List[VitalReading] = []
        for i in range(self.TOTAL_STEPS):
            timestamp = start_time + timedelta(minutes=self.INTERVAL_MINUTES * i)
            readings.append(
                VitalReading(
                    step_index=i,
                    timestamp=timestamp,
                    heart_rate=round(float(hr_clamped[i]), 1),
                    spo2=round(float(spo2_clamped[i]), 1),
                    sbp=round(float(sbp_clamped[i]), 1),
                    respiratory_rate=round(float(rr_clamped[i]), 1)
                )
            )
        return readings
