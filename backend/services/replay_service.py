"""
Replay Service - Asynchronous Clinical Data Replay & Streaming Engine.
Orchestrates virtual ICU patient vitals, sliding window buffers, playback speed controls, and anomaly shocks.
"""

import asyncio
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, AsyncGenerator
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.schemas.monitoring import (
    VitalsFrame,
    SlidingWindowVitals,
    SimulationStateResponse,
    StreamVitalsMessage
)

# Patient Profiles Metadata
DEFAULT_PATIENTS = [
    {
        "patient_id": "PAT-101",
        "name": "Elena Rostova",
        "age": 62,
        "gender": "F",
        "bed": "ICU Bed 01",
        "admission_time": "2026-08-20 06:30:00",
        "primary_diagnosis": "Post-operative Respiratory Monitoring",
        "trajectory": "normal",
        "medical_history": ["Hypertension", "Type 2 Diabetes"],
        "allergies": ["Penicillin"],
        "attending_physician": "Dr. A. Sharma (ICU Lead)"
    },
    {
        "patient_id": "PAT-102",
        "name": "Marcus Vance",
        "age": 54,
        "gender": "M",
        "bed": "ICU Bed 02",
        "admission_time": "2026-08-21 14:10:00",
        "primary_diagnosis": "Severe Bacterial Pneumonia with Sepsis",
        "trajectory": "septic_shock",
        "medical_history": ["COPD", "Tobacco Use Disorder"],
        "allergies": ["Sulfa Drugs"],
        "attending_physician": "Dr. A. Sharma (ICU Lead)"
    },
    {
        "patient_id": "PAT-103",
        "name": "Sarah Jenkins",
        "age": 71,
        "gender": "F",
        "bed": "ICU Bed 03",
        "admission_time": "2026-08-22 01:45:00",
        "primary_diagnosis": "Acute Respiratory Distress Syndrome (ARDS)",
        "trajectory": "ards",
        "medical_history": ["Asthma", "Coronary Artery Disease"],
        "allergies": ["No known allergies"],
        "attending_physician": "Dr. K. Patel (Pulmonary Specialist)"
    },
    {
        "patient_id": "PAT-104",
        "name": "David Chen",
        "age": 48,
        "gender": "M",
        "bed": "ICU Bed 04",
        "admission_time": "2026-08-22 09:00:00",
        "primary_diagnosis": "Post-Thoracic Surgery Recovery",
        "trajectory": "normal",
        "medical_history": ["Hyperlipidemia"],
        "allergies": ["Latex"],
        "attending_physician": "Dr. A. Sharma (ICU Lead)"
    }
]

VITALS_CONFIG = {
    'heart_rate': {'baseline': 75.0, 'min': 30.0, 'max': 200.0},
    'spo2': {'baseline': 98.0, 'min': 65.0, 'max': 100.0},
    'systolic_bp': {'baseline': 120.0, 'min': 50.0, 'max': 220.0},
    'diastolic_bp': {'baseline': 80.0, 'min': 30.0, 'max': 140.0},
    'respiratory_rate': {'baseline': 16.0, 'min': 6.0, 'max': 50.0},
    'body_temperature': {'baseline': 37.0, 'min': 33.0, 'max': 43.0}
}


class ClinicalReplayService:
    """
    Service managing real-time data playback, sliding window buffers, and anomaly simulation.
    """

    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.is_playing = True
        self.speed_multiplier = 1.0
        self.mode = "simulation"  # "simulation", "mimic_replay", "inference"
        self.current_step = 0
        self.start_sim_time = datetime(2026, 8, 22, 8, 0, 0)
        
        self.patients_meta = {p["patient_id"]: p for p in DEFAULT_PATIENTS}
        self.buffers: Dict[str, deque] = {}
        self.history_records: Dict[str, List[Dict]] = {}
        self.active_anomalies: Dict[str, List[str]] = {}
        self.what_if_overrides: Dict[str, Dict[str, float]] = {}

        self._initialize_synthetic_streams()

    def _generate_synthetic_point(self, patient_id: str, step: int) -> Dict[str, float]:
        """Generate a single physiological vital observation point."""
        profile = self.patients_meta[patient_id]
        traj = profile.get("trajectory", "normal")
        
        # Circadian sine wave & stochastic noise
        circ = np.sin(2 * np.pi * step / 120.0)
        noise_hr = np.random.normal(0, 1.2)
        noise_spo2 = np.random.normal(0, 0.3)
        noise_sbp = np.random.normal(0, 1.8)
        noise_rr = np.random.normal(0, 0.6)
        noise_temp = np.random.normal(0, 0.05)

        hr = VITALS_CONFIG['heart_rate']['baseline'] + 3.0 * circ + noise_hr
        spo2 = VITALS_CONFIG['spo2']['baseline'] - 0.2 * np.abs(circ) + noise_spo2
        sbp = VITALS_CONFIG['systolic_bp']['baseline'] + 4.0 * circ + noise_sbp
        dbp = VITALS_CONFIG['diastolic_bp']['baseline'] + 2.0 * circ + noise_sbp * 0.5
        rr = VITALS_CONFIG['respiratory_rate']['baseline'] + 0.8 * circ + noise_rr
        temp = VITALS_CONFIG['body_temperature']['baseline'] + 0.15 * circ + noise_temp

        # Apply Trajectory Alterations
        if traj == "septic_shock" and step > 20:
            ramp = min(1.0, (step - 20) / 40.0)
            hr += ramp * 55.0
            sbp -= ramp * 42.0
            dbp -= ramp * 25.0
            temp += ramp * 2.5
            rr += ramp * 14.0
            spo2 -= ramp * 5.0
        elif traj == "ards" and step > 15:
            ramp = min(1.0, (step - 15) / 35.0)
            spo2 -= ramp * 16.0
            rr += ramp * 20.0
            hr += ramp * 35.0
            sbp += ramp * 10.0

        # Apply Injected Manual Shock Anomalies
        anomalies = self.active_anomalies.get(patient_id, [])
        for anom in anomalies:
            if anom == "septic_spike":
                hr += 35.0
                sbp -= 30.0
                dbp -= 18.0
                temp += 2.2
                rr += 10.0
            elif anom == "hypoxia_drop":
                spo2 -= 14.0
                rr += 12.0
                hr += 25.0
            elif anom == "cardiac_arrhythmia":
                hr += 50.0
                sbp += 25.0

        # Apply What-If Overrides if present for the latest step
        overrides = self.what_if_overrides.get(patient_id, {})
        for k, v in overrides.items():
            if k == "heart_rate": hr = v
            elif k == "spo2": spo2 = v
            elif k == "systolic_bp": sbp = v
            elif k == "diastolic_bp": dbp = v
            elif k == "respiratory_rate": rr = v
            elif k == "body_temperature": temp = v

        # Physiological Bounds Clipping
        hr = float(np.clip(hr, VITALS_CONFIG['heart_rate']['min'], VITALS_CONFIG['heart_rate']['max']))
        spo2 = float(np.clip(spo2, VITALS_CONFIG['spo2']['min'], VITALS_CONFIG['spo2']['max']))
        sbp = float(np.clip(sbp, VITALS_CONFIG['systolic_bp']['min'], VITALS_CONFIG['systolic_bp']['max']))
        dbp = float(np.clip(dbp, VITALS_CONFIG['diastolic_bp']['min'], VITALS_CONFIG['diastolic_bp']['max']))
        rr = float(np.clip(rr, VITALS_CONFIG['respiratory_rate']['min'], VITALS_CONFIG['respiratory_rate']['max']))
        temp = float(np.clip(temp, VITALS_CONFIG['body_temperature']['min'], VITALS_CONFIG['body_temperature']['max']))

        # Derived metrics
        map_val = dbp + (sbp - dbp) / 3.0
        shock_idx = hr / sbp if sbp > 0 else 0.0

        obs_time = (self.start_sim_time + timedelta(minutes=step)).strftime("%Y-%m-%d %H:%M:%S")

        return {
            "timestamp": obs_time,
            "step": step,
            "heart_rate": round(hr, 2),
            "spo2": round(spo2, 2),
            "systolic_bp": round(sbp, 2),
            "diastolic_bp": round(dbp, 2),
            "respiratory_rate": round(rr, 2),
            "body_temperature": round(temp, 2),
            "mean_arterial_pressure": round(map_val, 2),
            "shock_index": round(shock_idx, 3)
        }

    def _initialize_synthetic_streams(self):
        """Seed rolling buffers with initial window observations."""
        self.buffers.clear()
        self.history_records.clear()
        self.active_anomalies.clear()
        self.what_if_overrides.clear()
        
        for pid in self.patients_meta:
            self.buffers[pid] = deque(maxlen=self.window_size)
            self.history_records[pid] = []
            self.active_anomalies[pid] = []
            
            for s in range(self.window_size):
                pt = self._generate_synthetic_point(pid, s)
                self.buffers[pid].append(pt)
                self.history_records[pid].append(pt)

        self.current_step = self.window_size

    def step(self) -> Dict[str, VitalsFrame]:
        """Advance simulation by 1 time-step across all patients."""
        if self.is_playing:
            self.current_step += 1
            
        results = {}
        for pid in self.patients_meta:
            if self.is_playing:
                pt = self._generate_synthetic_point(pid, self.current_step)
                self.buffers[pid].append(pt)
                self.history_records[pid].append(pt)
            else:
                pt = self.buffers[pid][-1]
                
            results[pid] = VitalsFrame(**pt)
            
        return results

    def get_latest_vitals(self, patient_id: str) -> Optional[VitalsFrame]:
        """Retrieve latest vital readings for a patient."""
        if patient_id in self.buffers and len(self.buffers[patient_id]) > 0:
            frame_dict = dict(self.buffers[patient_id][-1])
            overrides = self.what_if_overrides.get(patient_id, {})
            for k, v in overrides.items():
                if k in frame_dict:
                    frame_dict[k] = float(v)
            if 'systolic_bp' in frame_dict and 'diastolic_bp' in frame_dict:
                frame_dict['mean_arterial_pressure'] = round(frame_dict['diastolic_bp'] + (frame_dict['systolic_bp'] - frame_dict['diastolic_bp'])/3.0, 2)
            if 'heart_rate' in frame_dict and 'systolic_bp' in frame_dict and frame_dict['systolic_bp'] > 0:
                frame_dict['shock_index'] = round(frame_dict['heart_rate'] / frame_dict['systolic_bp'], 3)
            return VitalsFrame(**frame_dict)
        return None

    def get_sliding_window(self, patient_id: str) -> SlidingWindowVitals:
        """Retrieve full sliding window for model inference."""
        buf = [dict(f) for f in self.buffers.get(patient_id, [])]
        overrides = self.what_if_overrides.get(patient_id, {})
        if buf and overrides:
            for k, v in overrides.items():
                if k in buf[-1]:
                    buf[-1][k] = float(v)
        timestamps = [f["timestamp"] for f in buf]
        feature_keys = ['heart_rate', 'spo2', 'systolic_bp', 'respiratory_rate', 'body_temperature']
        seq = [{k: f[k] for k in feature_keys} for f in buf]
        
        return SlidingWindowVitals(
            patient_id=patient_id,
            window_size=len(buf),
            timestamps=timestamps,
            features=feature_keys,
            sequence=seq
        )

    def get_sliding_window_matrix(self, patient_id: str) -> np.ndarray:
        """Extract (24, 5) NumPy matrix ready for PyTorch model tensor."""
        buf = [dict(f) for f in self.buffers.get(patient_id, [])]
        overrides = self.what_if_overrides.get(patient_id, {})
        if buf and overrides:
            for k, v in overrides.items():
                if k in buf[-1]:
                    buf[-1][k] = float(v)
        keys = ['heart_rate', 'spo2', 'systolic_bp', 'respiratory_rate', 'body_temperature']
        mat = np.zeros((self.window_size, len(keys)), dtype=np.float32)
        for i, frame in enumerate(buf):
            for j, k in enumerate(keys):
                mat[i, j] = frame.get(k, VITALS_CONFIG[k]['baseline'])
        return mat

    def get_patient_history(self, patient_id: str, limit: int = 100) -> List[VitalsFrame]:
        """Retrieve historical vital readings."""
        records = self.history_records.get(patient_id, [])
        tail = records[-limit:]
        return [VitalsFrame(**r) for r in tail]

    def inject_anomaly(self, patient_id: str, anomaly_type: str):
        """Inject or clear physiological perturbation."""
        if patient_id not in self.active_anomalies:
            self.active_anomalies[patient_id] = []
            
        if anomaly_type == "reset":
            self.active_anomalies[patient_id].clear()
            self.what_if_overrides.pop(patient_id, None)
        elif anomaly_type not in self.active_anomalies[patient_id]:
            self.active_anomalies[patient_id].append(anomaly_type)

    def set_what_if_overrides(self, patient_id: str, overrides: Dict[str, float]):
        """Apply counterfactual what-if slider overrides to patient."""
        self.what_if_overrides[patient_id] = overrides

    def play(self):
        self.is_playing = True

    def pause(self):
        self.is_playing = False

    def reset(self):
        self.current_step = 0
        self._initialize_synthetic_streams()

    def set_speed(self, speed: float):
        self.speed_multiplier = max(0.1, min(20.0, float(speed)))

    def set_mode(self, mode: str):
        if mode in ["simulation", "mimic_replay", "inference"]:
            self.mode = mode

    def get_simulation_state(self) -> SimulationStateResponse:
        return SimulationStateResponse(
            is_playing=self.is_playing,
            speed_multiplier=self.speed_multiplier,
            mode=self.mode,
            current_step=self.current_step,
            active_anomalies=self.active_anomalies,
            status_message="Simulation active" if self.is_playing else "Simulation paused"
        )
